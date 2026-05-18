"""
batch.py

Multi-patient batch orchestration for the Poisson ridge encoding pipeline.
Wraps the core regression + reliability steps from neural_encoding.regression
and neural_encoding.reliability into patient-level and population-level runners.
"""

from __future__ import annotations

import os
import pickle
import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# Self-other beta correlation
# ---------------------------------------------------------------------------

def compute_self_other_beta_corr_from_results(results: Dict, region: str) -> pd.DataFrame:
    """
    Compute per-neuron Pearson correlation between self and other beta vectors.

    Parameters
    ----------
    results : dict
        Output from a patient-level regression run. Expected keys:
        ``{region}_self`` and ``{region}_other``, each containing a ``"coef"``
        DataFrame of shape (n_neurons × n_features).
    region : str
        Brain region name (e.g. ``"hippocampus"`` or ``"ACC"``).

    Returns
    -------
    pd.DataFrame with columns: neuron_index, true_corr, region.
    """
    key_s = f"{region}_self"
    key_o = f"{region}_other"
    if key_s not in results or key_o not in results:
        raise ValueError(f"Missing {key_s} or {key_o} in results.")

    df_self = results[key_s]["coef"]
    df_other = results[key_o]["coef"]

    common = df_self.index.intersection(df_other.index)
    df_self = df_self.loc[common].sort_index()
    df_other = df_other.loc[common].sort_index()

    if df_self.empty or df_other.empty:
        raise ValueError(f"No overlapping neurons in {region}.")

    corrs = [pearsonr(df_self.loc[i], df_other.loc[i])[0] for i in df_self.index]
    return pd.DataFrame({
        "neuron_index": df_self.index,
        "true_corr": corrs,
        "region": region.upper(),
    })


# ---------------------------------------------------------------------------
# NaN cleaning helper
# ---------------------------------------------------------------------------

def clean_XY(
    X: np.ndarray,
    Y: np.ndarray,
    label: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Drop rows that contain NaNs in either X or Y.

    Returns
    -------
    X_clean, Y_clean, keep_mask (bool array of length n_rows)
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"Shape mismatch{' for ' + label if label else ''}: "
            f"X rows ({X.shape[0]}) != Y rows ({Y.shape[0]})."
        )
    nan_mask = np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1)
    keep = ~nan_mask
    return X[keep], Y[keep], keep


# ---------------------------------------------------------------------------
# Single-patient runner (high-level)
# ---------------------------------------------------------------------------

def run_one_patient(
    patient_ID: str,
    embedding_file: str,
    duration_file: str,
    spike_base_dir: str,
    regions: Sequence[str],
    speakers: Sequence[str],
    *,
    get_self_and_other_features,   # callable from your pipeline
    load_spike_data,               # callable from your pipeline
    build_Y_matrices,              # callable from your pipeline
    run_all_conditions_for_patient,  # callable from your pipeline
    run_beta_reliability_all_neurons,  # callable from your pipeline
    target_speaker: str = "SPK1",
    n_components: int = 30,
    n_iterations: int = 10,
    n_jobs_reliability: int = 8,
    n_nulls_reliability: int = 100,
    alphas: Optional[np.ndarray] = None,
    results_root: str = "results",
    model_type: str = "BERT",
) -> Optional[Dict]:
    """
    End-to-end regression + reliability pipeline for one patient.

    Returns the results dict on success, or None if a required file is missing.
    """
    if alphas is None:
        alphas = np.logspace(-3, 2, 10)

    print(f"\n{'='*40}")
    print(f"Patient: {patient_ID}  ({model_type})")
    print(f"{'='*40}")

    for path in (embedding_file, duration_file):
        if not os.path.isfile(path):
            print(f"  Skipping {patient_ID}: missing file {path}")
            return None
    if not os.path.isdir(spike_base_dir):
        print(f"  Skipping {patient_ID}: missing spike dir {spike_base_dir}")
        return None

    # --- load X ---
    X_self, X_other, df_metadata, *_ = get_self_and_other_features(
        embedding_file, duration_file,
        n_components=n_components,
        target_speaker=target_speaker,
    )
    print(f"  X_self: {X_self.shape}  X_other: {X_other.shape}")

    # --- load spikes ---
    spike_data = load_spike_data(
        speakers=list(speakers),
        regions=list(regions),
        self_base_dir=spike_base_dir,
        mode="auto",
    )
    if not spike_data:
        print(f"  Skipping {patient_ID}: no spike data.")
        return None

    # --- build Y ---
    region_Ys = build_Y_matrices(df_metadata, spike_data, list(regions),
                                 target_speaker=target_speaker)
    if not region_Ys:
        print(f"  Skipping {patient_ID}: no region_Ys.")
        return None

    # --- clean X/Y and build dicts ---
    X_dict: Dict[str, np.ndarray] = {}
    Y_dict: Dict[str, np.ndarray] = {}

    for region in regions:
        for cond in ("self", "other"):
            key = f"{region}_{cond}"
            Y = region_Ys.get(region, {}).get(cond)
            X = X_self if cond == "self" else X_other
            if Y is None:
                continue
            try:
                Xc, Yc, _ = clean_XY(X, Y, label=key)
            except ValueError as e:
                print(f"  Skipping {key}: {e}")
                continue
            if Xc.shape[0] == 0:
                print(f"  Skipping {key}: all NaNs after cleaning.")
                continue
            X_dict[key] = Xc
            Y_dict[key] = Yc
            print(f"  {key}: X={Xc.shape}  Y={Yc.shape}")

    if not X_dict:
        print(f"  Skipping {patient_ID}: no valid conditions.")
        return None

    # --- regression ---
    results = run_all_conditions_for_patient(
        X_dict=X_dict,
        Y_dict=Y_dict,
        patient_id=patient_ID,
        region_list=list(regions),
        n_semantic_dims=n_components,
        n_iterations=n_iterations,
    )

    # --- save regression results ---
    patient_dir = os.path.join(results_root, patient_ID)
    os.makedirs(patient_dir, exist_ok=True)
    results_path = os.path.join(patient_dir, "ALL_CONDITIONS_RESULTS.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"  Regression saved: {results_path}")

    # --- beta correlations ---
    for region in regions:
        try:
            corr_df = compute_self_other_beta_corr_from_results(results, region)
            corr_csv = os.path.join(patient_dir, f"{region.upper()}_SELF_OTHER_CORRELATION.csv")
            corr_df.to_csv(corr_csv, index=False)
            print(f"  Beta corr saved: {corr_csv}")
        except Exception as e:
            print(f"  Beta corr skipped for {region}: {e}")

    # --- reliability ---
    for region in regions:
        ks = f"{region}_self"
        ko = f"{region}_other"
        if ks not in X_dict or ko not in X_dict:
            continue
        if X_dict[ks].shape[0] == 0 or X_dict[ko].shape[0] == 0:
            continue
        print(f"  Running reliability for {region}")
        try:
            rel_results = run_beta_reliability_all_neurons(
                X_self=X_dict[ks],
                X_other=X_dict[ko],
                Y_self=Y_dict[ks],
                Y_other=Y_dict[ko],
                alphas=alphas,
                n_nulls=n_nulls_reliability,
                neuron_mode="all",
                n_jobs=n_jobs_reliability,
            )
            rel_path = os.path.join(patient_dir, f"{region.upper()}_RELIABILITY_RESULTS.pkl")
            with open(rel_path, "wb") as f:
                pickle.dump(rel_results, f)
            print(f"  Reliability saved: {rel_path}")
        except Exception as e:
            print(f"  Reliability failed for {region}: {e}")

    return results


# ---------------------------------------------------------------------------
# Multi-patient batch runner
# ---------------------------------------------------------------------------

def run_batch_across_patients(
    patients_to_run: List[Dict],
    regions: Sequence[str],
    speakers: Sequence[str],
    *,
    get_self_and_other_features,
    load_spike_data,
    build_Y_matrices,
    run_all_conditions_for_patient,
    run_beta_reliability_all_neurons,
    model_type: str = "BERT",
    target_speaker: str = "SPK1",
    n_iterations: int = 10,
    n_jobs_reliability: int = 8,
    n_nulls_reliability: int = 100,
    alphas: Optional[np.ndarray] = None,
    results_root: str = "results",
    path_resolver=None,
) -> Dict[str, Optional[Dict]]:
    """
    Run the full encoding pipeline across a list of patients.

    Parameters
    ----------
    patients_to_run : list of dicts, each with keys:
        - ``patient_ID``  (str)  e.g. ``"PTYFF_task17"``
        - ``patient``     (str)  e.g. ``"ptYFF_task17"``  (used for file paths)
        - ``n_components`` (int) number of PCs
    path_resolver : callable(patient_ID, patient, model_type) -> dict, optional
        Returns a dict with keys ``embedding_file``, ``duration_file``,
        ``spike_base_dir``.  If None you must pre-populate those keys in each
        patient dict.

    Returns
    -------
    dict keyed by patient_ID with per-patient results (or None on failure).
    """
    all_results: Dict[str, Optional[Dict]] = {}

    for cfg in patients_to_run:
        pid = cfg["patient_ID"]
        patient = cfg.get("patient", pid)
        n_components = cfg.get("n_components", 30)

        if path_resolver is not None:
            paths = path_resolver(pid, patient, model_type)
        else:
            paths = {
                "embedding_file": cfg["embedding_file"],
                "duration_file": cfg["duration_file"],
                "spike_base_dir": cfg["spike_base_dir"],
            }

        try:
            result = run_one_patient(
                patient_ID=pid,
                embedding_file=paths["embedding_file"],
                duration_file=paths["duration_file"],
                spike_base_dir=paths["spike_base_dir"],
                regions=regions,
                speakers=speakers,
                get_self_and_other_features=get_self_and_other_features,
                load_spike_data=load_spike_data,
                build_Y_matrices=build_Y_matrices,
                run_all_conditions_for_patient=run_all_conditions_for_patient,
                run_beta_reliability_all_neurons=run_beta_reliability_all_neurons,
                target_speaker=target_speaker,
                n_components=n_components,
                n_iterations=n_iterations,
                n_jobs_reliability=n_jobs_reliability,
                n_nulls_reliability=n_nulls_reliability,
                alphas=alphas,
                results_root=results_root,
                model_type=model_type,
            )
            all_results[pid] = result
        except Exception as e:
            print(f"\nFAILED {pid}: {e}")
            all_results[pid] = None

    return all_results
