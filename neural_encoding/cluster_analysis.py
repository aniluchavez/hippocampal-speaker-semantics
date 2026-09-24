"""
cluster_analysis.py

Semantic cluster analysis for the hippocampal speaker-semantics pipeline.

Provides:
- compute_trial_activations        : predicted neural activations per trial
- compute_cluster_cosine_distances_from_activations : cluster-mean cosine similarity
- clean_and_sample_words           : word sampling per cluster (for labelling)
- plot_cosine_distances_rainbow_annotated : bar-plot of cosine distances by cluster
- run_clusterwise_cosine_distance  : per-neuron Poisson-ridge betas + cosine
                                     distance between self and other, by semantic
                                     cluster (most recent version with balancing,
                                     capping, optional half-splits, joblib parallel)
- report_cluster_balance           : chi-squared diagnostic of trial balance
- minimal_balancing                : iterative downsampling to chi-sq balance
"""

from __future__ import annotations

import os
import pickle
import random
import re
import warnings
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import cosine as cosine_distance
from scipy.stats import chi2_contingency
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Trial activations from model coefficients
# ---------------------------------------------------------------------------

def compute_trial_activations(
    results: Dict,
    X_dict: Dict[str, np.ndarray],
) -> Dict[str, pd.DataFrame]:
    """
    Compute per-trial predicted activations for all conditions in ``results``.

    Parameters
    ----------
    results : dict
        Output of ``run_all_conditions_for_patient``.  Each value must have a
        ``"coef"`` DataFrame of shape (n_neurons × n_features).
    X_dict : dict
        Maps condition keys (e.g. ``"ACC_self"``) to design matrices
        (n_trials × n_features).

    Returns
    -------
    dict mapping condition keys to DataFrames of shape (n_trials × n_neurons).
    """
    activations: Dict[str, pd.DataFrame] = {}
    for key in results:
        X = X_dict[key]
        B = results[key]["coef"].values.T          # (n_features × n_neurons)
        act = X @ B                                  # (n_trials × n_neurons)
        activations[key] = pd.DataFrame(
            act,
            columns=[f"Neuron_{i}" for i in range(act.shape[1])],
        )
    return activations


# ---------------------------------------------------------------------------
# Cluster-mean cosine distances from activations
# ---------------------------------------------------------------------------

def compute_cluster_cosine_distances_from_activations(
    metadata_self: pd.DataFrame,
    metadata_other: pd.DataFrame,
    activations_dict: Dict[str, pd.DataFrame],
    condition_prefix: str = "hippocampus",
    cluster_col: str = "clusID",
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    For each cluster compute cosine similarity between the self and other
    mean activation vectors.

    Returns
    -------
    (cosine_df, vec_self_last, vec_other_last)
        cosine_df has columns: clusID, cosine_similarity, cosine_distance.
    """
    key_self = f"{condition_prefix}_self"
    key_other = f"{condition_prefix}_other"

    merged_self = pd.concat(
        [metadata_self.reset_index(drop=True), activations_dict[key_self]], axis=1
    )
    merged_other = pd.concat(
        [metadata_other.reset_index(drop=True), activations_dict[key_other]], axis=1
    )

    neuron_cols = [c for c in merged_self.columns if c.startswith("Neuron_")]
    cluster_ids = sorted(
        set(merged_self[cluster_col]).intersection(set(merged_other[cluster_col]))
    )

    rows = []
    vec_s = vec_o = None
    for cid in cluster_ids:
        vec_s = merged_self[merged_self[cluster_col] == cid][neuron_cols].mean().values.reshape(1, -1)
        vec_o = merged_other[merged_other[cluster_col] == cid][neuron_cols].mean().values.reshape(1, -1)
        cos_sim = cosine_similarity(vec_s, vec_o)[0, 0]
        rows.append({"clusID": cid, "cosine_similarity": cos_sim, "cosine_distance": 1 - cos_sim})

    return pd.DataFrame(rows), vec_s, vec_o


# ---------------------------------------------------------------------------
# Word sampling helper
# ---------------------------------------------------------------------------

def clean_and_sample_words(group: pd.Series, used_words: Optional[set] = None, n: int = 5) -> str:
    """
    Return up to ``n`` unique, globally-unseen words from ``group``.

    Pass a shared ``used_words`` set to avoid repeats across clusters.
    """
    if used_words is None:
        used_words = set()
    cleaned = (
        group.dropna()
        .drop_duplicates()
        .apply(lambda w: re.sub(r"[^\w\s]", "", w).lower().strip())
    )
    available = [w for w in cleaned if w and w not in used_words]
    sampled = random.sample(available, min(n, len(available)))
    used_words.update(sampled)
    return ", ".join(sampled)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

CLUSTER_LABELS: Dict[int, str] = {
    1: "Body Parts", 2: "Places", 3: "Emotional", 4: "Mental", 5: "Social",
    6: "Objects", 7: "Visual", 8: "Numerical", 9: "Actions", 10: "Identity",
    11: "Function Words", 12: "Proper Nouns",
}


def plot_cosine_distances_rainbow_annotated(
    cosine_df: pd.DataFrame,
    title: str = "Self vs Other Cosine Distance by Cluster",
    top_n: Optional[int] = None,
    cluster_labels: Optional[Dict[int, str]] = None,
    cluster_id_col: str = "clusID",
    distance_col: str = "cosine_distance",
):
    """Bar-plot of cosine distances, one bar per semantic cluster."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if cluster_labels is None:
        cluster_labels = CLUSTER_LABELS

    df = cosine_df.sort_values(distance_col, ascending=True)
    if top_n:
        df = df.head(top_n)
    df = df.copy()
    df["Label"] = df[cluster_id_col].map(cluster_labels)

    plt.figure(figsize=(14, 6))
    palette = sns.color_palette("hls", len(df))
    sns.barplot(data=df, x="Label", y=distance_col, palette=palette)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Semantic Cluster")
    plt.ylabel("Cosine Distance (1 − similarity)")
    plt.title(title)
    sns.despine()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Cluster balance diagnostic
# ---------------------------------------------------------------------------

def report_cluster_balance(
    metadata_self: pd.DataFrame,
    metadata_other: pd.DataFrame,
    patient_ID: str,
    region: str,
    cluster_col: str = "ClusterID",
) -> Tuple[pd.DataFrame, float, float, pd.DataFrame]:
    """
    Chi-squared test of independence between cluster ID and condition (self/other).

    Prints a contingency table and standardised residuals.  Flags any cluster
    with |residual| > 2.

    Returns
    -------
    (contingency, chi2, p, residuals_df)
    """
    # ignore_index: metadata_self/metadata_other carry their own independent
    # positional indices (used elsewhere to map back into X_self/X_other);
    # concatenating them keeps overlapping labels (e.g. both start at 0),
    # which pandas' crosstab rejects ("duplicate labels") -- combined here is
    # only ever used for the contingency table, never returned.
    combined = pd.concat([
        metadata_self.assign(condition="self"),
        metadata_other.assign(condition="other"),
    ], ignore_index=True)
    contingency = pd.crosstab(combined[cluster_col], combined["condition"])
    chi2, p, _, expected = chi2_contingency(contingency)

    print(f"\n[Trial Balance] Patient: {patient_ID}  Region: {region}")
    print(contingency)
    print(f"Chi² = {chi2:.3f}  p = {p:.4f}")

    residuals = (contingency.values - expected) / np.sqrt(expected)
    residuals_df = pd.DataFrame(residuals, index=contingency.index, columns=contingency.columns)
    print("\nStandardised Residuals:")
    print(residuals_df.round(2))

    flagged = np.abs(residuals_df) > 2
    if flagged.any().any():
        print("\nProblematic clusters (|residual| > 2):")
        for cl in residuals_df.index[flagged.any(axis=1)]:
            print(f"  Cluster {cl}: {residuals_df.loc[cl].to_dict()}")

    return contingency, chi2, p, residuals_df


# ---------------------------------------------------------------------------
# Minimal balancing
# ---------------------------------------------------------------------------

def minimal_balancing(
    metadata_self: pd.DataFrame,
    metadata_other: pd.DataFrame,
    cluster_column: str = "ClusterID",
    verbose: bool = True,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Iteratively downsample the most imbalanced cluster/condition cell until
    the chi-squared test of independence is non-significant (p ≥ 0.05).

    Returns
    -------
    (metadata_self_balanced, metadata_other_balanced)
    """
    # ignore_index here too (see report_cluster_balance) -- this combined
    # frame is rebuilt fresh each loop iteration purely for the contingency
    # table; m_self/m_other (the values actually returned) keep their real
    # positional index throughout.
    combined = pd.concat([
        metadata_self.assign(condition="self"),
        metadata_other.assign(condition="other"),
    ], ignore_index=True)
    contingency = pd.crosstab(combined[cluster_column], combined["condition"])
    chi2, p, _, expected = chi2_contingency(contingency)
    if verbose:
        print(f"Initial  chi² = {chi2:.2f}  p = {p:.4f}")

    m_self = metadata_self.copy()
    m_other = metadata_other.copy()

    while p < 0.05 and not contingency.empty:
        residuals = (contingency.values - expected) / np.sqrt(expected)
        # Only downsample over-represented cells (positive residual). Picking
        # argmax(|residual|) directly can land on the under-represented cell
        # in the same row (its |residual| can exceed the over-represented
        # cell's, since they divide by different sqrt(expected)), which would
        # try to shrink the smaller side and immediately break the loop below.
        overrepresented = np.where(residuals > 0, residuals, -np.inf)
        if not np.isfinite(overrepresented).any():
            break
        r_idx, c_idx = np.unravel_index(overrepresented.argmax(), overrepresented.shape)
        problem_cluster = contingency.index[r_idx]
        problem_cond = contingency.columns[c_idx]

        if verbose:
            print(f"  Downsampling '{problem_cond}' in cluster {problem_cluster}")

        if problem_cond == "self":
            target = int(contingency.loc[problem_cluster, "other"])
            candidates = m_self[m_self[cluster_column] == problem_cluster]
            if len(candidates) <= target:
                break
            keep = candidates.sample(target, random_state=random_state).index
            m_self = pd.concat([
                m_self[m_self[cluster_column] != problem_cluster],
                m_self.loc[keep],
            ])
        else:
            target = int(contingency.loc[problem_cluster, "self"])
            candidates = m_other[m_other[cluster_column] == problem_cluster]
            if len(candidates) <= target:
                break
            keep = candidates.sample(target, random_state=random_state).index
            m_other = pd.concat([
                m_other[m_other[cluster_column] != problem_cluster],
                m_other.loc[keep],
            ])

        combined = pd.concat([
            m_self.assign(condition="self"),
            m_other.assign(condition="other"),
        ], ignore_index=True)
        contingency = pd.crosstab(combined[cluster_column], combined["condition"])
        chi2, p, _, expected = chi2_contingency(contingency)

    if verbose:
        print(f"Final    chi² = {chi2:.2f}  p = {p:.4f}")

    # Do NOT reset_index here: callers (run_clusterwise_cosine_distance) use
    # metadata_self.index / metadata_other.index as direct row positions into
    # the original X_self/Y_self/X_other/Y_other arrays. .sample()/concat above
    # preserve each surviving row's original positional index; resetting to a
    # fresh 0..M-1 RangeIndex would silently remap every downsampled cluster's
    # rows onto the WRONG positions in X/Y once any row has actually been
    # dropped (only a no-op when balancing makes no changes at all).
    return m_self, m_other


# ---------------------------------------------------------------------------
# Per-neuron Poisson-ridge clusterwise cosine distance (most recent version)
# ---------------------------------------------------------------------------

def run_clusterwise_cosine_distance(
    X_self: np.ndarray,
    X_other: np.ndarray,
    Y_self: np.ndarray,
    Y_other: np.ndarray,
    metadata_self: pd.DataFrame,
    metadata_other: pd.DataFrame,
    cluster_column: str,
    region_name: str,
    patient_id: str,
    n_components: int,
    results_root: str,
    run_poisson_ridge,   # callable: your project's run_poisson_ridge function
    *,
    function_word_cluster_id: int = 10,
    balance_function_words: bool = True,
    balance_all_clusters: bool = False,
    soft_balance: bool = False,
    soft_balance_target: str | int | None = "median",
    resample_to_balance_target: bool = False,
    min_trials_per_condition: int = 10,
    cap_large_clusters: bool = True,
    max_size_ratio: float = 5.0,
    cap_reference: str = "median",
    random_state: int = 42,
    save_csvs: bool = True,
    save_pickle: bool = True,
    compute_half_splits: bool = False,
    n_jobs: int = 4,
    print_trial_counts: bool = False,
    offset_self: "np.ndarray | None" = None,
    offset_other: "np.ndarray | None" = None,
) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
    """
    Per-neuron Poisson ridge regression within each semantic cluster,
    computing cosine distance between self-condition and other-condition beta
    vectors for each neuron.

    Supports:
    - Function-word cluster balancing / ratio capping
    - Global soft/hard balancing across all clusters
    - Optional within-condition half-split cosine reliability
    - Joblib parallelism across (cluster, neuron) tasks

    Parameters
    ----------
    X_self, X_other : np.ndarray (n_trials × n_features)
    Y_self, Y_other : np.ndarray (n_trials × n_neurons)
    metadata_self, metadata_other : pd.DataFrame
        Must contain the column ``cluster_column`` with integer cluster IDs.
    cluster_column : str
        Column name in metadata holding cluster IDs.
    region_name : str
        Used in saved file names.
    patient_id : str
        Used in saved file names.
    n_components : int
        Number of semantic PCs (passed to ``run_poisson_ridge``).
    results_root : str
        Root directory; results are saved under ``{results_root}/{patient_id}/``.
    run_poisson_ridge : callable
        Your project's ``run_poisson_ridge(X, Y, ...)`` function.

    Returns
    -------
    (cosine_df, all_betas, cluster_counts_df)
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    rng = np.random.RandomState(random_state)
    out_dir = os.path.join(results_root, patient_id)
    os.makedirs(out_dir, exist_ok=True)

    # ---- helpers ----
    def _safe_cosine(a, b):
        a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
        if a.size == 0 or b.size == 0 or np.allclose(a, 0) or np.allclose(b, 0):
            return np.nan
        return cosine_distance(a, b)

    def _target_count(counts, mode):
        counts = np.asarray(counts, dtype=int)
        counts = counts[counts > 0]
        if len(counts) == 0 or mode is None:
            return None
        if isinstance(mode, int):
            return int(mode)
        if mode == "median":
            return int(np.floor(np.median(counts)))
        if mode == "mean":
            return int(np.floor(np.mean(counts)))
        if isinstance(mode, str) and mode.startswith("q"):
            return int(np.floor(np.percentile(counts, float(mode[1:]))))
        raise ValueError(f"Unknown soft_balance_target: {mode}")

    def _reference_count(counts, mode):
        counts = np.asarray(counts, dtype=int)
        counts = counts[counts > 0]
        if len(counts) == 0:
            return None
        if mode == "median":
            return int(np.floor(np.median(counts)))
        if mode == "mean":
            return int(np.floor(np.mean(counts)))
        raise ValueError(f"Unknown cap_reference: {mode}")

    def _sample(ix, n, rng):
        ix = np.asarray(ix)
        return ix if len(ix) <= n else np.sort(rng.choice(ix, n, replace=False))

    def _sample_to_target(ix, n, rng):
        """Return exactly n indices, upsampling with replacement if needed."""
        ix = np.asarray(ix)
        if len(ix) == n:
            return ix
        if len(ix) > n:
            return np.sort(rng.choice(ix, n, replace=False))
        return np.sort(rng.choice(ix, n, replace=True))

    # ---- first pass: raw counts ----
    clusters = sorted(
        set(metadata_self[cluster_column].dropna()) |
        set(metadata_other[cluster_column].dropna())
    )
    cluster_info = []
    raw_min_counts = []
    raw_condition_counts = []

    for cid in clusters:
        ix_s = metadata_self.index[metadata_self[cluster_column] == cid].to_numpy()
        ix_o = metadata_other.index[metadata_other[cluster_column] == cid].to_numpy()
        n_min = min(len(ix_s), len(ix_o))
        cluster_info.append(dict(cluster_id=cid, ix_self=ix_s, ix_other=ix_o,
                                 n_self=len(ix_s), n_other=len(ix_o)))
        if len(ix_s) >= min_trials_per_condition and len(ix_o) >= min_trials_per_condition:
            raw_min_counts.append(n_min)
            raw_condition_counts.extend([len(ix_s), len(ix_o)])

    # ---- global balancing target ----
    global_target = None
    if soft_balance and balance_all_clusters:
        if soft_balance_target == "pooled_condition_median":
            global_target = _target_count(raw_condition_counts, "median")
        elif soft_balance_target == "pooled_condition_mean":
            global_target = _target_count(raw_condition_counts, "mean")
        else:
            global_target = _target_count(raw_min_counts, soft_balance_target)
        if global_target is not None:
            global_target = max(global_target, min_trials_per_condition)

    reference_count = _reference_count(raw_min_counts, cap_reference)
    max_allowed = None
    if cap_large_clusters and reference_count is not None:
        max_allowed = max(int(np.floor(max_size_ratio * reference_count)), min_trials_per_condition)

    # ---- second pass: build tasks ----
    tasks = []
    count_rows = []

    for info in cluster_info:
        cid = info["cluster_id"]
        ix_s, ix_o = info["ix_self"].copy(), info["ix_other"].copy()
        n_s, n_o = info["n_self"], info["n_other"]

        if n_s < min_trials_per_condition or n_o < min_trials_per_condition:
            count_rows.append(dict(cluster_id=cid, n_self_raw=n_s, n_other_raw=n_o,
                                   n_self_used=0, n_other_used=0, kept=False,
                                   drop_reason=f"below_min_{min_trials_per_condition}"))
            continue

        drop_reason = "kept"

        if balance_function_words and cid == function_word_cluster_id:
            n_t = min(n_s, n_o, max_allowed) if max_allowed else min(n_s, n_o)
            ix_s = _sample(ix_s, n_t, rng)
            ix_o = _sample(ix_o, n_t, rng)
            drop_reason = f"function_word_balanced_to_{n_t}"

        if cap_large_clusters and max_allowed is not None:
            n_cap = min(len(ix_s), len(ix_o), max_allowed)
            if n_cap < len(ix_s) or n_cap < len(ix_o):
                ix_s = _sample(ix_s, n_cap, rng)
                ix_o = _sample(ix_o, n_cap, rng)
                drop_reason = f"ratio_capped_to_{n_cap}"

        if balance_all_clusters:
            if soft_balance:
                gt = global_target or min(len(ix_s), len(ix_o))
                if resample_to_balance_target:
                    n_t = gt
                else:
                    n_t = min(len(ix_s), len(ix_o), gt)
            else:
                n_t = (min(raw_min_counts) if raw_min_counts else min(len(ix_s), len(ix_o)))
            if n_t < min_trials_per_condition:
                count_rows.append(dict(cluster_id=cid, n_self_raw=n_s, n_other_raw=n_o,
                                       n_self_used=0, n_other_used=0, kept=False,
                                       drop_reason=f"balance_target_below_min_{min_trials_per_condition}"))
                continue
            if soft_balance and resample_to_balance_target:
                ix_s = _sample_to_target(ix_s, n_t, rng)
                ix_o = _sample_to_target(ix_o, n_t, rng)
                drop_reason = f"soft_resampled_to_{n_t}"
            else:
                ix_s = _sample(ix_s, n_t, rng)
                ix_o = _sample(ix_o, n_t, rng)
                drop_reason = f"{'soft' if soft_balance else 'hard'}_balanced_to_{n_t}"

        Xs = X_self[ix_s];  Ys = Y_self[ix_s]
        Xo = X_other[ix_o]; Yo = Y_other[ix_o]
        off_s = offset_self[ix_s] if offset_self is not None else None
        off_o = offset_other[ix_o] if offset_other is not None else None
        n_su, n_ou = Xs.shape[0], Xo.shape[0]

        if n_su < min_trials_per_condition or n_ou < min_trials_per_condition:
            count_rows.append(dict(cluster_id=cid, n_self_raw=n_s, n_other_raw=n_o,
                                   n_self_used=n_su, n_other_used=n_ou, kept=False,
                                   drop_reason=f"post_sampling_below_min_{min_trials_per_condition}"))
            continue

        count_rows.append(dict(cluster_id=cid, n_self_raw=n_s, n_other_raw=n_o,
                               n_self_used=n_su, n_other_used=n_ou, kept=True,
                               drop_reason=drop_reason))

        for nidx in range(Ys.shape[1]):
            tasks.append((cid, nidx, Xs, Ys, Xo, Yo, off_s, off_o, n_su, n_ou))

    cluster_counts_df = pd.DataFrame(count_rows).sort_values("cluster_id").reset_index(drop=True)
    if print_trial_counts:
        print(f"\nCluster trial counts — {patient_id} | {region_name}")
        print(cluster_counts_df)
    if save_csvs:
        p = os.path.join(out_dir, f"{region_name}_cluster_trial_counts.csv")
        cluster_counts_df.to_csv(p, index=False)
        print(f"  Cluster counts saved: {p}")

    # ---- worker ----
    def _worker(cid, nidx, Xs, Ys, Xo, Yo, off_s, off_o, n_su, n_ou):
        try:
            ys = Ys[:, nidx:nidx+1]
            yo = Yo[:, nidx:nidx+1]

            beta_s = run_poisson_ridge(
                Xs, ys, patient_id=patient_id, neuron_idx=0,
                region_name=region_name, n_semantic_dims=n_components,
                results_dir=results_root, fast_beta_only=True, save_results=False,
                offset=off_s,
            ).filter(like="beta_").values.flatten()

            beta_o = run_poisson_ridge(
                Xo, yo, patient_id=patient_id, neuron_idx=0,
                region_name=region_name, n_semantic_dims=n_components,
                results_dir=results_root, fast_beta_only=True, save_results=False,
                offset=off_o,
            ).filter(like="beta_").values.flatten()

            dist = _safe_cosine(beta_s, beta_o)
            if not np.isfinite(dist):
                return None, None

            row: Dict = dict(cluster_id=cid, region=region_name, neuron=nidx,
                             cosine_distance=dist, n_self_used=n_su, n_other_used=n_ou)

            if compute_half_splits and Xs.shape[0] >= 4 and Xo.shape[0] >= 4:
                try:
                    is1, is2 = train_test_split(
                        np.arange(Xs.shape[0]), test_size=0.5,
                        random_state=random_state,
                    )
                    io1, io2 = train_test_split(
                        np.arange(Xo.shape[0]), test_size=0.5,
                        random_state=random_state,
                    )

                    def _beta(X, Y, offset):
                        return run_poisson_ridge(
                            X, Y, patient_id=patient_id, neuron_idx=0,
                            region_name=region_name, n_semantic_dims=n_components,
                            results_dir=results_root, fast_beta_only=True, save_results=False,
                            offset=offset,
                        ).filter(like="beta_").values.flatten()

                    bs1 = _beta(
                        Xs[is1], ys[is1],
                        off_s[is1] if off_s is not None else None,
                    )
                    bs2 = _beta(
                        Xs[is2], ys[is2],
                        off_s[is2] if off_s is not None else None,
                    )
                    bo1 = _beta(
                        Xo[io1], yo[io1],
                        off_o[io1] if off_o is not None else None,
                    )
                    bo2 = _beta(
                        Xo[io2], yo[io2],
                        off_o[io2] if off_o is not None else None,
                    )
                    row["self_halfsplit_cosine"] = 1 - _safe_cosine(bs1, bs2)
                    row["other_halfsplit_cosine"] = 1 - _safe_cosine(bo1, bo2)
                except Exception as e:
                    row["self_halfsplit_cosine"] = np.nan
                    row["other_halfsplit_cosine"] = np.nan

            beta_entry = ((cid, nidx), {"beta_self": beta_s, "beta_other": beta_o})
            return row, beta_entry

        except Exception as e:
            print(f"  Failed cluster {cid} neuron {nidx}: {e}")
            return None, None

    # ---- run ----
    raw_results = Parallel(n_jobs=n_jobs)(
        delayed(_worker)(*t) for t in tasks
    )

    all_cosine_rows = []
    all_betas: Dict = {}
    for row, beta_entry in raw_results:
        if row is not None:
            all_cosine_rows.append(row)
        if beta_entry is not None:
            all_betas[beta_entry[0]] = beta_entry[1]

    cosine_df = pd.DataFrame(all_cosine_rows)

    # ---- save ----
    if save_csvs and not cosine_df.empty:
        p = os.path.join(out_dir, f"{region_name}_clusterwise_cosine_distances.csv")
        cosine_df.to_csv(p, index=False)
        print(f"  Cosine distances saved: {p}")

    if save_pickle:
        p = os.path.join(out_dir, f"{region_name}_clusterwise_raw_betas.pkl")
        with open(p, "wb") as f:
            pickle.dump(all_betas, f)
        print(f"  Raw betas saved: {p}")

    return cosine_df, all_betas, cluster_counts_df


# ---------------------------------------------------------------------------
# Per-neuron Poisson-ridge clusterwise cosine distance, bootstrap-averaged
# ---------------------------------------------------------------------------

def run_clusterwise_cosine_distance_bootstrap(
    X_self: np.ndarray,
    X_other: np.ndarray,
    Y_self: np.ndarray,
    Y_other: np.ndarray,
    metadata_self: pd.DataFrame,
    metadata_other: pd.DataFrame,
    cluster_column: str,
    region_name: str,
    patient_id: str,
    n_components: int,
    results_root: str,
    run_poisson_ridge,   # callable: your project's run_poisson_ridge function
    *,
    min_trials_per_condition: int = 10,
    balance_self_other: bool = True,
    cross_cluster_cap: "int | str | None" = None,
    adaptive_n_components: bool = False,
    cv_splits_for_pc_rule: int = 5,
    min_n_comp: int = 2,
    n_bootstrap: int = 20,
    n_halfsplit_repeats: int = 5,
    random_state: int = 42,
    save_csvs: bool = True,
    save_pickle: bool = True,
    compute_half_splits: bool = False,
    n_jobs: int = 4,
    print_trial_counts: bool = False,
) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
    """
    Per-neuron Poisson ridge regression within each semantic cluster, computing
    cosine distance between self-condition and other-condition beta vectors.

    Unlike ``run_clusterwise_cosine_distance``, this does NOT force every
    cluster down to a shared cross-cluster size (no global median/ratio cap,
    no special-cased function-word cluster). Each cluster's balance target is
    its own natural ``min(n_self, n_other)`` -- e.g. a function-word cluster
    with 1000 self trials and 300 other trials still uses 300 self trials per
    fit (self/other must be equal-sized for a fair beta-vs-beta comparison),
    not a tiny cross-patient median, so far less data gets discarded overall.

    To avoid that 300-trial subset being one arbitrary, noisy draw: the
    SMALLER side has no spare trials, so it's fixed and fit once; the LARGER
    side is re-sampled and re-fit ``n_bootstrap`` times (different random
    subset each draw), and the reported cosine distance is the mean across
    draws -- using far more of the larger side's data over the course of the
    run than a single fixed draw would, at the cost of extra fits for the
    larger side only. Half-split reliability for both sides also gets
    averaged over ``n_halfsplit_repeats`` repeats rather than one arbitrary
    50/50 split (the large side's repeats reuse the first
    ``n_halfsplit_repeats`` bootstrap draws' already-sampled subset instead of
    drawing fresh ones, to avoid doubling the resampling cost).

    Set ``balance_self_other=False`` to skip self/other balancing entirely:
    each side is fit once on its full, natural trial count (no resampling, no
    bootstrap averaging -- n_bootstrap/n_halfsplit_repeats are then ignored
    for the large side, though the small-side half-split repeats still run).
    This maximizes data use but means self and other betas are fit on very
    different N for imbalanced clusters, so any resulting distance difference
    is confounded with that N difference unless controlled for downstream
    (e.g. via the same log(n_used) covariate approach used elsewhere).
    ``min_trials_per_condition`` (checked independently per side) still
    applies as a floor.

    ``cross_cluster_cap`` extends the same bootstrap-averaging trick to make
    cosine distances directly comparable *across* clusters too, instead of
    only controlling for cluster size after the fact via a log(n) regression
    covariate. By default (``None``) each cluster keeps its own natural
    ``min(n_self, n_other)`` as before -- clusters stay different sizes. Set
    it to ``"min_kept"`` to cap every kept cluster down to the *smallest*
    kept cluster's natural size, or to an explicit int for a specific cap.
    Either side of any cluster whose natural count exceeds the cap gets
    bootstrap-resampled down to the cap and re-fit ``n_bootstrap`` times
    (same averaging-over-many-draws principle as the self/other balance,
    applied here per-side instead of only to whichever side happens to be
    larger) -- so this is the same fix the self/other-balance bootstrap was
    for, applied one level up: no single arbitrary downsample of any cluster,
    every side just gets resampled and averaged whenever it has more data
    than the common target. Only meaningful when ``balance_self_other=True``
    (ignored otherwise, since "every cluster the same size" presupposes
    self/other are already equalized within each cluster first).

    ``adaptive_n_components`` guards against fitting more PCA dimensions
    than a cluster's trial count can support -- the same safeguard
    ``scripts/semantic_glm.py`` already applies to its own (whole-condition)
    fits (``n_comp_eff = min(n_components, n_trials // N_OUTER - 2)``,
    skipping if below 2), which this function never inherited despite
    reusing the same fitter. With the default ``n_components=100`` and
    cluster sizes as low as 10-20 trials, most clusters were being fit at
    5-50x more dimensions than trials support -- severely underdetermined
    even with ridge regularization, and liable to push betas toward
    near-random/near-orthogonal directions regardless of true signal. When
    enabled, each cluster's *final* target size (after any balancing/cap)
    determines its own ``n_comp_eff = min(X.shape[1], n_target //
    cv_splits_for_pc_rule - 2)``; the cluster's already-PCA'd feature
    columns are truncated to the leading ``n_comp_eff`` (PCA orders
    components by explained variance, so this keeps the most informative
    ones), and the cluster is dropped entirely if ``n_comp_eff < min_n_comp``.
    ``cv_splits_for_pc_rule`` should match whatever inner CV fold count the
    fitter itself uses for alpha selection (5, to match
    ``select_alpha_cv``'s default ``n_splits``).

    Returns
    -------
    (cosine_df, all_betas, cluster_counts_df)
        ``all_betas`` stores each side's fixed beta (if that side wasn't
        bootstrapped) or its *first* bootstrap draw's beta (if it was) per
        (cluster, neuron) -- enough for downstream sanity checks, not the
        full bootstrap distribution (the per-draw distances are what get
        averaged into ``cosine_distance``).
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    out_dir = os.path.join(results_root, patient_id)
    os.makedirs(out_dir, exist_ok=True)

    def _safe_cosine(a, b):
        a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
        if a.size == 0 or b.size == 0 or np.allclose(a, 0) or np.allclose(b, 0):
            return np.nan
        return cosine_distance(a, b)

    def _fit_beta(X, Y_col):
        return run_poisson_ridge(
            X, Y_col, patient_id=patient_id, neuron_idx=0,
            region_name=region_name, n_semantic_dims=n_components,
            results_dir=results_root, fast_beta_only=True, save_results=False,
        ).filter(like="beta_").values.flatten()

    def _n_comp_eff(n_target, ceiling):
        return min(ceiling, n_target // cv_splits_for_pc_rule - 2)

    # ---- first pass: natural per-cluster counts ----
    clusters = sorted(
        set(metadata_self[cluster_column].dropna()) |
        set(metadata_other[cluster_column].dropna())
    )
    cluster_info = []
    count_rows = []

    for cid in clusters:
        ix_s = metadata_self.index[metadata_self[cluster_column] == cid].to_numpy()
        ix_o = metadata_other.index[metadata_other[cluster_column] == cid].to_numpy()
        n_s, n_o = len(ix_s), len(ix_o)
        n_target_natural = min(n_s, n_o)

        if n_target_natural < min_trials_per_condition:
            count_rows.append(dict(cluster_id=cid, n_self_raw=n_s, n_other_raw=n_o,
                                   n_self_used=0, n_other_used=0, kept=False,
                                   drop_reason=f"below_min_{min_trials_per_condition}"))
            continue

        cluster_info.append(dict(cid=cid, ix_s=ix_s, ix_o=ix_o, n_s=n_s, n_o=n_o,
                                  n_target_natural=n_target_natural))

    # ---- cross-cluster cap (only meaningful when self/other are already
    # balanced within each cluster -- see docstring) ----
    cap = None
    if balance_self_other and cross_cluster_cap is not None:
        if cross_cluster_cap == "min_kept":
            cap = min(info["n_target_natural"] for info in cluster_info) if cluster_info else None
        else:
            cap = int(cross_cluster_cap)

    tasks = []
    for info in cluster_info:
        cid, ix_s, ix_o, n_s, n_o = info["cid"], info["ix_s"], info["ix_o"], info["n_s"], info["n_o"]
        X_self_full, Y_self_full = X_self[ix_s], Y_self[ix_s]
        X_other_full, Y_other_full = X_other[ix_o], Y_other[ix_o]

        if not balance_self_other:
            n_comp_eff = _n_comp_eff(min(n_s, n_o), X_self.shape[1]) if adaptive_n_components else X_self.shape[1]
            if adaptive_n_components and n_comp_eff < min_n_comp:
                count_rows.append(dict(cluster_id=cid, n_self_raw=n_s, n_other_raw=n_o,
                                       n_self_used=0, n_other_used=0, kept=False,
                                       drop_reason=f"n_comp_eff_{max(n_comp_eff,0)}_below_min_{min_n_comp}"))
                continue
            pc_note = f"_pc{n_comp_eff}" if adaptive_n_components else ""
            count_rows.append(dict(cluster_id=cid, n_self_raw=n_s, n_other_raw=n_o,
                                   n_self_used=n_s, n_other_used=n_o, kept=True,
                                   drop_reason=f"no_balancing_full_n{pc_note}"))
            X_self_t = X_self_full[:, :n_comp_eff] if adaptive_n_components else X_self_full
            X_other_t = X_other_full[:, :n_comp_eff] if adaptive_n_components else X_other_full
            for nidx in range(Y_self.shape[1]):
                tasks.append((cid, nidx, X_self_t, Y_self_full, X_other_t, Y_other_full,
                              n_s, n_o))
            continue

        n_target = info["n_target_natural"] if cap is None else min(info["n_target_natural"], cap)
        if n_target < min_trials_per_condition:
            count_rows.append(dict(cluster_id=cid, n_self_raw=n_s, n_other_raw=n_o,
                                   n_self_used=0, n_other_used=0, kept=False,
                                   drop_reason=f"cross_cluster_cap_below_min_{min_trials_per_condition}"))
            continue

        n_comp_eff = _n_comp_eff(n_target, X_self.shape[1]) if adaptive_n_components else X_self.shape[1]
        if adaptive_n_components and n_comp_eff < min_n_comp:
            count_rows.append(dict(cluster_id=cid, n_self_raw=n_s, n_other_raw=n_o,
                                   n_self_used=0, n_other_used=0, kept=False,
                                   drop_reason=f"n_comp_eff_{max(n_comp_eff,0)}_below_min_{min_n_comp}"))
            continue

        bs_note = "" if (n_s == n_target and n_o == n_target) else f"_bootstrapped_x{n_bootstrap}"
        cap_note = "" if cap is None else f"_capped_{cap}"
        pc_note = f"_pc{n_comp_eff}" if adaptive_n_components else ""
        count_rows.append(dict(cluster_id=cid, n_self_raw=n_s, n_other_raw=n_o,
                               n_self_used=n_target, n_other_used=n_target, kept=True,
                               drop_reason=f"target_{n_target}{cap_note}{bs_note}{pc_note}"))

        X_self_t = X_self_full[:, :n_comp_eff] if adaptive_n_components else X_self_full
        X_other_t = X_other_full[:, :n_comp_eff] if adaptive_n_components else X_other_full
        for nidx in range(Y_self.shape[1]):
            tasks.append((cid, nidx, X_self_t, Y_self_full, X_other_t, Y_other_full,
                          n_target, n_target))

    cluster_counts_df = pd.DataFrame(count_rows).sort_values("cluster_id").reset_index(drop=True)
    if print_trial_counts:
        print(f"\nCluster trial counts (bootstrap) — {patient_id} | {region_name}")
        print(cluster_counts_df)
    if save_csvs:
        p = os.path.join(out_dir, f"{region_name}_cluster_trial_counts.csv")
        cluster_counts_df.to_csv(p, index=False)
        print(f"  Cluster counts saved: {p}")

    # ---- worker ----
    # Self and other are now treated symmetrically: each side independently
    # is either "fixed" (its full count already equals its own target, so it
    # contributes one fit reused every draw) or "bootstrapped" (it has spare
    # trials beyond the target, so it gets resampled and re-fit each draw).
    # With cross_cluster_cap=None this collapses to the original small/large
    # split (exactly one side fixed, the other bootstrapped); with a cap, a
    # cluster can have BOTH sides bootstrapped (if both exceed the cap) or
    # neither (if both are already at or below it).
    def _worker(cid, nidx, X_self_full, Y_self_full, X_other_full, Y_other_full,
                n_target_self, n_target_other):
        try:
            y_self_full = Y_self_full[:, nidx:nidx + 1]
            y_other_full = Y_other_full[:, nidx:nidx + 1]
            n_self_full = X_self_full.shape[0]
            n_other_full = X_other_full.shape[0]

            self_fixed = n_self_full == n_target_self
            other_fixed = n_other_full == n_target_other
            n_draws = 1 if (self_fixed and other_fixed) else n_bootstrap

            beta_self_fixed = _fit_beta(X_self_full, y_self_full) if self_fixed else None
            beta_other_fixed = _fit_beta(X_other_full, y_other_full) if other_fixed else None

            dists = []
            self_half_corrs, other_half_corrs = [], []
            beta_self_first, beta_other_first = beta_self_fixed, beta_other_fixed

            for b in range(n_draws):
                if self_fixed:
                    X_s, y_s, beta_self = X_self_full, y_self_full, beta_self_fixed
                else:
                    rng_s = np.random.RandomState(random_state + cid * 100_000 + nidx * 1_000 + b)
                    idx_s = rng_s.choice(n_self_full, size=n_target_self, replace=False)
                    X_s, y_s = X_self_full[idx_s], y_self_full[idx_s]
                    beta_self = _fit_beta(X_s, y_s)
                    if beta_self_first is None:
                        beta_self_first = beta_self

                if other_fixed:
                    X_o, y_o, beta_other = X_other_full, y_other_full, beta_other_fixed
                else:
                    rng_o = np.random.RandomState(random_state + 1 + cid * 100_000 + nidx * 1_000 + b)
                    idx_o = rng_o.choice(n_other_full, size=n_target_other, replace=False)
                    X_o, y_o = X_other_full[idx_o], y_other_full[idx_o]
                    beta_other = _fit_beta(X_o, y_o)
                    if beta_other_first is None:
                        beta_other_first = beta_other

                d = _safe_cosine(beta_self, beta_other)
                if np.isfinite(d):
                    dists.append(d)

                # Reuse this draw's already-sampled subset for half-split
                # reliability instead of drawing a fresh one, for whichever
                # side(s) are actually being bootstrapped this draw.
                if compute_half_splits and b < n_halfsplit_repeats:
                    if not self_fixed and n_target_self >= 4:
                        try:
                            Xs1, Xs2, Ys1, Ys2 = train_test_split(
                                X_s, y_s, test_size=0.5,
                                random_state=int(np.random.RandomState(
                                    random_state + 7 + cid * 100_000 + nidx * 1_000 + b
                                ).randint(0, 2**31 - 1)))
                            c = _safe_cosine(_fit_beta(Xs1, Ys1), _fit_beta(Xs2, Ys2))
                            if np.isfinite(c):
                                self_half_corrs.append(1 - c)
                        except Exception:
                            pass
                    if not other_fixed and n_target_other >= 4:
                        try:
                            Xo1, Xo2, Yo1, Yo2 = train_test_split(
                                X_o, y_o, test_size=0.5,
                                random_state=int(np.random.RandomState(
                                    random_state + 8 + cid * 100_000 + nidx * 1_000 + b
                                ).randint(0, 2**31 - 1)))
                            c = _safe_cosine(_fit_beta(Xo1, Yo1), _fit_beta(Xo2, Yo2))
                            if np.isfinite(c):
                                other_half_corrs.append(1 - c)
                        except Exception:
                            pass

            if not dists:
                return None, None

            row: Dict = dict(
                cluster_id=cid, region=region_name, neuron=nidx,
                cosine_distance=float(np.mean(dists)),
                cosine_distance_sem=(float(np.std(dists, ddof=1) / np.sqrt(len(dists)))
                                      if len(dists) > 1 else 0.0),
                n_bootstrap_draws=len(dists),
                n_self_used=n_target_self, n_other_used=n_target_other,
            )

            if compute_half_splits:
                # Independent repeated random half-splits for whichever
                # side(s) are FIXED (no bootstrap draws to reuse from).
                if self_fixed and n_target_self >= 4:
                    for r in range(n_halfsplit_repeats):
                        try:
                            rng_h = np.random.RandomState(random_state + 13 + cid * 100_000 + nidx * 1_000 + r)
                            Xs1, Xs2, Ys1, Ys2 = train_test_split(
                                X_self_full, y_self_full, test_size=0.5,
                                random_state=int(rng_h.randint(0, 2**31 - 1)))
                            c = _safe_cosine(_fit_beta(Xs1, Ys1), _fit_beta(Xs2, Ys2))
                            if np.isfinite(c):
                                self_half_corrs.append(1 - c)
                        except Exception:
                            pass
                if other_fixed and n_target_other >= 4:
                    for r in range(n_halfsplit_repeats):
                        try:
                            rng_h = np.random.RandomState(random_state + 14 + cid * 100_000 + nidx * 1_000 + r)
                            Xo1, Xo2, Yo1, Yo2 = train_test_split(
                                X_other_full, y_other_full, test_size=0.5,
                                random_state=int(rng_h.randint(0, 2**31 - 1)))
                            c = _safe_cosine(_fit_beta(Xo1, Yo1), _fit_beta(Xo2, Yo2))
                            if np.isfinite(c):
                                other_half_corrs.append(1 - c)
                        except Exception:
                            pass

                row["self_halfsplit_cosine"] = float(np.mean(self_half_corrs)) if self_half_corrs else np.nan
                row["other_halfsplit_cosine"] = float(np.mean(other_half_corrs)) if other_half_corrs else np.nan

            beta_entry = ((cid, nidx), {
                "beta_self": beta_self_first,
                "beta_other": beta_other_first,
            })
            return row, beta_entry

        except Exception as e:
            print(f"  Failed cluster {cid} neuron {nidx}: {e}")
            return None, None

    # ---- run ----
    raw_results = Parallel(n_jobs=n_jobs)(
        delayed(_worker)(*t) for t in tasks
    )

    all_cosine_rows = []
    all_betas: Dict = {}
    for row, beta_entry in raw_results:
        if row is not None:
            all_cosine_rows.append(row)
        if beta_entry is not None:
            all_betas[beta_entry[0]] = beta_entry[1]

    cosine_df = pd.DataFrame(all_cosine_rows)

    if save_csvs and not cosine_df.empty:
        p = os.path.join(out_dir, f"{region_name}_clusterwise_cosine_distances.csv")
        cosine_df.to_csv(p, index=False)
        print(f"  Cosine distances saved: {p}")

    if save_pickle:
        p = os.path.join(out_dir, f"{region_name}_clusterwise_raw_betas.pkl")
        with open(p, "wb") as f:
            pickle.dump(all_betas, f)
        print(f"  Raw betas saved: {p}")

    return cosine_df, all_betas, cluster_counts_df
