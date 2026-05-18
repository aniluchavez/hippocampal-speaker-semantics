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
    combined = pd.concat([
        metadata_self.assign(condition="self"),
        metadata_other.assign(condition="other"),
    ])
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
    combined = pd.concat([
        metadata_self.assign(condition="self"),
        metadata_other.assign(condition="other"),
    ])
    contingency = pd.crosstab(combined[cluster_column], combined["condition"])
    chi2, p, _, expected = chi2_contingency(contingency)
    if verbose:
        print(f"Initial  chi² = {chi2:.2f}  p = {p:.4f}")

    m_self = metadata_self.copy()
    m_other = metadata_other.copy()

    while p < 0.05 and not contingency.empty:
        residuals = (contingency.values - expected) / np.sqrt(expected)
        r_idx, c_idx = np.unravel_index(np.abs(residuals).argmax(), residuals.shape)
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
        ])
        contingency = pd.crosstab(combined[cluster_column], combined["condition"])
        chi2, p, _, expected = chi2_contingency(contingency)

    if verbose:
        print(f"Final    chi² = {chi2:.2f}  p = {p:.4f}")

    return m_self.reset_index(drop=True), m_other.reset_index(drop=True)


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

    # ---- first pass: raw counts ----
    clusters = sorted(
        set(metadata_self[cluster_column].dropna()) |
        set(metadata_other[cluster_column].dropna())
    )
    cluster_info = []
    raw_min_counts = []

    for cid in clusters:
        ix_s = metadata_self.index[metadata_self[cluster_column] == cid].to_numpy()
        ix_o = metadata_other.index[metadata_other[cluster_column] == cid].to_numpy()
        n_min = min(len(ix_s), len(ix_o))
        cluster_info.append(dict(cluster_id=cid, ix_self=ix_s, ix_other=ix_o,
                                 n_self=len(ix_s), n_other=len(ix_o)))
        if len(ix_s) >= min_trials_per_condition and len(ix_o) >= min_trials_per_condition:
            raw_min_counts.append(n_min)

    # ---- global balancing target ----
    global_target = None
    if soft_balance and balance_all_clusters:
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
                n_t = min(len(ix_s), len(ix_o), gt)
            else:
                n_t = (min(raw_min_counts) if raw_min_counts else min(len(ix_s), len(ix_o)))
            if n_t < min_trials_per_condition:
                count_rows.append(dict(cluster_id=cid, n_self_raw=n_s, n_other_raw=n_o,
                                       n_self_used=0, n_other_used=0, kept=False,
                                       drop_reason=f"balance_target_below_min_{min_trials_per_condition}"))
                continue
            ix_s = _sample(ix_s, n_t, rng)
            ix_o = _sample(ix_o, n_t, rng)
            drop_reason = f"{'soft' if soft_balance else 'hard'}_balanced_to_{n_t}"

        Xs = X_self[ix_s];  Ys = Y_self[ix_s]
        Xo = X_other[ix_o]; Yo = Y_other[ix_o]
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
            tasks.append((cid, nidx, Xs, Ys, Xo, Yo, n_su, n_ou))

    cluster_counts_df = pd.DataFrame(count_rows).sort_values("cluster_id").reset_index(drop=True)
    if print_trial_counts:
        print(f"\nCluster trial counts — {patient_id} | {region_name}")
        print(cluster_counts_df)
    if save_csvs:
        p = os.path.join(out_dir, f"{region_name}_cluster_trial_counts.csv")
        cluster_counts_df.to_csv(p, index=False)
        print(f"  Cluster counts saved: {p}")

    # ---- worker ----
    def _worker(cid, nidx, Xs, Ys, Xo, Yo, n_su, n_ou):
        try:
            ys = Ys[:, nidx:nidx+1]
            yo = Yo[:, nidx:nidx+1]

            beta_s = run_poisson_ridge(
                Xs, ys, patient_id=patient_id, neuron_idx=0,
                region_name=region_name, n_semantic_dims=n_components,
                results_dir=results_root, fast_beta_only=True, save_results=False,
            ).filter(like="beta_").values.flatten()

            beta_o = run_poisson_ridge(
                Xo, yo, patient_id=patient_id, neuron_idx=0,
                region_name=region_name, n_semantic_dims=n_components,
                results_dir=results_root, fast_beta_only=True, save_results=False,
            ).filter(like="beta_").values.flatten()

            dist = _safe_cosine(beta_s, beta_o)
            if not np.isfinite(dist):
                return None, None

            row: Dict = dict(cluster_id=cid, region=region_name, neuron=nidx,
                             cosine_distance=dist, n_self_used=n_su, n_other_used=n_ou)

            if compute_half_splits and Xs.shape[0] >= 4 and Xo.shape[0] >= 4:
                try:
                    Xs1, Xs2, Ys1, Ys2 = train_test_split(Xs, ys, test_size=0.5, random_state=random_state)
                    Xo1, Xo2, Yo1, Yo2 = train_test_split(Xo, yo, test_size=0.5, random_state=random_state)

                    def _beta(X, Y):
                        return run_poisson_ridge(
                            X, Y, patient_id=patient_id, neuron_idx=0,
                            region_name=region_name, n_semantic_dims=n_components,
                            results_dir=results_root, fast_beta_only=True, save_results=False,
                        ).filter(like="beta_").values.flatten()

                    bs1, bs2 = _beta(Xs1, Ys1), _beta(Xs2, Ys2)
                    bo1, bo2 = _beta(Xo1, Yo1), _beta(Xo2, Yo2)
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
