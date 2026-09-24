#!/usr/bin/env python3
"""
cluster_frequency_diagnostic.py

Diagnostic: is the clusterwise self-vs-other cosine distance
(scripts/cluster_glm_reliability.py's output) actually tracking semantic
category, or just word frequency? Function-word clusters are far more
frequent in English than content-word clusters, so a naive cross-cluster
comparison conflates the two.

For each patient/region, joins log_word_freq (computed once by
extract_control_features.py into ConvoDATAS/ControlFeatures/*.csv) onto the
same per-trial cluster assignment used for the cosine-distance fits, via the
identical onset-based row alignment cluster_glm_reliability.py uses for
FinalClusterID. Aggregates to one row per (patient, region, cluster):
mean log_word_freq, the per-cluster fit size (n_target = min(self, other)
actually used by run_clusterwise_cosine_distance_bootstrap, read from its
saved cluster_trial_counts.csv), and mean cosine_distance.

Controls for cluster size when testing the frequency relationship, not just
reports a raw correlation: with n_components=100 semantic dimensions, a
cluster fit on only ~12-30 trials is in a low-sample/high-dimension regime
where ridge-regularized betas get shrunk toward near-random directions,
which mechanically inflates cosine distance toward 1 independent of any real
semantic signal. Small clusters are disproportionately low-frequency-word
clusters (frequent words generate more trials), so a naive frequency
correlation would partly be re-detecting that sample-size artifact. n_target
is used as the control (not the bootstrap's cosine_distance_sem) because it's
exogenous -- set by how many words of that category exist, not by the same
noisy fitting procedure that produced the outcome -- and is uniformly defined
for every cluster (unlike cosine_distance_sem, which is forced to exactly 0
for already-balanced clusters that needed no bootstrap resampling at all,
not because those estimates are unusually precise).

Reports both a simple pooled correlation (for reference) and an OLS
regression of cosine_distance on log_word_freq + log(n_target), with
patient x region fixed effects, so frequency only "counts" as a confound if
it predicts distance beyond what cluster size alone already explains.

No transcript/word-level values are printed -- only per-cluster aggregate
statistics and regression coefficients (PHI-safe).

Usage:
    cd /scratch/aniluchavez/hippocampal-speaker-semantics
    python3 -u scripts/cluster_frequency_diagnostic.py
"""

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import cluster_glm_reliability as glm  # noqa: E402

CONTROL_FEATURES_DIR = "/scratch/aniluchavez/ConvoDATAS/ControlFeatures"
RESULTS_ROOT = glm.RESULTS_ROOT


def load_log_word_freq(patient_ID, n_words_expected):
    """log_word_freq from ControlFeatures CSV, aligned to the embedding-cache
    row order via the same onset dropna/sort transform as load_cluster_ids."""
    path = os.path.join(CONTROL_FEATURES_DIR, f"{patient_ID}_control_features.csv")
    if not os.path.exists(path):
        print(f"  {patient_ID}: no control-features file at {path}", flush=True)
        return None
    df = pd.read_csv(path)
    if "log_word_freq" not in df.columns or "onset" not in df.columns:
        print(f"  {patient_ID}: missing log_word_freq/onset column", flush=True)
        return None
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df = df.dropna(subset=["onset"]).sort_values("onset").reset_index(drop=True)
    if len(df) != n_words_expected:
        print(f"  [WARN] {patient_ID}: control-feature rows ({len(df)}) != "
              f"embedding-cache rows ({n_words_expected}) -- skipping", flush=True)
        return None
    return df["log_word_freq"].values


def main():
    rows = []

    for cfg in glm.PATIENTS:
        patient_ID = cfg["patient_ID"]
        for region in cfg["region_ranges"]:
            cosine_path = os.path.join(RESULTS_ROOT, patient_ID, f"{region}_clusterwise_cosine_distances.csv")
            counts_path = os.path.join(RESULTS_ROOT, patient_ID, f"{region}_cluster_trial_counts.csv")
            if not os.path.exists(cosine_path) or not os.path.exists(counts_path):
                continue

            data = glm.build_patient_region_data(cfg, region, glm.LAYER, glm.N_COMPONENTS)
            if data is None:
                continue

            n_words_expected = len(data["mask_self"])
            freq_full = load_log_word_freq(patient_ID, n_words_expected)
            if freq_full is None:
                continue

            freq_self = freq_full[data["mask_self"]][data["valid_self"]]
            freq_other = freq_full[data["mask_other"]][data["valid_other"]]
            cluster_self = data["metadata_self"]["ClusterID"].values
            cluster_other = data["metadata_other"]["ClusterID"].values

            # log_word_freq is averaged over ALL trials assigned to the cluster
            # (the vocabulary's true frequency composition), not just whichever
            # subset a given bootstrap draw happened to sample.
            trial_freq = pd.DataFrame({
                "cluster_id": np.concatenate([cluster_self, cluster_other]),
                "log_word_freq": np.concatenate([freq_self, freq_other]),
            })
            mean_freq_by_cluster = trial_freq.groupby("cluster_id")["log_word_freq"].mean()

            # n_target: the per-cluster fit size actually used by
            # run_clusterwise_cosine_distance_bootstrap (n_self_used ==
            # n_other_used for every kept cluster) -- the exogenous size
            # covariate, not the raw pre-balancing trial count.
            counts_df = pd.read_csv(counts_path)
            counts_df = counts_df[counts_df["kept"] == True]
            n_target_by_cluster = counts_df.set_index("cluster_id")["n_self_used"]

            cosine_df = pd.read_csv(cosine_path)
            mean_dist_by_cluster = cosine_df.groupby("cluster_id")["cosine_distance"].mean()

            for cid in mean_dist_by_cluster.index:
                if cid not in mean_freq_by_cluster.index or cid not in n_target_by_cluster.index:
                    continue
                rows.append(dict(
                    patient=patient_ID, region=region, cluster_id=cid,
                    mean_log_word_freq=mean_freq_by_cluster[cid],
                    mean_cosine_distance=mean_dist_by_cluster[cid],
                    n_target=int(n_target_by_cluster[cid]),
                ))

    if not rows:
        print("No overlapping patient/region data found -- "
              "has scripts/cluster_glm_reliability.py finished running?")
        return

    df = pd.DataFrame(rows)
    df["log_n_target"] = np.log(df["n_target"])
    df["group"] = df["patient"] + "_" + df["region"]

    out_path = os.path.join(RESULTS_ROOT, "cluster_frequency_diagnostic_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved per-(patient, region, cluster) summary: {out_path}")
    print(f"  {len(df)} cluster-level data points across "
          f"{df['patient'].nunique()} patients, {df['region'].nunique()} region(s)")

    # ---- raw pooled correlations (reference only -- conflates patient
    # baseline differences AND cluster-size noise with any real frequency
    # effect; the regression below is the one that actually separates them) --
    r_pooled, p_pooled = pearsonr(df["mean_log_word_freq"], df["mean_cosine_distance"])
    rho_pooled, p_rho_pooled = spearmanr(df["mean_log_word_freq"], df["mean_cosine_distance"])
    r_size, p_size = pearsonr(df["log_n_target"], df["mean_cosine_distance"])
    print(f"\nRaw pooled correlations (reference only, not confound-controlled):")
    print(f"  freq  vs distance:  Pearson r = {r_pooled:+.3f}  p = {p_pooled:.4g}  "
          f"(Spearman rho = {rho_pooled:+.3f}  p = {p_rho_pooled:.4g})")
    print(f"  log(n_target) vs distance:  Pearson r = {r_size:+.3f}  p = {p_size:.4g}")

    # ---- OLS: does frequency predict distance beyond cluster size and
    # patient/region baseline? This is the controlled test. ----
    fe_df = df[df.groupby("group")["cluster_id"].transform("size") > 1].copy()
    if len(fe_df) <= fe_df["group"].nunique() + 2:
        print("\nNot enough multi-cluster patient/region groups for a controlled regression.")
        return

    model = smf.ols(
        "mean_cosine_distance ~ mean_log_word_freq + log_n_target + C(group)",
        data=fe_df,
    ).fit()

    b_freq = model.params["mean_log_word_freq"]
    p_freq = model.pvalues["mean_log_word_freq"]
    b_size = model.params["log_n_target"]
    p_size_ctrl = model.pvalues["log_n_target"]
    print(f"\nOLS (cosine_distance ~ log_word_freq + log(n_target) + patient/region FE, "
          f"n={len(fe_df)}):")
    print(f"  log_word_freq coef = {b_freq:+.4f}  p = {p_freq:.4g}")
    print(f"  log(n_target) coef = {b_size:+.4f}  p = {p_size_ctrl:.4g}")
    print(f"  R-squared = {model.rsquared:.3f}")

    print(
        "\nInterpretation: log(n_target)'s coefficient is the expected small-cluster "
        "noise-inflation artifact (negative -> smaller clusters show larger distance "
        "just from estimation noise). log_word_freq's coefficient is what matters for "
        "the confound question: if it's significant even after controlling for cluster "
        "size and patient/region baseline, the clusterwise cosine-distance result is "
        "confounded by word frequency and needs a frequency-matched null "
        "(see scripts/stratified_perm_prototype.py for the existing pattern) before it "
        "can be read as evidence for semantic-category-specific coding."
    )


if __name__ == "__main__":
    main()
