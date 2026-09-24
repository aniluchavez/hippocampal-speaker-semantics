#!/usr/bin/env python3
"""
normalize_clusterwise_global_most_populated.py

Post-processing pass over scripts/cluster_glm_reliability.py's per-patient
{region}_clusterwise_cosine_distances.csv outputs: normalizes each
neuron/cluster's raw self-vs-other cosine distance by a "noise floor" derived
from whichever semantic cluster has the most trials *summed across all
patients* (the global most-populated cluster), per region.

Ported from resultsofsemantic.ipynb's normalization_mode="global_most_populated"
(cell 8) -- same noise-floor formula (1 - min(self_halfsplit_mean,
other_halfsplit_mean) for the reference cluster), same per-cluster one-way
ANOVA on the normalized distances -- adapted to read cosine_distance directly
from the CSV (already computed) instead of recomputing it from the raw-betas
pickle, and generalized to loop over whichever regions a patient actually has.

Requires scripts/cluster_glm_reliability.py to have been run with
compute_half_splits=True (self_halfsplit_cosine/other_halfsplit_cosine columns
must be present).

Usage:
    cd /scratch/aniluchavez/hippocampal-speaker-semantics
    python3 -u scripts/normalize_clusterwise_global_most_populated.py
"""

import os
import argparse

import numpy as np
import pandas as pd
from scipy.stats import f_oneway

DEFAULT_RESULTS_ROOT = "/scratch/aniluchavez/ConvoDATAS/SemanticGLM/clusterwise"
HALFSPLIT_COLS = ["self_halfsplit_cosine", "other_halfsplit_cosine"]


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=str, default=DEFAULT_RESULTS_ROOT,
                   help="Root directory containing per-patient clusterwise cosine CSVs")
    p.add_argument("--regions", type=str, default="hippocampus,ACC",
                    help="Comma-separated regions to process")
    return p.parse_args()


def find_global_majority_cluster(patient_dirs, region):
    """Cluster_id with the most halfsplit-valid trials, summed across all
    patients, for this region."""
    cluster_counts = {}
    for pdir in patient_dirs:
        csv_path = os.path.join(pdir, f"{region}_clusterwise_cosine_distances.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if not set(HALFSPLIT_COLS).issubset(df.columns):
            continue
        df = df.dropna(subset=HALFSPLIT_COLS)
        for k, v in df["cluster_id"].value_counts().items():
            cluster_counts[k] = cluster_counts.get(k, 0) + v
    if not cluster_counts:
        return None
    return max(cluster_counts, key=cluster_counts.get)


def normalize_patient_region(csv_path, global_majority_cluster, region):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if not set(HALFSPLIT_COLS).issubset(df.columns):
        print(f"  {csv_path}: missing halfsplit columns "
              f"(re-run cluster_glm_reliability.py with compute_half_splits=True) -- skip")
        return None

    df_valid = df.dropna(subset=HALFSPLIT_COLS)
    ref_df = df_valid[df_valid["cluster_id"] == global_majority_cluster]

    noise_floor = np.nan
    if not ref_df.empty:
        self_mean = ref_df["self_halfsplit_cosine"].mean(skipna=True)
        other_mean = ref_df["other_halfsplit_cosine"].mean(skipna=True)
        if np.isfinite(self_mean) or np.isfinite(other_mean):
            noise_floor = 1 - np.nanmin([self_mean, other_mean])

    if np.isfinite(noise_floor) and noise_floor > 0:
        df["normalized_distance"] = df["cosine_distance"] / noise_floor
    else:
        df["normalized_distance"] = np.nan

    df["reference_cluster"] = global_majority_cluster
    df["noise_floor"] = noise_floor

    n_valid = df["normalized_distance"].notna().sum()
    print(f"  {os.path.basename(os.path.dirname(csv_path))}/{region}: "
          f"ref_cluster={global_majority_cluster} noise_floor={noise_floor:.4f} "
          f"({n_valid}/{len(df)} rows normalized)" if np.isfinite(noise_floor) else
          f"  {os.path.basename(os.path.dirname(csv_path))}/{region}: "
          f"ref_cluster={global_majority_cluster} has no usable halfsplit data -- not normalized")

    return df


def run_anova(df, value_col="normalized_distance", group_col="cluster_id"):
    groups = [
        g[value_col].dropna().values
        for _, g in df.groupby(group_col)
        if g[value_col].notna().sum() > 1
    ]
    if len(groups) > 1:
        f_val, p_val = f_oneway(*groups)
        return float(f_val), float(p_val)
    return np.nan, np.nan


def main():
    args = _args()
    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    results_root = args.results_root

    patient_dirs = sorted(
        os.path.join(results_root, d) for d in os.listdir(results_root)
        if os.path.isdir(os.path.join(results_root, d))
    )

    for region in regions:
        print(f"\n{'='*60}\nRegion: {region}")
        global_majority_cluster = find_global_majority_cluster(patient_dirs, region)
        if global_majority_cluster is None:
            print(f"  No usable data for region {region} -- skip")
            continue
        print(f"  Global most-populated cluster: {global_majority_cluster}")

        anova_rows = []
        for pdir in patient_dirs:
            patient_id = os.path.basename(pdir)
            csv_path = os.path.join(pdir, f"{region}_clusterwise_cosine_distances.csv")
            if not os.path.exists(csv_path):
                continue

            df_norm = normalize_patient_region(csv_path, global_majority_cluster, region)
            if df_norm is None:
                continue

            out_path = os.path.join(pdir, f"{region}_clusterwise_cosine_distances_normalized.csv")
            df_norm.to_csv(out_path, index=False)

            f_val, p_val = run_anova(df_norm)
            anova_rows.append({
                "patient": patient_id, "region": region,
                "reference_cluster": global_majority_cluster,
                "noise_floor": df_norm["noise_floor"].iloc[0],
                "f_val": f_val, "p_val": p_val,
            })

        if anova_rows:
            anova_df = pd.DataFrame(anova_rows)
            anova_path = os.path.join(results_root, f"{region}_anova_summary_global_most_populated.csv")
            anova_df.to_csv(anova_path, index=False)
            print(f"  Saved ANOVA summary: {anova_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
