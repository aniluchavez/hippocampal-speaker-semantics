"""
aggregate_cluster_results.py

Load per-patient clusterwise cosine distance CSVs / raw-beta PKLs,
concatenate across patients, run a one-way ANOVA across clusters,
and plot mean cosine distance per semantic category.

Usage
-----
Edit the CONFIG section and run:
    python scripts/aggregate_cluster_results.py
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# CONFIG — edit these before running
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("/projects/bhayden/anilu/Language_docs/RegressionRESULTSBERT150_200_sema10_sb_pr_HS3_Newclustering/allwords")
REGION = "hippocampus"   # or "ACC"
PATIENT_IDS = [
    "PTYEU_task147", "PTYFF_task17", "PTYFG_task18", "PTYFI_task81", "PTYFA_task25",
    "PTYEY_task86",  "PTYEV_task37", "PTYEZ_task60",  "PTYFK_task40", "PTYFC_task28",
]
CLUSTER_CATEGORY_MAP = {
    1: "Body Parts", 2: "Places", 3: "Emotional", 4: "Mental", 5: "Social",
    6: "Objects",    7: "Visual", 8: "Numerical", 9: "Actions", 10: "Identity",
    11: "Function Words", 12: "Proper Nouns",
}
# If True, compute cosine distances from raw beta PKLs; if False, load pre-computed CSV.
USE_RAW_BETAS = True
# ---------------------------------------------------------------------------


def load_from_raw_betas(results_dir: Path, patient_id: str, region: str) -> pd.DataFrame:
    pkl_path = results_dir / patient_id / f"{region}_clusterwise_raw_betas.pkl"
    if not pkl_path.exists():
        print(f"  Missing: {pkl_path}")
        return pd.DataFrame()

    with open(pkl_path, "rb") as f:
        beta_dict = pickle.load(f)

    rows = []
    for cluster_id, betas in beta_dict.items():
        for i, (b_self, b_other) in enumerate(zip(betas["self"], betas["other"])):
            b_self = np.asarray(b_self)
            b_other = np.asarray(b_other)
            if np.allclose(b_self, 0) or np.allclose(b_other, 0):
                continue
            cos_dist = 1 - cosine_similarity(b_self.reshape(1, -1), b_other.reshape(1, -1))[0, 0]
            rows.append(dict(patient=patient_id, cluster_id=cluster_id,
                             neuron_id=i, cosine_distance=cos_dist))
    return pd.DataFrame(rows)


def load_from_csv(results_dir: Path, patient_id: str, region: str) -> pd.DataFrame:
    csv_path = results_dir / patient_id / f"{region}_clusterwise_cosine_distances.csv"
    if not csv_path.exists():
        print(f"  Missing: {csv_path}")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df["patient"] = patient_id
    return df


def main():
    all_dfs = []
    for pid in PATIENT_IDS:
        print(f"Loading {pid}...")
        if USE_RAW_BETAS:
            df = load_from_raw_betas(RESULTS_DIR, pid, REGION)
        else:
            df = load_from_csv(RESULTS_DIR, pid, REGION)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError("No patient data loaded.")

    concat_df = pd.concat(all_dfs, ignore_index=True)
    concat_df["Category"] = concat_df["cluster_id"].map(CLUSTER_CATEGORY_MAP).fillna("Unknown")

    # save
    out_csv = RESULTS_DIR / f"{REGION}_ALLPATIENTS_clusterwise_cosine_distance_table.csv"
    concat_df.to_csv(out_csv, index=False)
    print(f"\nSaved concatenated table: {out_csv}")
    print(f"Shape: {concat_df.shape}")
    print(concat_df["cluster_id"].value_counts())

    # ANOVA
    groups = [g["cosine_distance"].values
              for _, g in concat_df.groupby("cluster_id") if len(g) > 1]
    if len(groups) > 1:
        f_val, p_val = f_oneway(*groups)
    else:
        f_val, p_val = np.nan, np.nan

    print(f"\nOne-way ANOVA ({REGION}): F = {f_val:.3f}  p = {p_val:.4g}")

    anova_out = RESULTS_DIR / f"{REGION}_ALLPATIENTS_anova_summary.txt"
    with open(anova_out, "w") as fh:
        fh.write(f"Region: {REGION}\nF = {f_val:.3f}\np = {p_val:.4g}\n")
    print(f"ANOVA summary saved: {anova_out}")

    # mean per cluster
    mean_df = (
        concat_df.groupby(["cluster_id", "Category"])["cosine_distance"]
        .mean().reset_index()
        .sort_values("cosine_distance")
    )
    mean_out = RESULTS_DIR / f"{REGION}_ALLPATIENTS_clusterwise_mean_cosine_distance.csv"
    mean_df.to_csv(mean_out, index=False)
    print(f"Mean distances saved: {mean_out}")

    # plot
    mean_df["cluster_id"] = mean_df["cluster_id"].astype(str)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=mean_df, x="cluster_id", y="cosine_distance",
                hue="Category", palette="Set2", dodge=False)
    plt.title(f"Mean Cosine Distance Across Patients — {REGION}")
    plt.xlabel("Cluster ID")
    plt.ylabel("Mean Cosine Distance")
    plt.legend(title="Semantic Category", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
