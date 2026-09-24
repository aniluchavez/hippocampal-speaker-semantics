#!/usr/bin/env python3
"""Other-other clusterwise GLM using transferred legacy BERT/spike_sum inputs.

This reproduces the data source used by
``RegressionRESULTSBERT150_200_sema10_OTHERS_new`` while choosing the two
largest non-Speaker1 speakers per patient, rather than hardcoding SPK3/SPK5.
"""

from __future__ import annotations

import argparse
import ast
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
import cluster_glm_reliability as glm  # noqa: E402


_cluster_mod = glm._cluster_mod
run_clusterwise_cosine_distance = _cluster_mod.run_clusterwise_cosine_distance
report_cluster_balance = _cluster_mod.report_cluster_balance


PATIENTS = [
    {"patient_ID": "PTYEU_task147", "patient": "ptYEU_task147", "n_components": 30},
    {"patient_ID": "PTYEY_task86", "patient": "ptYEY_task86", "n_components": 30},
    {"patient_ID": "PTYFA_task25", "patient": "ptYFA_task25", "n_components": 30},
    {"patient_ID": "PTYFC_task28", "patient": "ptYFC_task28", "n_components": 30},
    {"patient_ID": "PTYFF_task17", "patient": "ptYFF_task17", "n_components": 30},
    # YFG intentionally excluded by default: transferred hippocampus file has
    # 48 columns, inconsistent with the curated hippocampus map.
    {"patient_ID": "PTYFK_task40", "patient": "ptYFK_task40", "n_components": 30},
]


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--patient", default=None)
    p.add_argument("--region", default="hippocampus")
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--n-components", type=int, default=30)
    p.add_argument("--embed-root", default="/scratch/aniluchavez/ConvoDATAS/BERTEmbeds")
    p.add_argument("--spike-root", default="/scratch/aniluchavez/appwindow_BERT_spikes_202509")
    p.add_argument("--results-root", default="/scratch/aniluchavez/ConvoDATAS/SemanticGLM/transferred_bert_other_other_top2_pc30_excl_yfg")
    p.add_argument("--min-trials-per-condition", type=int, default=10)
    p.add_argument("--no-balance", action="store_true", default=True)
    return p.parse_args()


def _norm_spk(speaker: str) -> str:
    return "SPK" + speaker.replace("Speaker", "")


def _load_embeddings(embedding_csv_path, n_components):
    df = pd.read_csv(embedding_csv_path)
    df["Parsed_Embedding"] = df["Embedding"].apply(ast.literal_eval)
    emb = np.array(df["Parsed_Embedding"].tolist())
    if "onset" in df.columns:
        df = df.sort_values(by="onset").reset_index(drop=True)
    pcs = PCA(n_components=n_components).fit_transform(emb)
    pcs_df = pd.DataFrame(pcs, columns=[f"PC{i+1}" for i in range(n_components)])
    return df.reset_index(drop=True), pcs_df


def _build_features(pcs_df, durations):
    pcs = pcs_df.values
    dur = durations.values.reshape(-1, 1)
    x = np.hstack([pcs, dur, pcs * dur])
    return StandardScaler().fit_transform(x)


def _speaker_counts(spike_dir, region):
    rows = []
    for sp_dir in sorted(os.listdir(spike_dir)):
        if not sp_dir.startswith("Speaker") or sp_dir == "Speaker1":
            continue
        path = os.path.join(spike_dir, sp_dir, f"{region}_spike_sum.npy")
        if not os.path.exists(path):
            continue
        arr = np.load(path, mmap_mode="r")
        rows.append((sp_dir, int(arr.shape[0]), int(arr.shape[1])))
    return sorted(rows, key=lambda x: (-x[1], x[0]))


def _load_condition(spike_dir, speaker, region, x_all, meta, cluster_vals):
    spk_norm = _norm_spk(speaker)
    mask = meta["Speaker"].astype(str).str.upper().str.replace(" ", "", regex=False) == spk_norm
    idx = np.where(mask.to_numpy())[0]
    y_path = os.path.join(spike_dir, speaker, f"{region}_spike_sum.npy")
    y = np.load(y_path)
    if y.shape[0] != len(idx):
        raise ValueError(f"{speaker}: Y rows={y.shape[0]} != metadata rows={len(idx)}")
    x = x_all[idx]
    clusters = cluster_vals.iloc[idx].reset_index(drop=True)
    valid = ~(np.isnan(x).any(axis=1) | np.isnan(y).any(axis=1))
    return (
        x[valid],
        y[valid].astype(np.float64),
        pd.DataFrame({"ClusterID": clusters[valid].to_numpy()}),
    )


def main():
    args = _args()
    run_poisson_ridge = glm.make_beta_fit_adapter(glm.ALPHAS)

    patients = [p for p in PATIENTS if args.patient is None or p["patient_ID"] == args.patient]
    for cfg in patients:
        patient_ID = cfg["patient_ID"]
        patient = cfg["patient"]
        n_components = args.n_components or cfg["n_components"]

        print(f"\n{'='*60}\n  {patient_ID} / {args.region}", flush=True)

        emb_dir = os.path.join(args.embed_root, f"{patient_ID}_words_english_only")
        embedding_file = os.path.join(emb_dir, f"{patient_ID}_aligned_embeddings_withNP.csv")
        cluster_file = os.path.join(emb_dir, f"{patient_ID}_filtered_used_rows_withNP_withClusterIDNew.xlsx")
        if not os.path.exists(cluster_file):
            newest_cluster_file = os.path.join(
                emb_dir, f"{patient_ID}_filtered_used_rows_withNP_withClusterIDNewest.xlsx"
            )
            if os.path.exists(newest_cluster_file):
                cluster_file = newest_cluster_file
        spike_dir = os.path.join(args.spike_root, f"output_{patient}_english_only_dur_10_410_500")
        duration_file = os.path.join(spike_dir, f"{patient}_with_regress_dur.xlsx")
        if not all(os.path.exists(p) for p in [embedding_file, cluster_file, spike_dir, duration_file]):
            print(f"  missing inputs -- skip", flush=True)
            continue

        speaker_rows = _speaker_counts(spike_dir, args.region)
        if len(speaker_rows) < 2:
            print("  fewer than two non-Speaker1 speakers -- skip", flush=True)
            continue
        speaker_a, n_a, _ = speaker_rows[0]
        speaker_b, n_b, _ = speaker_rows[1]
        print(f"  top2: {speaker_a}={n_a}, {speaker_b}={n_b}", flush=True)

        meta, pcs_df = _load_embeddings(embedding_file, n_components)
        dur_df = pd.read_excel(duration_file).reset_index(drop=True)
        meta = meta.reset_index(drop=True)
        meta["regress_dur"] = dur_df["regress_dur"].to_numpy()
        x_all = _build_features(pcs_df, meta["regress_dur"])

        cluster_df = pd.read_excel(cluster_file)
        cluster_col = "FinalClusterID" if "FinalClusterID" in cluster_df.columns else "ClusterID"
        cluster_vals = cluster_df[cluster_col].reset_index(drop=True)

        x_a, y_a, meta_a = _load_condition(spike_dir, speaker_a, args.region, x_all, meta, cluster_vals)
        x_b, y_b, meta_b = _load_condition(spike_dir, speaker_b, args.region, x_all, meta, cluster_vals)

        spike_ok = (y_a.sum(0) >= glm.MIN_SPIKES) & (y_b.sum(0) >= glm.MIN_SPIKES)
        if not spike_ok.any():
            print("  0 neurons pass spike filter -- skip", flush=True)
            continue
        y_a = y_a[:, spike_ok]
        y_b = y_b[:, spike_ok]
        print(
            f"  rows: {speaker_a}={x_a.shape[0]}, {speaker_b}={x_b.shape[0]}, "
            f"neurons={int(spike_ok.sum())}",
            flush=True,
        )

        out_root = os.path.join(args.results_root, f"{speaker_a}_vs_{speaker_b}")
        pair_dir = os.path.join(out_root, patient_ID)
        os.makedirs(pair_dir, exist_ok=True)
        pd.DataFrame([{
            "patient": patient_ID,
            "region": args.region,
            "speaker_a": speaker_a,
            "speaker_b": speaker_b,
            "speaker_a_rows": x_a.shape[0],
            "speaker_b_rows": x_b.shape[0],
            "n_neurons": int(spike_ok.sum()),
            "n_components": n_components,
            "feature_dim": x_a.shape[1],
            "spike_source": "spike_sum",
            "all_other_counts": repr(speaker_rows),
        }]).to_csv(os.path.join(pair_dir, f"{args.region}_speaker_pair_summary.csv"), index=False)

        report_cluster_balance(meta_a, meta_b, patient_ID, args.region)

        run_clusterwise_cosine_distance(
            X_self=x_a,
            X_other=x_b,
            Y_self=y_a,
            Y_other=y_b,
            metadata_self=meta_a,
            metadata_other=meta_b,
            cluster_column="ClusterID",
            region_name=args.region,
            patient_id=patient_ID,
            n_components=n_components,
            results_root=out_root,
            run_poisson_ridge=run_poisson_ridge,
            balance_function_words=False,
            compute_half_splits=True,
            n_jobs=args.n_jobs,
            cap_large_clusters=False,
            min_trials_per_condition=args.min_trials_per_condition,
            print_trial_counts=True,
            max_size_ratio=2,
            cap_reference="median",
            balance_all_clusters=False,
            soft_balance=False,
        )

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
