#!/usr/bin/env python3
"""Clusterwise beta-cosine analysis for other-vs-other speaker pairs.

For each patient, choose the two non-Speaker1 speakers with the most transcript
words, then run the same separate-category Poisson-ridge beta cosine pipeline
used for self-vs-other, treating speaker A/B as the two conditions.

Default setup is the BERT bidirectional max-context original fixed window:
non-Speaker1 spike windows are onset+200 ms, length 500 ms in
``tshift-150_tlen500_oshift+200_olen500``.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
import cluster_glm_reliability as glm  # noqa: E402


RESULTS_ROOT = "/scratch/aniluchavez/ConvoDATAS/SemanticGLM/clusterwise_bertbase_bidirmax_other_other_softbalanced"
N_COMPONENTS = 10
MIN_SPIKES = glm.MIN_SPIKES


CANDIDATE_PATIENTS = [
    "PTYEU_task147",
    "PTYEY_task86",
    "PTYFA_task25",
    "PTYFC_task28",
    "PTYFF_task17",
    "PTYFG_task18",
    "PTYFK_task40",
    "PTYFM_task104",
    "PTYFP_task88",
    "PTYFR_task91",
    "PTYFS_task95",
    "PTYFU_task224",
    "PTYEZ_task60",
]


_cluster_mod = glm._cluster_mod
report_cluster_balance = _cluster_mod.report_cluster_balance
minimal_balancing = _cluster_mod.minimal_balancing
run_clusterwise_cosine_distance = _cluster_mod.run_clusterwise_cosine_distance


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--patient", type=str, default=None)
    p.add_argument("--region", type=str, default="hippocampus")
    p.add_argument("--n_jobs", type=int, default=8)
    p.add_argument("--results-root", type=str, default=RESULTS_ROOT)
    p.add_argument("--model-tag", type=str, default="bert-base-bidirmax")
    p.add_argument("--context-tag", type=str, default="")
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--n-components", type=int, default=N_COMPONENTS)
    p.add_argument("--speaker-a", type=str, default=None,
                   help="Force the first non-Speaker1 condition, e.g. Speaker3.")
    p.add_argument("--speaker-b", type=str, default=None,
                   help="Force the second non-Speaker1 condition, e.g. Speaker5.")
    p.add_argument("--window-tag", type=str, default="tshift-150_tlen500_oshift+200_olen500")
    p.add_argument("--min-trials-per-condition", type=int, default=10)
    p.add_argument("--no-balance", action="store_true",
                   help="Use true/natural counts: no self-other matching, no median capping.")
    p.add_argument("--legacy-minimal-balance", action="store_true",
                   help="Apply notebook-style minimal_balancing before fitting, with X/Y rows kept aligned.")
    p.add_argument("--legacy-minimal-plus-median-cap", action="store_true",
                   help="Apply legacy minimal_balancing first, then run the usual soft median category capping.")
    p.add_argument("--legacy-minimal-resample-to-median", action="store_true",
                   help="Apply legacy minimal_balancing first, then resample every retained category/condition to the patient-specific median matched category size.")
    p.add_argument("--legacy-minimal-resample-fixed-target", type=int, default=None,
                   help="Apply legacy minimal_balancing first, then resample every retained category/condition to this fixed target.")
    p.add_argument("--sample-manifest", type=str, default=None,
                   help="CSV manifest of sampled row indices to apply before fitting. Columns: patient,condition,row_idx.")
    p.add_argument("--soft-balance-target", type=str, default="median",
                   help="Soft-balance cap target passed to run_clusterwise_cosine_distance. Examples: median, mean, q75.")
    p.add_argument("--resample-seed", type=int, default=42)
    return p.parse_args()


def _speaker_sort_key(speaker: str) -> int:
    return int(speaker.replace("Speaker", "").strip())


def _top_two_other_speakers(dir_membership, spike_dir, region):
    rows = []
    for speaker, mask in dir_membership.items():
        if speaker == "Speaker1":
            continue
        y = glm.load_spike_matrix(spike_dir, speaker, region)
        if y is None or y.ndim < 2:
            continue
        n_words = int(mask.sum())
        if y.shape[0] != n_words:
            print(
                f"  [WARN] {speaker}: transcript words={n_words} "
                f"but spike rows={y.shape[0]}; using spike rows for ranking",
                flush=True,
            )
            n_words = int(y.shape[0])
        rows.append((speaker, n_words))
    rows = sorted(rows, key=lambda x: (-x[1], _speaker_sort_key(x[0])))
    return rows[:2], rows


def _load_speaker_condition(spike_dir, speaker, region, mask, x_pca, cluster_ids_full):
    y_mat = glm.load_spike_matrix(spike_dir, speaker, region)
    if y_mat is None or y_mat.ndim < 2:
        return None
    if y_mat.shape[0] != int(mask.sum()):
        print(
            f"  {speaker}: row mismatch: Y rows={y_mat.shape[0]} "
            f"mask words={int(mask.sum())}",
            flush=True,
        )
        return None
    valid = ~np.isnan(y_mat).any(axis=1)
    x = StandardScaler().fit_transform(x_pca[mask][valid])
    y = y_mat[valid].astype(np.float64)
    clusters = cluster_ids_full[mask][valid]
    return x, y, pd.DataFrame({"ClusterID": clusters})


def build_other_other_data(cfg, region, layer, n_components, speaker_a_override=None, speaker_b_override=None):
    patient_ID = cfg["patient_ID"]
    patient = cfg["patient"]

    npy_path = os.path.join(
        glm.EMBED_DIR, f"{patient_ID}_{glm.MODEL_TAG}{glm.CONTEXT_TAG}_word_emb_layers.npy"
    )
    spike_dir = glm.find_spike_dir(patient)
    if not os.path.exists(npy_path) or spike_dir is None:
        print(f"  {patient_ID}: missing embeddings or spike dir -- skip", flush=True)
        return None

    try:
        _, _, _, dir_membership = glm.load_speaker_assignment(spike_dir)
    except FileNotFoundError as e:
        print(f"  {patient_ID}: {e} -- skip", flush=True)
        return None

    top2, all_other_counts = _top_two_other_speakers(dir_membership, spike_dir, region)
    if speaker_a_override is not None or speaker_b_override is not None:
        if speaker_a_override is None or speaker_b_override is None:
            raise ValueError("Must provide both --speaker-a and --speaker-b, or neither.")
        speaker_a = speaker_a_override
        speaker_b = speaker_b_override
        if speaker_a == "Speaker1" or speaker_b == "Speaker1" or speaker_a == speaker_b:
            raise ValueError("Forced other-other speakers must be distinct non-Speaker1 labels.")
        if speaker_a not in dir_membership or speaker_b not in dir_membership:
            print(
                f"  {patient_ID}/{region}: forced speakers {speaker_a}/{speaker_b} "
                "not present in duration file -- skip",
                flush=True,
            )
            return None
        ya = glm.load_spike_matrix(spike_dir, speaker_a, region)
        yb = glm.load_spike_matrix(spike_dir, speaker_b, region)
        if ya is None or yb is None:
            print(
                f"  {patient_ID}/{region}: missing spike matrix for forced "
                f"{speaker_a}/{speaker_b} -- skip",
                flush=True,
            )
            return None
        n_a = int(ya.shape[0])
        n_b = int(yb.shape[0])
    else:
        if len(top2) < 2:
            print(f"  {patient_ID}/{region}: fewer than 2 non-Speaker1 speakers with data", flush=True)
            return None
        speaker_a, n_a = top2[0]
        speaker_b, n_b = top2[1]

    embedding_cache = np.load(npy_path, mmap_mode="r")
    x_layer_raw = (
        embedding_cache[layer] if embedding_cache.ndim == 3 else embedding_cache
    ).astype(np.float32)

    cluster_ids_full = glm.load_cluster_ids(patient_ID, x_layer_raw.shape[0])
    if cluster_ids_full is None:
        return None

    x_pca = PCA(n_components=n_components).fit_transform(
        np.asarray(x_layer_raw, dtype=np.float64)
    )

    cond_a = _load_speaker_condition(
        spike_dir, speaker_a, region, dir_membership[speaker_a], x_pca, cluster_ids_full
    )
    cond_b = _load_speaker_condition(
        spike_dir, speaker_b, region, dir_membership[speaker_b], x_pca, cluster_ids_full
    )
    if cond_a is None or cond_b is None:
        return None

    x_a, y_a_raw, meta_a = cond_a
    x_b, y_b_raw, meta_b = cond_b

    spike_ok = (y_a_raw.sum(0) >= MIN_SPIKES) & (y_b_raw.sum(0) >= MIN_SPIKES)
    if spike_ok.sum() == 0:
        print(f"  {patient_ID}/{region}: 0 neurons pass spike filter -- skip", flush=True)
        return None

    y_a = y_a_raw[:, spike_ok].astype(np.float64)
    y_b = y_b_raw[:, spike_ok].astype(np.float64)

    print(
        f"  {patient_ID}/{region}: {speaker_a}={x_a.shape[0]} words, "
        f"{speaker_b}={x_b.shape[0]} words, neurons={int(spike_ok.sum())}",
        flush=True,
    )

    return dict(
        X_a=x_a,
        X_b=x_b,
        Y_a=y_a,
        Y_b=y_b,
        metadata_a=meta_a,
        metadata_b=meta_b,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        speaker_a_word_count=n_a,
        speaker_b_word_count=n_b,
        all_other_counts=all_other_counts,
    )


def _apply_legacy_minimal_balance(data):
    """Apply the notebook minimal_balancing routine while preserving X/Y row alignment."""
    meta_a = data["metadata_a"].copy().reset_index(drop=True)
    meta_b = data["metadata_b"].copy().reset_index(drop=True)
    meta_a["_row_idx"] = np.arange(len(meta_a))
    meta_b["_row_idx"] = np.arange(len(meta_b))

    balanced_a, balanced_b = minimal_balancing(
        meta_a, meta_b, cluster_column="ClusterID", verbose=True
    )

    keep_a = balanced_a["_row_idx"].to_numpy(dtype=int)
    keep_b = balanced_b["_row_idx"].to_numpy(dtype=int)

    out = dict(data)
    out["X_a"] = data["X_a"][keep_a]
    out["Y_a"] = data["Y_a"][keep_a]
    out["X_b"] = data["X_b"][keep_b]
    out["Y_b"] = data["Y_b"][keep_b]
    out["metadata_a"] = balanced_a.drop(columns=["_row_idx"], errors="ignore").reset_index(drop=True)
    out["metadata_b"] = balanced_b.drop(columns=["_row_idx"], errors="ignore").reset_index(drop=True)
    out["legacy_minimal_balance_n_a"] = len(keep_a)
    out["legacy_minimal_balance_n_b"] = len(keep_b)
    return out


def _resample_to_patient_median(data, seed=42, fixed_target=None):
    """Downsample large categories and upsample small ones to a common target.

    The target is the median, within this patient, of min(n_speaker_a, n_speaker_b)
    across retained categories. Each condition/category cell is sampled to exactly
    that target, using replacement only when the original cell is smaller.
    """
    rng = np.random.default_rng(seed)
    meta_a = data["metadata_a"].copy().reset_index(drop=True)
    meta_b = data["metadata_b"].copy().reset_index(drop=True)
    counts = []
    for cid in sorted(set(meta_a["ClusterID"].dropna()) & set(meta_b["ClusterID"].dropna())):
        idx_a = meta_a.index[meta_a["ClusterID"] == cid].to_numpy()
        idx_b = meta_b.index[meta_b["ClusterID"] == cid].to_numpy()
        if len(idx_a) > 0 and len(idx_b) > 0:
            counts.append(min(len(idx_a), len(idx_b)))

    if not counts:
        raise ValueError("No shared categories available for median resampling.")

    target = int(fixed_target) if fixed_target is not None else int(np.median(counts))
    target = max(1, target)

    keep_a_parts = []
    keep_b_parts = []
    clusters_kept = []
    for cid in sorted(set(meta_a["ClusterID"].dropna()) & set(meta_b["ClusterID"].dropna())):
        idx_a = meta_a.index[meta_a["ClusterID"] == cid].to_numpy()
        idx_b = meta_b.index[meta_b["ClusterID"] == cid].to_numpy()
        if len(idx_a) == 0 or len(idx_b) == 0:
            continue
        keep_a_parts.append(rng.choice(idx_a, size=target, replace=len(idx_a) < target))
        keep_b_parts.append(rng.choice(idx_b, size=target, replace=len(idx_b) < target))
        clusters_kept.append(cid)

    keep_a = np.concatenate(keep_a_parts)
    keep_b = np.concatenate(keep_b_parts)

    out = dict(data)
    out["X_a"] = data["X_a"][keep_a]
    out["Y_a"] = data["Y_a"][keep_a]
    out["X_b"] = data["X_b"][keep_b]
    out["Y_b"] = data["Y_b"][keep_b]
    out["metadata_a"] = meta_a.iloc[keep_a].reset_index(drop=True)
    out["metadata_b"] = meta_b.iloc[keep_b].reset_index(drop=True)
    out["median_resample_target"] = target
    out["median_resample_n_categories"] = len(clusters_kept)
    out["median_resample_clusters"] = ",".join(str(c) for c in clusters_kept)
    return out


def _apply_sample_manifest(data, manifest_path, patient_ID):
    """Apply a precomputed row-sampling manifest to one patient's A/B matrices.

    The manifest may contain repeated row indices, enabling pseudopopulation
    upsampling with replacement. Row indices refer to the post-valid-filter
    metadata/X/Y rows produced by ``build_other_other_data``.
    """
    manifest = pd.read_csv(manifest_path)
    sub = manifest[manifest["patient"] == patient_ID].copy()
    if sub.empty:
        raise ValueError(f"{patient_ID}: no rows found in sample manifest {manifest_path}")

    keep = {}
    for condition in ("a", "b"):
        rows = sub[sub["condition"] == condition]
        if rows.empty:
            raise ValueError(f"{patient_ID}: no condition={condition} rows in sample manifest")
        keep[condition] = rows["row_idx"].to_numpy(dtype=int)

    out = dict(data)
    out["X_a"] = data["X_a"][keep["a"]]
    out["Y_a"] = data["Y_a"][keep["a"]]
    out["metadata_a"] = data["metadata_a"].iloc[keep["a"]].reset_index(drop=True)
    out["X_b"] = data["X_b"][keep["b"]]
    out["Y_b"] = data["Y_b"][keep["b"]]
    out["metadata_b"] = data["metadata_b"].iloc[keep["b"]].reset_index(drop=True)
    out["sample_manifest"] = manifest_path
    out["sample_manifest_n_a"] = len(keep["a"])
    out["sample_manifest_n_b"] = len(keep["b"])
    return out


def main():
    args = _args()
    glm.MODEL_TAG = args.model_tag
    glm.CONTEXT_TAG = args.context_tag
    glm.WINDOW_TAG = args.window_tag

    run_poisson_ridge = glm.make_beta_fit_adapter(glm.ALPHAS)

    patients = [
        p for p in glm.PATIENTS
        if p["patient_ID"] in CANDIDATE_PATIENTS
        and (args.patient is None or p["patient_ID"] == args.patient)
    ]

    for cfg in patients:
        patient_ID = cfg["patient_ID"]
        if args.region not in cfg["region_ranges"]:
            print(f"{patient_ID}: no {args.region} region -- skip", flush=True)
            continue

        print(f"\n{'=' * 60}\n  {patient_ID} / {args.region}", flush=True)
        data = build_other_other_data(
            cfg,
            args.region,
            args.layer,
            args.n_components,
            speaker_a_override=args.speaker_a,
            speaker_b_override=args.speaker_b,
        )
        if data is None:
            continue

        if args.sample_manifest is not None:
            data = _apply_sample_manifest(data, args.sample_manifest, patient_ID)

        if (
            args.legacy_minimal_balance
            or args.legacy_minimal_plus_median_cap
            or args.legacy_minimal_resample_to_median
            or args.legacy_minimal_resample_fixed_target is not None
        ):
            data = _apply_legacy_minimal_balance(data)

        if args.legacy_minimal_resample_to_median or args.legacy_minimal_resample_fixed_target is not None:
            data = _resample_to_patient_median(
                data,
                seed=args.resample_seed,
                fixed_target=args.legacy_minimal_resample_fixed_target,
            )

        pair_label = f"{data['speaker_a']}_vs_{data['speaker_b']}"
        out_root = os.path.join(args.results_root, pair_label)

        pair_dir = os.path.join(out_root, patient_ID)
        os.makedirs(pair_dir, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "patient": patient_ID,
                    "region": args.region,
                    "speaker_a": data["speaker_a"],
                    "speaker_b": data["speaker_b"],
                    "speaker_a_word_count": data["speaker_a_word_count"],
                    "speaker_b_word_count": data["speaker_b_word_count"],
                    "all_other_counts": repr(data["all_other_counts"]),
                    "window_tag": args.window_tag,
                    "model_tag": args.model_tag,
                    "layer": args.layer,
                    "n_components": args.n_components,
                    "speaker_a_override": args.speaker_a,
                    "speaker_b_override": args.speaker_b,
                    "legacy_minimal_balance": bool(
                        args.legacy_minimal_balance
                        or args.legacy_minimal_plus_median_cap
                        or args.legacy_minimal_resample_to_median
                        or args.legacy_minimal_resample_fixed_target is not None
                    ),
                    "legacy_minimal_plus_median_cap": bool(args.legacy_minimal_plus_median_cap),
                    "legacy_minimal_resample_to_median": bool(args.legacy_minimal_resample_to_median),
                    "legacy_minimal_resample_fixed_target": args.legacy_minimal_resample_fixed_target,
                    "sample_manifest": data.get("sample_manifest", ""),
                    "sample_manifest_n_a": data.get("sample_manifest_n_a", np.nan),
                    "sample_manifest_n_b": data.get("sample_manifest_n_b", np.nan),
                    "legacy_minimal_balance_n_a": data.get("legacy_minimal_balance_n_a", np.nan),
                    "legacy_minimal_balance_n_b": data.get("legacy_minimal_balance_n_b", np.nan),
                    "median_resample_target": data.get("median_resample_target", np.nan),
                    "median_resample_n_categories": data.get("median_resample_n_categories", np.nan),
                    "median_resample_clusters": data.get("median_resample_clusters", ""),
                }
            ]
        ).to_csv(os.path.join(pair_dir, f"{args.region}_speaker_pair_summary.csv"), index=False)

        report_cluster_balance(
            data["metadata_a"], data["metadata_b"], patient_ID, args.region
        )

        run_clusterwise_cosine_distance(
            X_self=data["X_a"],
            X_other=data["X_b"],
            Y_self=data["Y_a"],
            Y_other=data["Y_b"],
            metadata_self=data["metadata_a"],
            metadata_other=data["metadata_b"],
            cluster_column="ClusterID",
            region_name=args.region,
            patient_id=patient_ID,
            n_components=args.n_components,
            results_root=out_root,
            run_poisson_ridge=run_poisson_ridge,
            balance_function_words=True,
            compute_half_splits=True,
            n_jobs=args.n_jobs,
            cap_large_clusters=False,
            min_trials_per_condition=args.min_trials_per_condition,
            print_trial_counts=True,
            max_size_ratio=2,
            cap_reference="median",
            balance_all_clusters=(
                not args.no_balance
                and args.sample_manifest is None
                and not args.legacy_minimal_balance
                and not args.legacy_minimal_resample_to_median
                and args.legacy_minimal_resample_fixed_target is None
            ),
            soft_balance=(
                not args.no_balance
                and args.sample_manifest is None
                and not args.legacy_minimal_balance
                and not args.legacy_minimal_resample_to_median
                and args.legacy_minimal_resample_fixed_target is None
            ),
            soft_balance_target=args.soft_balance_target,
        )

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
