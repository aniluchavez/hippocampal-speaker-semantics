#!/usr/bin/env python3
"""Build an other-other manifest that only downsamples Function Words.

All non-Function-Word rows are retained. Function Words (ClusterID=11) are
downsampled globally, separately for speaker condition A and B, to the largest
available non-Function-Word category count for that condition unless explicit
targets are provided.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import cluster_glm_other_other as oo
import cluster_glm_reliability as glm


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--region", default="hippocampus")
    p.add_argument("--function-cluster", type=int, default=11)
    p.add_argument("--target-a", type=int, default=None)
    p.add_argument("--target-b", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-tag", default="bert-base-bidirmax")
    p.add_argument("--context-tag", default="")
    p.add_argument("--window-tag", default="tshift-150_tlen500_oshift+200_olen500")
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--n-components", type=int, default=10)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = _args()
    glm.MODEL_TAG = args.model_tag
    glm.CONTEXT_TAG = args.context_tag
    glm.WINDOW_TAG = args.window_tag

    rng = np.random.default_rng(args.seed)
    rows = []
    patients = [p for p in glm.PATIENTS if p["patient_ID"] in oo.CANDIDATE_PATIENTS]

    for cfg in patients:
        patient_ID = cfg["patient_ID"]
        if args.region not in cfg["region_ranges"]:
            continue
        print(f"Loading {patient_ID}", flush=True)
        data = oo.build_other_other_data(cfg, args.region, args.layer, args.n_components)
        if data is None:
            continue
        for condition, meta_key, speaker_key in (
            ("a", "metadata_a", "speaker_a"),
            ("b", "metadata_b", "speaker_b"),
        ):
            meta = data[meta_key].reset_index(drop=True)
            for row_idx, cid in enumerate(meta["ClusterID"].to_numpy()):
                if pd.isna(cid):
                    continue
                rows.append(
                    {
                        "patient": patient_ID,
                        "condition": condition,
                        "speaker": data[speaker_key],
                        "cluster_id": int(cid),
                        "row_idx": row_idx,
                    }
                )

    pool = pd.DataFrame(rows)
    if pool.empty:
        raise RuntimeError("No rows available.")

    summary_before = (
        pool.groupby(["cluster_id", "condition"])
        .size()
        .reset_index(name="available_n")
        .sort_values(["cluster_id", "condition"])
    )

    non_fw = summary_before[summary_before["cluster_id"] != args.function_cluster]
    inferred = non_fw.groupby("condition")["available_n"].max().to_dict()
    targets = {
        "a": int(args.target_a) if args.target_a is not None else int(inferred["a"]),
        "b": int(args.target_b) if args.target_b is not None else int(inferred["b"]),
    }

    keep_parts = []
    summary_rows = []
    for (cid, condition), group in pool.groupby(["cluster_id", "condition"]):
        if cid == args.function_cluster:
            target = min(targets[condition], len(group))
            chosen = rng.choice(group.index.to_numpy(), size=target, replace=False)
            kept = pool.loc[chosen].copy()
            action = "downsample_function_words"
        else:
            kept = group.copy()
            target = len(group)
            action = "keep_raw"
        kept["draw_index"] = np.arange(len(kept))
        kept["target"] = target
        kept["available_n"] = len(group)
        kept["sampled_with_replacement"] = False
        keep_parts.append(kept)
        summary_rows.append(
            {
                "cluster_id": cid,
                "condition": condition,
                "available_n": len(group),
                "target_or_kept_n": target,
                "action": action,
                "n_patients_available": group["patient"].nunique(),
            }
        )

    manifest = pd.concat(keep_parts, ignore_index=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)
    summary = pd.DataFrame(summary_rows).sort_values(["cluster_id", "condition"])
    summary_path = out.with_name(out.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"Wrote {out}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Function-word targets: {targets}", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
