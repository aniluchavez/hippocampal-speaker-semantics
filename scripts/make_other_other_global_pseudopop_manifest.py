#!/usr/bin/env python3
"""Build a global other-other pseudopopulation sampling manifest.

This balances counts globally across all candidate patients, not within each
patient. For each semantic category and speaker condition, it samples exactly
``target`` rows pooled across patients. Sampling is with replacement only when
the pooled cell has fewer than ``target`` rows.
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
    p.add_argument("--target", type=int, default=300)
    p.add_argument("--exclude-cluster", type=int, action="append", default=[])
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
    pools = []
    patients = [
        p for p in glm.PATIENTS
        if p["patient_ID"] in oo.CANDIDATE_PATIENTS
    ]

    for cfg in patients:
        patient_ID = cfg["patient_ID"]
        if args.region not in cfg["region_ranges"]:
            continue
        print(f"Loading {patient_ID}", flush=True)
        data = oo.build_other_other_data(
            cfg, args.region, args.layer, args.n_components
        )
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
                cid = int(cid)
                if cid in set(args.exclude_cluster):
                    continue
                pools.append(
                    {
                        "patient": patient_ID,
                        "condition": condition,
                        "speaker": data[speaker_key],
                        "cluster_id": cid,
                        "row_idx": row_idx,
                    }
                )

    pool_df = pd.DataFrame(pools)
    if pool_df.empty:
        raise RuntimeError("No rows available for manifest.")

    manifest_parts = []
    summary_rows = []
    for (cid, condition), group in pool_df.groupby(["cluster_id", "condition"]):
        idx = group.index.to_numpy()
        replace = len(idx) < args.target
        chosen = rng.choice(idx, size=args.target, replace=replace)
        sampled = pool_df.loc[chosen].copy()
        sampled["draw_index"] = np.arange(args.target)
        sampled["target"] = args.target
        sampled["available_n"] = len(idx)
        sampled["sampled_with_replacement"] = replace
        manifest_parts.append(sampled)
        summary_rows.append(
            {
                "cluster_id": cid,
                "condition": condition,
                "available_n": len(idx),
                "target": args.target,
                "sampled_with_replacement": replace,
                "n_patients_available": group["patient"].nunique(),
            }
        )

    manifest = pd.concat(manifest_parts, ignore_index=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)
    summary = pd.DataFrame(summary_rows).sort_values(["cluster_id", "condition"])
    summary_path = out.with_name(out.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"Wrote {out}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
