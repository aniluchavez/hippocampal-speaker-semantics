#!/usr/bin/env python3
"""Joint category×speaker interaction GLM for other-vs-other controls.

This mirrors ``joint_category_interaction_glm.py`` but replaces self/other with
the two non-Speaker1 speakers with the most words for each patient. The key
held-out comparison is the full model versus a reduced model with the
category-specific speaker×semantic-PC deviation terms removed.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import cosine

sys.path.insert(0, os.path.dirname(__file__))
import cluster_glm_other_other as oo  # noqa: E402
import cluster_glm_reliability as glm  # noqa: E402
from joint_category_interaction_glm import (  # noqa: E402
    ALPHAS,
    NAMES,
    choose_alpha,
    fit_neuron,
    make_design,
)


def _args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", required=True)
    ap.add_argument("--region", default="hippocampus")
    ap.add_argument("--model-tag", default="bert-base-bidirmax")
    ap.add_argument("--context-tag", default="")
    ap.add_argument("--window-tag", default="tshift-150_tlen500_oshift+200_olen500")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--n-components", type=int, default=30)
    ap.add_argument("--deviation-penalty-multiplier", type=float, default=10.0)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--results-root", required=True)
    return ap.parse_args()


def main():
    args = _args()
    glm.MODEL_TAG = args.model_tag
    glm.CONTEXT_TAG = args.context_tag
    glm.WINDOW_TAG = args.window_tag

    cfg = next(p for p in glm.PATIENTS if p["patient_ID"] == args.patient)
    if args.region not in cfg["region_ranges"]:
        print(f"{args.patient}: no {args.region}; skipping", flush=True)
        return

    data = oo.build_other_other_data(cfg, args.region, args.layer, args.n_components)
    if data is None:
        print(f"{args.patient}: no usable other-other data", flush=True)
        return

    X = np.vstack([data["X_a"], data["X_b"]])
    Y = np.vstack([data["Y_a"], data["Y_b"]])
    category = np.concatenate([
        data["metadata_a"]["ClusterID"].to_numpy(),
        data["metadata_b"]["ClusterID"].to_numpy(),
    ]).astype(int)
    condition = np.concatenate([
        np.zeros(len(data["X_a"]), dtype=int),
        np.ones(len(data["X_b"]), dtype=int),
    ])

    categories = sorted(np.unique(category))
    strata = np.asarray([f"{c}_{q}" for c, q in zip(category, condition)])
    deviation_scale = 1 / np.sqrt(args.deviation_penalty_multiplier)

    D, semantic_start = make_design(
        X, category, condition, categories, deviation_scale
    )
    baseline = D[:, :semantic_start]
    reduced = D[:, :-len(categories) * args.n_components]

    fitted = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(fit_neuron)(n, D, reduced, baseline, Y[:, n], strata)
        for n in range(Y.shape[1])
    )

    detail = pd.DataFrame([
        {k: v for k, v in row.items() if k != "coef"} for row in fitted
    ])

    p = args.n_components
    ncat = len(categories)
    cosine_rows = []
    for row in fitted:
        coef = row["coef"]
        sem = coef[semantic_start:]
        shared = sem[:p]
        cond_shared = sem[p:2*p]
        cat_dev = sem[2*p:2*p+ncat*p].reshape(ncat, p) * deviation_scale
        cat_cond = sem[2*p+ncat*p:].reshape(ncat, p) * deviation_scale
        for j, cid in enumerate(categories):
            beta_a = shared + cat_dev[j]
            beta_b = shared + cond_shared + cat_dev[j] + cat_cond[j]
            distance = (
                cosine(beta_a, beta_b)
                if np.linalg.norm(beta_a) and np.linalg.norm(beta_b)
                else np.nan
            )
            cosine_rows.append({
                "neuron": row["neuron"],
                "cluster_id": cid,
                "category": NAMES.get(cid, str(cid)),
                "cosine_distance": distance,
            })

    cosine_df = pd.DataFrame(cosine_rows)
    summary = pd.DataFrame([{
        "patient": args.patient,
        "region": args.region,
        "speaker_a": data["speaker_a"],
        "speaker_b": data["speaker_b"],
        "speaker_a_word_count": data["speaker_a_word_count"],
        "speaker_b_word_count": data["speaker_b_word_count"],
        "all_other_counts": repr(data["all_other_counts"]),
        "model_tag": args.model_tag,
        "context_tag": args.context_tag,
        "window_tag": args.window_tag,
        "layer": args.layer,
        "n_components": p,
        "n_trials": len(X),
        "n_neurons": Y.shape[1],
        "n_categories": ncat,
        "deviation_penalty_multiplier": args.deviation_penalty_multiplier,
        "total_heldout_ll_improvement": detail.heldout_ll_improvement.sum(),
        "median_neuron_ll_improvement": detail.heldout_ll_improvement.median(),
        "pct_neurons_positive": 100 * (detail.heldout_ll_improvement > 0).mean(),
        "median_positive_fold_fraction": detail.positive_fold_fraction.median(),
        "total_full_minus_reduced_ll": detail.full_minus_reduced_ll.sum(),
        "median_full_minus_reduced_ll": detail.full_minus_reduced_ll.median(),
        "pct_neurons_full_beats_reduced":
            100 * (detail.full_minus_reduced_ll > 0).mean(),
        "median_full_beats_reduced_fold_fraction":
            detail.full_beats_reduced_fold_fraction.median(),
    }])

    pair_label = f"{data['speaker_a']}_vs_{data['speaker_b']}"
    out = os.path.join(args.results_root, pair_label, args.patient)
    os.makedirs(out, exist_ok=True)
    detail.to_csv(os.path.join(out, f"{args.region}_joint_glm_nested_cv.csv"), index=False)
    cosine_df.to_csv(
        os.path.join(out, f"{args.region}_joint_glm_category_cosine_distances.csv"),
        index=False,
    )
    summary.to_csv(os.path.join(out, f"{args.region}_joint_glm_summary.csv"), index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
