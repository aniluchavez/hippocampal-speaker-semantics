#!/usr/bin/env python3
"""Disjoint-subset beta stability learning curves for separate category GLMs."""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import cosine

sys.path.insert(0, os.path.dirname(__file__))
import cluster_glm_reliability as glm

WINDOW_TAG = "tref-offset_shift-200_tlen500_oref-onset_shift+100_olen500"
ALPHAS = np.asarray([1e-3, 1e-2, 1e-1, 1, 10, 100, 300, 1000, 3000])
SIZES = np.asarray([5, 8, 10, 15, 20, 30, 40, 60, 80, 100, 150, 200])
NAMES = {
    1: "Body Parts", 2: "Places", 3: "Emotional", 4: "Mental",
    5: "Social", 6: "Objects", 7: "Visual", 8: "Numerical",
    9: "Actions", 10: "Identity", 11: "Function Words",
}


def fit_one(patient, cid, category_name, condition, neuron, X, y, repeats):
    alpha = glm.select_alpha_cv(X, y, ALPHAS, n_splits=3, random_state=0)
    rows = []
    for n in SIZES[SIZES * 2 <= len(X)]:
        for repeat in range(repeats):
            rng = np.random.RandomState(
                100000 + cid*10000 + neuron*100 + repeat +
                (0 if condition == "self" else 500000))
            selected = rng.choice(len(X), 2*n, replace=False)
            ia, ib = selected[:n], selected[n:]
            ba = glm.fit_poisson_ridge_beta_sklearn(X[ia], y[ia], alpha)
            bb = glm.fit_poisson_ridge_beta_sklearn(X[ib], y[ib], alpha)
            # Matched chance null: preserve each subset's X and y marginals,
            # alpha, neuron, category, condition, and N, but destroy X↔y pairing.
            na_perm = rng.permutation(n)
            nb_perm = rng.permutation(n)
            ba_null = glm.fit_poisson_ridge_beta_sklearn(
                X[ia][na_perm], y[ia], alpha)
            bb_null = glm.fit_poisson_ridge_beta_sklearn(
                X[ib][nb_perm], y[ib], alpha)
            na, nb = np.linalg.norm(ba), np.linalg.norm(bb)
            similarity = 1 - cosine(ba, bb) if na and nb else np.nan
            nna, nnb = np.linalg.norm(ba_null), np.linalg.norm(bb_null)
            null_similarity = (
                1 - cosine(ba_null, bb_null) if nna and nnb else np.nan)
            # Variance around the two-fit mean, normalized by beta energy.
            relative_variance = (
                np.sum((ba - bb)**2) / (2 * (na**2 + nb**2))
                if na or nb else np.nan)
            rows.append({
                "patient": patient, "cluster_id": cid,
                "category": category_name, "condition": condition,
                "neuron": neuron, "available_trials": len(X),
                "subset_trials": int(n), "repeat": repeat, "alpha": alpha,
                "beta_cosine_similarity": similarity,
                "null_beta_cosine_similarity": null_similarity,
                "excess_beta_similarity": similarity - null_similarity,
                "relative_beta_variance": relative_variance,
            })
    return rows


def main():
    global SIZES, ALPHAS
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="PTYEZ_task60")
    ap.add_argument("--region", default="hippocampus")
    ap.add_argument("--repeats", type=int, default=25)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--model-tag", default="gpt2-large")
    ap.add_argument("--window-tag", default=WINDOW_TAG)
    ap.add_argument("--context-tag", default=None)
    ap.add_argument("--layer", type=int, default=36)
    ap.add_argument("--n-components", type=int, default=10)
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES.tolist())
    ap.add_argument("--alphas", type=float, nargs="+", default=ALPHAS.tolist())
    args = ap.parse_args()
    SIZES = np.asarray(args.sizes, dtype=int)
    ALPHAS = np.asarray(args.alphas, dtype=float)
    glm.MODEL_TAG = args.model_tag
    glm.WINDOW_TAG = args.window_tag
    if args.context_tag is not None:
        glm.CONTEXT_TAG = args.context_tag
    cfg = next(p for p in glm.PATIENTS if p["patient_ID"] == args.patient)
    data = glm.build_patient_region_data(
        cfg, args.region, args.layer, args.n_components)
    if data is None:
        raise SystemExit(f"No data for {args.patient} / {args.region}")
    tasks = []
    for condition in ("self", "other"):
        clusters = data[f"metadata_{condition}"]["ClusterID"].to_numpy()
        for cid in sorted(np.unique(clusters)):
            idx = np.flatnonzero(clusters == cid)
            if len(idx) < 10:
                continue
            X, Y = data[f"X_{condition}"][idx], data[f"Y_{condition}"][idx]
            for neuron in range(Y.shape[1]):
                tasks.append((
                    args.patient, int(cid), NAMES.get(cid, str(cid)), condition,
                    neuron, X, Y[:, neuron], args.repeats))
    nested = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(fit_one)(*task) for task in tasks)
    detail = pd.DataFrame([row for block in nested for row in block])
    summary = detail.groupby(
        ["cluster_id", "category", "condition", "available_trials",
         "subset_trials"], as_index=False
    ).agg(
        n_neurons=("neuron", "nunique"),
        median_beta_similarity=("beta_cosine_similarity", "median"),
        median_null_similarity=("null_beta_cosine_similarity", "median"),
        median_excess_similarity=("excess_beta_similarity", "median"),
        q25_beta_similarity=("beta_cosine_similarity", lambda x: x.quantile(.25)),
        median_relative_variance=("relative_beta_variance", "median"),
        q75_relative_variance=("relative_beta_variance", lambda x: x.quantile(.75)),
    )
    overall = detail.groupby("subset_trials", as_index=False).agg(
        cells=("cluster_id", "size"),
        median_beta_similarity=("beta_cosine_similarity", "median"),
        median_null_similarity=("null_beta_cosine_similarity", "median"),
        median_excess_similarity=("excess_beta_similarity", "median"),
        median_relative_variance=("relative_beta_variance", "median"),
    )
    repeat_stats = detail.groupby(
        ["cluster_id", "category", "condition", "available_trials",
         "subset_trials", "repeat"], as_index=False
    ).agg(
        observed_stat=("beta_cosine_similarity", "median"),
        null_stat=("null_beta_cosine_similarity", "median"),
    )
    repeat_stats["delta"] = repeat_stats.observed_stat-repeat_stats.null_stat
    inference_rows = []
    key_cols = ["cluster_id", "category", "condition",
                "available_trials", "subset_trials"]
    for keys, group in repeat_stats.groupby(key_cols):
        observed = group.observed_stat.median()
        p_emp = (1 + np.sum(group.null_stat >= observed)) / (len(group) + 1)
        inference_rows.append(dict(
            zip(key_cols, keys),
            observed_median_similarity=observed,
            null_median_similarity=group.null_stat.median(),
            median_excess_similarity=group.delta.median(),
            delta_ci_2_5=group.delta.quantile(.025),
            delta_ci_97_5=group.delta.quantile(.975),
            empirical_p=p_emp,
        ))
    inference = pd.DataFrame(inference_rows)
    out = os.path.join(args.results_root, args.patient)
    os.makedirs(out, exist_ok=True)
    detail.to_csv(os.path.join(
        out, "hippocampus_beta_variance_learning_curve_detail.csv"), index=False)
    summary.to_csv(os.path.join(
        out, "hippocampus_beta_variance_learning_curve_summary.csv"), index=False)
    overall.to_csv(os.path.join(
        out, "hippocampus_beta_variance_learning_curve_overall.csv"), index=False)
    inference.to_csv(os.path.join(
        out, "hippocampus_beta_variance_learning_curve_null_inference.csv"),
        index=False)
    print(overall.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
