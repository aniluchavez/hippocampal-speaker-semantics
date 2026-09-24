#!/usr/bin/env python3
"""Repeated-fold beta stability for the PTYEZ GPT-2 L36 fixed-window run."""

import argparse
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import cosine
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(__file__))
import cluster_glm_reliability as glm

REGION = "hippocampus"
WINDOW_TAG = "tref-offset_shift-200_tlen500_oref-onset_shift+100_olen500"
OUT_ROOT = ("/scratch/aniluchavez/ConvoDATAS/SemanticGLM/"
            "clusterwise_gpt2large_L36_fixed_custom_hardbalanced_min30")
ALPHAS = np.asarray([1e-3, 1e-2, 1e-1, 1, 10, 100, 300, 1000, 3000])
NAMES = {
    1: "Body Parts", 2: "Places", 3: "Emotional", 4: "Mental",
    5: "Social", 6: "Objects", 7: "Visual", 8: "Numerical",
    9: "Actions", 10: "Identity", 11: "Function Words",
}


def prepare_cells(data, min_trials=30, balance_mode="hard"):
    ms, mo = data["metadata_self"].copy(), data["metadata_other"].copy()
    ms["_orig_ix"], mo["_orig_ix"] = np.arange(len(ms)), np.arange(len(mo))
    ms, mo = glm._cluster_mod.minimal_balancing(ms, mo, cluster_column="ClusterID")
    out = {}
    for cond, meta in (("self", ms), ("other", mo)):
        ix = meta["_orig_ix"].to_numpy()
        out[cond] = {
            "X": data[f"X_{cond}"][ix],
            "Y": data[f"Y_{cond}"][ix],
            "cluster": meta["ClusterID"].to_numpy(),
        }
    eligible = sorted(
        c for c in set(out["self"]["cluster"]) | set(out["other"]["cluster"])
        if np.sum(out["self"]["cluster"] == c) >= min_trials
        and np.sum(out["other"]["cluster"] == c) >= min_trials
    )
    matched = [
        min(np.sum(out["self"]["cluster"] == c),
            np.sum(out["other"]["cluster"] == c))
        for c in eligible
    ]
    target = (min(matched) if balance_mode == "hard"
              else int(np.floor(np.median(matched))))
    rng = np.random.RandomState(42)
    cells = {}
    for cid in eligible:
        cells[cid] = {}
        for cond in ("self", "other"):
            ix = np.flatnonzero(out[cond]["cluster"] == cid)
            if len(ix) > target:
                ix = np.sort(rng.choice(ix, target, replace=False))
            cells[cid][cond] = (out[cond]["X"][ix], out[cond]["Y"][ix])
    return cells, target


def fit_one(patient, cid, condition, neuron, X, y):
    alpha = glm.select_alpha_cv(
        X, y, ALPHAS, n_splits=3, random_state=0)
    folds = KFold(5, shuffle=True, random_state=42)
    betas = [
        glm.fit_poisson_ridge_beta_sklearn(X[tr], y[tr], alpha)
        for tr, _ in folds.split(X)
    ]
    betas = np.asarray(betas)
    sims = [
        1 - cosine(betas[a], betas[b])
        for a, b in combinations(range(5), 2)
        if np.linalg.norm(betas[a]) and np.linalg.norm(betas[b])
    ]
    centered = np.mean(np.sum((betas - betas.mean(0)) ** 2, axis=1))
    total = np.mean(np.sum(betas ** 2, axis=1))
    return {
        "patient": patient, "region": REGION, "cluster_id": cid,
        "category": NAMES[cid], "condition": condition, "neuron": neuron,
        "n_trials": len(X), "outer_train_trials": 4 * len(X) // 5,
        "alpha": alpha,
        "mean_pairwise_beta_cosine": np.mean(sims) if sims else np.nan,
        "min_pairwise_beta_cosine": np.min(sims) if sims else np.nan,
        "relative_beta_variance": centered / total if total else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="PTYEZ_task60")
    ap.add_argument("--results-root", default=OUT_ROOT)
    ap.add_argument("--balance-mode", choices=["hard", "soft"], default="hard")
    ap.add_argument("--min-trials", type=int, default=30)
    args = ap.parse_args()
    glm.MODEL_TAG = "gpt2-large"
    glm.WINDOW_TAG = WINDOW_TAG
    cfg = next(p for p in glm.PATIENTS if p["patient_ID"] == args.patient)
    data = glm.build_patient_region_data(cfg, REGION, 36, 10)
    cells, target = prepare_cells(
        data, min_trials=args.min_trials, balance_mode=args.balance_mode)
    tasks = []
    for cid, conditions in cells.items():
        for condition, (X, Y) in conditions.items():
            for neuron in range(Y.shape[1]):
                tasks.append((args.patient, cid, condition, neuron, X, Y[:, neuron]))
    rows = Parallel(n_jobs=8, verbose=10)(
        delayed(fit_one)(*task) for task in tasks)
    detail = pd.DataFrame(rows)
    summary = detail.groupby(
        ["cluster_id", "category", "condition", "n_trials", "outer_train_trials"],
        as_index=False,
    ).agg(
        neurons=("neuron", "nunique"),
        median_beta_cosine=("mean_pairwise_beta_cosine", "median"),
        q25_beta_cosine=("mean_pairwise_beta_cosine", lambda x: x.quantile(.25)),
        pct_neurons_below_05=("mean_pairwise_beta_cosine", lambda x: 100*(x < .5).mean()),
        median_relative_beta_variance=("relative_beta_variance", "median"),
    )
    out = os.path.join(args.results_root, args.patient)
    detail.to_csv(os.path.join(out, f"{REGION}_beta_fold_stability_fast_detail.csv"),
                  index=False)
    summary.to_csv(os.path.join(out, f"{REGION}_beta_fold_stability_fast_summary.csv"),
                   index=False)
    print(f"Hard-balance target: {target}", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
