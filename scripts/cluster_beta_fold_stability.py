#!/usr/bin/env python3
"""Five-fold coefficient stability diagnostic for one word-duration GLM patient."""

import argparse
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from scipy.spatial.distance import cosine

sys.path.insert(0, os.path.dirname(__file__))
import cluster_glm_worddur_offset as wd


CATEGORY = {
    1: "Body Parts", 2: "Places", 3: "Emotional", 4: "Mental",
    5: "Social", 6: "Objects", 7: "Visual", 8: "Numerical",
    9: "Actions", 10: "Identity", 11: "Function Words",
}


def balanced_data(data, seed=42):
    """Reproduce minimal balancing followed by median soft balancing."""
    ms = data["metadata_self"].copy()
    mo = data["metadata_other"].copy()
    ms["_orig_ix"] = np.arange(len(ms))
    mo["_orig_ix"] = np.arange(len(mo))
    ms, mo = wd.minimal_balancing(ms, mo, cluster_column="ClusterID")

    out = {}
    for cond, meta in (("self", ms), ("other", mo)):
        ix = meta["_orig_ix"].to_numpy()
        out[cond] = {
            "X": data[f"X_{cond}"][ix],
            "Y": data[f"Y_{cond}"][ix],
            "offset": data[f"offset_{cond}"][ix],
            "cluster": meta["ClusterID"].to_numpy(),
        }

    eligible = []
    for cid in sorted(set(out["self"]["cluster"]) | set(out["other"]["cluster"])):
        ns = np.sum(out["self"]["cluster"] == cid)
        no = np.sum(out["other"]["cluster"] == cid)
        if ns >= 10 and no >= 10:
            eligible.append(min(ns, no))
    target = max(int(np.floor(np.median(eligible))), 10)

    rng = np.random.RandomState(seed)
    cells = {}
    for cid in sorted(set(out["self"]["cluster"]) | set(out["other"]["cluster"])):
        ns = np.flatnonzero(out["self"]["cluster"] == cid)
        no = np.flatnonzero(out["other"]["cluster"] == cid)
        if len(ns) < 10 or len(no) < 10:
            continue
        n = min(len(ns), len(no), target)
        ns = ns if len(ns) <= n else np.sort(rng.choice(ns, n, replace=False))
        no = no if len(no) <= n else np.sort(rng.choice(no, n, replace=False))
        cells[cid] = {
            "self": tuple(out["self"][k][ns] for k in ("X", "Y", "offset")),
            "other": tuple(out["other"][k][no] for k in ("X", "Y", "offset")),
        }
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="PTYEZ_task60")
    ap.add_argument("--region", default="hippocampus")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fast-alpha", action="store_true",
                    help="Use a 9-value broad alpha grid and 3-fold inner CV.")
    args = ap.parse_args()
    alphas = (np.asarray([1e-3, 1e-2, 1e-1, 1, 10, 100, 300, 1000, 3000])
              if args.fast_alpha else wd.glm.ALPHAS)
    inner_folds = 3 if args.fast_alpha else 5
    suffix = "_fast" if args.fast_alpha else ""

    cfg = next(p for p in wd.glm.PATIENTS if p["patient_ID"] == args.patient)
    data = wd.build_patient_region_data_worddur(
        cfg, args.region, wd.glm.LAYER, wd.N_COMPONENTS)
    cells = balanced_data(data)
    rows = []

    for cid, conditions in cells.items():
        for condition, (X, Y, offset) in conditions.items():
            folds = list(KFold(args.folds, shuffle=True, random_state=42).split(X))
            for neuron in range(Y.shape[1]):
                y = Y[:, neuron]
                # Select alpha once on the complete cell, then hold it fixed.
                # This isolates coefficient sensitivity to trial membership.
                alpha = wd.select_alpha_cv_gpu(
                    X, y, alphas, offset=offset,
                    n_splits=min(inner_folds, len(X) // 2), random_state=0)
                betas = []
                for train, _ in folds:
                    betas.append(wd.fit_poisson_ridge_beta_gpu(
                        X[train], y[train], alpha, offset=offset[train]))
                betas = np.asarray(betas)
                similarities = [
                    1.0 - cosine(betas[a], betas[b])
                    for a, b in combinations(range(len(betas)), 2)
                    if np.linalg.norm(betas[a]) and np.linalg.norm(betas[b])
                ]
                centered_ss = np.mean(np.sum(
                    (betas - betas.mean(axis=0)) ** 2, axis=1))
                total_ss = np.mean(np.sum(betas ** 2, axis=1))
                rows.append({
                    "patient": args.patient, "region": args.region,
                    "cluster_id": cid, "category": CATEGORY.get(cid, str(cid)),
                    "condition": condition, "neuron": neuron,
                    "n_trials": len(X), "alpha": alpha,
                    "mean_pairwise_beta_cosine": np.mean(similarities),
                    "min_pairwise_beta_cosine": np.min(similarities),
                    "relative_beta_variance": centered_ss / total_ss
                    if total_ss > 0 else np.nan,
                })
        print(f"finished {cid}: {CATEGORY.get(cid, cid)}", flush=True)

    detail = pd.DataFrame(rows)
    summary = detail.groupby(
        ["patient", "region", "cluster_id", "category", "condition", "n_trials"],
        as_index=False,
    ).agg(
        neurons=("neuron", "nunique"),
        median_beta_cosine=("mean_pairwise_beta_cosine", "median"),
        q25_beta_cosine=("mean_pairwise_beta_cosine", lambda x: x.quantile(.25)),
        median_relative_beta_variance=("relative_beta_variance", "median"),
        q75_relative_beta_variance=("relative_beta_variance", lambda x: x.quantile(.75)),
    )
    out = os.path.join(wd.RESULTS_ROOT, args.patient)
    detail.to_csv(os.path.join(
        out, f"{args.region}_beta_fold_stability{suffix}_detail.csv"), index=False)
    summary.to_csv(os.path.join(
        out, f"{args.region}_beta_fold_stability{suffix}_summary.csv"), index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
