#!/usr/bin/env python3
"""Joint partially-pooled category interaction GLM for one patient."""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import cosine
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(__file__))
import cluster_glm_reliability as glm

WINDOW_TAG = "tref-offset_shift-200_tlen500_oref-onset_shift+100_olen500"
ALPHAS = np.asarray([1e-3, 1e-2, 1e-1, 1, 10, 100, 300, 1000, 3000, 10000])
NAMES = {
    1: "Body Parts", 2: "Places", 3: "Emotional", 4: "Mental",
    5: "Social", 6: "Objects", 7: "Visual", 8: "Numerical",
    9: "Actions", 10: "Identity", 11: "Function Words",
}


def make_design(X, category, condition, categories, deviation_scale):
    """Shared slopes plus shrunken category and category×condition deviations."""
    onehot = np.column_stack([category == c for c in categories]).astype(float)
    cond = condition[:, None].astype(float)
    baseline = np.column_stack([condition, onehot, onehot * cond])
    shared = np.column_stack([X, X * cond])
    cat_pc = np.concatenate([X * onehot[:, j:j+1]
                             for j in range(len(categories))], axis=1)
    cat_cond_pc = cat_pc * np.tile(cond, (1, X.shape[1] * len(categories)))
    semantic = np.column_stack([
        shared, deviation_scale * cat_pc, deviation_scale * cat_cond_pc])
    return np.column_stack([baseline, semantic]), baseline.shape[1]


def choose_alpha(D, y, strata, alphas, seed):
    splits = StratifiedKFold(3, shuffle=True, random_state=seed)
    best_alpha, best_ll = alphas[0], -np.inf
    for alpha in alphas:
        score = 0.0
        for train, test in splits.split(D, strata):
            model = PoissonRegressor(alpha=alpha, max_iter=1000)
            model.fit(D[train], y[train])
            score += glm.poisson_ll_numpy(y[test], model.predict(D[test]))
        if score > best_ll:
            best_alpha, best_ll = alpha, score
    return float(best_alpha)


def fit_neuron(neuron, D, reduced, baseline, y, strata):
    outer = StratifiedKFold(5, shuffle=True, random_state=42)
    ll_model = ll_reduced = ll_null = 0.0
    positive_folds = 0
    full_beats_reduced_folds = 0
    for fold, (train, test) in enumerate(outer.split(D, strata)):
        alpha = choose_alpha(D[train], y[train], strata[train], ALPHAS, fold)
        alpha_reduced = choose_alpha(
            reduced[train], y[train], strata[train], ALPHAS, 50 + fold)
        alpha0 = choose_alpha(
            baseline[train], y[train], strata[train], ALPHAS, 100 + fold)
        model = PoissonRegressor(alpha=alpha, max_iter=1000).fit(
            D[train], y[train])
        reduced_model = PoissonRegressor(
            alpha=alpha_reduced, max_iter=1000).fit(reduced[train], y[train])
        null = PoissonRegressor(alpha=alpha0, max_iter=1000).fit(
            baseline[train], y[train])
        lm = glm.poisson_ll_numpy(y[test], model.predict(D[test]))
        lr = glm.poisson_ll_numpy(
            y[test], reduced_model.predict(reduced[test]))
        ln = glm.poisson_ll_numpy(y[test], null.predict(baseline[test]))
        ll_model += lm
        ll_reduced += lr
        ll_null += ln
        positive_folds += lm > ln
        full_beats_reduced_folds += lm > lr

    alpha = choose_alpha(D, y, strata, ALPHAS, 999)
    final = PoissonRegressor(alpha=alpha, max_iter=2000).fit(D, y)
    return {
        "neuron": neuron, "heldout_ll_model": ll_model,
        "heldout_ll_null": ll_null,
        "heldout_ll_improvement": ll_model - ll_null,
        "heldout_ll_reduced": ll_reduced,
        "full_minus_reduced_ll": ll_model - ll_reduced,
        "positive_fold_fraction": positive_folds / 5,
        "full_beats_reduced_fold_fraction": full_beats_reduced_folds / 5,
        "alpha": alpha, "coef": final.coef_,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="PTYEZ_task60")
    ap.add_argument("--n-components", type=int, default=10)
    ap.add_argument("--deviation-penalty-multiplier", type=float, default=10.0)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--results-root", required=True)
    args = ap.parse_args()

    glm.MODEL_TAG = "gpt2-large"
    glm.WINDOW_TAG = WINDOW_TAG
    cfg = next(p for p in glm.PATIENTS if p["patient_ID"] == args.patient)
    data = glm.build_patient_region_data(
        cfg, "hippocampus", 36, args.n_components)

    X = np.vstack([data["X_self"], data["X_other"]])
    Y = np.vstack([data["Y_self"], data["Y_other"]])
    category = np.concatenate([
        data["metadata_self"]["ClusterID"].to_numpy(),
        data["metadata_other"]["ClusterID"].to_numpy(),
    ]).astype(int)
    condition = np.concatenate([
        np.zeros(len(data["X_self"]), dtype=int),
        np.ones(len(data["X_other"]), dtype=int),
    ])
    categories = sorted(np.unique(category))
    strata = np.asarray([f"{c}_{q}" for c, q in zip(category, condition)])
    deviation_scale = 1 / np.sqrt(args.deviation_penalty_multiplier)
    D, semantic_start = make_design(
        X, category, condition, categories, deviation_scale)
    baseline = D[:, :semantic_start]
    # The final n_categories*n_components columns are the
    # category×condition×PC deviations. Removing them forces the semantic
    # self/other slope difference to be shared across categories.
    reduced = D[:, :-len(categories) * args.n_components]

    fitted = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(fit_neuron)(n, D, reduced, baseline, Y[:, n], strata)
        for n in range(Y.shape[1]))
    detail = pd.DataFrame([{k: v for k, v in row.items() if k != "coef"}
                           for row in fitted])

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
            beta_self = shared + cat_dev[j]
            beta_other = shared + cond_shared + cat_dev[j] + cat_cond[j]
            distance = (cosine(beta_self, beta_other)
                        if np.linalg.norm(beta_self) and np.linalg.norm(beta_other)
                        else np.nan)
            cosine_rows.append({
                "neuron": row["neuron"], "cluster_id": cid,
                "category": NAMES.get(cid, str(cid)),
                "cosine_distance": distance,
            })
    cosine_df = pd.DataFrame(cosine_rows)
    summary = pd.DataFrame([{
        "patient": args.patient, "n_trials": len(X), "n_neurons": Y.shape[1],
        "n_categories": ncat, "n_components": p,
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
    out = os.path.join(args.results_root, args.patient)
    os.makedirs(out, exist_ok=True)
    detail.to_csv(os.path.join(out, "hippocampus_joint_glm_nested_cv.csv"),
                  index=False)
    cosine_df.to_csv(os.path.join(
        out, "hippocampus_joint_glm_category_cosine_distances.csv"), index=False)
    summary.to_csv(os.path.join(out, "hippocampus_joint_glm_summary.csv"),
                   index=False)
    print(summary.to_string(index=False), flush=True)
    print(cosine_df.groupby(["cluster_id", "category"]).cosine_distance
          .agg(["count", "mean", "std"]).to_string(), flush=True)


if __name__ == "__main__":
    main()
