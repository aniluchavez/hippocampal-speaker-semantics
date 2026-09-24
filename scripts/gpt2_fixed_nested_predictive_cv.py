#!/usr/bin/env python3
"""True nested held-out prediction for soft-balanced cluster GLMs."""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
import cluster_glm_reliability as glm
from gpt2_fixed_beta_fold_stability import prepare_cells, NAMES, WINDOW_TAG, ALPHAS

REGION = "hippocampus"


def poisson_saturated_ll(y):
    y = np.asarray(y, float)
    mu = np.where(y > 0, y, 1.0)
    return glm.poisson_ll_numpy(y, mu)


def fit_one(patient, cid, condition, neuron, X, y, alphas):
    ll_model = ll_null = ll_sat = 0.0
    fold_rows = []
    outer = KFold(5, shuffle=True, random_state=42)
    for fold, (train, test) in enumerate(outer.split(X)):
        Xtr, Xte, ytr, yte = X[train], X[test], y[train], y[test]
        alpha = glm.select_alpha_cv(
            Xtr, ytr, alphas, n_splits=3, random_state=fold)
        scaler = StandardScaler().fit(Xtr)
        model = PoissonRegressor(alpha=alpha, max_iter=1000)
        model.fit(scaler.transform(Xtr), ytr)
        mu = model.predict(scaler.transform(Xte))
        mu0 = np.full(len(test), max(float(np.mean(ytr)), 1e-10))
        lm = glm.poisson_ll_numpy(yte, mu)
        ln = glm.poisson_ll_numpy(yte, mu0)
        ls = poisson_saturated_ll(yte)
        ll_model += lm
        ll_null += ln
        ll_sat += ls
        fold_rows.append((fold, alpha, lm - ln))
    null_deviance = 2 * (ll_sat - ll_null)
    model_deviance = 2 * (ll_sat - ll_model)
    deviance_explained = (
        1 - model_deviance / null_deviance if null_deviance > 1e-12 else np.nan)
    return {
        "patient": patient, "region": REGION, "cluster_id": cid,
        "category": NAMES[cid], "condition": condition, "neuron": neuron,
        "n_trials": len(X), "heldout_ll_model": ll_model,
        "heldout_ll_null": ll_null,
        "heldout_ll_improvement": ll_model - ll_null,
        "heldout_deviance_explained": deviance_explained,
        "positive_heldout_improvement": ll_model > ll_null,
        "median_selected_alpha": np.median([x[1] for x in fold_rows]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", required=True)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--n-components", type=int, default=10)
    ap.add_argument("--extended-alpha", action="store_true")
    ap.add_argument("--content-macro-categories", action="store_true")
    ap.add_argument("--all-natural-counts", action="store_true")
    args = ap.parse_args()
    glm.MODEL_TAG = "gpt2-large"
    glm.WINDOW_TAG = WINDOW_TAG
    cfg = next(p for p in glm.PATIENTS if p["patient_ID"] == args.patient)
    data = glm.build_patient_region_data(cfg, REGION, 36, args.n_components)
    if args.content_macro_categories:
        macro_map = {
            1: 1, 2: 1, 6: 1,
            3: 2, 4: 2, 5: 2, 10: 2,
            7: 3, 8: 3, 9: 3,
        }
        for cond in ("self", "other"):
            meta = data[f"metadata_{cond}"]
            keep = meta["ClusterID"].isin(macro_map).to_numpy()
            data[f"X_{cond}"] = data[f"X_{cond}"][keep]
            data[f"Y_{cond}"] = data[f"Y_{cond}"][keep]
            data[f"metadata_{cond}"] = pd.DataFrame({
                "ClusterID": meta.loc[keep, "ClusterID"].map(macro_map).to_numpy()
            })
        NAMES.update({
            1: "Concrete/referential",
            2: "Socio-affective/cognitive",
            3: "Perceptual/action",
        })
    if args.all_natural_counts:
        cells = {}
        for cid in sorted(set(data["metadata_self"]["ClusterID"]) |
                          set(data["metadata_other"]["ClusterID"])):
            iself = np.flatnonzero(
                data["metadata_self"]["ClusterID"].to_numpy() == cid)
            iother = np.flatnonzero(
                data["metadata_other"]["ClusterID"].to_numpy() == cid)
            if len(iself) < 10 or len(iother) < 10:
                continue
            cells[cid] = {
                "self": (data["X_self"][iself], data["Y_self"][iself]),
                "other": (data["X_other"][iother], data["Y_other"][iother]),
            }
        target = np.nan
    else:
        cells, target = prepare_cells(data, min_trials=10, balance_mode="soft")
    alphas = (np.asarray([
        1e-3, 1e-2, 1e-1, 1, 10, 100, 300, 1000, 3000,
        10000, 30000, 100000,
    ]) if args.extended_alpha else ALPHAS)
    tasks = []
    for cid, conditions in cells.items():
        for condition, (X, Y) in conditions.items():
            for neuron in range(Y.shape[1]):
                tasks.append((
                    args.patient, cid, condition, neuron, X, Y[:, neuron], alphas))
    rows = Parallel(n_jobs=8, verbose=10)(
        delayed(fit_one)(*task) for task in tasks)
    detail = pd.DataFrame(rows)
    summary = detail.groupby(
        ["patient", "region", "cluster_id", "category", "condition", "n_trials"],
        as_index=False,
    ).agg(
        neurons=("neuron", "nunique"),
        total_heldout_ll_improvement=("heldout_ll_improvement", "sum"),
        median_heldout_ll_improvement=("heldout_ll_improvement", "median"),
        median_heldout_deviance_explained=("heldout_deviance_explained", "median"),
        pct_neurons_positive=("positive_heldout_improvement", lambda x: 100*x.mean()),
    )
    out = os.path.join(args.results_root, args.patient)
    suffix = f"_pc{args.n_components}"
    if args.extended_alpha:
        suffix += "_strongalpha"
    detail.to_csv(os.path.join(out, f"{REGION}_nested_predictive_cv{suffix}_detail.csv"),
                  index=False)
    summary.to_csv(os.path.join(out, f"{REGION}_nested_predictive_cv{suffix}_summary.csv"),
                   index=False)
    print(f"Balance target: {target}", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
