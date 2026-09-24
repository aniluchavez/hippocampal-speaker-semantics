#!/usr/bin/env python3
"""Parametric bootstrap null for category-specific self/other interactions."""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import PoissonRegressor

sys.path.insert(0, os.path.dirname(__file__))
import cluster_glm_reliability as glm
from joint_category_interaction_glm import (
    make_design, choose_alpha, WINDOW_TAG, ALPHAS,
)
from joint_turn_permutation import derive_cosines, repeated_measures_f


def fit_models(Y, D, reduced, alpha_full, alpha_reduced, semantic_start,
               p, categories, deviation_scale):
    ll_delta = 0.0
    coefs = []
    for neuron in range(Y.shape[1]):
        full = PoissonRegressor(
            alpha=float(alpha_full[neuron]), max_iter=2000).fit(D, Y[:, neuron])
        red = PoissonRegressor(
            alpha=float(alpha_reduced[neuron]), max_iter=2000
        ).fit(reduced, Y[:, neuron])
        ll_delta += (
            glm.poisson_ll_numpy(Y[:, neuron], full.predict(D)) -
            glm.poisson_ll_numpy(Y[:, neuron], red.predict(reduced)))
        coefs.append(full.coef_)
    distances = derive_cosines(
        coefs, semantic_start, p, categories, deviation_scale)
    category_means = {
        f"distance_cluster_{cid}": float(np.nanmean(distances[:, j]))
        for j, cid in enumerate(categories)
    }
    return 2 * ll_delta, repeated_measures_f(distances), category_means


def one_bootstrap(seed, mu_null, D, reduced, alpha_full, alpha_reduced,
                  semantic_start, p, categories, deviation_scale):
    rng = np.random.default_rng(seed)
    Ysim = rng.poisson(np.clip(mu_null, 1e-10, 1e6))
    deviance, geometry_f, category_means = fit_models(
        Ysim, D, reduced, alpha_full, alpha_reduced, semantic_start,
        p, categories, deviation_scale)
    return {
        "interaction_deviance": deviance,
        "geometry_rm_f": geometry_f,
        **category_means,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="PTYEZ_task60")
    ap.add_argument("--n-bootstraps", type=int, default=1000)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--n-components", type=int, default=10)
    ap.add_argument("--deviation-penalty-multiplier", type=float, default=10)
    ap.add_argument("--results-root", required=True)
    args = ap.parse_args()
    glm.MODEL_TAG, glm.WINDOW_TAG = "gpt2-large", WINDOW_TAG
    cfg = next(p for p in glm.PATIENTS if p["patient_ID"] == args.patient)
    data = glm.build_patient_region_data(
        cfg, "hippocampus", 36, args.n_components)
    X = np.vstack([data["X_self"], data["X_other"]])
    Y = np.vstack([data["Y_self"], data["Y_other"]])
    category = np.concatenate([
        data["metadata_self"]["ClusterID"], data["metadata_other"]["ClusterID"]
    ]).astype(int)
    condition = np.r_[np.zeros(len(data["X_self"]), int),
                      np.ones(len(data["X_other"]), int)]
    categories = sorted(np.unique(category))
    strata = np.asarray([f"{c}_{q}" for c, q in zip(category, condition)])
    deviation_scale = 1 / np.sqrt(args.deviation_penalty_multiplier)
    D, semantic_start = make_design(
        X, category, condition, categories, deviation_scale)
    reduced = D[:, :-len(categories) * args.n_components]

    alpha_full, alpha_reduced, mu_null = [], [], []
    for neuron in range(Y.shape[1]):
        af = choose_alpha(D, Y[:, neuron], strata, ALPHAS, 700 + neuron)
        ar = choose_alpha(
            reduced, Y[:, neuron], strata, ALPHAS, 900 + neuron)
        alpha_full.append(af)
        alpha_reduced.append(ar)
        red = PoissonRegressor(alpha=ar, max_iter=2000).fit(
            reduced, Y[:, neuron])
        mu_null.append(red.predict(reduced))
    alpha_full = np.asarray(alpha_full)
    alpha_reduced = np.asarray(alpha_reduced)
    mu_null = np.column_stack(mu_null)

    observed_deviance, observed_f, observed_category_means = fit_models(
        Y, D, reduced, alpha_full, alpha_reduced, semantic_start,
        args.n_components, categories, deviation_scale)
    rows = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(one_bootstrap)(
            80000 + i, mu_null, D, reduced, alpha_full, alpha_reduced,
            semantic_start, args.n_components, categories, deviation_scale)
        for i in range(args.n_bootstraps))
    null = pd.DataFrame(rows)
    p_dev = (1 + np.sum(null.interaction_deviance >= observed_deviance)) / (
        len(null) + 1)
    p_geom = (1 + np.sum(null.geometry_rm_f >= observed_f)) / (len(null) + 1)
    summary = pd.DataFrame([{
        "patient": args.patient, "n_bootstraps": len(null),
        "observed_interaction_deviance": observed_deviance,
        "null_deviance_95pct": null.interaction_deviance.quantile(.95),
        "interaction_empirical_p": p_dev,
        "observed_geometry_rm_f": observed_f,
        "null_geometry_f_95pct": null.geometry_rm_f.quantile(.95),
        "geometry_empirical_p": p_geom,
        **{
            f"observed_{key}": value
            for key, value in observed_category_means.items()
        },
    }])
    out = os.path.join(args.results_root, args.patient)
    null.to_csv(os.path.join(
        out, "hippocampus_joint_reduced_model_bootstrap_null.csv"), index=False)
    summary.to_csv(os.path.join(
        out, "hippocampus_joint_reduced_model_bootstrap_summary.csv"), index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
