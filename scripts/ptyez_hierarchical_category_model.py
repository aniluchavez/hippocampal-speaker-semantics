#!/usr/bin/env python3
"""Crossed Gaussian partial-pooling pilot for PTYEZ hippocampus."""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import chi2

ROOT = "/scratch/aniluchavez/ConvoDATAS/SemanticGLM/clusterwise_worddur_offset"
PATIENT = "PTYEZ_task60"
REGION = "hippocampus"
NAMES = {
    1: "Body Parts", 2: "Places", 3: "Emotional", 4: "Mental",
    5: "Social", 6: "Objects", 7: "Visual", 8: "Numerical",
    9: "Actions", 10: "Identity", 11: "Function Words",
}


def fit_vc(y, X, Zc, Zn, include_category=True, reml=True):
    n, p = X.shape

    def objective(theta):
        vc = np.exp(theta[0]) if include_category else 0.0
        vn = np.exp(theta[-2])
        ve = np.exp(theta[-1])
        V = vc * (Zc @ Zc.T) + vn * (Zn @ Zn.T) + ve * np.eye(n)
        try:
            cf = cho_factor(V, lower=True, check_finite=False)
            vi_x = cho_solve(cf, X, check_finite=False)
            vi_y = cho_solve(cf, y, check_finite=False)
            xtvix = X.T @ vi_x
            beta = np.linalg.solve(xtvix, X.T @ vi_y)
            resid = y - X @ beta
            quad = resid @ cho_solve(cf, resid, check_finite=False)
            logdet_v = 2 * np.log(np.diag(cf[0])).sum()
            val = logdet_v + quad
            if reml:
                sign, logdet_x = np.linalg.slogdet(xtvix)
                if sign <= 0:
                    return 1e100
                val += logdet_x
                denom = n - p
            else:
                denom = n
            return 0.5 * (val + denom * np.log(2 * np.pi))
        except (np.linalg.LinAlgError, ValueError):
            return 1e100

    start = np.log([0.01, 0.01, 0.1] if include_category else [0.01, 0.1])
    result = minimize(objective, start, method="L-BFGS-B",
                      bounds=[(-14, 3)] * len(start))
    theta = result.x
    vc = np.exp(theta[0]) if include_category else 0.0
    vn, ve = np.exp(theta[-2]), np.exp(theta[-1])
    V = vc * (Zc @ Zc.T) + vn * (Zn @ Zn.T) + ve * np.eye(n)
    cf = cho_factor(V, lower=True, check_finite=False)
    vi_x = cho_solve(cf, X, check_finite=False)
    vi_y = cho_solve(cf, y, check_finite=False)
    beta = np.linalg.solve(X.T @ vi_x, X.T @ vi_y)
    resid = y - X @ beta
    ucat = vc * Zc.T @ cho_solve(cf, resid, check_finite=False)
    return dict(result=result, nll=objective(theta), beta=beta, vc=vc,
                vn=vn, ve=ve, ucat=ucat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default=ROOT)
    ap.add_argument("--exclude-clusters", type=int, nargs="*", default=[4])
    ap.add_argument("--output-tag", default="")
    args = ap.parse_args()
    p = os.path.join(args.results_root, PATIENT,
                     f"{REGION}_clusterwise_cosine_distances.csv")
    d = pd.read_csv(p)
    d = d[~d["cluster_id"].isin(args.exclude_clusters)].copy()
    categories = sorted(d["cluster_id"].unique())
    neurons = sorted(d["neuron"].unique())
    ci = {v: i for i, v in enumerate(categories)}
    ni = {v: i for i, v in enumerate(neurons)}
    Zc = np.zeros((len(d), len(categories)))
    Zn = np.zeros((len(d), len(neurons)))
    for row, (c, n) in enumerate(zip(d.cluster_id, d.neuron)):
        Zc[row, ci[c]] = 1
        Zn[row, ni[n]] = 1

    effective_n = np.sqrt(
        d["n_self_used"].to_numpy(float) * d["n_other_used"].to_numpy(float))
    logn = np.log(effective_n)
    logn_mean, logn_sd = logn.mean(), logn.std()
    logn_z = (logn - logn_mean) / logn_sd
    X = np.column_stack([np.ones(len(d)), logn_z])
    y = d["cosine_distance"].to_numpy(float)

    full = fit_vc(y, X, Zc, Zn, include_category=True, reml=True)
    full_ml = fit_vc(y, X, Zc, Zn, include_category=True, reml=False)
    null_ml = fit_vc(y, X, Zc, Zn, include_category=False, reml=False)
    lr = max(0.0, 2 * (null_ml["nll"] - full_ml["nll"]))
    # Category-variance null is on the boundary: 0.5*chi0 + 0.5*chi1.
    p_boundary = 0.5 * chi2.sf(lr, 1)

    raw_means = d.groupby("cluster_id")["cosine_distance"].mean()
    nself = d.groupby("cluster_id")["n_self_used"].first()
    nother = d.groupby("cluster_id")["n_other_used"].first()
    ntrials = np.sqrt(nself * nother)
    category_logn_z = (np.log(ntrials.to_numpy(float)) - logn_mean) / logn_sd
    population_part = full["beta"][0] + full["beta"][1] * category_logn_z
    estimates = population_part + full["ucat"]
    out = pd.DataFrame({
        "cluster_id": categories,
        "category": [NAMES[c] for c in categories],
        "n_self": nself.reindex(categories).to_numpy(),
        "n_other": nother.reindex(categories).to_numpy(),
        "effective_n_geometric_mean": ntrials.reindex(categories).to_numpy(),
        "raw_category_mean": raw_means.reindex(categories).to_numpy(),
        "partially_pooled_mean": estimates,
        "category_deviation_blup": full["ucat"],
    })
    out["shrinkage_toward_count_adjusted_mean"] = (
        out["raw_category_mean"] - out["partially_pooled_mean"])
    out = out.sort_values("partially_pooled_mean")

    summary = pd.DataFrame([{
        "patient": PATIENT, "region": REGION,
        "excluded_clusters": ",".join(map(str, args.exclude_clusters)) or "none",
        "n_observations": len(d), "n_categories": len(categories),
        "n_neurons": len(neurons), "intercept": full["beta"][0],
        "log_trial_count_z_coefficient": full["beta"][1],
        "category_variance": full["vc"], "neuron_variance": full["vn"],
        "residual_variance": full["ve"], "category_variance_lr": lr,
        "category_variance_boundary_p": p_boundary,
        "converged": full["result"].success,
    }])
    out_dir = os.path.join(args.results_root, PATIENT)
    suffix = f"_{args.output_tag}" if args.output_tag else ""
    out.to_csv(os.path.join(
        out_dir, f"{REGION}_hierarchical_category_partial_pooling{suffix}.csv"), index=False)
    summary.to_csv(os.path.join(
        out_dir, f"{REGION}_hierarchical_category_model_summary{suffix}.csv"), index=False)
    print(summary.to_string(index=False))
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
