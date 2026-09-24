#!/usr/bin/env python3
"""Full semantic-link permutation test for natural-count cluster GLMs."""

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
NAMES = {
    1: "Body Parts", 2: "Places", 3: "Emotional", 4: "Mental",
    5: "Social", 6: "Objects", 7: "Visual", 8: "Numerical",
    9: "Actions", 10: "Identity", 11: "Function Words",
}


def fit_betas(X, Y):
    betas = []
    for neuron in range(Y.shape[1]):
        y = Y[:, neuron]
        alpha = glm.select_alpha_cv(X, y, ALPHAS, n_splits=3, random_state=0)
        betas.append(glm.fit_poisson_ridge_beta_sklearn(X, y, alpha))
    return np.asarray(betas)


def run_one(perm_id, seed, cells):
    rng = np.random.RandomState(seed)
    fitted = {}
    for cid, conditions in cells.items():
        fitted[cid] = {}
        for condition, (X, Y) in conditions.items():
            if perm_id >= 0:
                X = X[rng.permutation(len(X))]
            fitted[cid][condition] = fit_betas(X, Y)

    category_rows = []
    neuron_distances = []
    for cid, b in fitted.items():
        bs, bo = b["self"], b["other"]
        dist = np.asarray([
            cosine(bs[i], bo[i])
            if np.linalg.norm(bs[i]) > 0 and np.linalg.norm(bo[i]) > 0
            else np.nan
            for i in range(min(len(bs), len(bo)))
        ])
        valid = np.isfinite(dist)
        neuron_distances.extend((cid, x) for x in dist[valid])
        category_rows.append({
            "permutation": perm_id, "cluster_id": cid,
            "category": NAMES.get(cid, str(cid)),
            "mean_cosine_distance": np.nanmean(dist),
            "n_neurons": int(valid.sum()),
        })

    # Patient-level omnibus statistic, matching the original neuron-row ANOVA.
    groups = {}
    for cid, value in neuron_distances:
        groups.setdefault(cid, []).append(value)
    grand = np.mean([v for values in groups.values() for v in values])
    ss_between = sum(len(v) * (np.mean(v) - grand) ** 2 for v in groups.values())
    ss_within = sum(sum((np.asarray(v) - np.mean(v)) ** 2) for v in groups.values())
    k = len(groups)
    n = sum(map(len, groups.values()))
    f_stat = (ss_between / (k - 1)) / (ss_within / (n - k))
    for row in category_rows:
        row["patient_anova_f"] = f_stat
    return category_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", required=True)
    ap.add_argument("--n-permutations", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--content-macro-categories", action="store_true")
    args = ap.parse_args()

    glm.MODEL_TAG = "gpt2-large"
    glm.WINDOW_TAG = WINDOW_TAG
    cfg = next(p for p in glm.PATIENTS if p["patient_ID"] == args.patient)
    data = glm.build_patient_region_data(cfg, "hippocampus", 36, 10)
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
    cells = {}
    for cid in sorted(
        set(data["metadata_self"]["ClusterID"]) |
        set(data["metadata_other"]["ClusterID"])
    ):
        iself = np.flatnonzero(data["metadata_self"]["ClusterID"].to_numpy() == cid)
        iother = np.flatnonzero(data["metadata_other"]["ClusterID"].to_numpy() == cid)
        if len(iself) < 10 or len(iother) < 10:
            continue
        cells[cid] = {
            "self": (data["X_self"][iself], data["Y_self"][iself]),
            "other": (data["X_other"][iother], data["Y_other"][iother]),
        }

    jobs = [(-1, 42)] + [(i, 10000 + i) for i in range(args.n_permutations)]
    nested = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(run_one)(pid, seed, cells) for pid, seed in jobs)
    out = pd.DataFrame([row for block in nested for row in block])
    out.insert(0, "patient", args.patient)
    observed_f = out.loc[out.permutation == -1, "patient_anova_f"].iloc[0]
    null_f = out.loc[out.permutation >= 0].groupby("permutation").patient_anova_f.first()
    p_emp = (1 + np.sum(null_f >= observed_f)) / (len(null_f) + 1)
    summary = pd.DataFrame([{
        "patient": args.patient, "n_permutations": len(null_f),
        "observed_anova_f": observed_f, "null_f_mean": null_f.mean(),
        "null_f_95pct": null_f.quantile(.95), "empirical_p": p_emp,
        "n_categories": len(cells),
    }])
    out_dir = os.path.join(args.results_root, args.patient)
    os.makedirs(out_dir, exist_ok=True)
    out.to_csv(os.path.join(out_dir, "hippocampus_full_pipeline_permutations.csv"),
               index=False)
    summary.to_csv(os.path.join(
        out_dir, "hippocampus_full_pipeline_permutation_summary.csv"), index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
