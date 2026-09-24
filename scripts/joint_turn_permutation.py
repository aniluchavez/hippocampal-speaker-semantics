#!/usr/bin/env python3
"""Turn-level condition-label permutation null for joint category geometry."""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import cosine
from sklearn.linear_model import PoissonRegressor

sys.path.insert(0, os.path.dirname(__file__))
import cluster_glm_reliability as glm
from joint_category_interaction_glm import make_design, WINDOW_TAG, NAMES


def repeated_measures_f(values):
    """One-factor repeated-measures F for neuron × category matrix."""
    values = np.asarray(values, float)
    keep = np.isfinite(values).all(axis=1)
    x = values[keep]
    n, k = x.shape
    grand = x.mean()
    ss_total = np.sum((x - grand) ** 2)
    ss_category = n * np.sum((x.mean(0) - grand) ** 2)
    ss_neuron = k * np.sum((x.mean(1) - grand) ** 2)
    ss_error = ss_total - ss_category - ss_neuron
    return (ss_category / (k - 1)) / (ss_error / ((k - 1) * (n - 1)))


def derive_cosines(coefs, semantic_start, p, categories, deviation_scale):
    ncat = len(categories)
    matrix = np.full((len(coefs), ncat), np.nan)
    for neuron, coef in enumerate(coefs):
        sem = coef[semantic_start:]
        shared = sem[:p]
        cond_shared = sem[p:2*p]
        cat_dev = sem[2*p:2*p+ncat*p].reshape(ncat, p) * deviation_scale
        cat_cond = sem[2*p+ncat*p:].reshape(ncat, p) * deviation_scale
        for j in range(ncat):
            bs = shared + cat_dev[j]
            bo = shared + cond_shared + cat_dev[j] + cat_cond[j]
            if np.linalg.norm(bs) and np.linalg.norm(bo):
                matrix[neuron, j] = cosine(bs, bo)
    return matrix


def fit_permutation(perm_id, seed, X, Y, category, turn_ids, turn_condition,
                    categories, p, deviation_scale, alphas):
    if perm_id < 0:
        condition = np.asarray([turn_condition[t] for t in turn_ids], int)
    else:
        rng = np.random.RandomState(seed)
        turns = np.asarray(sorted(turn_condition))
        labels = np.asarray([turn_condition[t] for t in turns])
        shuffled = dict(zip(turns, rng.permutation(labels)))
        condition = np.asarray([shuffled[t] for t in turn_ids], int)
    D, semantic_start = make_design(
        X, category, condition, categories, deviation_scale)
    coefs = []
    for neuron in range(Y.shape[1]):
        model = PoissonRegressor(
            alpha=float(alphas[neuron]), max_iter=2000).fit(D, Y[:, neuron])
        coefs.append(model.coef_)
    distances = derive_cosines(
        coefs, semantic_start, p, categories, deviation_scale)
    row = {
        "permutation": perm_id,
        "rm_anova_f": repeated_measures_f(distances),
        "n_turns": len(turn_condition),
    }
    for j, cid in enumerate(categories):
        row[f"distance_cluster_{cid}"] = np.nanmean(distances[:, j])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="PTYEZ_task60")
    ap.add_argument("--n-permutations", type=int, default=1000)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--n-components", type=int, default=10)
    ap.add_argument("--deviation-penalty-multiplier", type=float, default=10)
    ap.add_argument("--joint-results-root", required=True)
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
    categories = sorted(np.unique(category))

    # Recover original word positions, then define turns as contiguous runs
    # from the same transcript speaker.
    spike_dir = glm.find_spike_dir(cfg["patient"])
    assignment, _, _, _ = glm.load_speaker_assignment(spike_dir)
    full_turn = np.zeros(len(assignment), dtype=int)
    turn = -1
    previous = object()
    for i, speaker in enumerate(assignment):
        if speaker != previous:
            turn += 1
            previous = speaker
        full_turn[i] = turn
    idx_self = np.flatnonzero(data["mask_self"])[data["valid_self"]]
    idx_other = np.flatnonzero(data["mask_other"])[data["valid_other"]]
    turn_ids = np.concatenate([full_turn[idx_self], full_turn[idx_other]])
    turn_condition = {}
    for t, c in zip(turn_ids, np.r_[
            np.zeros(len(idx_self), int), np.ones(len(idx_other), int)]):
        turn_condition[int(t)] = int(c)

    nested_path = os.path.join(
        args.joint_results_root, args.patient,
        "hippocampus_joint_glm_nested_cv.csv")
    alpha_df = pd.read_csv(nested_path).sort_values("neuron")
    alphas = alpha_df["alpha"].to_numpy()
    deviation_scale = 1 / np.sqrt(args.deviation_penalty_multiplier)
    jobs = [(-1, 42)] + [(i, 20000 + i) for i in range(args.n_permutations)]
    rows = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(fit_permutation)(
            pid, seed, X, Y, category, turn_ids, turn_condition, categories,
            args.n_components, deviation_scale, alphas)
        for pid, seed in jobs)
    out = pd.DataFrame(rows)
    observed = out.loc[out.permutation == -1, "rm_anova_f"].iloc[0]
    null = out.loc[out.permutation >= 0, "rm_anova_f"]
    p_emp = (1 + np.sum(null >= observed)) / (len(null) + 1)
    summary = pd.DataFrame([{
        "patient": args.patient, "n_permutations": len(null),
        "n_turns": rows[0]["n_turns"], "observed_rm_f": observed,
        "null_mean_f": null.mean(), "null_95pct_f": null.quantile(.95),
        "empirical_p": p_emp,
    }])
    out_dir = os.path.join(args.joint_results_root, args.patient)
    out.to_csv(os.path.join(
        out_dir, "hippocampus_joint_turn_permutations.csv"), index=False)
    summary.to_csv(os.path.join(
        out_dir, "hippocampus_joint_turn_permutation_summary.csv"), index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
