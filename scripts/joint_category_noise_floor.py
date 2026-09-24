#!/usr/bin/env python3
"""Repeated split-half category noise floors for the joint interaction GLM."""

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


def slopes_from_coef(coef, semantic_start, p, categories, deviation_scale):
    ncat = len(categories)
    sem = coef[semantic_start:]
    shared, cond_shared = sem[:p], sem[p:2*p]
    cat_dev = sem[2*p:2*p+ncat*p].reshape(ncat, p) * deviation_scale
    cat_cond = sem[2*p+ncat*p:].reshape(ncat, p) * deviation_scale
    self_beta = shared[None, :] + cat_dev
    other_beta = self_beta + cond_shared[None, :] + cat_cond
    return self_beta, other_beta


def one_split(split_id, X, Y, category, condition, turn_ids, turn_condition,
              categories, p, deviation_scale, alphas):
    rng = np.random.RandomState(50000 + split_id)
    halves = [[], []]
    # Keep every contiguous speaker turn intact. Within each condition,
    # greedily assign shuffled turns to balance the number of words per half.
    for cond in (0, 1):
        turns = [t for t, c in turn_condition.items() if c == cond]
        turns = list(rng.permutation(turns))
        half_word_counts = [0, 0]
        for turn in turns:
            idx = np.flatnonzero(turn_ids == turn)
            side = int(half_word_counts[1] < half_word_counts[0])
            halves[side].extend(idx)
            half_word_counts[side] += len(idx)
    fitted = []
    for idx in halves:
        idx = np.asarray(sorted(idx))
        D, semantic_start = make_design(
            X[idx], category[idx], condition[idx], categories, deviation_scale)
        sb, ob = [], []
        for neuron in range(Y.shape[1]):
            model = PoissonRegressor(
                alpha=float(alphas[neuron]), max_iter=2000).fit(D, Y[idx, neuron])
            s, o = slopes_from_coef(
                model.coef_, semantic_start, p, categories, deviation_scale)
            sb.append(s)
            ob.append(o)
        fitted.append((np.asarray(sb), np.asarray(ob)))
    rows = []
    for neuron in range(Y.shape[1]):
        for j, cid in enumerate(categories):
            bs1, bs2 = fitted[0][0][neuron, j], fitted[1][0][neuron, j]
            bo1, bo2 = fitted[0][1][neuron, j], fitted[1][1][neuron, j]
            rows.append({
                "split": split_id, "neuron": neuron, "cluster_id": cid,
                "category": NAMES.get(cid, str(cid)),
                "self_halfsplit_distance": cosine(bs1, bs2),
                "other_halfsplit_distance": cosine(bo1, bo2),
                "cross_half1_distance": cosine(bs1, bo1),
                "cross_half2_distance": cosine(bs2, bo2),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="PTYEZ_task60")
    ap.add_argument("--n-splits", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--n-components", type=int, default=10)
    ap.add_argument("--deviation-penalty-multiplier", type=float, default=10)
    ap.add_argument("--joint-results-root", required=True)
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
    for t, c in zip(turn_ids, condition):
        turn_condition[int(t)] = int(c)
    out_dir = os.path.join(args.joint_results_root, args.patient)
    alpha = pd.read_csv(os.path.join(
        out_dir, "hippocampus_joint_glm_nested_cv.csv")).sort_values("neuron")
    alphas = alpha["alpha"].to_numpy()
    deviation_scale = 1 / np.sqrt(args.deviation_penalty_multiplier)
    nested = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(one_split)(
            i, X, Y, category, condition, turn_ids, turn_condition, categories,
            args.n_components, deviation_scale, alphas)
        for i in range(args.n_splits))
    detail = pd.DataFrame([row for block in nested for row in block])
    detail["matched_half_cross_distance"] = detail[
        ["cross_half1_distance", "cross_half2_distance"]].mean(axis=1)
    detail["mean_within_condition_noise"] = detail[
        ["self_halfsplit_distance", "other_halfsplit_distance"]].mean(axis=1)
    detail["matched_cross_minus_noise"] = (
        detail.matched_half_cross_distance -
        detail.mean_within_condition_noise)
    cross = pd.read_csv(os.path.join(
        out_dir, "hippocampus_joint_glm_category_cosine_distances.csv"))
    summary = detail.groupby(["cluster_id", "category"], as_index=False).agg(
        self_noise_floor=("self_halfsplit_distance", "median"),
        other_noise_floor=("other_halfsplit_distance", "median"),
        cross_half1=("cross_half1_distance", "median"),
        cross_half2=("cross_half2_distance", "median"),
    )
    cross_mean = cross.groupby("cluster_id").cosine_distance.mean()
    summary["cross_condition_distance"] = summary.cluster_id.map(cross_mean)
    summary["mean_within_condition_noise"] = summary[
        ["self_noise_floor", "other_noise_floor"]].mean(axis=1)
    summary["matched_half_cross_distance"] = summary[
        ["cross_half1", "cross_half2"]].mean(axis=1)
    summary["matched_cross_minus_noise"] = (
        summary.matched_half_cross_distance -
        summary.mean_within_condition_noise)
    split_delta = detail.groupby(
        ["cluster_id", "category", "split"], as_index=False
    ).matched_cross_minus_noise.mean()
    delta_ci = split_delta.groupby(
        ["cluster_id", "category"], as_index=False
    ).agg(
        mean_matched_cross_minus_noise=("matched_cross_minus_noise", "mean"),
        delta_ci_2_5=("matched_cross_minus_noise", lambda x: x.quantile(.025)),
        delta_ci_97_5=("matched_cross_minus_noise", lambda x: x.quantile(.975)),
        pct_splits_delta_positive=(
            "matched_cross_minus_noise", lambda x: 100*(x > 0).mean()),
    )
    summary = summary.merge(delta_ci, on=["cluster_id", "category"])
    detail.to_csv(os.path.join(
        out_dir, "hippocampus_joint_category_noise_floor_detail.csv"), index=False)
    summary.to_csv(os.path.join(
        out_dir, "hippocampus_joint_category_noise_floor_summary.csv"), index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
