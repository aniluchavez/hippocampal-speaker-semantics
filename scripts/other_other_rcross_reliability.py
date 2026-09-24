#!/usr/bin/env python3
"""Other-vs-other beta r_cross and split-half reliability.

Uses the same reliability engine as the self/other analyses, but the two
conditions are the top two non-Speaker1 speakers for each patient.
"""

from __future__ import annotations

import argparse
import importlib.util as ilu
import os
import pickle
import sys

import numpy as np
import pandas as pd

PROJECT = "/scratch/aniluchavez/hippocampal-speaker-semantics"
sys.path.insert(0, os.path.join(PROJECT, "scripts"))

import cluster_glm_other_other as oo  # noqa: E402
import cluster_glm_reliability as glm  # noqa: E402

_rel_spec = ilu.spec_from_file_location(
    "nn_reliability", os.path.join(PROJECT, "neural_encoding", "reliability.py")
)
_rel = ilu.module_from_spec(_rel_spec)
sys.modules["nn_reliability"] = _rel
_rel_spec.loader.exec_module(_rel)


def _args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", required=True)
    ap.add_argument("--region", default="hippocampus")
    ap.add_argument("--model-tag", default="bert-base-bidirmax")
    ap.add_argument("--context-tag", default="")
    ap.add_argument("--window-tag", default="tshift-150_tlen500_oshift+200_olen500")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--n-components", type=int, default=10)
    ap.add_argument("--n-null", type=int, default=100)
    ap.add_argument("--n-half-splits", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--random-state", type=int, default=2026)
    ap.add_argument("--results-root", required=True)
    return ap.parse_args()


def _mean_sem(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    sem = np.nanstd(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return float(np.nanmean(x)), float(sem)


def main():
    args = _args()
    glm.MODEL_TAG = args.model_tag
    glm.CONTEXT_TAG = args.context_tag
    glm.WINDOW_TAG = args.window_tag

    cfg_patient = next(p for p in glm.PATIENTS if p["patient_ID"] == args.patient)
    if args.region not in cfg_patient["region_ranges"]:
        print(f"{args.patient}: no {args.region}; skip", flush=True)
        return

    data = oo.build_other_other_data(
        cfg_patient, args.region, args.layer, args.n_components
    )
    if data is None:
        print(f"{args.patient}: no usable other-other data", flush=True)
        return

    cfg = _rel.ReliabilityConfig(
        n_null=args.n_null,
        n_half_splits=args.n_half_splits,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
    )
    res = _rel.run_beta_reliability_all_neurons(
        X_self=data["X_a"],
        X_other=data["X_b"],
        Y_self=data["Y_a"],
        Y_other=data["Y_b"],
        cfg=cfg,
        verbose=False,
    )

    detail = pd.DataFrame(res)
    # Normalize names for the other-other interpretation while keeping original
    # reliability keys untouched for compatibility with plotting/table scripts.
    r_cross_mean, r_cross_sem = _mean_sem(detail.get("r_cross", pd.Series(dtype=float)))
    a_rel_mean, a_rel_sem = _mean_sem(detail.get("self_reliability_mean", pd.Series(dtype=float)))
    b_rel_mean, b_rel_sem = _mean_sem(detail.get("other_reliability_mean", pd.Series(dtype=float)))
    ceil_vals = np.sqrt(
        np.maximum(detail.get("self_reliability_mean", pd.Series(dtype=float)), 0)
        * np.maximum(detail.get("other_reliability_mean", pd.Series(dtype=float)), 0)
    )
    ceil_mean, ceil_sem = _mean_sem(ceil_vals)
    ratio_vals = detail["r_cross"].to_numpy(float) / ceil_vals
    ratio_mean, ratio_sem = _mean_sem(ratio_vals)

    summary = pd.DataFrame([{
        "patient": args.patient,
        "region": args.region,
        "speaker_a": data["speaker_a"],
        "speaker_b": data["speaker_b"],
        "speaker_a_word_count": data["speaker_a_word_count"],
        "speaker_b_word_count": data["speaker_b_word_count"],
        "all_other_counts": repr(data["all_other_counts"]),
        "model_tag": args.model_tag,
        "context_tag": args.context_tag,
        "window_tag": args.window_tag,
        "layer": args.layer,
        "n_components": args.n_components,
        "n_trials_a": len(data["X_a"]),
        "n_trials_b": len(data["X_b"]),
        "n_neurons": int(data["Y_a"].shape[1]),
        "n_neurons_valid_r_cross": int(np.isfinite(detail["r_cross"]).sum()),
        "mean_r_cross": r_cross_mean,
        "sem_r_cross": r_cross_sem,
        "mean_speaker_a_reliability": a_rel_mean,
        "sem_speaker_a_reliability": a_rel_sem,
        "mean_speaker_b_reliability": b_rel_mean,
        "sem_speaker_b_reliability": b_rel_sem,
        "mean_noise_ceiling": ceil_mean,
        "sem_noise_ceiling": ceil_sem,
        "mean_r_cross_over_ceiling": ratio_mean,
        "sem_r_cross_over_ceiling": ratio_sem,
        "n_null": args.n_null,
        "n_half_splits": args.n_half_splits,
    }])

    pair_label = f"{data['speaker_a']}_vs_{data['speaker_b']}"
    out = os.path.join(args.results_root, pair_label, args.patient)
    os.makedirs(out, exist_ok=True)
    detail.to_csv(os.path.join(out, f"{args.region}_other_other_reliability_detail.csv"), index=False)
    summary.to_csv(os.path.join(out, f"{args.region}_other_other_reliability_summary.csv"), index=False)
    with open(os.path.join(out, f"{args.region}_other_other_reliability.pkl"), "wb") as f:
        pickle.dump({"summary": summary, "reliability": res}, f)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
