#!/usr/bin/env python3
"""Recompute PC50 r_cross summaries using only above-null significant neurons."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


BASE = Path("/scratch/aniluchavez/hippocampal-speaker-semantics/results")
PC = 50


def sem(x) -> float:
    x = pd.Series(x).dropna()
    return x.std(ddof=1) / np.sqrt(x.count()) if x.count() > 1 else np.nan


def main() -> None:
    above = pd.read_csv(
        BASE / "bert_l12_fixed_unbalanced_pc_sweep_rcross_above_null_no_fdr_neurons.csv"
    )
    rel = pd.read_csv(
        BASE / "bert_l12_fixed_unbalanced_pc_sweep_reliability_neurons.csv"
    )

    sig = above[(above["pc"] == PC) & (above["raw_sig_p05"].astype(bool))].copy()
    rel50 = rel[rel["pc"] == PC].copy()
    merged = sig.merge(
        rel50[
            [
                "pc",
                "patient",
                "neuron",
                "ceiling",
                "self_reliability_mean",
                "other_reliability_mean",
            ]
        ],
        on=["pc", "patient", "neuron"],
        how="left",
    )
    merged["ceiling_minus_rcross"] = merged["ceiling"] - merged["r_cross"]
    merged["obs_minus_null"] = merged["r_cross"] - merged["null_mean"]

    pat = merged.groupby("patient", as_index=False).agg(
        n_sig_neurons=("r_cross", "size"),
        mean_r_cross_sig=("r_cross", "mean"),
        median_r_cross_sig=("r_cross", "median"),
        mean_null_mean_sig=("null_mean", "mean"),
        mean_diff_obs_minus_null_sig=("obs_minus_null", "mean"),
        mean_ceiling_sig=("ceiling", "mean"),
        median_ceiling_sig=("ceiling", "median"),
        mean_ceiling_minus_rcross_sig=("ceiling_minus_rcross", "mean"),
    )

    all_patients = sorted(rel50["patient"].unique())
    missing = sorted(set(all_patients) - set(pat["patient"]))

    diff_null = pat["mean_diff_obs_minus_null_sig"].to_numpy(float)
    diff_ceil = pat["mean_ceiling_minus_rcross_sig"].to_numpy(float)
    if len(diff_null) > 1:
        w_null, p_null = stats.wilcoxon(
            diff_null, alternative="greater", zero_method="wilcox", method="auto"
        )
        w_ceil, p_ceil = stats.wilcoxon(
            diff_ceil, alternative="greater", zero_method="wilcox", method="auto"
        )
    else:
        w_null = p_null = w_ceil = p_ceil = np.nan

    summary = pd.DataFrame(
        [
            {
                "pc": PC,
                "n_patients_total": len(all_patients),
                "n_patients_with_sig_neurons": pat["patient"].nunique(),
                "n_sig_neurons": len(merged),
                "n_total_neurons": len(rel50),
                "pct_neurons_retained": 100 * len(merged) / len(rel50),
                "neuron_mean_r_cross_sig": merged["r_cross"].mean(),
                "neuron_median_r_cross_sig": merged["r_cross"].median(),
                "neuron_mean_null_mean_sig": merged["null_mean"].mean(),
                "neuron_mean_diff_obs_minus_null_sig": merged["obs_minus_null"].mean(),
                "neuron_mean_ceiling_sig": merged["ceiling"].mean(),
                "neuron_median_ceiling_sig": merged["ceiling"].median(),
                "neuron_mean_ceiling_minus_rcross_sig": merged[
                    "ceiling_minus_rcross"
                ].mean(),
                "patient_mean_r_cross_sig": pat["mean_r_cross_sig"].mean(),
                "patient_sem_r_cross_sig": sem(pat["mean_r_cross_sig"]),
                "patient_median_r_cross_sig": pat["mean_r_cross_sig"].median(),
                "patient_mean_null_mean_sig": pat["mean_null_mean_sig"].mean(),
                "patient_mean_diff_obs_minus_null_sig": pat[
                    "mean_diff_obs_minus_null_sig"
                ].mean(),
                "patient_mean_ceiling_sig": pat["mean_ceiling_sig"].mean(),
                "patient_sem_ceiling_sig": sem(pat["mean_ceiling_sig"]),
                "patient_mean_ceiling_minus_rcross_sig": pat[
                    "mean_ceiling_minus_rcross_sig"
                ].mean(),
                "wilcoxon_W_obs_minus_null_greater": w_null,
                "wilcoxon_p_obs_minus_null_greater": p_null,
                "wilcoxon_W_ceiling_minus_obs_greater": w_ceil,
                "wilcoxon_p_ceiling_minus_obs_greater": p_ceil,
                "patients_without_sig_neurons": ";".join(missing),
            }
        ]
    )

    out_sum = BASE / "bert_l12_fixed_pc50_sigonly_rcross_summary.csv"
    out_pat = BASE / "bert_l12_fixed_pc50_sigonly_rcross_patient.csv"
    out_neu = BASE / "bert_l12_fixed_pc50_sigonly_rcross_neurons.csv"
    summary.to_csv(out_sum, index=False)
    pat.to_csv(out_pat, index=False)
    merged.to_csv(out_neu, index=False)

    print("SUMMARY")
    print(summary.to_string(index=False))
    print("\nPER_PATIENT")
    print(pat.to_string(index=False))
    print("\nWROTE")
    print(out_sum)
    print(out_pat)
    print(out_neu)


if __name__ == "__main__":
    main()
