#!/usr/bin/env python3
"""Paired patient-level tests for layer-wise encoding differences.

Each observation is one patient's median cross-validated pseudo-R². Tests are
paired across patients. FDR is applied separately within each model/condition.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


MODEL_ORDER = ["GPT-2 Large", "GPT-2 XL", "Llama 3.1 8B", "BERT-base"]
CONDITION_COLORS = {"self": "#8b3a62", "other": "#c35a24"}


def paired_result(x: np.ndarray, y: np.ndarray) -> dict:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    difference = y - x
    n = len(difference)
    if n < 2:
        return {
            "n_patients": n,
            "mean_difference": np.nan,
            "median_difference": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "t": np.nan,
            "p_raw": np.nan,
            "cohen_dz": np.nan,
        }
    result = stats.ttest_rel(y, x, nan_policy="omit")
    mean_difference = float(np.mean(difference))
    sd_difference = float(np.std(difference, ddof=1))
    sem = stats.sem(difference)
    critical = stats.t.ppf(0.975, n - 1)
    return {
        "n_patients": n,
        "mean_difference": mean_difference,
        "median_difference": float(np.median(difference)),
        "ci95_low": float(mean_difference - critical * sem),
        "ci95_high": float(mean_difference + critical * sem),
        "t": float(result.statistic),
        "p_raw": float(result.pvalue),
        "cohen_dz": (
            float(mean_difference / sd_difference)
            if sd_difference > 0
            else np.nan
        ),
    }


def fdr_column(frame: pd.DataFrame, p_column: str, output_column: str) -> None:
    frame[output_column] = np.nan
    valid = frame[p_column].notna()
    if valid.any():
        frame.loc[valid, output_column] = multipletests(
            frame.loc[valid, p_column], method="fdr_bh"
        )[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=Path(
            "/scratch/aniluchavez/ConvoDATAS/SemanticGLM/plots/"
            "all15_bestfixed_model_layer_performance_patient.csv"
        ),
    )
    ap.add_argument(
        "--output_prefix",
        type=Path,
        default=Path(
            "/scratch/aniluchavez/ConvoDATAS/SemanticGLM/plots/"
            "all15_bestfixed_layer_differences"
        ),
    )
    args = ap.parse_args()

    data = pd.read_csv(args.input)
    data = data.loc[~data["model"].eq("fasttext-wiki")].copy()
    pairwise_rows = []
    omnibus_rows = []

    for (model, display, condition), group in data.groupby(
        ["model", "model_display", "condition"], sort=False
    ):
        pivot = group.pivot(
            index="patient", columns="layer", values="patient_median_r2"
        ).sort_index(axis=1)
        layers = list(pivot.columns.astype(int))

        complete = pivot.dropna()
        if len(complete) >= 2:
            friedman = stats.friedmanchisquare(
                *(complete[layer].to_numpy() for layer in layers)
            )
            omnibus_rows.append(
                {
                    "model": model,
                    "model_display": display,
                    "condition": condition,
                    "n_patients": len(complete),
                    "n_layers": len(layers),
                    "friedman_chi2": float(friedman.statistic),
                    "friedman_p": float(friedman.pvalue),
                }
            )

        for layer_a, layer_b in combinations(layers, 2):
            result = paired_result(
                pivot[layer_a].to_numpy(), pivot[layer_b].to_numpy()
            )
            pairwise_rows.append(
                {
                    "model": model,
                    "model_display": display,
                    "condition": condition,
                    "layer_a": layer_a,
                    "layer_b": layer_b,
                    "layer_distance": layer_b - layer_a,
                    "is_adjacent": layer_b == layer_a + 1,
                    "is_vs_first": layer_a == layers[0],
                    **result,
                }
            )

    pairwise = pd.DataFrame(pairwise_rows)
    corrected_groups = []
    for _, group in pairwise.groupby(["model", "condition"], sort=False):
        group = group.copy()
        fdr_column(group, "p_raw", "p_fdr_all_pairs")

        group["p_fdr_adjacent"] = np.nan
        adjacent = group["is_adjacent"]
        if adjacent.any():
            adjacent_frame = group.loc[adjacent].copy()
            fdr_column(adjacent_frame, "p_raw", "p_fdr_adjacent")
            group.loc[adjacent, "p_fdr_adjacent"] = adjacent_frame[
                "p_fdr_adjacent"
            ].to_numpy()

        group["p_fdr_vs_first"] = np.nan
        vs_first = group["is_vs_first"]
        if vs_first.any():
            first_frame = group.loc[vs_first].copy()
            fdr_column(first_frame, "p_raw", "p_fdr_vs_first")
            group.loc[vs_first, "p_fdr_vs_first"] = first_frame[
                "p_fdr_vs_first"
            ].to_numpy()
        corrected_groups.append(group)

    pairwise = pd.concat(corrected_groups, ignore_index=True)
    pairwise["significant_all_pairs"] = pairwise["p_fdr_all_pairs"] < 0.05
    pairwise["significant_adjacent"] = pairwise["p_fdr_adjacent"] < 0.05
    pairwise["significant_vs_first"] = pairwise["p_fdr_vs_first"] < 0.05
    adjacent = pairwise.loc[pairwise["is_adjacent"]].copy()
    vs_first = pairwise.loc[pairwise["is_vs_first"]].copy()
    omnibus = pd.DataFrame(omnibus_rows)
    fdr_column(omnibus, "friedman_p", "friedman_p_fdr")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        "pairwise": args.output_prefix.with_name(
            args.output_prefix.name + "_pairwise_ttests.csv"
        ),
        "adjacent": args.output_prefix.with_name(
            args.output_prefix.name + "_adjacent_ttests.csv"
        ),
        "vs_first": args.output_prefix.with_name(
            args.output_prefix.name + "_vs_first_ttests.csv"
        ),
        "omnibus": args.output_prefix.with_name(
            args.output_prefix.name + "_omnibus_friedman.csv"
        ),
    }
    pairwise.to_csv(outputs["pairwise"], index=False)
    adjacent.to_csv(outputs["adjacent"], index=False)
    vs_first.to_csv(outputs["vs_first"], index=False)
    omnibus.to_csv(outputs["omnibus"], index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, display in zip(axes.flat, MODEL_ORDER):
        subset = adjacent.loc[adjacent["model_display"].eq(display)]
        for condition in ["self", "other"]:
            group = subset.loc[subset["condition"].eq(condition)].sort_values(
                "layer_b"
            )
            ax.plot(
                group["layer_b"],
                group["mean_difference"],
                marker="o",
                markersize=3.5,
                linewidth=1.5,
                color=CONDITION_COLORS[condition],
                label=condition,
            )
            significant = group["significant_adjacent"]
            ax.scatter(
                group.loc[significant, "layer_b"],
                group.loc[significant, "mean_difference"],
                s=55,
                facecolors="none",
                edgecolors=CONDITION_COLORS[condition],
                linewidths=1.8,
            )
        ax.axhline(0, color="0.4", linestyle="--", linewidth=0.8)
        ax.set_title(display, fontweight="bold")
        ax.set_xlabel("Upper layer in adjacent comparison")
        ax.set_ylabel("Mean paired Δ pseudo-R²")
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(
        "Adjacent-layer encoding changes across patients\n"
        "Open circles: paired t-test significant after within-model/condition FDR",
        fontweight="bold",
    )
    png = args.output_prefix.with_suffix(".png")
    pdf = args.output_prefix.with_suffix(".pdf")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    for name, output in outputs.items():
        print(f"{name}: {output}")
    print(f"plot: {png}")
    print(f"plot: {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
