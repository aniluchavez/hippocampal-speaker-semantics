#!/usr/bin/env python3
"""Cleaner marker-only SVG for model/layer semantic encoding performance.

This intentionally avoids connecting lines while keeping enough visual structure
to avoid the "random confetti" look: small layer points, model-specific horizontal
jitter, and a ring around each model's best layer within each condition.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SUMMARY = Path(
    "/scratch/aniluchavez/ConvoDATAS/SemanticGLM/plots/"
    "all15_bestfixed_model_layer_performance_summary.csv"
)
OUT = Path(
    "/scratch/aniluchavez/ConvoDATAS/SemanticGLM/plots/"
    "all15_bestfixed_model_layer_performance_clean_nolines.svg"
)

MODELS = [
    ("bert-base", "BERT-base", "#238b45", "D", -0.018),
    ("gpt2-large", "GPT-2 Large", "#2878b5", "o", -0.006),
    ("gpt2-xl", "GPT-2 XL", "#6a51a3", "s", 0.006),
    ("llama-3.1-8b", "LLaMA 3.1 8B", "#d95f0e", "^", 0.018),
    ("fasttext-wiki", "fastText static", "#555555", "X", 0.000),
]


def main() -> int:
    df = pd.read_csv(SUMMARY)
    df["relative_depth"] = np.where(
        df["n_layers"].gt(1),
        df["layer"] / (df["n_layers"] - 1),
        0.5,
    )

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "svg.fonttype": "none",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.35), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.80, bottom=0.17, top=0.80, wspace=0.12)

    for ax, condition, title in zip(
        axes, ["self", "other"], ["Speaking/self", "Listening/other"]
    ):
        sub = df.loc[df["condition"].eq(condition)].copy()

        for model, label, color, marker, jitter in MODELS:
            g = sub.loc[sub["model"].eq(model)].sort_values("layer")
            if g.empty:
                continue
            x = g["relative_depth"].to_numpy()
            y = g["median_r2"].to_numpy()
            x_plot = x + (0.0 if model == "fasttext-wiki" else jitter)

            # Faint larger underlay gives each model a visual footprint without
            # drawing any line between layers.
            if model != "fasttext-wiki":
                ax.scatter(
                    x_plot,
                    y,
                    s=75,
                    marker=marker,
                    color=color,
                    alpha=0.10,
                    linewidth=0,
                    zorder=1,
                )

            ax.scatter(
                x_plot,
                y,
                s=28 if model != "fasttext-wiki" else 64,
                marker=marker,
                facecolor=color,
                edgecolor="white",
                linewidth=0.35,
                alpha=0.84,
                label=label,
                zorder=2,
            )

            if np.isfinite(y).any():
                best = int(np.nanargmax(y))
                ax.scatter(
                    [x_plot[best]],
                    [y[best]],
                    s=96 if model != "fasttext-wiki" else 86,
                    marker=marker,
                    facecolor="none",
                    edgecolor=color,
                    linewidth=1.45,
                    zorder=4,
                )

        ax.axhline(0, color="0.72", linewidth=0.7, linestyle=(0, (2, 2)), zorder=0)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Relative model depth")
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks([0, 0.5, 1.0])
        ax.grid(False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["left"].set_color("0.25")
        ax.spines["bottom"].set_color("0.25")
        ax.tick_params(axis="both", colors="0.20", length=3)

    axes[0].set_ylabel(
        "Cross-validated McFadden pseudo-R²\nmedian of patient medians"
    )
    handles, labels = axes[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[1].legend(
        by_label.values(),
        by_label.keys(),
        frameon=False,
        bbox_to_anchor=(1.02, 1.02),
        loc="upper left",
        title="Model",
    )
    fig.suptitle(
        "Semantic encoding by model layer at selected fixed windows\n"
        "self: −300 to +200 ms; other: +20 to +520 ms",
        fontweight="bold",
        y=0.97,
    )
    fig.text(
        0.085,
        0.03,
        "Open circles mark the best layer within each model/condition. Points are not connected.",
        fontsize=8.5,
        color="0.25",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
