#!/usr/bin/env python3
"""Marker-only SVG for best-fixed-window model/layer encoding performance."""

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
    "all15_bestfixed_model_layer_performance_nolines.svg"
)

MODEL_ORDER = [
    ("gpt2-large", "GPT-2 Large", "#2878b5", "o"),
    ("gpt2-xl", "GPT-2 XL", "#6a51a3", "s"),
    ("llama-3.1-8b", "Llama 3.1 8B", "#d95f0e", "^"),
    ("bert-base", "BERT-base", "#238b45", "D"),
    ("fasttext-wiki", "fastText static", "#555555", "X"),
]


def main() -> int:
    df = pd.read_csv(SUMMARY)
    df["relative_depth"] = np.where(
        df["n_layers"].gt(1),
        df["layer"] / (df["n_layers"] - 1),
        0.5,
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(11.5, 4.8), sharey=True, constrained_layout=True
    )

    for ax, condition, title in zip(
        axes, ["self", "other"], ["Speaking/self", "Listening/other"]
    ):
        subset = df.loc[df["condition"].eq(condition)].copy()
        for model, label, color, marker in MODEL_ORDER:
            group = subset.loc[subset["model"].eq(model)].sort_values("layer")
            if group.empty:
                continue
            size = 60 if model == "fasttext-wiki" else 34
            ax.scatter(
                group["relative_depth"],
                group["median_r2"],
                s=size,
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.45,
                alpha=0.92,
                label=label,
                zorder=3,
            )

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Relative model depth")
        ax.set_xlim(-0.04, 1.04)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)

    axes[0].set_ylabel(
        "Cross-validated McFadden pseudo-R²\nmedian of patient medians"
    )
    axes[1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle(
        "Semantic encoding by model layer at the selected fixed windows\n"
        "self: −300 to +200 ms; other: +20 to +520 ms",
        fontweight="bold",
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
