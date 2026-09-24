#!/usr/bin/env python3
"""Build Fig. 1L bestfixed patient-dot panel with an extended y-axis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/scratch/aniluchavez/hippocampal-speaker-semantics")
SOURCE = Path(
    "/scratch/aniluchavez/ConvoDATAS/SemanticGLM/plots/"
    "bestfixed_bert_L12_ll_diff_barplot_paired_patient_values.csv"
)
OUT_PREFIX = ROOT / "figshare" / "SourceData" / "fig1L_dots_overlay_bestfixed"


def main() -> int:
    paired = pd.read_csv(SOURCE)
    conditions = ["self", "other"]
    colors = {"self": "#ff0000", "other": "#0000ff"}

    fig = plt.figure(figsize=(1.9, 2.6), facecolor="white")
    ax = fig.add_axes([0.326, 0.149, 0.595, 0.793])

    x_positions = np.arange(len(conditions), dtype=float)
    rng = np.random.default_rng(20260917)
    jitter = {
        cond: rng.uniform(-0.16, 0.16, len(paired))
        for cond in conditions
    }

    for x, condition in zip(x_positions, conditions):
        values = paired[condition].to_numpy(dtype=float)
        mean = float(np.mean(values))
        sem = float(pd.Series(values).sem())

        ax.bar(x, mean, width=0.60, color=colors[condition], zorder=1)
        ax.errorbar(
            x,
            mean,
            yerr=sem,
            fmt="none",
            color="black",
            lw=1.5,
            capsize=0,
            zorder=4,
        )
        ax.scatter(
            x + jitter[condition],
            values,
            s=14,
            facecolor="black",
            edgecolor="white",
            linewidth=0.3,
            zorder=5,
        )

    for _, row in paired.iterrows():
        ax.plot(
            x_positions + [jitter["self"][row.name], jitter["other"][row.name]],
            [row["self"], row["other"]],
            color="0.5",
            alpha=0.6,
            lw=0.6,
            zorder=2,
        )

    ax.set_xlim(-0.38, 1.38)
    ax.set_ylim(0, 160)
    ax.set_yticks([0, 40, 80, 120, 160])
    ax.set_xticks(x_positions, conditions)
    ax.set_ylabel("log likelihood (LLH)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)

    bracket_y0, bracket_y1 = 143, 150
    ax.plot([0, 0, 1, 1], [bracket_y0, bracket_y1, bracket_y1, bracket_y0], color="black", lw=1)
    ax.text(0.5, 153, "p=0.03", ha="center", va="bottom", fontsize=8)

    OUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".pdf", ".svg", ".png"]:
        fig.savefig(OUT_PREFIX.with_suffix(suffix), dpi=300)
    plt.close(fig)

    print(f"Wrote {OUT_PREFIX.with_suffix('.pdf')}")
    print(f"Max plotted value: {paired[['self', 'other']].to_numpy().max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
