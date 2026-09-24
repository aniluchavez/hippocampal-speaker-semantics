#!/usr/bin/env python3
"""Visualize neurons encoding speaker identity, semantics, either, or both."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd


RUN_DIR = Path(
    "/scratch/aniluchavez/ConvoDATAS/SpeakerSemanticJoint/"
    "llama-3.1-8b_ctx200_L14_pc50_fixed_self0_other0_len500_"
    "blockcv_yfoldcirc_perm100"
)
FIGURE_DIR = Path(
    "/scratch/aniluchavez/hippocampal-speaker-semantics/figures"
)

COLORS = {
    "Neither": "#d6d6d6",
    "Speaker identity only": "#2b8cbe",
    "Semantics only": "#e67e22",
    "Both": "#7b3294",
}
ORDER = ["Speaker identity only", "Semantics only", "Both", "Neither"]


parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", type=Path, default=RUN_DIR)
parser.add_argument(
    "--criterion",
    choices=["global_fdr", "raw_p05"],
    default="global_fdr",
)
parser.add_argument("--window_label", default="common 0–500 ms window")
parser.add_argument("--output_prefix", default="speaker_semantic_neuron_populations")
args = parser.parse_args()

data = pd.read_csv(args.run_dir / "all15_neuron_results.csv")
summary = pd.read_csv(args.run_dir / "all15_overlap_summary.csv")
if args.criterion == "global_fdr":
    speaker = data["speaker_global_fdr_sig"].astype(bool)
    semantic = data["semantic_global_fdr_sig"].astype(bool)
    criterion_label = "Global FDR q<0.05"
    summary_criterion = "global_fdr"
    output_suffix = "globalFDR"
else:
    speaker = data["speaker_raw_sig"].astype(bool)
    semantic = data["semantic_raw_sig"].astype(bool)
    criterion_label = "Raw p<0.05 (uncorrected)"
    summary_criterion = "raw_p05"
    output_suffix = "rawP05"
data["category"] = np.select(
    [
        speaker & semantic,
        speaker & ~semantic,
        ~speaker & semantic,
    ],
    ["Both", "Speaker identity only", "Semantics only"],
    default="Neither",
)
counts = data["category"].value_counts().reindex(ORDER, fill_value=0)
n_total = len(data)

fig = plt.figure(figsize=(16, 9), constrained_layout=True)
grid = fig.add_gridspec(2, 2, width_ratios=[0.9, 1.5], height_ratios=[1, 1])
ax_venn = fig.add_subplot(grid[0, 0])
ax_bars = fig.add_subplot(grid[1, 0])
ax_scatter = fig.add_subplot(grid[:, 1])

# Panel A: Venn-style summary.
ax_venn.set_aspect("equal")
ax_venn.add_patch(
    Circle(
        (-0.43, 0),
        0.82,
        facecolor=COLORS["Speaker identity only"],
        edgecolor="#155b7a",
        linewidth=2,
        alpha=0.72,
    )
)
ax_venn.add_patch(
    Circle(
        (0.43, 0),
        0.82,
        facecolor=COLORS["Semantics only"],
        edgecolor="#a9500b",
        linewidth=2,
        alpha=0.72,
    )
)
ax_venn.text(
    -0.72,
    0.76,
    "Speaker identity",
    ha="center",
    va="center",
    fontsize=15,
    fontweight="bold",
)
ax_venn.text(
    0.72,
    0.76,
    "Semantics",
    ha="center",
    va="center",
    fontsize=15,
    fontweight="bold",
)
ax_venn.text(
    -0.64,
    0,
    f"{counts['Speaker identity only']}",
    ha="center",
    va="center",
    fontsize=28,
    fontweight="bold",
    color="white",
)
ax_venn.text(
    0.64,
    0,
    f"{counts['Semantics only']}",
    ha="center",
    va="center",
    fontsize=28,
    fontweight="bold",
    color="white",
)
ax_venn.text(
    0,
    0,
    f"{counts['Both']}",
    ha="center",
    va="center",
    fontsize=28,
    fontweight="bold",
    color="white",
    bbox=dict(
        boxstyle="circle,pad=0.35",
        facecolor=COLORS["Both"],
        edgecolor="white",
        linewidth=1.5,
    ),
)
global_row = summary.loc[
    summary["criterion"].eq(summary_criterion)
].iloc[0]
ax_venn.text(
    0,
    -1.04,
    f"Neither: {counts['Neither']}   •   overlap expected: "
    f"{global_row['expected_overlap']:.1f}\n"
    f"enrichment = {global_row['enrichment']:.2f}, "
    f"permutation p = {global_row['permutation_p']:.3f}",
    ha="center",
    va="center",
    fontsize=11,
)
ax_venn.set_xlim(-1.45, 1.45)
ax_venn.set_ylim(-1.25, 1.12)
ax_venn.axis("off")
ax_venn.set_title("A  Encoding populations", loc="left", fontweight="bold")

# Panel B: patient-wise category proportions.
patient_counts = (
    data.groupby(["patient", "category"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=ORDER, fill_value=0)
)
patient_props = patient_counts.div(patient_counts.sum(axis=1), axis=0)
patient_props = patient_props.sort_values(
    ["Both", "Speaker identity only", "Semantics only"], ascending=False
)
left = np.zeros(len(patient_props))
y = np.arange(len(patient_props))
for category in ORDER:
    values = patient_props[category].to_numpy()
    ax_bars.barh(
        y,
        values,
        left=left,
        color=COLORS[category],
        edgecolor="white",
        linewidth=0.5,
        label=category,
    )
    left += values
ax_bars.set_yticks(y)
ax_bars.set_yticklabels(
    [name.replace("_task", " · ") for name in patient_props.index], fontsize=8
)
ax_bars.invert_yaxis()
ax_bars.set_xlim(0, 1)
ax_bars.set_xlabel("Proportion of hippocampal neurons")
ax_bars.set_title("B  Encoding profile by patient", loc="left", fontweight="bold")
ax_bars.spines[["top", "right", "left"]].set_visible(False)
ax_bars.grid(axis="x", alpha=0.15)

# Panel C: every neuron in unique-effect space.
plot_order = ["Neither", "Speaker identity only", "Semantics only", "Both"]
sizes = {
    "Neither": 18,
    "Speaker identity only": 35,
    "Semantics only": 35,
    "Both": 58,
}
alphas = {"Neither": 0.35, "Speaker identity only": 0.72, "Semantics only": 0.72, "Both": 0.95}
for category in plot_order:
    group = data.loc[data["category"].eq(category)]
    ax_scatter.scatter(
        group["unique_speaker_delta_ll_per_word"],
        group["unique_semantic_delta_ll_per_word"],
        s=sizes[category],
        color=COLORS[category],
        alpha=alphas[category],
        edgecolor="white" if category != "Neither" else "none",
        linewidth=0.5,
        label=f"{category} (n={len(group)})",
        zorder=3 if category == "Both" else 2,
    )
ax_scatter.axhline(0, color="0.55", linewidth=1, linestyle="--")
ax_scatter.axvline(0, color="0.55", linewidth=1, linestyle="--")
ax_scatter.set_xlabel("Unique speaker-identity encoding (ΔLL / word)")
ax_scatter.set_ylabel("Unique semantic encoding (ΔLL / word)")
ax_scatter.set_title(
    "C  Every neuron in unique encoding space",
    loc="left",
    fontweight="bold",
)
ax_scatter.spines[["top", "right"]].set_visible(False)
ax_scatter.grid(alpha=0.12)
ax_scatter.legend(
    frameon=False,
    loc="upper right",
    title=criterion_label,
    fontsize=10,
)

fig.suptitle(
    "Speaker identity and semantic encoding in human hippocampal neurons\n"
    f"Llama 3.1 8B layer 14 • {args.window_label} • "
    f"{n_total} neurons across 15 patients",
    fontsize=18,
    fontweight="bold",
)

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
png = FIGURE_DIR / f"{args.output_prefix}_{output_suffix}.png"
pdf = FIGURE_DIR / f"{args.output_prefix}_{output_suffix}.pdf"
fig.savefig(png, dpi=240, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
print(png)
print(pdf)
print(counts.to_string())
