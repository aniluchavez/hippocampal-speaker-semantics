#!/usr/bin/env python3
"""Plot only the speaker/semantic encoding Venn summary."""

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


def overlap_stats(data, speaker, semantic, n_perm, seed):
    observed = int((speaker & semantic).sum())
    groups = list(data.groupby("patient").indices.values())
    expected = sum(
        speaker[idx].sum() * semantic[idx].sum() / len(idx)
        for idx in groups
    )
    rng = np.random.default_rng(seed)
    null = np.zeros(n_perm, dtype=np.int32)
    for permutation in range(n_perm):
        shuffled = semantic.copy()
        for idx in groups:
            shuffled[idx] = rng.permutation(shuffled[idx])
        null[permutation] = int((speaker & shuffled).sum())
    p = (1 + np.sum(null >= observed)) / (n_perm + 1)
    return {
        "expected_overlap": expected,
        "enrichment": observed / expected if expected > 0 else np.nan,
        "permutation_p": p,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, default=RUN_DIR)
    parser.add_argument(
        "--criterion",
        choices=["global_fdr", "raw_p05"],
        default="raw_p05",
    )
    parser.add_argument("--output_prefix", default="speaker_semantic_venn_only")
    parser.add_argument("--model_label", default="")
    parser.add_argument("--n_perm", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    all_neurons = args.run_dir / "all15_neuron_results.csv"
    if all_neurons.exists():
        data = pd.read_csv(all_neurons)
    else:
        files = sorted(args.run_dir.glob("PTY*_joint_results.csv"))
        if len(files) != 15:
            raise SystemExit(f"Expected 15 patient files, found {len(files)}")
        data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)

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
    speaker = speaker.to_numpy(bool)
    semantic = semantic.to_numpy(bool)

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
    summary_path = args.run_dir / "all15_overlap_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        summary_row = summary.loc[summary["criterion"].eq(summary_criterion)].iloc[0]
    else:
        summary_row = overlap_stats(
            data, speaker, semantic, args.n_perm, args.seed
        )

    fig, ax = plt.subplots(figsize=(6.6, 5.4), constrained_layout=True)
    ax.set_aspect("equal")
    ax.add_patch(
        Circle(
            (-0.43, 0),
            0.82,
            facecolor=COLORS["Speaker identity only"],
            edgecolor="#155b7a",
            linewidth=2,
            alpha=0.72,
        )
    )
    ax.add_patch(
        Circle(
            (0.43, 0),
            0.82,
            facecolor=COLORS["Semantics only"],
            edgecolor="#a9500b",
            linewidth=2,
            alpha=0.72,
        )
    )
    ax.text(
        -0.72,
        0.76,
        "Speaker identity",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        0.72,
        0.76,
        "Semantics",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        -0.64,
        0,
        f"{counts['Speaker identity only']}",
        ha="center",
        va="center",
        fontsize=30,
        fontweight="bold",
        color="white",
    )
    ax.text(
        0.64,
        0,
        f"{counts['Semantics only']}",
        ha="center",
        va="center",
        fontsize=30,
        fontweight="bold",
        color="white",
    )
    ax.text(
        0,
        0,
        f"{counts['Both']}",
        ha="center",
        va="center",
        fontsize=30,
        fontweight="bold",
        color="white",
        bbox=dict(
            boxstyle="circle,pad=0.35",
            facecolor=COLORS["Both"],
            edgecolor="white",
            linewidth=1.5,
        ),
    )
    ax.text(
        0,
        -1.05,
        f"Neither: {counts['Neither']}   |   expected overlap: "
        f"{summary_row['expected_overlap']:.1f}\n"
        f"enrichment = {summary_row['enrichment']:.2f}, "
        f"permutation p = {summary_row['permutation_p']:.3f}",
        ha="center",
        va="center",
        fontsize=10.5,
    )
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.25, 1.12)
    ax.axis("off")
    ax.set_title(
        "Speaker identity and semantic encoding\n"
        f"{args.model_label + ' | ' if args.model_label else ''}{criterion_label}",
        fontweight="bold",
        fontsize=15,
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png = FIGURE_DIR / f"{args.output_prefix}_{output_suffix}.png"
    pdf = FIGURE_DIR / f"{args.output_prefix}_{output_suffix}.pdf"
    svg = FIGURE_DIR / f"{args.output_prefix}_{output_suffix}.svg"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    print(png)
    print(pdf)
    print(svg)
    print(counts.to_string())


if __name__ == "__main__":
    main()
