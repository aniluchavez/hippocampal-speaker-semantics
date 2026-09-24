#!/usr/bin/env python3
"""Plot across-patient fixed- and variable-window semantic GLM sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_task_csvs(root: Path, job_ids: list[str]) -> pd.DataFrame:
    frames = []
    for job_id in job_ids:
        for path in (root / job_id).glob("*.csv"):
            frame = pd.read_csv(path)
            # The first PTYEU array predates the patient column.
            if "patient" not in frame or frame["patient"].isna().all():
                frame["patient"] = "PTYEU_task147"
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No task CSVs found for jobs: {job_ids}")
    return pd.concat(frames, ignore_index=True)


def collapse_condition(
    frame: pd.DataFrame, condition: str, window_columns: list[str]
) -> pd.DataFrame:
    return (
        frame.loc[frame["condition"].eq(condition)]
        .groupby(["patient", *window_columns], as_index=False)
        .agg(median_r2=("median_r2", "mean"))
    )


def summarize(frame: pd.DataFrame, window_columns: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(window_columns, as_index=False)
        .agg(
            median=("median_r2", "median"),
            q25=("median_r2", lambda x: x.quantile(0.25)),
            q75=("median_r2", lambda x: x.quantile(0.75)),
            n_patients=("patient", "nunique"),
        )
        .sort_values(window_columns)
    )


def plot_line(ax, data, x_col, label, color):
    x = data[x_col].to_numpy(dtype=float)
    median = data["median"].to_numpy(dtype=float)
    q25 = data["q25"].to_numpy(dtype=float)
    q75 = data["q75"].to_numpy(dtype=float)
    ax.plot(x, median, marker="o", linewidth=2.2, label=label, color=color)
    ax.fill_between(x, q25, q75, color=color, alpha=0.14, linewidth=0)


def style_axis(ax, title, xlabel):
    ax.axhline(0, color="0.35", linewidth=0.9, linestyle="--", zorder=0)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cross-validated McFadden pseudo-R²")
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed_jobs", default="495559,495657")
    ap.add_argument("--variable_jobs", default="495566,495661")
    ap.add_argument(
        "--task_root",
        type=Path,
        default=Path(
            "/scratch/aniluchavez/ConvoDATAS/SemanticGLM/plots/window_sweep_tasks"
        ),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/scratch/aniluchavez/ConvoDATAS/SemanticGLM/plots/"
            "all14_gpt2large_L36_pc50_window_sweep_pattern.png"
        ),
    )
    args = ap.parse_args()

    fixed = load_task_csvs(args.task_root, args.fixed_jobs.split(","))
    variable = load_task_csvs(args.task_root, args.variable_jobs.split(","))

    fixed_self = summarize(
        collapse_condition(fixed, "self", ["length_ms", "self_start_ms"]),
        ["length_ms", "self_start_ms"],
    )
    fixed_other = summarize(
        collapse_condition(fixed, "other", ["length_ms", "other_start_ms"]),
        ["length_ms", "other_start_ms"],
    )
    variable_self = summarize(
        collapse_condition(
            variable, "self", ["self_start_ms", "self_end_shift_ms"]
        ),
        ["self_start_ms", "self_end_shift_ms"],
    )
    variable_other = summarize(
        collapse_condition(
            variable, "other", ["other_start_ms", "other_end_shift_ms"]
        ),
        ["other_start_ms", "other_end_shift_ms"],
    )

    n_patients = min(fixed["patient"].nunique(), variable["patient"].nunique())
    colors = {200: "#8fb9dd", 300: "#367eaa", 500: "#173f5f"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    for length, group in fixed_self.groupby("length_ms"):
        plot_line(
            axes[0, 0],
            group,
            "self_start_ms",
            f"{int(length)} ms",
            colors[int(length)],
        )
    style_axis(
        axes[0, 0], "A  Speaking/self — fixed windows", "Start relative to onset (ms)"
    )
    axes[0, 0].legend(title="Window length", frameon=False)

    for length, group in fixed_other.groupby("length_ms"):
        plot_line(
            axes[0, 1],
            group,
            "other_start_ms",
            f"{int(length)} ms",
            colors[int(length)],
        )
    style_axis(
        axes[0, 1],
        "B  Listening/other — fixed windows",
        "Start relative to onset (ms)",
    )
    axes[0, 1].legend(title="Window length", frameon=False)

    for end_shift, group in variable_self.groupby("self_end_shift_ms"):
        plot_line(
            axes[1, 0],
            group,
            "self_start_ms",
            f"End = offset {int(end_shift):+d} ms",
            "#8b3a62",
        )
    style_axis(
        axes[1, 0],
        "C  Speaking/self — variable windows",
        "Start relative to onset (ms)",
    )
    axes[1, 0].legend(frameon=False)

    for end_shift, group in variable_other.groupby("other_end_shift_ms"):
        plot_line(
            axes[1, 1],
            group,
            "other_start_ms",
            f"End = offset {int(end_shift):+d} ms",
            "#c35a24",
        )
    style_axis(
        axes[1, 1],
        "D  Listening/other — variable windows",
        "Start relative to onset (ms)",
    )
    axes[1, 1].legend(frameon=False)

    fig.suptitle(
        f"Semantic encoding across window definitions (GPT-2 Large, L36, 50 PCs; n={n_patients})\n"
        "Lines: median across patient-level median R²; shading: interquartile range",
        fontsize=14,
        fontweight="bold",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    pdf_path = args.output.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(args.output)
    print(pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
