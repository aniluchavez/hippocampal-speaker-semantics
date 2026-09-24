#!/usr/bin/env python3
"""Summarize GPT-2 XL context-window encoding sweep."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULT_ROOT = Path("/scratch/aniluchavez/ConvoDATAS/SemanticGLM")
PLOT_ROOT = Path("/scratch/aniluchavez/ConvoDATAS/SemanticGLM/plots")
MODEL = "gpt2-xl"
LAYER = 48
PC = 50
SPIKE_TAG = "bestfixed_selfm300_len500_otherp20_len500"
CONTEXTS = [0, 25, 50, 100, 200, 400, 800]


def result_dir(context_words: int) -> Path:
    return (
        RESULT_ROOT
        / f"{MODEL}_ctx{context_words}_{SPIKE_TAG}_notebookexact_shuf_xcirc_r2only"
        / f"pc{PC}"
    )


def load_rows(context_words: int) -> tuple[list[pd.DataFrame], list[str]]:
    folder = result_dir(context_words)
    rows: list[pd.DataFrame] = []
    missing: list[str] = []
    for path in sorted(folder.glob(f"*_L{LAYER:02d}_sem.pkl")):
        if path.name.startswith("L"):
            continue
        with path.open("rb") as handle:
            obj = pickle.load(handle)
        frame = obj["df"] if isinstance(obj, dict) else obj
        frame = frame.copy()
        frame["context_words"] = context_words
        frame["patient"] = frame["patient"].fillna(path.name.split("_L")[0])
        rows.append(frame)
    if not rows:
        missing.append(str(folder))
    return rows, missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_prefix", type=Path,
                    default=PLOT_ROOT / "all15_gpt2xl_L48_pc50_context_window_sweep")
    ap.add_argument("--contexts", type=str, default=",".join(map(str, CONTEXTS)))
    args = ap.parse_args()
    contexts = [int(x) for x in args.contexts.split(",") if x.strip()]

    all_rows: list[pd.DataFrame] = []
    missing: list[str] = []
    for ctx in contexts:
        rows, miss = load_rows(ctx)
        all_rows.extend(rows)
        missing.extend(miss)

    if not all_rows:
        raise RuntimeError("No context-sweep result PKLs found yet.")

    detailed = pd.concat(all_rows, ignore_index=True)
    patient_level = (
        detailed.groupby(["context_words", "patient", "condition"], as_index=False)
        .agg(patient_median_r2=("r2", "median"), n_neurons=("neuron_idx", "nunique"))
    )
    summary = (
        patient_level.groupby(["context_words", "condition"], as_index=False)
        .agg(
            median_r2=("patient_median_r2", "median"),
            q25_r2=("patient_median_r2", lambda x: x.quantile(0.25)),
            q75_r2=("patient_median_r2", lambda x: x.quantile(0.75)),
            mean_r2=("patient_median_r2", "mean"),
            n_patients=("patient", "nunique"),
            n_neurons=("n_neurons", "sum"),
        )
    )
    summary["context_label"] = summary["context_words"].map(
        lambda x: "word only" if x == 0 else str(x)
    )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(args.output_prefix.with_name(args.output_prefix.name + "_detailed.csv"), index=False)
    patient_level.to_csv(args.output_prefix.with_name(args.output_prefix.name + "_patient.csv"), index=False)
    summary.to_csv(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharey=True, constrained_layout=True)
    colors = {"self": "#984ea3", "other": "#377eb8"}
    titles = {"self": "Speaking/self", "other": "Listening/other"}
    for ax, condition in zip(axes, ["self", "other"]):
        sub = summary.loc[summary["condition"].eq(condition)].sort_values("context_words")
        x = np.arange(len(sub))
        ax.plot(x, sub["median_r2"], marker="o", linewidth=2.2, color=colors[condition])
        ax.fill_between(x, sub["q25_r2"], sub["q75_r2"], color=colors[condition], alpha=0.18)
        ax.axhline(0, color="0.45", linestyle=":", linewidth=1)
        ax.set_xticks(x, sub["context_label"], rotation=35, ha="right")
        ax.set_xlabel("Causal context window (preceding words)")
        ax.set_title(titles[condition], fontweight="bold")
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Held-out McFadden pseudo-R²\nmedian of patient medians; band = IQR")
    fig.suptitle(
        "GPT-2 XL L48 semantic encoding vs causal context length\n"
        "fixed hippocampal windows: self −300→+200 ms, other +20→+520 ms; 50 PCs",
        fontweight="bold",
    )
    png = args.output_prefix.with_suffix(".png")
    pdf = args.output_prefix.with_suffix(".pdf")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    print(summary.to_string(index=False))
    if missing:
        print("\nMissing result folders:")
        for item in missing:
            print(f"  {item}")
    print(f"\nSaved:\n  {png}\n  {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
