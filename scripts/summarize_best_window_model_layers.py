#!/usr/bin/env python3
"""Collect and plot model-by-layer encoding performance at the best fixed window."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULT_ROOT = Path("/scratch/aniluchavez/ConvoDATAS/SemanticGLM")
MODELS = [
    ("gpt2-large", "_ctx200", 37, "GPT-2 Large", "#2878b5"),
    ("gpt2-xl", "_ctx200", 49, "GPT-2 XL", "#6a51a3"),
    ("llama-3.1-8b", "_ctx200", 33, "Llama 3.1 8B", "#d95f0e"),
    ("bert-base", "_ctx200", 13, "BERT-base", "#238b45"),
    ("fasttext-wiki", "", 1, "fastText static", "#555555"),
]


def result_dir(model: str, context: str, tag: str) -> Path:
    return (
        RESULT_ROOT
        / f"{model}{context}_{tag}_notebookexact_shuf_xcirc_r2only"
        / "pc50"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(
            "/scratch/aniluchavez/ConvoDATAS/SemanticGLM/plots/"
            "best_fixed_window_all15.json"
        ),
    )
    ap.add_argument(
        "--output_prefix",
        type=Path,
        default=Path(
            "/scratch/aniluchavez/ConvoDATAS/SemanticGLM/plots/"
            "all15_bestfixed_model_layer_performance"
        ),
    )
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    tag = config["spike_tag"]

    rows = []
    missing = []
    for model, context, n_layers, display, _ in MODELS:
        folder = result_dir(model, context, tag)
        for layer in range(n_layers):
            layer_files = sorted(folder.glob(f"*_L{layer:02d}_sem.pkl"))
            patient_files = [
                path for path in layer_files if not path.name.startswith("L")
            ]
            if not patient_files:
                missing.append((model, layer))
                continue
            for path in patient_files:
                with path.open("rb") as handle:
                    obj = pickle.load(handle)
                frame = obj["df"] if isinstance(obj, dict) else obj
                frame = frame.copy()
                frame["model"] = model
                frame["model_display"] = display
                frame["layer"] = layer
                frame["n_layers"] = n_layers
                frame["patient"] = frame["patient"].fillna(path.name.split("_L")[0])
                rows.append(frame)

    if missing:
        raise RuntimeError(f"Missing model/layer results: {missing[:20]}")
    detailed = pd.concat(rows, ignore_index=True)
    patient_level = (
        detailed.groupby(
            ["model", "model_display", "n_layers", "layer", "patient", "condition"],
            as_index=False,
        )
        .agg(patient_median_r2=("r2", "median"))
    )
    summary = (
        patient_level.groupby(
            ["model", "model_display", "n_layers", "layer", "condition"],
            as_index=False,
        )
        .agg(
            median_r2=("patient_median_r2", "median"),
            q25_r2=("patient_median_r2", lambda x: x.quantile(0.25)),
            q75_r2=("patient_median_r2", lambda x: x.quantile(0.75)),
            mean_r2=("patient_median_r2", "mean"),
            n_patients=("patient", "nunique"),
        )
    )
    summary["relative_depth"] = np.where(
        summary["n_layers"] > 1,
        summary["layer"] / (summary["n_layers"] - 1),
        np.nan,
    )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    patient_level.to_csv(
        args.output_prefix.with_name(args.output_prefix.name + "_patient.csv"),
        index=False,
    )
    summary.to_csv(
        args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"),
        index=False,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, constrained_layout=True)
    color_map = {model: color for model, _, _, _, color in MODELS}
    for ax, condition, title in zip(
        axes, ["self", "other"], ["Speaking/self", "Listening/other"]
    ):
        subset = summary.loc[summary["condition"].eq(condition)]
        static = subset.loc[subset["model"].eq("fasttext-wiki")]
        if not static.empty:
            y = float(static["median_r2"].iloc[0])
            ax.axhline(
                y,
                color=color_map["fasttext-wiki"],
                linestyle="--",
                linewidth=1.8,
                label="fastText static",
            )
        for model, _, n_layers, display, color in MODELS[:-1]:
            group = subset.loc[subset["model"].eq(model)].sort_values("layer")
            ax.plot(
                group["relative_depth"],
                group["median_r2"],
                color=color,
                linewidth=2,
                label=display,
            )
        ax.axhline(0, color="0.4", linewidth=0.8, linestyle=":")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Relative model depth")
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Cross-validated McFadden pseudo-R²\n(median of patient medians)")
    axes[1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle(
        "Semantic encoding by model layer at the selected fixed windows\n"
        f"self: {config['self_start_ms']:+d} to "
        f"{config['self_start_ms'] + config['self_length_ms']:+d} ms; "
        f"other: {config['other_start_ms']:+d} to "
        f"{config['other_start_ms'] + config['other_length_ms']:+d} ms",
        fontweight="bold",
    )
    png = args.output_prefix.with_suffix(".png")
    pdf = args.output_prefix.with_suffix(".pdf")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(png)
    print(pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
