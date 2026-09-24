#!/usr/bin/env python3
"""Plot layer-by-layer LLH improvement for speaker-tag BERT fixed windows."""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


RESULT_DIR = Path(
    "/scratch/aniluchavez/ConvoDATAS/SemanticGLM/"
    "bert-base-causal_ctx200spktag_tshift-150_tlen500_oshift+200_olen500_"
    "notebookexact_shuf_xcirc_r2only/pc50"
)
PLOT_DIR = Path("/scratch/aniluchavez/ConvoDATAS/SemanticGLM/plots")
FIG_DIR = Path("/scratch/aniluchavez/hippocampal-speaker-semantics/figures")
OUT_PREFIX = PLOT_DIR / "bert_causal_spktag_tshiftm150_otherp200_layer_llh"


def load_results() -> pd.DataFrame:
    frames = []
    missing_layers = []
    for layer in range(13):
        paths = sorted(RESULT_DIR.glob(f"*_L{layer:02d}_sem.pkl"))
        paths = [path for path in paths if not path.name.startswith("L")]
        if len(paths) != 15:
            missing_layers.append((layer, len(paths)))
        for path in paths:
            with path.open("rb") as handle:
                obj = pickle.load(handle)
            frame = obj["df"] if isinstance(obj, dict) else obj
            frame = frame.copy()
            frame["ll_diff"] = frame["ll_real"] - frame["ll_null"]
            frames.append(frame)
    if missing_layers:
        raise FileNotFoundError(
            "Expected 15 patient files per layer, got: "
            + ", ".join(f"L{layer:02d}={count}" for layer, count in missing_layers)
        )
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_results()
    per_neuron = df[
        [
            "patient",
            "region",
            "condition",
            "neuron_idx",
            "layer",
            "ll_real",
            "ll_null",
            "ll_diff",
            "r2",
            "n_spikes",
        ]
    ]
    patient = (
        per_neuron.groupby(["patient", "region", "condition", "layer"], as_index=False)
        .agg(
            patient_mean_ll_diff=("ll_diff", "mean"),
            patient_median_ll_diff=("ll_diff", "median"),
            patient_mean_r2=("r2", "mean"),
            patient_median_r2=("r2", "median"),
            n_neurons=("neuron_idx", "count"),
        )
    )
    summary = (
        patient.groupby(["condition", "layer"], as_index=False)
        .agg(
            mean_patient_mean_ll_diff=("patient_mean_ll_diff", "mean"),
            sem_patient_mean_ll_diff=("patient_mean_ll_diff", lambda x: x.sem()),
            mean_patient_median_ll_diff=("patient_median_ll_diff", "mean"),
            sem_patient_median_ll_diff=("patient_median_ll_diff", lambda x: x.sem()),
            n_patients=("patient", "nunique"),
        )
    )

    colors = {"other": "#0000ff", "self": "#ff0000"}
    labels = {"other": "listening", "self": "speaking"}
    fig, ax = plt.subplots(figsize=(4.05, 4.05))
    for condition in ["other", "self"]:
        sub = summary.loc[summary["condition"].eq(condition)].sort_values("layer")
        x = sub["layer"].to_numpy()
        y = sub["mean_patient_mean_ll_diff"].to_numpy()
        sem = sub["sem_patient_mean_ll_diff"].fillna(0).to_numpy()
        ax.errorbar(
            x,
            y,
            yerr=sem,
            color=colors[condition],
            linewidth=3.0,
            marker="o",
            markersize=5.5,
            markerfacecolor=colors[condition],
            markeredgecolor=colors[condition],
            capsize=0,
            elinewidth=2.2,
            label=labels[condition],
        )

    ax.set_xlabel("BERT layer", fontsize=15)
    ax.set_ylabel("model improvement (LLH difference)", fontsize=13)
    ax.set_xlim(-0.55, 12.55)
    ax.set_xticks(np.arange(0, 13, 2))
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    ax.axhline(0, color="black", linewidth=0.8)
    ymin = float(
        (summary["mean_patient_mean_ll_diff"] - summary["sem_patient_mean_ll_diff"].fillna(0)).min()
    )
    ymax = float(
        (summary["mean_patient_mean_ll_diff"] + summary["sem_patient_mean_ll_diff"].fillna(0)).max()
    )
    pad = 0.06 * (ymax - ymin if ymax > ymin else 1.0)
    ax.set_ylim(ymin - pad, ymax + pad)

    handles = [
        Line2D([0], [0], color=colors["other"], linewidth=5, marker=r"$\sim$",
               markersize=22, markeredgewidth=0, label="listening"),
        Line2D([0], [0], color=colors["self"], linewidth=5, marker=r"$\sim$",
               markersize=22, markeredgewidth=0, label="speaking"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=11,
              handlelength=2.1, borderaxespad=0.2)
    ax.text(
        0.98,
        0.07,
        f"n={patient['patient'].nunique()} patients, "
        f"{per_neuron.loc[per_neuron['layer'].eq(0)].shape[0]:,} neurons",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
    )

    fig.tight_layout()
    for suffix in [".png", ".pdf", ".svg"]:
        fig.savefig(OUT_PREFIX.with_suffix(suffix), dpi=300, bbox_inches="tight")
        fig.savefig((FIG_DIR / OUT_PREFIX.name).with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)

    per_neuron.to_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_per_neuron.csv"), index=False)
    patient.to_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_patient.csv"), index=False)
    summary.to_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_summary.csv"), index=False)
    print(OUT_PREFIX)
    print(summary.sort_values(["condition", "mean_patient_mean_ll_diff"]).tail().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
