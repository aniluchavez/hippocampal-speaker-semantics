#!/usr/bin/env python3
"""Summarize held-out LL improvement for bestfixed BERT layers."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("/scratch/aniluchavez/ConvoDATAS/SemanticGLM")
DEFAULT_TAG = "bestfixed_selfm300_len500_otherp20_len500"
DEFAULT_MODEL_DIR = (
    "bert-base_ctx200_"
    f"{DEFAULT_TAG}_notebookexact_shuf_xcirc_r2only"
    "/pc50"
)
DEFAULT_OUT = Path(
    "/scratch/aniluchavez/ConvoDATAS/SemanticGLM/plots/"
    "bestfixed_bert_layer_ll_diff"
)


def load_layers(model_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(model_dir.glob("L*_all.pkl")):
        with path.open("rb") as handle:
            obj = pickle.load(handle)
        frame = obj["df"] if isinstance(obj, dict) else obj
        frame = frame.copy()
        missing = {"ll_real", "ll_null", "layer"} - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frame["ll_diff"] = frame["ll_real"] - frame["ll_null"]
        rows.append(frame)
    if not rows:
        raise FileNotFoundError(f"No L*_all.pkl files found in {model_dir}")
    return pd.concat(rows, ignore_index=True)


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    patient = (
        df.groupby(["patient", "region", "condition", "layer"], as_index=False)
        .agg(
            patient_median_ll_diff=("ll_diff", "median"),
            patient_mean_ll_diff=("ll_diff", "mean"),
            patient_median_r2=("r2", "median"),
            n_neurons=("neuron_idx", "count"),
        )
    )
    summary = (
        patient.groupby(["region", "condition", "layer"], as_index=False)
        .agg(
            mean_patient_median_ll_diff=("patient_median_ll_diff", "mean"),
            sem_patient_median_ll_diff=(
                "patient_median_ll_diff",
                lambda x: x.sem(),
            ),
            median_patient_median_ll_diff=("patient_median_ll_diff", "median"),
            mean_patient_median_r2=("patient_median_r2", "mean"),
            median_patient_median_r2=("patient_median_r2", "median"),
            n_patients=("patient", "nunique"),
        )
    )
    return patient, summary


def plot_summary(
    summary: pd.DataFrame,
    detailed: pd.DataFrame,
    output_prefix: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {"other": "#0000ff", "self": "#ff0000"}
    labels = {"other": "listening", "self": "speaking"}

    fig, ax = plt.subplots(figsize=(4.05, 4.05))
    for condition in ["other", "self"]:
        sub = summary.loc[summary["condition"].eq(condition)].sort_values("layer")
        x = sub["layer"].to_numpy()
        y = sub["mean_patient_median_ll_diff"].to_numpy()
        sem = np.nan_to_num(sub["sem_patient_median_ll_diff"].to_numpy(), nan=0.0)
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
        (
            summary["mean_patient_median_ll_diff"]
            - summary["sem_patient_median_ll_diff"].fillna(0)
        ).min()
    )
    ymax = float(
        (
            summary["mean_patient_median_ll_diff"]
            + summary["sem_patient_median_ll_diff"].fillna(0)
        ).max()
    )
    pad = 0.06 * (ymax - ymin if ymax > ymin else 1.0)
    ax.set_ylim(ymin - pad, ymax + pad)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=colors["other"],
            linewidth=5,
            marker=r"$\sim$",
            markersize=22,
            markeredgewidth=0,
            label="listening",
        ),
        Line2D(
            [0],
            [0],
            color=colors["self"],
            linewidth=5,
            marker=r"$\sim$",
            markersize=22,
            markeredgewidth=0,
            label="speaking",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper left",
        fontsize=11,
        handlelength=2.1,
        borderaxespad=0.2,
    )

    n_patients = detailed["patient"].nunique()
    n_neurons = (
        detailed.loc[detailed["layer"].eq(detailed["layer"].min())]
        .groupby(["patient", "region", "condition", "neuron_idx"])
        .ngroups
    )
    ax.text(
        0.98,
        0.07,
        f"n={n_patients} patients, {n_neurons:,} neurons",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
    )

    right = ax.twinx()
    right.set_ylim(ax.get_ylim())
    right.set_ylabel("likelihood (LLH)", fontsize=13)
    right.tick_params(axis="y", which="major", labelsize=12)
    right.spines["top"].set_visible(False)

    fig.tight_layout()
    for suffix in [".png", ".pdf", ".svg"]:
        fig.savefig(output_prefix.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def scaled_patient_summary(patient: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, cond_df in patient.groupby("condition"):
        piv = cond_df.pivot_table(
            index=["patient", "region"],
            columns="layer",
            values="patient_median_ll_diff",
        )
        if 12 not in piv.columns:
            continue
        denom = piv[12].replace(0, np.nan)
        norm = piv.div(denom, axis=0).clip(lower=0)
        l12_raw_mean = cond_df.loc[
            cond_df["layer"].eq(12), "patient_median_ll_diff"
        ].mean()
        scaled = norm * l12_raw_mean
        long = scaled.reset_index().melt(
            id_vars=["patient", "region"],
            var_name="layer",
            value_name="scaled_ll_diff",
        )
        long["condition"] = condition
        rows.append(long)
    return pd.concat(rows, ignore_index=True)


def plot_scaled_summary(
    patient: pd.DataFrame,
    detailed: pd.DataFrame,
    output_prefix: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    scaled = scaled_patient_summary(patient)
    summary = (
        scaled.groupby(["condition", "layer"], as_index=False)
        .agg(
            mean_scaled_ll_diff=("scaled_ll_diff", "mean"),
            sem_scaled_ll_diff=("scaled_ll_diff", lambda x: x.sem()),
        )
    )
    colors = {"other": "#0000ff", "self": "#ff0000"}
    fig, ax = plt.subplots(figsize=(4.05, 4.05))
    for condition in ["other", "self"]:
        sub = summary.loc[summary["condition"].eq(condition)].sort_values("layer")
        x = sub["layer"].to_numpy(dtype=float)
        y = sub["mean_scaled_ll_diff"].to_numpy(dtype=float)
        sem = np.nan_to_num(sub["sem_scaled_ll_diff"].to_numpy(dtype=float), nan=0.0)
        lower = np.minimum(sem, y)
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([lower, sem]),
            color=colors[condition],
            linewidth=3.0,
            marker="o",
            markersize=5.5,
            markerfacecolor=colors[condition],
            markeredgecolor=colors[condition],
            capsize=0,
            elinewidth=2.0,
        )

    ax.set_xlabel("BERT layer", fontsize=15)
    ax.set_ylabel("model improvement (LLH difference)", fontsize=13)
    ax.set_xlim(-0.55, 12.55)
    ax.set_xticks(np.arange(0, 13, 2))
    ax.set_ylim(0, float((summary["mean_scaled_ll_diff"] + summary["sem_scaled_ll_diff"].fillna(0)).max()) + 1.5)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    ax.axhline(0, color="black", linewidth=0.8)

    legend_handles = [
        Line2D([0], [0], color=colors["other"], linewidth=5, marker=r"$\sim$", markersize=22, markeredgewidth=0, label="listening"),
        Line2D([0], [0], color=colors["self"], linewidth=5, marker=r"$\sim$", markersize=22, markeredgewidth=0, label="speaking"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper left", fontsize=11, handlelength=2.1, borderaxespad=0.2)

    n_patients = detailed["patient"].nunique()
    n_neurons = (
        detailed.loc[detailed["layer"].eq(detailed["layer"].min())]
        .groupby(["patient", "region", "condition", "neuron_idx"])
        .ngroups
    )
    ax.text(0.98, 0.07, f"n={n_patients} patients, {n_neurons:,} neurons", transform=ax.transAxes, ha="right", va="bottom", fontsize=11)

    right = ax.twinx()
    right.set_ylim(ax.get_ylim())
    right.set_ylabel("likelihood (LLH)", fontsize=13)
    right.tick_params(axis="y", which="major", labelsize=12)
    right.spines["top"].set_visible(False)

    fig.tight_layout()
    for suffix in [".png", ".pdf", ".svg"]:
        fig.savefig(output_prefix.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)
    scaled.to_csv(output_prefix.with_name(output_prefix.name + "_patient_scaled.csv"), index=False)
    summary.to_csv(output_prefix.with_name(output_prefix.name + "_summary_scaled.csv"), index=False)


def normalized_patient_summary(patient: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, cond_df in patient.groupby("condition"):
        piv = cond_df.pivot_table(
            index=["patient", "region"],
            columns="layer",
            values="patient_median_ll_diff",
        )
        # Normalize each patient/condition by that patient's strongest layer.
        denom = piv.clip(lower=0).max(axis=1).replace(0, np.nan)
        norm = piv.clip(lower=0).div(denom, axis=0)
        long = norm.reset_index().melt(
            id_vars=["patient", "region"],
            var_name="layer",
            value_name="normalized_ll_diff",
        )
        long["condition"] = condition
        rows.append(long)
    return pd.concat(rows, ignore_index=True).dropna(subset=["normalized_ll_diff"])


def plot_normalized_summary(
    patient: pd.DataFrame,
    detailed: pd.DataFrame,
    output_prefix: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    normalized = normalized_patient_summary(patient)
    summary = (
        normalized.groupby(["condition", "layer"], as_index=False)
        .agg(
            mean_normalized_ll_diff=("normalized_ll_diff", "mean"),
            sem_normalized_ll_diff=("normalized_ll_diff", lambda x: x.sem()),
            n_patients=("patient", "nunique"),
        )
    )
    colors = {"other": "#0000ff", "self": "#ff0000"}
    fig, ax = plt.subplots(figsize=(4.05, 4.05))
    for condition in ["other", "self"]:
        sub = summary.loc[summary["condition"].eq(condition)].sort_values("layer")
        x = sub["layer"].to_numpy(dtype=float)
        y = sub["mean_normalized_ll_diff"].to_numpy(dtype=float)
        sem = np.nan_to_num(sub["sem_normalized_ll_diff"].to_numpy(dtype=float), nan=0.0)
        lower = np.minimum(sem, y)
        upper = np.minimum(sem, 1.0 - y)
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([lower, upper]),
            color=colors[condition],
            linewidth=3.0,
            marker="o",
            markersize=5.5,
            markerfacecolor=colors[condition],
            markeredgecolor=colors[condition],
            capsize=0,
            elinewidth=2.0,
        )

    ax.set_xlabel("BERT layer", fontsize=15)
    ax.set_ylabel("normalized LLH improvement", fontsize=13)
    ax.set_xlim(-0.55, 12.55)
    ax.set_xticks(np.arange(0, 13, 2))
    ax.set_ylim(0, 1.08)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    ax.axhline(0, color="black", linewidth=0.8)

    legend_handles = [
        Line2D([0], [0], color=colors["other"], linewidth=5, marker=r"$\sim$", markersize=22, markeredgewidth=0, label="listening"),
        Line2D([0], [0], color=colors["self"], linewidth=5, marker=r"$\sim$", markersize=22, markeredgewidth=0, label="speaking"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper left", fontsize=11, handlelength=2.1, borderaxespad=0.2)

    n_patients = normalized["patient"].nunique()
    n_neurons = (
        detailed.loc[detailed["layer"].eq(detailed["layer"].min())]
        .groupby(["patient", "region", "condition", "neuron_idx"])
        .ngroups
    )
    ax.text(
        0.98,
        0.07,
        f"n={n_patients} patients, {n_neurons:,} neurons",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
    )

    fig.tight_layout()
    for suffix in [".png", ".pdf", ".svg"]:
        fig.savefig(output_prefix.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)
    normalized.to_csv(output_prefix.with_name(output_prefix.name + "_patient_normalized.csv"), index=False)
    summary.to_csv(output_prefix.with_name(output_prefix.name + "_summary_normalized.csv"), index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    model_dir = args.model_dir or (args.root / DEFAULT_MODEL_DIR)
    out = args.output_prefix
    out.parent.mkdir(parents=True, exist_ok=True)

    detailed = load_layers(model_dir)
    patient, summary = summarize(detailed)

    detailed[
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
    ].to_csv(out.with_name(out.name + "_per_neuron.csv"), index=False)
    patient.to_csv(out.with_name(out.name + "_patient.csv"), index=False)
    summary.to_csv(out.with_name(out.name + "_summary.csv"), index=False)
    plot_summary(summary, detailed, out)
    plot_scaled_summary(patient, detailed, out.with_name(out.name + "_within_patient_scaled"))
    plot_normalized_summary(patient, detailed, out.with_name(out.name + "_within_patient_normalized"))

    print(f"Read: {model_dir}")
    print(f"Wrote: {out.parent}")
    for condition in ["self", "other"]:
        sub = summary.loc[summary["condition"].eq(condition)]
        best = sub.loc[sub["mean_patient_median_ll_diff"].idxmax()]
        print(
            f"{condition}: best layer {int(best['layer'])}, "
            f"mean patient-median ll_diff "
            f"{best['mean_patient_median_ll_diff']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
