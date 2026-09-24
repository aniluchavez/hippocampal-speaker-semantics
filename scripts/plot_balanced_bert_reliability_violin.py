#!/usr/bin/env python3
"""Balanced-trial BERT hippocampal reliability violin plot."""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


ROOT = Path("/scratch/aniluchavez/ConvoDATAS/SemanticGLM")
PROJECT = Path("/scratch/aniluchavez/hippocampal-speaker-semantics")

FOLDER = (
    ROOT
    / "bert-base_ctx200_fixed_selfm200_otherp200_len500_notebookexact_shuf_xcirc_r2only_relbal"
    / "pc30"
)
LAYER = 12
MODEL_LABEL = "BERT balanced"
METRICS = ["Speaking", "Listening", "Cross-condition"]
COLORS = {
    "Speaking": "#c83e3e",
    "Listening": "#3656b3",
    "Cross-condition": "#b449b5",
}


def p_label(p: float) -> str:
    if not np.isfinite(p):
        return "p=n/a"
    if p < 0.0001:
        return "p<0.0001"
    if p < 0.001:
        return f"p={p:.4f}"
    return f"p={p:.3f}"


def one_sided_wilcoxon(x: np.ndarray, y: np.ndarray | None = None) -> float:
    x = np.asarray(x, dtype=float)
    if y is None:
        valid = np.isfinite(x)
        difference = x[valid]
    else:
        y = np.asarray(y, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        difference = x[valid] - y[valid]
    if len(difference) < 2 or np.allclose(difference, 0):
        return np.nan
    return float(
        stats.wilcoxon(
            difference,
            alternative="greater",
            zero_method="wilcox",
            method="auto",
        ).pvalue
    )


rows = []
for path in sorted(FOLDER.glob(f"PTY*_L{LAYER:02d}_sem.pkl")):
    patient = path.name.split(f"_L{LAYER:02d}")[0]
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    reliability = obj.get("reliability", {}).get("hippocampus", [])
    meta = obj.get("reliability_meta", {}).get("hippocampus", {})
    for result in reliability:
        speaking = float(result.get("self_reliability_mean", np.nan))
        listening = float(result.get("other_reliability_mean", np.nan))
        cross = float(result.get("r_cross", np.nan))
        rows.append(
            {
                "model": MODEL_LABEL,
                "patient": patient,
                "neuron": int(result.get("neuron", len(rows))),
                "Speaking": speaking,
                "Listening": listening,
                "Cross-condition": cross,
                "bracket_ceiling": np.nanmin([speaking, listening]),
                "n_self_used": meta.get("n_self_used", np.nan),
                "n_other_used": meta.get("n_other_used", np.nan),
                "n_self_raw": meta.get("n_self_raw", np.nan),
                "n_other_raw": meta.get("n_other_raw", np.nan),
            }
        )

wide = pd.DataFrame(rows)
if wide.empty:
    raise SystemExit(f"No reliability results found in {FOLDER}")

long = wide.melt(
    id_vars=[
        "model",
        "patient",
        "neuron",
        "bracket_ceiling",
        "n_self_used",
        "n_other_used",
        "n_self_raw",
        "n_other_raw",
    ],
    value_vars=METRICS,
    var_name="metric",
    value_name="correlation",
).dropna(subset=["correlation"])

patient = (
    wide.groupby("patient", as_index=False)
    .agg(
        cross=("Cross-condition", "median"),
        ceiling=("bracket_ceiling", "median"),
        speaking=("Speaking", "median"),
        listening=("Listening", "median"),
        n_self_used=("n_self_used", "first"),
        n_other_used=("n_other_used", "first"),
        n_self_raw=("n_self_raw", "first"),
        n_other_raw=("n_other_raw", "first"),
    )
)
p_cross_zero = one_sided_wilcoxon(patient["cross"].to_numpy())
p_ceiling_cross = one_sided_wilcoxon(
    patient["ceiling"].to_numpy(), patient["cross"].to_numpy()
)
stats_df = pd.DataFrame(
    [
        {
            "model": MODEL_LABEL,
            "n_patients": patient["patient"].nunique(),
            "n_neurons": len(wide),
            "median_speaking": wide["Speaking"].median(),
            "median_listening": wide["Listening"].median(),
            "median_cross": wide["Cross-condition"].median(),
            "mean_cross": wide["Cross-condition"].mean(),
            "patient_mean_cross": wide.groupby("patient")["Cross-condition"].mean().mean(),
            "median_bracket_ceiling": wide["bracket_ceiling"].median(),
            "p_cross_greater_zero_patient_wilcoxon": p_cross_zero,
            "p_ceiling_greater_cross_patient_wilcoxon": p_ceiling_cross,
        }
    ]
)

sns.set_theme(style="white", context="talk")
fig, ax = plt.subplots(1, 1, figsize=(7.6, 6.4), constrained_layout=True)
rng = np.random.default_rng(42)

sns.violinplot(
    data=long,
    x="metric",
    y="correlation",
    order=METRICS,
    hue="metric",
    hue_order=METRICS,
    palette=COLORS,
    inner=None,
    cut=0,
    linewidth=1,
    legend=False,
    ax=ax,
)

for x_position, metric in enumerate(METRICS):
    values = wide[metric].dropna().to_numpy()
    keep = (
        rng.choice(len(values), size=min(260, len(values)), replace=False)
        if len(values) > 260
        else np.arange(len(values))
    )
    jitter = rng.uniform(-0.16, 0.16, size=len(keep))
    ax.scatter(
        x_position + jitter,
        values[keep],
        s=9,
        color="black",
        alpha=0.28,
        linewidths=0,
        zorder=3,
    )
    median = float(np.median(values))
    ax.plot(
        [x_position - 0.23, x_position + 0.23],
        [median, median],
        color="black",
        linewidth=4,
        solid_capstyle="butt",
        zorder=4,
    )

cross_median = float(stats_df.loc[0, "median_cross"])
ceiling_median = float(stats_df.loc[0, "median_bracket_ceiling"])
bracket_x = 2.48
ax.plot(
    [bracket_x, bracket_x],
    [0, ceiling_median],
    color="black",
    linewidth=1.5,
    clip_on=False,
)
for y in [0, cross_median, ceiling_median]:
    ax.plot(
        [bracket_x - 0.07, bracket_x],
        [y, y],
        color="black",
        linewidth=1.5,
        clip_on=False,
    )
ax.text(
    bracket_x + 0.05,
    (0 + cross_median) / 2,
    p_label(p_cross_zero),
    va="center",
    fontsize=10,
)
ax.text(
    bracket_x + 0.05,
    (cross_median + ceiling_median) / 2,
    p_label(p_ceiling_cross),
    va="center",
    fontsize=10,
)

ax.axhline(0, color="0.82", linewidth=7, zorder=0)
ax.set_title(
    f"{MODEL_LABEL}, n={len(wide)} neurons",
    fontweight="bold",
)
ax.set_xlabel("")
ax.set_ylabel("correlation (r)")
ax.set_ylim(-0.35, 1.03)
ax.tick_params(axis="x", labelrotation=8, labelsize=11)
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle(
    "Hippocampal beta reliability with balanced trial counts\n"
    "BERT L12, 30 PCs; fixed 500 ms windows: self onset−200 ms, other onset+200 ms\n"
    "bracket ceiling = min(speaking, listening) per neuron; "
    "p-values use paired patient-level Wilcoxon tests",
    fontsize=13,
    fontweight="bold",
)

figure_dir = PROJECT / "figures"
result_dir = PROJECT / "results"
figure_dir.mkdir(parents=True, exist_ok=True)
result_dir.mkdir(parents=True, exist_ok=True)
stem = "bert_balanced_fixed_pc30_reliability_hippocampus_violin"
png = figure_dir / f"{stem}.png"
pdf = figure_dir / f"{stem}.pdf"
svg = figure_dir / f"{stem}.svg"
data_csv = result_dir / f"{stem}_neurons.csv"
patient_csv = result_dir / f"{stem}_patients.csv"
stats_csv = result_dir / f"{stem}_stats.csv"
fig.savefig(png, dpi=220, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
fig.savefig(svg, bbox_inches="tight")
wide.to_csv(data_csv, index=False)
patient.to_csv(patient_csv, index=False)
stats_df.to_csv(stats_csv, index=False)
print(png)
print(pdf)
print(svg)
print(data_csv)
print(patient_csv)
print(stats_csv)
