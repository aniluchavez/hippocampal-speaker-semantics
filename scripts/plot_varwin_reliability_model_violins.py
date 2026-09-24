#!/usr/bin/env python3
"""Four-model hippocampal reliability violin subplots."""

from __future__ import annotations

import argparse
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

RUNS = [
    ("BERT", "bert-base_ctx200", 12),
    ("BERT + speaker tag", "bert-base-causal_ctx200spktag", 12),
    ("Llama 3.1 8B", "llama-3.1-8b_ctx200", 14),
    ("GPT-2 XL", "gpt2-xl_ctx200", 45),
]
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
parser = argparse.ArgumentParser()
parser.add_argument(
    "--tag", default="varwin_selfm300tooffset0_other0tooffsetp100"
)
parser.add_argument(
    "--window_label",
    default="self: onset−300 ms → offset; other: onset → offset+100 ms",
)
parser.add_argument(
    "--output_stem",
    default="varwin_reliability_models_hippocampus_violin_subplots",
)
args = parser.parse_args()
suffix = f"_{args.tag}_notebookexact_shuf_xcirc_r2only/pc50"

for model_label, folder_prefix, layer in RUNS:
    folder = ROOT / f"{folder_prefix}{suffix}"
    files = sorted(folder.glob(f"PTY*_L{layer:02d}_sem.pkl"))
    for path in files:
        patient = path.name.split(f"_L{layer:02d}")[0]
        with path.open("rb") as handle:
            obj = pickle.load(handle)
        reliability = obj.get("reliability", {}).get("hippocampus", [])
        for result in reliability:
            speaking = float(result.get("self_reliability_mean", np.nan))
            listening = float(result.get("other_reliability_mean", np.nan))
            cross = float(result.get("r_cross", np.nan))
            rows.append(
                {
                    "model": model_label,
                    "patient": patient,
                    "neuron": int(result.get("neuron", len(rows))),
                    "Speaking": speaking,
                    "Listening": listening,
                    "Cross-condition": cross,
                    "bracket_ceiling": np.nanmin([speaking, listening]),
                }
            )

wide = pd.DataFrame(rows)
if wide.empty:
    raise SystemExit("No reliability results found")

long = wide.melt(
    id_vars=["model", "patient", "neuron", "bracket_ceiling"],
    value_vars=METRICS,
    var_name="metric",
    value_name="correlation",
).dropna(subset=["correlation"])

stats_rows = []
for model_label, group in wide.groupby("model", sort=False):
    patient = (
        group.groupby("patient", as_index=False)
        .agg(
            cross=("Cross-condition", "median"),
            ceiling=("bracket_ceiling", "median"),
        )
    )
    p_cross_zero = one_sided_wilcoxon(patient["cross"].to_numpy())
    p_ceiling_cross = one_sided_wilcoxon(
        patient["ceiling"].to_numpy(), patient["cross"].to_numpy()
    )
    stats_rows.append(
        {
            "model": model_label,
            "n_patients": patient["patient"].nunique(),
            "n_neurons": len(group),
            "median_speaking": group["Speaking"].median(),
            "median_listening": group["Listening"].median(),
            "median_cross": group["Cross-condition"].median(),
            "median_bracket_ceiling": group["bracket_ceiling"].median(),
            "p_cross_greater_zero_patient_wilcoxon": p_cross_zero,
            "p_ceiling_greater_cross_patient_wilcoxon": p_ceiling_cross,
        }
    )
stats_df = pd.DataFrame(stats_rows)

sns.set_theme(style="white", context="talk")
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True, constrained_layout=True)
rng = np.random.default_rng(42)

for ax, (model_label, _, _) in zip(axes.flat, RUNS):
    model_long = long.loc[long["model"].eq(model_label)]
    model_wide = wide.loc[wide["model"].eq(model_label)]
    model_stats = stats_df.loc[stats_df["model"].eq(model_label)].iloc[0]

    sns.violinplot(
        data=model_long,
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
        values = model_wide[metric].dropna().to_numpy()
        keep = (
            rng.choice(len(values), size=min(220, len(values)), replace=False)
            if len(values) > 220
            else np.arange(len(values))
        )
        jitter = rng.uniform(-0.16, 0.16, size=len(keep))
        ax.scatter(
            x_position + jitter,
            values[keep],
            s=8,
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

    cross_median = float(model_stats["median_cross"])
    ceiling_median = float(model_stats["median_bracket_ceiling"])
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
        p_label(model_stats["p_cross_greater_zero_patient_wilcoxon"]),
        va="center",
        fontsize=10,
    )
    ax.text(
        bracket_x + 0.05,
        (cross_median + ceiling_median) / 2,
        p_label(model_stats["p_ceiling_greater_cross_patient_wilcoxon"]),
        va="center",
        fontsize=10,
    )

    ax.axhline(0, color="0.82", linewidth=7, zorder=0)
    ax.set_title(
        f"{model_label}, n={int(model_stats['n_neurons'])} neurons",
        fontweight="bold",
    )
    ax.set_xlabel("")
    ax.set_ylabel("correlation (r)")
    ax.set_ylim(-0.35, 1.03)
    ax.tick_params(axis="x", labelrotation=8, labelsize=11)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle(
    "Hippocampal beta reliability across semantic models\n"
    f"{args.window_label}\n"
    "bracket ceiling = min(speaking, listening) per neuron; "
    "p-values use paired patient-level Wilcoxon tests",
    fontsize=15,
    fontweight="bold",
)

figure_dir = PROJECT / "figures"
result_dir = PROJECT / "results"
figure_dir.mkdir(parents=True, exist_ok=True)
result_dir.mkdir(parents=True, exist_ok=True)
png = figure_dir / f"{args.output_stem}.png"
pdf = figure_dir / f"{args.output_stem}.pdf"
svg = figure_dir / f"{args.output_stem}.svg"
data_csv = result_dir / f"{args.output_stem}_neurons.csv"
stats_csv = result_dir / f"{args.output_stem}_stats.csv"
fig.savefig(png, dpi=220, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
fig.savefig(svg, bbox_inches="tight")
wide.to_csv(data_csv, index=False)
stats_df.to_csv(stats_csv, index=False)
print(png)
print(pdf)
print(svg)
print(data_csv)
print(stats_csv)
