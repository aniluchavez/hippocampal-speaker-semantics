#!/usr/bin/env python3
"""
Other-other (external speaker vs external speaker) reliability, hippocampus,
BERT bidirectional-max-context embeddings WITH speaker tags, PC10, patient
PTYEZ excluded. Apples-to-apples spktag counterpart of figure 13 (same
attention type, window, pc, and layer -- only the tag differs).
Bracket-violin format matched to the self/other reliability figures
(plot_rcross_bert_pc30_sigVP.py / plot_rcross_bert_spktag_allneurons.py).

Source data: SemanticGLM/other_other_rcross_reliability_bert_spktag_pc10/
             */*/hippocampus_other_other_reliability_detail.csv

Output: figures/14_other_other_exclude_ptyez_pc10_reliability_violin_spktag_matched_format.pdf
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

PROJECT  = Path("/scratch/aniluchavez/hippocampal-speaker-semantics")
DATA_DIR = Path("/scratch/aniluchavez/ConvoDATAS/SemanticGLM/other_other_rcross_reliability_bert_spktag_pc10")
FIG_DIR  = PROJECT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── build wide-format per-neuron data directly from the detail CSVs ───────────
rows = []
for f in sorted(DATA_DIR.glob("*/*/hippocampus_other_other_reliability_detail.csv")):
    patient = f.parts[-2]
    pair = f.parts[-3]
    df = pd.read_csv(
        f, usecols=["neuron", "r_cross", "self_reliability_mean", "other_reliability_mean"]
    )
    df["patient"] = patient
    df["pair"] = pair
    rows.append(df)
wide_raw = pd.concat(rows, ignore_index=True)
wide_raw = wide_raw.loc[~wide_raw["patient"].str.startswith("PTYEZ")]

wide_raw = wide_raw.rename(
    columns={
        "self_reliability_mean": "External speaker A",
        "other_reliability_mean": "External speaker B",
    }
)
wide_raw["lower_ceiling"] = wide_raw[["External speaker A", "External speaker B"]].min(axis=1)
wide_raw["Cross-speaker"] = wide_raw["r_cross"]

df_sig = wide_raw

n_neurons  = len(df_sig)
n_patients = df_sig["patient"].nunique()
print(f"n={n_neurons} neurons ({n_patients} patients)")

# ── stats helpers (same as self/other bracket scripts) ────────────────────────
def p_label(p):
    if not np.isfinite(p): return "p=n/a"
    if p < 0.0001: return "p<0.0001"
    if p < 0.001:  return f"p={p:.4f}"
    return f"p={p:.3f}"

def one_sided_wilcoxon(x, y=None):
    x = np.asarray(x, float)
    if y is None:
        diff = x[np.isfinite(x)]
    else:
        y = np.asarray(y, float)
        ok = np.isfinite(x) & np.isfinite(y)
        diff = (x - y)[ok]
    if len(diff) < 2 or np.allclose(diff, 0): return np.nan
    return float(stats.wilcoxon(diff, alternative="greater",
                                zero_method="wilcox", method="auto").pvalue)

patient_stats = (
    df_sig.groupby("patient", as_index=False)
    .agg(cross=("r_cross", "median"), ceiling=("lower_ceiling", "median"))
)
p_cross_zero = one_sided_wilcoxon(patient_stats["cross"].values)
p_cross_ceil = one_sided_wilcoxon(patient_stats["ceiling"].values,
                                   patient_stats["cross"].values)

median_cross = df_sig["r_cross"].median()
median_ceil  = df_sig["lower_ceiling"].median()

print(f"median r_cross={median_cross:.3f}, ceiling={median_ceil:.3f}")
print(f"p(r_cross>0)={p_cross_zero:.2e}")
print(f"p(ceiling>r_cross)={p_cross_ceil:.2e}")

# ── figure ────────────────────────────────────────────────────────────────────
METRICS = ["External speaker A", "External speaker B", "Cross-speaker"]
COLORS  = {
    "External speaker A": "#2a9d8f",
    "External speaker B": "#e2795c",
    "Cross-speaker":      "#7b3fa0",
}

long = df_sig.melt(
    id_vars=["patient", "neuron", "lower_ceiling"],
    value_vars=METRICS, var_name="metric", value_name="correlation",
).dropna(subset=["correlation"])

XPOS = {"External speaker A": 0.0, "External speaker B": 1.6, "Cross-speaker": 3.2}
long["x_pos"] = long["metric"].map(XPOS)

# Target Illustrator placement: W 3.1191in x H 1.7218in (aspect ratio 1.8114:1).
# Render at 2.5x that size (same exact ratio) so uniform scale-to-fit in
# Illustrator introduces no distortion.
TARGET_W, TARGET_H = 3.1191, 1.7218
SCALE = 2.5

sns.set_theme(style="white", context="paper")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
fig, ax = plt.subplots(figsize=(TARGET_W * SCALE, TARGET_H * SCALE), constrained_layout=True)

sns.violinplot(
    data=long, x="x_pos", y="correlation", native_scale=True,
    hue="metric", hue_order=METRICS,
    palette=COLORS, inner=None, cut=2, linewidth=0.8, legend=False, ax=ax,
    width=0.7, dodge=False,
)
for collection in ax.collections:
    collection.set_edgecolor("black")

for metric in METRICS:
    xi = XPOS[metric]
    vals = long.loc[long["metric"] == metric, "correlation"].values
    med = float(np.median(vals))
    ax.plot([xi - 0.18, xi + 0.18], [med, med],
            color="black", linewidth=2, solid_capstyle="butt", zorder=4)

bx = XPOS["Cross-speaker"] + 0.65
ax.plot([bx, bx], [0, median_ceil], color="black", linewidth=1.0, clip_on=False)
for y in [0, median_cross, median_ceil]:
    ax.plot([bx - 0.08, bx], [y, y], color="black", linewidth=1.0, clip_on=False)
ax.text(bx + 0.06, median_cross / 2, p_label(p_cross_zero), va="center", fontsize=7)
ax.text(bx + 0.06, (median_cross + median_ceil) / 2, p_label(p_cross_ceil), va="center", fontsize=7)

ax.axhline(0, color="0.82", linewidth=5, zorder=0)
ax.set_xticks([XPOS[m] for m in METRICS])
ax.set_xticklabels(METRICS, fontsize=8)
ax.set_xlabel("")
ax.set_ylabel("correlation (r)", fontsize=8)
ax.tick_params(axis="y", labelsize=7)
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.text(0.02, 0.02, f"other-other, {n_neurons} neurons, {n_patients} patients (PTYEZ excluded)",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=5.5)
ax.set_title(
    "Hippocampal beta reliability between external speakers, BERT bidirectional max context + speaker tag, PC10, PTYEZ excluded",
    fontsize=6.5, fontweight="bold",
)
ax.spines[["top", "right"]].set_visible(False)

out_pdf = FIG_DIR / "14_other_other_exclude_ptyez_pc10_reliability_violin_spktag_matched_format.pdf"
out_png = FIG_DIR / "14_other_other_exclude_ptyez_pc10_reliability_violin_spktag_matched_format.png"
out_svg = FIG_DIR / "14_other_other_exclude_ptyez_pc10_reliability_violin_spktag_matched_format.svg"
fig.savefig(out_pdf)
fig.savefig(out_png, dpi=220)
fig.savefig(out_svg)
print(f"\nSaved: {out_png}")
