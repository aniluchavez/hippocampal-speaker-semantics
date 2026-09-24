#!/usr/bin/env python3
"""
Bracket violin plot for self/other reliability, ALL hippocampus neurons
(no significance/VP filtering), using BERT speaker-tag embeddings
(bert-base-causal, ctx200spktag, bestfixed window, pc50, L12).

Output: figures/11_brackets_bert_spktag_allneurons_L12.pdf

No variance-partitioning run exists yet for the speaker-tag embeddings, so
unlike plot_rcross_bert_pc30_sigVP.py this does not filter to unique-semantic
significant neurons -- all neurons with reliability data are included.
"""

import pickle, glob
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
GLM_DIR  = Path("/scratch/aniluchavez/ConvoDATAS/SemanticGLM")
FIG_DIR  = PROJECT / "figures"
FIG_DIR.mkdir(exist_ok=True)

REL_DIR = GLM_DIR / ("bert-base-causal_ctx200spktag_bestfixed_selfm300_len500_otherp20_len500"
                     "_notebookexact_shuf_xcirc_r2only") / "pc50"

# ── load reliability ──────────────────────────────────────────────────────────
rows = []
for f in sorted(glob.glob(str(REL_DIR / "PTY*_L12_sem.pkl"))):
    patient = Path(f).name.replace("_L12_sem.pkl", "")
    obj = pickle.load(open(f, "rb"))
    for r in obj.get("reliability", {}).get("hippocampus", []):
        rows.append({
            "patient": patient,
            "neuron_idx": int(r["neuron"]),
            "r_cross": float(r["r_cross"]),
            "self_reliability_mean": float(r["self_reliability_mean"]),
            "other_reliability_mean": float(r["other_reliability_mean"]),
            "lower_ceiling": float(min(r["self_reliability_mean"],
                                       r["other_reliability_mean"])),
        })
df_sig = pd.DataFrame(rows)

print(f"r_cross rows (all neurons, no sig filter): {len(df_sig)}")

# ── stats helpers ─────────────────────────────────────────────────────────────
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

n_neurons  = len(df_sig)
n_patients = df_sig["patient"].nunique()
median_cross = df_sig["r_cross"].median()
median_ceil  = df_sig["lower_ceiling"].median()
pct_pos      = 100 * (df_sig["r_cross"] > 0).mean()

print(f"\nn={n_neurons} neurons ({n_patients} patients)")
print(f"median r_cross={median_cross:.3f}, ceiling={median_ceil:.3f}")
print(f"% positive={pct_pos:.1f}%")
print(f"p(r_cross>0)={p_cross_zero:.2e}")
print(f"p(ceiling>r_cross)={p_cross_ceil:.2e}")

# ── figure ────────────────────────────────────────────────────────────────────
METRICS = ["speaking", "listening", "cross-condition"]
COLORS  = {
    "speaking":       "#ff2b2b",
    "listening":      "#1f1fd0",
    "cross-condition": "#e619cc",
}

wide = df_sig[["patient", "neuron_idx",
               "self_reliability_mean", "other_reliability_mean",
               "r_cross", "lower_ceiling"]].copy()
wide.columns = ["patient", "neuron_idx",
                "speaking", "listening",
                "cross-condition", "lower_ceiling"]

XPOS = {"speaking": 0.0, "listening": 1.6, "cross-condition": 3.2}

long = wide.melt(
    id_vars=["patient", "neuron_idx", "lower_ceiling"],
    value_vars=METRICS, var_name="metric", value_name="correlation",
).dropna(subset=["correlation"])
long["x_pos"] = long["metric"].map(XPOS)

# Target Illustrator placement: W 3.1191in x H 1.7218in (aspect ratio 1.8114:1),
# matching figures 13/14 (other-other). Render at 2.5x that size (same exact
# ratio) so uniform scale-to-fit in Illustrator introduces no distortion.
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
    vals = wide[metric].dropna().values
    med = float(np.median(vals))
    ax.plot([xi - 0.18, xi + 0.18], [med, med],
            color="black", linewidth=2, solid_capstyle="butt", zorder=4)

bx = XPOS["cross-condition"] + 0.65
ax.plot([bx, bx], [0, median_ceil], color="black", linewidth=1.0, clip_on=False)
for y in [0, median_cross, median_ceil]:
    ax.plot([bx - 0.08, bx], [y, y], color="black", linewidth=1.0, clip_on=False)
ax.text(bx + 0.06, median_cross / 2, p_label(p_cross_zero), va="center", fontsize=7)
ax.text(bx + 0.06, (median_cross + median_ceil) / 2, p_label(p_cross_ceil), va="center", fontsize=7)

ax.axhline(0, color="0.82", linewidth=5, zorder=0)
ax.set_ylim(-0.4, 1.1)
ax.set_xticks([XPOS[m] for m in METRICS])
ax.set_xticklabels(METRICS, fontsize=8)
ax.set_xlabel("")
ax.set_ylabel("subspace correlation (r)", fontsize=8)
ax.tick_params(axis="y", labelsize=7)
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.text(0.02, 0.02, f"all patients, {n_neurons} neurons",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=5.5)
ax.spines[["top", "right"]].set_visible(False)

out_pdf = FIG_DIR / "11_brackets_bert_spktag_allneurons_L12.pdf"
out_png = FIG_DIR / "11_brackets_bert_spktag_allneurons_L12.png"
out_svg = FIG_DIR / "11_brackets_bert_spktag_allneurons_L12.svg"
fig.savefig(out_pdf)
fig.savefig(out_png, dpi=220)
fig.savefig(out_svg)
print(f"\nSaved: {out_png}")
