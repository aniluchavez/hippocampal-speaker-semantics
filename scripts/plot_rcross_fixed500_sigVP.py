#!/usr/bin/env python3
"""
Bracket violin plot for self/other reliability, filtered to hippocampus neurons
significant (raw p<0.05, either self or other condition) in the fixed 500ms
window variance partitioning (self -200ms, other +100ms from onset).

Output: figures/09_brackets_fixed500_sigVP_L36.pdf
"""

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

PROJECT  = Path("/scratch/aniluchavez/hippocampal-speaker-semantics")
VPR_DIR  = Path("/scratch/aniluchavez/ConvoDATAS/VPResults")
GLM_DIR  = Path("/scratch/aniluchavez/ConvoDATAS/SemanticGLM")
FIG_DIR  = PROJECT / "figures"
RES_DIR  = PROJECT / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

VP_DIR = VPR_DIR / "gpt2-xl_ctx200_tshift-200_tlen500_oshift+100_olen500_xcirc" / "pc100"

# ── load VP significance data ─────────────────────────────────────────────────

df_vp = pickle.load(open(VP_DIR / "L36_VP_all.pkl", "rb"))

# ── load reliability from bestfixed window GLM (L45, pc50) ───────────────────
# Window: self onset-300ms to onset+200ms; other onset+20ms to onset+520ms.
# Slightly different from VP window (self -200ms, other +100ms) — acceptable.

import glob as _glob

REL_DIR = GLM_DIR / ("gpt2-xl_ctx200_bestfixed_selfm300_len500_otherp20_len500"
                     "_notebookexact_shuf_xcirc_r2only") / "pc50"
rows = []
for f in sorted(_glob.glob(str(REL_DIR / "PTY*_L45_sem.pkl"))):
    patient = Path(f).name.replace("_L45_sem.pkl", "")
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
df_rc = pd.DataFrame(rows)

# ── build significance mask: raw p<0.05 in EITHER self OR other ───────────────

hippo_vp = df_vp[df_vp["region"] == "hippocampus"].copy()

sig_self  = set(zip(
    hippo_vp.loc[(hippo_vp["condition"] == "self")  & (hippo_vp["p_perm"] < 0.05), "patient"],
    hippo_vp.loc[(hippo_vp["condition"] == "self")  & (hippo_vp["p_perm"] < 0.05), "neuron_idx"],
))
sig_other = set(zip(
    hippo_vp.loc[(hippo_vp["condition"] == "other") & (hippo_vp["p_perm"] < 0.05), "patient"],
    hippo_vp.loc[(hippo_vp["condition"] == "other") & (hippo_vp["p_perm"] < 0.05), "neuron_idx"],
))
sig_either = sig_self | sig_other

mask = df_rc.apply(lambda r: (r["patient"], r["neuron_idx"]) in sig_either, axis=1)
df_sig = df_rc[mask].copy()

print(f"Sig self:  {len(sig_self)}")
print(f"Sig other: {len(sig_other)}")
print(f"Sig either (used): {len(sig_either)}")
print(f"r_cross rows after filter: {len(df_sig)}")

# ── stats helpers ─────────────────────────────────────────────────────────────

def p_label(p):
    if not np.isfinite(p):
        return "p=n/a"
    if p < 0.0001:
        return "p<0.0001"
    if p < 0.001:
        return f"p={p:.4f}"
    return f"p={p:.3f}"


def one_sided_wilcoxon(x, y=None):
    x = np.asarray(x, float)
    if y is None:
        diff = x[np.isfinite(x)]
    else:
        y = np.asarray(y, float)
        ok = np.isfinite(x) & np.isfinite(y)
        diff = (x - y)[ok]
    if len(diff) < 2 or np.allclose(diff, 0):
        return np.nan
    return float(stats.wilcoxon(diff, alternative="greater",
                                zero_method="wilcox", method="auto").pvalue)


# ── patient-level Wilcoxon (same approach as prior figures) ───────────────────

patient_stats = (
    df_sig.groupby("patient", as_index=False)
    .agg(
        cross=("r_cross", "median"),
        ceiling=("lower_ceiling", "median"),
    )
)
p_cross_zero   = one_sided_wilcoxon(patient_stats["cross"].values)
p_cross_ceil   = one_sided_wilcoxon(patient_stats["ceiling"].values,
                                     patient_stats["cross"].values)

n_neurons      = len(df_sig)
n_patients     = df_sig["patient"].nunique()
median_cross   = df_sig["r_cross"].median()
median_ceil    = df_sig["lower_ceiling"].median()
pct_pos        = 100 * (df_sig["r_cross"] > 0).mean()

print(f"\nn={n_neurons} neurons ({n_patients} patients)")
print(f"median r_cross={median_cross:.3f}, median ceiling={median_ceil:.3f}")
print(f"% positive r_cross={pct_pos:.1f}%")
print(f"p(r_cross>0)={p_cross_zero:.2e}")
print(f"p(ceiling>r_cross)={p_cross_ceil:.2e}")

# ── build long df for violins ─────────────────────────────────────────────────

METRICS = ["Speaking\nreliability", "Listening\nreliability", "Cross-condition\n(r_cross)"]
COLORS  = {
    "Speaking\nreliability":   "#c83e3e",
    "Listening\nreliability":  "#3656b3",
    "Cross-condition\n(r_cross)": "#b449b5",
}

wide = df_sig[["patient", "neuron_idx",
               "self_reliability_mean", "other_reliability_mean", "r_cross",
               "lower_ceiling"]].copy()
wide.columns = ["patient", "neuron_idx",
                "Speaking\nreliability", "Listening\nreliability",
                "Cross-condition\n(r_cross)", "lower_ceiling"]

long = wide.melt(
    id_vars=["patient", "neuron_idx", "lower_ceiling"],
    value_vars=METRICS,
    var_name="metric",
    value_name="correlation",
).dropna(subset=["correlation"])

# ── figure ────────────────────────────────────────────────────────────────────

sns.set_theme(style="white", context="talk")
fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
rng = np.random.default_rng(42)

sns.violinplot(
    data=long, x="metric", y="correlation",
    order=METRICS, hue="metric", hue_order=METRICS,
    palette=COLORS, inner=None, cut=0, linewidth=1, legend=False, ax=ax,
)

for xi, metric in enumerate(METRICS):
    vals = wide[metric].dropna().values
    keep = (rng.choice(len(vals), size=min(220, len(vals)), replace=False)
            if len(vals) > 220 else np.arange(len(vals)))
    jitter = rng.uniform(-0.16, 0.16, size=len(keep))
    ax.scatter(xi + jitter, vals[keep], s=8, color="black",
               alpha=0.28, linewidths=0, zorder=3)
    med = float(np.median(vals))
    ax.plot([xi - 0.23, xi + 0.23], [med, med],
            color="black", linewidth=4, solid_capstyle="butt", zorder=4)

# bracket on the right of Cross-condition (index 2)
bx = 2.48
ax.plot([bx, bx], [0, median_ceil], color="black", linewidth=1.5, clip_on=False)
for y in [0, median_cross, median_ceil]:
    ax.plot([bx - 0.07, bx], [y, y], color="black", linewidth=1.5, clip_on=False)
ax.text(bx + 0.05, median_cross / 2,
        p_label(p_cross_zero), va="center", fontsize=10)
ax.text(bx + 0.05, (median_cross + median_ceil) / 2,
        p_label(p_cross_ceil), va="center", fontsize=10)

ax.axhline(0, color="0.82", linewidth=7, zorder=0)
ax.set_ylim(-0.35, 1.03)
ax.set_xlabel("")
ax.set_ylabel("correlation (r)")
ax.set_title(
    f"Hippocampus — semantic-significant neurons\n"
    f"VP: fixed 500ms (self −200ms, other +100ms), xcirc+block, L36\n"
    f"Reliability: bestfixed window, GPT-2 XL L45\n"
    f"n={n_neurons} neurons, {n_patients} patients  |  raw p<0.05 in self or other",
    fontsize=10, fontweight="bold",
)
ax.spines[["top", "right"]].set_visible(False)

out_pdf = FIG_DIR / "09_brackets_fixed500_sigVP_L36.pdf"
out_png = FIG_DIR / "09_brackets_fixed500_sigVP_L36.png"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, dpi=220, bbox_inches="tight")
print(f"\nSaved: {out_pdf}")
print(f"Saved: {out_png}")
