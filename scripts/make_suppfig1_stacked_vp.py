#!/usr/bin/env python3
"""
Supplementary Figure 1: per-hippocampal-neuron unique variance (semantic /
lexical / syntactic / acoustic), ranked in descending order by unique
semantic R^2, for BERT-base L12 and GPT2-XL L36.

Source: variance_partitioning.py output under
  VPResults/{bert-base_ctx200,gpt2-xl_ctx200}_worddur_xshuffle_cvshuffle_symperm/pc30/
(trial-shuffle null with shuffled cross-validation -- the same null used for
Table S4). Neurons are selected per condition (self/other) with
p_perm < 0.05 & unique_semantic > 0 & r2_full >= 0.02; the r2_full floor keeps
the panel to well-fit neurons for visualization and reproduces the published
panel counts exactly: BERT self n=45/other n=75, GPT2-XL self n=63/other n=109.

Usage: python3 -u scripts/make_suppfig1_stacked_vp.py [--out OUT.png]
"""
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VPR = "/scratch/aniluchavez/ConvoDATAS/VPResults"

COLORS = {
    "unique_semantic": "#2a78d6",
    "unique_lexical": "#1baf7a",
    "unique_syntactic": "#c98200",
    "unique_acoustic": "#008300",
}
LABELS = {
    "unique_semantic": "Semantic",
    "unique_lexical": "Lexical",
    "unique_syntactic": "Syntactic",
    "unique_acoustic": "Acoustic",
}
STACK_COLS = ["unique_semantic", "unique_lexical", "unique_syntactic", "unique_acoustic"]
R2_FULL_MIN = 0.02


def get_sig(df):
    hippo = df[df.region == "hippocampus"].copy()
    out = {}
    for cond in ["self", "other"]:
        sub = hippo[
            (hippo.condition == cond)
            & (hippo.p_perm < 0.05)
            & (hippo.unique_semantic > 0)
            & (hippo.r2_full >= R2_FULL_MIN)
        ].copy()
        sub = sub.sort_values("unique_semantic", ascending=False).reset_index(drop=True)
        out[cond] = sub
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/suppfig1_stacked_vp.png")
    args = ap.parse_args()

    bert = pickle.load(open(f"{VPR}/bert-base_ctx200_worddur_xshuffle_cvshuffle_symperm/pc30/L12_VP_all.pkl", "rb"))
    gpt2 = pickle.load(open(f"{VPR}/gpt2-xl_ctx200_worddur_xshuffle_cvshuffle_symperm/pc30/L36_VP_all.pkl", "rb"))

    bert_sig = get_sig(bert)
    gpt2_sig = get_sig(gpt2)

    fig, axes = plt.subplots(2, 2, figsize=(15.6, 10.8))
    panels = [
        ("A", bert_sig["self"], "Self", axes[0, 0]),
        ("B", bert_sig["other"], "Other", axes[0, 1]),
        ("C", gpt2_sig["self"], "Self", axes[1, 0]),
        ("D", gpt2_sig["other"], "Other", axes[1, 1]),
    ]

    for letter, sub, cond_label, ax in panels:
        n = len(sub)
        bottom = np.zeros(n)
        x = np.arange(n)
        for col in STACK_COLS:
            vals = sub[col].clip(lower=0).values
            ax.bar(x, vals, bottom=bottom, width=1.0, color=COLORS[col], label=LABELS[col])
            bottom += vals
        ax.set_title(f"{cond_label}  (n={n})", loc="left", fontsize=15, fontweight="bold")
        ax.set_xlabel("Neuron rank")
        ax.set_ylabel("Unique R²")
        ax.legend(loc="upper right", frameon=False, fontsize=9)
        ax.text(-0.12, 1.08, letter, transform=ax.transAxes, fontsize=26, fontweight="bold", va="top")
        ax.set_xlim(-0.5, n - 0.5)
        print(f"panel {letter} ({cond_label}): n={n}")

    plt.tight_layout(rect=[0.02, 0.02, 1, 1])
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
