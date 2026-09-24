#!/usr/bin/env python3
"""
Table S4: hippocampal variance partitioning, % significant neurons and mean
unique R^2 per feature family (semantic/lexical/syntactic/acoustic), by model
(GPT-2 XL / BERT L12) and condition (self/other), with patient-level
inferential statistics.

Source: variance_partitioning.py output under
  VPResults/{bert-base_ctx200,gpt2-xl_ctx200}_worddur_xshuffle_cvshuffle_symperm/pc30/
(trial-shuffle null with shuffled cross-validation).

Per (model, condition, family):
  - n_sig / n_total and pooled % significant (p_perm < 0.05 & unique_X > 0)
  - mean unique R^2 among significant neurons, patient-mean with bootstrap 95% CI
  - inferential test: is the per-patient percentage of significant neurons
    above the nominal 5% chance rate? One-sided Wilcoxon signed-rank test
    (patient-level), with rank-biserial r as effect size.

Usage: python3 -u scripts/compute_table_s4_stats.py
"""
import pickle
import numpy as np
import pandas as pd
from scipy import stats

VPR = "/scratch/aniluchavez/ConvoDATAS/VPResults"
CHANCE_PCT = 5.0
N_BOOT = 10000
BOOT_SEED = 0

FAMILIES = [
    ("Semantic", "p_perm", "unique_semantic"),
    ("Lexical", "p_perm_lexical", "unique_lexical"),
    ("Syntactic", "p_perm_syntactic", "unique_syntactic"),
    ("Acoustic", "p_perm_acoustic", "unique_acoustic"),
]

MODELS = [
    ("GPT-2 XL", f"{VPR}/gpt2-xl_ctx200_worddur_xshuffle_cvshuffle_symperm/pc30/L36_VP_all.pkl"),
    ("BERT L12", f"{VPR}/bert-base_ctx200_worddur_xshuffle_cvshuffle_symperm/pc30/L12_VP_all.pkl"),
]


def wilcoxon_rb_vs(vals, ref):
    vals = np.asarray(vals, float)
    diff = vals - ref
    diff = diff[diff != 0]
    res = stats.wilcoxon(diff, alternative="greater", zero_method="wilcox", method="auto")
    ranks = stats.rankdata(np.abs(diff))
    pos = ranks[diff > 0].sum()
    neg = ranks[diff < 0].sum()
    r_rb = (pos - neg) / (pos + neg)
    return res.statistic, r_rb, res.pvalue


def boot_ci(vals, n_iter=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals)
    boots = [np.mean(rng.choice(vals, size=len(vals), replace=True)) for _ in range(n_iter)]
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def main():
    rows = []
    for model_name, path in MODELS:
        df = pickle.load(open(path, "rb"))
        hippo = df[df.region == "hippocampus"]
        for cond, cond_label in [("self", "Self"), ("other", "Other")]:
            sub = hippo[hippo.condition == cond]
            n_total = len(sub)
            for fam, pcol, ucol in FAMILIES:
                mask = (sub[pcol] < 0.05) & (sub[ucol] > 0)
                n_sig = int(mask.sum())
                sub2 = sub.copy()
                sub2["sig"] = mask
                per_pat_pct = sub2.groupby("patient")["sig"].mean() * 100
                W, r_rb, p = wilcoxon_rb_vs(per_pat_pct.values, CHANCE_PCT)
                pct_lo, pct_hi = boot_ci(per_pat_pct.values)
                sig_df = sub2[sub2.sig]
                per_pat_r2 = sig_df.groupby("patient")[ucol].mean()
                r2_lo, r2_hi = boot_ci(per_pat_r2.values)
                rows.append({
                    "model": model_name, "condition": cond_label, "family": fam,
                    "n_sig": n_sig, "n_total": n_total,
                    "pooled_pct": 100 * n_sig / n_total,
                    "patient_mean_pct": per_pat_pct.mean(), "pct_ci_lo": pct_lo, "pct_ci_hi": pct_hi,
                    "mean_unique_r2": per_pat_r2.mean(), "r2_ci_lo": r2_lo, "r2_ci_hi": r2_hi,
                    "wilcoxon_W": W, "rank_biserial_r": r_rb, "p": p,
                    "n_patients": len(per_pat_pct),
                })
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    main()
