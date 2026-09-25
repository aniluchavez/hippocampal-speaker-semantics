#!/usr/bin/env python3
"""
Table S6: linguistic encoding strength in hippocampus comparing different
cross-validation/null constructs (BERT L12, bestfixed window, PC50).

Source: per-patient semantic_glm.py output pkls (`df` key) under
  SemanticGLM/bert-base_ctx200_bestfixed_selfm300_len500_otherp20_len500_notebookexact{,_shuf}_{xcirc,xshuffle}/pc50/
produced by scripts/run_table_s6_cv_null_sweep.sbatch (permutation testing
enabled -- the previously cached bestfixed pkl for this window had
permutation testing disabled via --r2_only, so its p_perm was hardcoded to
1.0 and unusable for this table).

For each (CV/null combo, condition):
  - Train/Test R^2 and percent-significant (raw p_perm<0.05), patient means +/- SEM
  - Test R^2 vs 0: two-sided Wilcoxon signed-rank test on patient means,
    rank-biserial r as effect size
  - percent-significant vs the nominal 5% chance rate: one-sided Wilcoxon
    signed-rank test, rank-biserial r as effect size

Usage: python3 -u scripts/compute_table_s6_stats.py
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

GLM_DIR = Path("/scratch/aniluchavez/ConvoDATAS/SemanticGLM")
CHANCE_PCT = 5.0
N_BOOT = 10000
BOOT_SEED = 0

COMBOS = [
    ("Block CV, circular null", "bert-base_ctx200_bestfixed_selfm300_len500_otherp20_len500_notebookexact_xcirc"),
    ("Block CV, shuffled null", "bert-base_ctx200_bestfixed_selfm300_len500_otherp20_len500_notebookexact_xshuffle"),
    ("Shuffle CV, shuffled null", "bert-base_ctx200_bestfixed_selfm300_len500_otherp20_len500_notebookexact_shuf_xshuffle"),
]
CONDITIONS = [("self", "Speaking (self)"), ("other", "Listening (other)")]


def wilcoxon_rb(vals, ref=0.0, alternative="two-sided"):
    vals = np.asarray(vals, float)
    diff = vals - ref
    diff = diff[diff != 0]
    res = stats.wilcoxon(diff, alternative=alternative, zero_method="wilcox", method="auto")
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
    for label, dirname in COMBOS:
        d = GLM_DIR / dirname / "pc50"
        dfs = [pickle.load(open(f, "rb"))["df"] for f in sorted(d.glob("PTY*_L12_sem.pkl"))]
        all_df = pd.concat(dfs, ignore_index=True)
        hippo = all_df[all_df.region == "hippocampus"]
        for cond, cond_label in CONDITIONS:
            sub = hippo[hippo.condition == cond]
            per_pat = sub.groupby("patient").apply(lambda g: pd.Series({
                "train": g.r2_train.mean(),
                "test": g.r2.mean(),
                "sig_pct": 100 * (g.p_perm < 0.05).mean(),
            }))
            n = len(per_pat)
            sem = lambda x: x.std(ddof=1) / np.sqrt(n)

            W_r2, r_r2, p_r2 = wilcoxon_rb(per_pat["test"].values, ref=0.0, alternative="two-sided")
            W_sig, r_sig, p_sig = wilcoxon_rb(per_pat["sig_pct"].values, ref=CHANCE_PCT, alternative="greater")

            n_total = len(sub)
            n_sig = int((sub.p_perm < 0.05).sum())
            n_sig_pos = int(((sub.p_perm < 0.05) & (sub.r2 > 0)).sum())
            median_r2_sig = sub.loc[sub.p_perm < 0.05, "r2"].median()

            rows.append({
                "combo": label, "condition": cond_label, "n_patients": n,
                "train_mean": per_pat["train"].mean(), "train_sem": sem(per_pat["train"]),
                "test_mean": per_pat["test"].mean(), "test_sem": sem(per_pat["test"]),
                "sig_mean": per_pat["sig_pct"].mean(), "sig_sem": sem(per_pat["sig_pct"]),
                "wilcoxon_W_r2vs0": W_r2, "rank_biserial_r_r2vs0": r_r2, "p_r2vs0": p_r2,
                "wilcoxon_W_sigvs5pct": W_sig, "rank_biserial_r_sigvs5pct": r_sig, "p_sigvs5pct": p_sig,
                "n_total_neurons": n_total, "n_sig_neurons": n_sig, "n_sig_r2_positive": n_sig_pos,
                "median_r2_among_sig": median_r2_sig,
            })
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    main()
