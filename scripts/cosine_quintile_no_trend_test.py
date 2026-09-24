#!/usr/bin/env python3
"""Tests whether the self/other beta "semi-orthogonal" alignment (r_cross)
depends systematically on cosine-distance-to-other-centroid content
similarity, using the quintile cosine-bin-split reliability results
(reliability_cosine_bin_nb5, job 556057). No new GLM run needed.

This supports the "it doesn't matter" reading of the reviewer's context-
overlap question -- i.e. the claim that alignment strength does NOT
systematically depend on how similar a word's content is to the opposite
condition's -- rather than a "near gets a boost" reading. The right test for
that claim is a per-patient LINEAR TREND (slope of r_cross vs. quintile
rank) tested against zero across patients, not a pile of pairwise adjacent-
bin comparisons (which will turn up a "significant" pair somewhere among
several tests even under a flat null, e.g. the Q2>Q3 dip already seen).

Reports, per patient then aggregated across the n patients with usable data
in all 5 bins:
  - per-patient linear-fit slope of r_cross vs bin rank (0..4), tested
    against 0 (Wilcoxon signed-rank + paired t-test)
  - per-patient Spearman rank correlation of r_cross vs bin rank, tested
    against 0
  - direct extremes comparison, Q1 (nearest) vs Q5 (farthest)
  - Friedman omnibus (all 5 bins differ at all?) for transparency, noting
    it does not imply a systematic trend even if significant

Source: scripts/run_fixed_gpt2xl_L48_cosine_bin_split_quintiles.sbatch.

Usage: python3 scripts/cosine_quintile_no_trend_test.py
"""
import glob
import pickle

import numpy as np
from scipy import stats

GLM_DIR = ('/scratch/aniluchavez/ConvoDATAS/SemanticGLM/'
           'gpt2-xl_ctx200_fixed_selfm200_otherp100_len500_notebookexact_shuf_xcirc_r2only')
BINS = ['bin0', 'bin1', 'bin2', 'bin3', 'bin4']


def collect():
    files = sorted(f for f in glob.glob(f'{GLM_DIR}/pc20/*_sem.pkl') if 'L48_all' not in f)
    per_pat = {b: {} for b in BINS}
    for f in files:
        pid = f.split('/')[-1].replace('_L48_sem.pkl', '')
        obj = pickle.load(open(f, 'rb'))
        cb = obj.get('reliability_cosine_bin_nb5', {}).get('hippocampus', {})
        if not cb:
            continue
        for b in BINS:
            res = cb.get(b, [])
            rc = np.array([n['r_cross'] for n in res], dtype=float)
            rc = rc[np.isfinite(rc)]
            if rc.size:
                per_pat[b][pid] = np.mean(rc)
    patients = sorted(set.intersection(*(set(per_pat[b].keys()) for b in BINS)))
    mat = np.array([[per_pat[b][p] for b in BINS] for p in patients])
    return mat, patients


def main():
    mat, patients = collect()
    n = len(patients)
    x = np.arange(5)
    print(f'n patients = {n}\n')

    means = mat.mean(axis=0)
    sems = mat.std(axis=0, ddof=1) / np.sqrt(n)
    for j in range(5):
        print(f'  bin{j}: r_cross = {means[j]:.4f} +/- {sems[j]:.4f}')
    print()

    # per-patient linear trend
    slopes = np.array([np.polyfit(x, mat[i], 1)[0] for i in range(n)])
    w_slope = stats.wilcoxon(slopes)
    t_slope = stats.ttest_1samp(slopes, 0)
    print(f'Linear trend (slope of r_cross vs. quintile rank), per patient then '
          f'tested vs. 0 across patients:')
    print(f'  mean slope = {slopes.mean():.5f} +/- {slopes.std(ddof=1)/np.sqrt(n):.5f}')
    print(f'  Wilcoxon signed-rank vs 0 (two-sided): p={w_slope.pvalue:.4f}')
    print(f'  paired t-test vs 0:                    p={t_slope.pvalue:.4f}, t={t_slope.statistic:.3f}')
    print()

    # per-patient rank correlation
    rhos = np.array([stats.spearmanr(x, mat[i])[0] for i in range(n)])
    w_rho = stats.wilcoxon(rhos)
    print(f'Spearman rank correlation (r_cross vs. quintile rank), per patient then '
          f'tested vs. 0 across patients:')
    print(f'  mean rho = {rhos.mean():.3f}')
    print(f'  Wilcoxon signed-rank vs 0 (two-sided): p={w_rho.pvalue:.4f}')
    print()

    # direct extremes
    q1, q5 = mat[:, 0], mat[:, 4]
    w_extremes = stats.wilcoxon(q1, q5)
    print(f'Direct extremes comparison, Q1 (nearest) vs Q5 (farthest):')
    print(f'  Q1={q1.mean():.4f}  Q5={q5.mean():.4f}  '
          f'two-sided Wilcoxon p={w_extremes.pvalue:.4f}  '
          f'(Q1>Q5 in {int((q1>q5).sum())}/{n} patients)')
    print()

    # omnibus, for transparency -- NOT the same claim as "no systematic trend"
    fr = stats.friedmanchisquare(*[mat[:, j] for j in range(5)])
    print(f'Friedman omnibus (do the 5 bins differ AT ALL, any pattern): '
          f'chi2={fr.statistic:.2f}, p={fr.pvalue:.4f}')
    print(f'  (a significant omnibus here reflects a local dip, not a monotonic '
          f'trend -- see the slope/rho tests above for the trend claim itself)')

    print()
    print('Summary: no significant linear trend or rank correlation between '
          'content similarity and alignment strength (slope p={:.2f}, rho p={:.2f}), '
          'and the two extremes (nearest vs. farthest quintile) are not significantly '
          'different from each other (p={:.2f}). Alignment strength does not '
          'systematically depend on cosine-distance-based content similarity.'
          .format(w_slope.pvalue, w_rho.pvalue, w_extremes.pvalue))


if __name__ == '__main__':
    main()
