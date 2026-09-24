#!/usr/bin/env python3
"""Self/other beta correlation (r_cross) vs PC count, gpt2-xl L48, hippocampus.

Per-patient means (not pooled-per-neuron) averaged across the 15 patients,
with SEM error bars, plotted alongside the per-patient-averaged noise
ceiling (sqrt(SB(r_self)*SB(r_other)) from split-half within-condition
reliability). Source: the _reliability sbatch sweep (n_null=50,
n_half_splits=50 per neuron).
"""
import glob
import pickle

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 130})

GLM_DIR = ('/scratch/aniluchavez/ConvoDATAS/SemanticGLM/'
           'gpt2-xl_ctx200_fixed_selfm200_otherp100_len500_notebookexact_shuf_xcirc_r2only')
PC_COUNTS = [5, 10, 20, 50, 100, 200, 300, 500]
FIG_DIR = '/scratch/aniluchavez/hippocampal-speaker-semantics/figures'


def per_patient_means(pc):
    files = sorted(glob.glob(f'{GLM_DIR}/pc{pc}/*_sem.pkl'))
    rcross_means, ceil_means = [], []
    for f in files:
        obj = pickle.load(open(f, 'rb'))
        rel = obj.get('reliability', {}).get('hippocampus', [])
        rc = np.array([n['r_cross'] for n in rel], dtype=float)
        ce = np.array([n['ceil_median'] for n in rel], dtype=float)
        rc, ce = rc[np.isfinite(rc)], ce[np.isfinite(ce)]
        if rc.size:
            rcross_means.append(np.mean(rc))
        if ce.size:
            ceil_means.append(np.mean(ce))
    return np.array(rcross_means), np.array(ceil_means)


def mean_sem(x):
    return np.mean(x), np.std(x, ddof=1) / np.sqrt(len(x))


r_mean, r_sem, c_mean, c_sem = [], [], [], []
for pc in PC_COUNTS:
    rc, ce = per_patient_means(pc)
    m, s = mean_sem(rc)
    r_mean.append(m); r_sem.append(s)
    m, s = mean_sem(ce)
    c_mean.append(m); c_sem.append(s)

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.errorbar(PC_COUNTS, c_mean, yerr=c_sem, color='#666666', marker='^',
            linewidth=2, markersize=7, linestyle='--', capsize=3,
            label='noise ceiling  sqrt(SB(r_self)·SB(r_other))')
ax.errorbar(PC_COUNTS, r_mean, yerr=r_sem, color='#2166ac', marker='o',
            linewidth=2, markersize=7, capsize=3,
            label='r_cross = corr(β_self, β_other)')

ax.axvline(100, color='gray', linewidth=1, linestyle=':', label='pc=100 (used)')
ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
ax.set_xlabel('Number of PCs')
ax.set_ylabel('Correlation')
ax.set_title('hippocampus — self/other β correlation vs. noise ceiling')
ax.legend(fontsize=9, frameon=False, loc='upper right')
ax.set_xscale('log')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.set_xticks(PC_COUNTS)
ax.set_ylim(-0.05, 0.75)

plt.suptitle('gpt2-xl L48 | fixed self[onset-200,+300] other[onset+100,+600] | '
             'per-patient mean ± SEM (n=15 patients) | n_null=n_half_splits=50',
             fontsize=8.5, y=1.0)
plt.tight_layout()
out = f'{FIG_DIR}/rcross_vs_pc_fixed_gpt2xl_L48.pdf'
plt.savefig(out, bbox_inches='tight')
print(f'Saved -> {out}')
for pc, rm, cm in zip(PC_COUNTS, r_mean, c_mean):
    print(f'pc={pc:4d}  r_cross={rm:.4f}  ceiling={cm:.4f}  ratio={rm/cm:.3f}  below_ceiling={rm < cm}')
