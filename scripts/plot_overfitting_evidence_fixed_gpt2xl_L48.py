#!/usr/bin/env python3
"""Point 5: two-panel 'we are not overfitting' figure for gpt2-xl L48, fixed
window, hippocampus. Left: train/test/shuffled-null R2 vs PC count (shows the
raw train/test gap opening up). Right: % significant neurons (calibrated
xcirc permutation test) vs PC count (shows the properly cross-validated
metric stays flat/stable across the same PC range) -- the pairing is the
argument: the growing train/test gap in raw R2 does not translate into a
growing false-discovery problem once you look at the metric that's actually
built to detect overfitting (per-neuron permutation significance).
"""
import glob
import pickle

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 130})

GLM_DIR = ('/scratch/aniluchavez/ConvoDATAS/SemanticGLM/'
           'gpt2-xl_ctx200_fixed_selfm200_otherp100_len500_notebookexact_shuf_xcirc')
PC_COUNTS = [5, 10, 20, 50, 100, 200, 300, 500]
CONDITIONS = ['self', 'other']
COLORS = {'self': '#2166ac', 'other': '#d6604d'}
FIG_DIR = '/scratch/aniluchavez/hippocampal-speaker-semantics/figures'


def load_pc(pc):
    files = sorted(glob.glob(f'{GLM_DIR}/pc{pc}/*_L48_sem.pkl'))
    rows = {cond: {'test': [], 'train': [], 'null': [], 'sig': []} for cond in CONDITIONS}
    for f in files:
        obj = pickle.load(open(f, 'rb'))
        df = obj['df']
        pn = obj['perm_nulls']
        for cond in CONDITIONS:
            sub = df[(df['region'] == 'hippocampus') & (df['condition'] == cond)]
            rows[cond]['test'].append(sub['r2'].to_numpy())
            rows[cond]['train'].append(sub['r2_train'].to_numpy())
            rows[cond]['sig'].append(sub['raw_significant'].to_numpy())
            d = pn[('hippocampus', cond)]
            ll_null, ll_perms = d['ll_null'], d['ll_perms']
            valid = ll_null < -0.1
            with np.errstate(invalid='ignore', divide='ignore'):
                r2_perm = 1.0 - ll_perms[:, valid] / ll_null[None, valid]
            rows[cond]['null'].append(r2_perm.ravel())
    return {cond: {k: np.concatenate(v) for k, v in d.items()} for cond, d in rows.items()}


data = {pc: load_pc(pc) for pc in PC_COUNTS}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

for cond in CONDITIONS:
    c = COLORS[cond]
    med_test = [np.nanmedian(data[pc][cond]['test']) for pc in PC_COUNTS]
    med_train = [np.nanmedian(data[pc][cond]['train']) for pc in PC_COUNTS]
    med_null = [np.nanmedian(data[pc][cond]['null']) for pc in PC_COUNTS]
    pct_sig = [100 * np.mean(data[pc][cond]['sig']) for pc in PC_COUNTS]

    ax1.plot(PC_COUNTS, med_test, color=c, marker='o', linewidth=2, markersize=6, label=f'{cond} test')
    ax1.plot(PC_COUNTS, med_train, color=c, marker='s', linewidth=2, linestyle='--',
              markersize=5, alpha=0.5, label=f'{cond} train')
    ax1.plot(PC_COUNTS, med_null, color=c, marker='x', linewidth=1.5, linestyle=':',
              markersize=6, alpha=0.85, label=f'{cond} shuffled null')

    ax2.plot(PC_COUNTS, pct_sig, color=c, marker='o', linewidth=2.5, markersize=7, label=cond)

ax1.axvline(100, color='gray', linewidth=1, linestyle=':')
ax1.axhline(0, color='k', linewidth=0.8, linestyle='--')
ax1.set_xlabel('Number of PCs')
ax1.set_ylabel('Median McFadden R²')
ax1.set_title('Raw R²: train/test gap widens\nwith more PCs')
ax1.legend(fontsize=7.5, frameon=False, loc='upper left')
ax1.set_xscale('log')
ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax1.set_xticks(PC_COUNTS)

ax2.axvline(100, color='gray', linewidth=1, linestyle=':', label='pc=100 (used)')
ax2.axhline(5, color='k', linewidth=1, linestyle='--', label='5% (chance)')
ax2.set_xlabel('Number of PCs')
ax2.set_ylabel('% significant neurons\n(calibrated xcirc permutation test)')
ax2.set_title('Calibrated significance: stable\nacross the same PC range')
ax2.legend(fontsize=8, frameon=False, loc='center right')
ax2.set_xscale('log')
ax2.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax2.set_xticks(PC_COUNTS)
ax2.set_ylim(0, 30)

plt.suptitle('gpt2-xl L48 | fixed windows | hippocampus | the widening raw train/test '
             'gap does not create a growing false-discovery problem', fontsize=10, y=1.02)
plt.tight_layout()
out = f'{FIG_DIR}/overfitting_evidence_fixed_gpt2xl_L48.pdf'
plt.savefig(out, bbox_inches='tight')
print(f'Saved -> {out}')
