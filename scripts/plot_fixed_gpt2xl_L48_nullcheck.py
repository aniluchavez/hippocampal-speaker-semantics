#!/usr/bin/env python3
"""Train/test/shuffled-null R2 curve for gpt2-xl L48, fixed window, hippocampus.

Rebuilds 01_train_test_curve_fixed_gpt2xl_L48.pdf with a third curve: the
median R2 of the xcirc circular-shift permutation null (100 perms), pooled
across all permutations and neurons, at each PC count. Source data is the
_nullcheck sbatch run (same CV/window/model config as the original figure,
minus --r2_only).
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

fig, ax = plt.subplots(figsize=(7, 5.5))
for cond in CONDITIONS:
    c = COLORS[cond]
    med_test = [np.nanmedian(data[pc][cond]['test']) for pc in PC_COUNTS]
    med_train = [np.nanmedian(data[pc][cond]['train']) for pc in PC_COUNTS]
    med_null = [np.nanmedian(data[pc][cond]['null']) for pc in PC_COUNTS]
    pct_sig = [100 * np.mean(data[pc][cond]['sig']) for pc in PC_COUNTS]

    ax.plot(PC_COUNTS, med_test, color=c, marker='o', linewidth=2,
            markersize=7, label=f'{cond} test')
    ax.plot(PC_COUNTS, med_train, color=c, marker='s', linewidth=2,
            linestyle='--', markersize=6, alpha=0.5, label=f'{cond} train')
    ax.plot(PC_COUNTS, med_null, color=c, marker='x', linewidth=1.5,
            linestyle=':', markersize=7, alpha=0.85, label=f'{cond} shuffled null')

    pc100_i = PC_COUNTS.index(100)
    print(f'{cond:6s}  pc=100  test={med_test[pc100_i]:.4f}  '
          f'null={med_null[pc100_i]:.4f}  raw_sig={pct_sig[pc100_i]:.1f}% '
          f'(vs 5% expected by chance)')

ax.axvline(100, color='gray', linewidth=1, linestyle=':', label='pc=100 (used)')
ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
ax.set_xlabel('Number of PCs')
ax.set_ylabel('Median McFadden R²')
ax.set_title('hippocampus — train / test / shuffled-null R²')
ax.legend(fontsize=8.5, frameon=False, ncol=1, loc='upper left')
ax.set_xscale('log')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.set_xticks(PC_COUNTS)

plt.suptitle('gpt2-xl L48 | fixed self[onset-200,+300] other[onset+100,+600] | '
             '5-fold shuffle CV | xcirc null (n_perm=100) | pc=100 (used)', fontsize=9, y=1.0)
plt.tight_layout()
out = f'{FIG_DIR}/01_train_test_curve_fixed_gpt2xl_L48_with_null.pdf'
plt.savefig(out, bbox_inches='tight')
print(f'Saved -> {out}')
