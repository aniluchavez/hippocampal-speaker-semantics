#!/usr/bin/env python3
"""Self/other beta correlation (r_cross) across cosine-distance-to-other-
centroid terciles (near/mid/far), gpt2-xl L48, hippocampus. Answers the
reviewer question of whether the self/other "semi-orthogonal" geometry is
just an artifact of self/other words living in different conversational
contexts. Source: the tercile cosine-bin-split sbatch sweep
(reliability_cosine_bin_nb3, n_null=n_half_splits=50).
"""
import glob
import pickle

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 130})

GLM_DIR = ('/scratch/aniluchavez/ConvoDATAS/SemanticGLM/'
           'gpt2-xl_ctx200_fixed_selfm200_otherp100_len500_notebookexact_shuf_xcirc_r2only')
BINS = ['near', 'mid', 'far']
FIG_DIR = '/scratch/aniluchavez/hippocampal-speaker-semantics/figures'
R_CROSS_COLOR = '#2166ac'
CEIL_COLOR = '#666666'


def collect():
    files = sorted(f for f in glob.glob(f'{GLM_DIR}/pc20/*_sem.pkl') if 'L48_all' not in f)
    per_pat = {b: {} for b in BINS}
    for f in files:
        pid = f.split('/')[-1].replace('_L48_sem.pkl', '')
        obj = pickle.load(open(f, 'rb'))
        cb = obj.get('reliability_cosine_bin_nb3', {}).get('hippocampus', {})
        if not cb:
            continue
        for b in BINS:
            res = cb.get(b, [])
            rc = np.array([n['r_cross'] for n in res], dtype=float)
            ce = np.array([n['ceil_median'] for n in res], dtype=float)
            rc, ce = rc[np.isfinite(rc)], ce[np.isfinite(ce)]
            if rc.size:
                per_pat[b][pid] = (np.mean(rc), np.mean(ce) if ce.size else np.nan)
    patients = sorted(set.intersection(*(set(per_pat[b].keys()) for b in BINS)))
    return per_pat, patients


def mean_sem(x):
    x = np.asarray(x, dtype=float)
    return np.mean(x), np.std(x, ddof=1) / np.sqrt(len(x))


per_pat, patients = collect()
n = len(patients)
x = np.arange(len(BINS))

rcross_mat = np.array([[per_pat[b][p][0] for b in BINS] for p in patients])
ceil_mat = np.array([[per_pat[b][p][1] for b in BINS] for p in patients])

fig, ax = plt.subplots(figsize=(6.5, 5.5))

# per-patient paired lines (thin, recessive) to show individual-level consistency
for i in range(n):
    ax.plot(x, rcross_mat[i], color=R_CROSS_COLOR, alpha=0.15, linewidth=1, zorder=1)

ceil_mean = [mean_sem(ceil_mat[:, j])[0] for j in range(3)]
ceil_sem = [mean_sem(ceil_mat[:, j])[1] for j in range(3)]
rcross_mean = [mean_sem(rcross_mat[:, j])[0] for j in range(3)]
rcross_sem = [mean_sem(rcross_mat[:, j])[1] for j in range(3)]

ax.errorbar(x, ceil_mean, yerr=ceil_sem, color=CEIL_COLOR, marker='^', linestyle='--',
            linewidth=2, markersize=8, capsize=4, zorder=3,
            label='noise ceiling  sqrt(SB(r_self)·SB(r_other))')
ax.errorbar(x, rcross_mean, yerr=rcross_sem, color=R_CROSS_COLOR, marker='o', linestyle='-',
            linewidth=2.5, markersize=9, capsize=4, zorder=4,
            label='r_cross = corr(β_self, β_other)')

# significance brackets from the paired Wilcoxon tests already computed
def sig_bracket(xi, xj, y, p):
    ax.plot([xi, xi, xj, xj], [y, y + 0.012, y + 0.012, y], color='black', linewidth=1)
    stars = 'n.s.' if p >= 0.05 else ('*' if p >= 0.01 else '**')
    ax.text((xi + xj) / 2, y + 0.016, stars, ha='center', va='bottom', fontsize=10)

series_top = max(max(rcross_mean[j] + rcross_sem[j], ceil_mean[j] + ceil_sem[j]) for j in range(3))
bracket1_y = series_top + 0.03
bracket2_y = series_top + 0.09
ymax = bracket2_y + 0.05
sig_bracket(0, 1, bracket1_y, 0.0052)   # near vs mid, p=0.0052
sig_bracket(1, 2, bracket2_y, 0.706)    # mid vs far, p=0.71 (n.s.)

ax.axhline(0, color='k', linewidth=0.8, linestyle=':')
ax.set_xticks(x)
ax.set_xticklabels(['near\n(context resembles\nopposite condition)',
                     'mid',
                     'far\n(context most dissimilar\nfrom opposite condition)'])
ax.set_ylabel('Correlation')
ax.set_title('hippocampus — self/other β correlation by\ncosine-distance-to-other-centroid tercile')
ax.legend(fontsize=8.5, frameon=False, loc='lower right')
ax.set_xlim(-0.3, 2.3)
ax.set_ylim(-0.05, ymax)

plt.suptitle(f'gpt2-xl L48 | fixed windows | pc=20 | n={n} patients (2 dropped, too few words) | '
             'thin lines = individual patients', fontsize=8, y=1.0)
plt.tight_layout()
out = f'{FIG_DIR}/cosine_tercile_rcross_fixed_gpt2xl_L48.pdf'
plt.savefig(out, bbox_inches='tight')
print(f'Saved -> {out}')
for j, b in enumerate(BINS):
    print(f'{b:5s}: r_cross={rcross_mean[j]:.4f}+/-{rcross_sem[j]:.4f}  '
          f'ceiling={ceil_mean[j]:.4f}+/-{ceil_sem[j]:.4f}  ratio={rcross_mean[j]/ceil_mean[j]:.3f}')
