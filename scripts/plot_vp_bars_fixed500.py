#!/usr/bin/env python3
"""
VP stacked bar figures for fixed 500ms window (self -200ms, other +100ms).

Figure 1 — 10_vp_bars_by_family_fixed500.pdf
  Stacked bars: unique_semantic / unique_controls / shared, by control family
  (all controls, lexical, syntactic, acoustic) — mean R² clipped ≥ 0.
  Title row shows raw p<0.05 % significant.

Figure 2 — 10_pct_sig_by_family_fixed500.pdf
  Grouped bars: % neurons with raw p<0.05 per family, region x condition.
"""

import glob, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 130})

VP_DIR  = '/scratch/aniluchavez/ConvoDATAS/VPResults/gpt2-xl_ctx200_tshift-200_tlen500_oshift+100_olen500_xcirc/pc100'
FIG_DIR = '/scratch/aniluchavez/hippocampal-speaker-semantics/figures'

REGIONS     = ['hippocampus', 'ACC']
CONDITIONS  = ['self', 'other']
COMPARISONS = ['all_controls', 'lexical', 'syntactic', 'acoustic']
COMP_LABELS = {'all_controls': 'vs all controls', 'lexical': 'vs lexical',
               'syntactic': 'vs syntactic', 'acoustic': 'vs acoustic'}
COMBOS      = [('hippocampus', 'self'), ('hippocampus', 'other'),
               ('ACC', 'self'), ('ACC', 'other')]
COND_COLORS = {'self': '#2166ac', 'other': '#d6604d'}
PAL         = {'unique_semantic': '#2166ac', 'unique_controls': '#d6604d',
               'shared': '#92c5de'}
VP_COLS     = ['unique_semantic', 'unique_controls', 'shared']

# ── load data ─────────────────────────────────────────────────────────────────
data = pickle.load(open(f'{VP_DIR}/L36_VP_all.pkl', 'rb'))
print(f'{len(data)} rows, {data["patient"].nunique()} patients')

# ── helper: extract components for a given comparison ─────────────────────────
def vp_components(sub, comparison):
    r2_sem = sub['r2_semantic']
    if comparison == 'all_controls':
        u_sem  = sub['unique_semantic']
        r2_x   = sub['r2_controls']
        p_raw  = sub['p_perm']
    else:
        u_sem  = sub[f'unique_semantic_{comparison}']
        r2_x   = sub[f'r2_{comparison}']
        p_raw  = sub[f'p_perm_{comparison}']
    valid = r2_x.between(-2, 2) & r2_sem.between(-2, 2) & u_sem.between(-2, 2)
    n_excl = (~valid).sum()
    u_sem  = u_sem[valid]
    r2_x   = r2_x[valid]
    r2_sem = r2_sem[valid]
    p_raw  = p_raw[valid]
    u_ctrl  = u_sem + r2_x - r2_sem
    shared  = r2_sem - u_sem
    sig_raw = p_raw < 0.05
    return u_sem, u_ctrl, shared, sig_raw, n_excl

# ── FIGURE 1: stacked R² bars by control family ───────────────────────────────
fig, axes = plt.subplots(len(COMPARISONS), 4,
                         figsize=(13, 4.2 * len(COMPARISONS)), sharey='row')

for row, comparison in enumerate(COMPARISONS):
    for ax, (region, cond) in zip(axes[row], COMBOS):
        sub = data[(data['region'] == region) & (data['condition'] == cond)]
        if sub.empty:
            ax.set_title(f'{region}/{cond}\n(no data)'); continue

        u_sem, u_ctrl, shared, sig_raw, n_excl = vp_components(sub, comparison)
        means = {
            'unique_semantic': u_sem.clip(lower=0).mean(),
            'unique_controls': u_ctrl.clip(lower=0).mean(),
            'shared':          shared.clip(lower=0).mean(),
        }
        pct_sig = 100 * sig_raw.mean()

        bottom = 0
        for col in VP_COLS:
            v = means[col]
            ax.bar(0, v, bottom=bottom, color=PAL[col], width=0.6)
            if v > 0.0005:
                ax.text(0, bottom + v / 2, f'{v:.4f}',
                        ha='center', va='center', fontsize=8,
                        color='white', fontweight='bold')
            bottom += v

        excl_note = f', {n_excl} excl.' if n_excl else ''
        title = (f'{region}\n{cond}\n{pct_sig:.0f}% sig{excl_note}'
                 if row == 0 else f'{pct_sig:.0f}% sig{excl_note}')
        ax.set_title(title, fontsize=9.5)
        ax.set_xticks([])
        ax.set_xlim(-0.5, 0.5)

    axes[row, 0].set_ylabel(f'{COMP_LABELS[comparison]}\nMean R² (clipped ≥ 0)',
                            fontsize=9.5)

handles = [mpatches.Patch(facecolor=PAL[c], label=c.replace('_', ' ')) for c in VP_COLS]
fig.legend(handles=handles, loc='lower center', ncol=3,
           fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
plt.suptitle('gpt2-xl L36  |  Fixed 500ms window (self −200ms, other +100ms)'
             '  |  Variance partitioning, by control family\n'
             'raw p<0.05, xcirc null + block CV', y=1.01)
plt.tight_layout()
out1 = f'{FIG_DIR}/10_vp_bars_by_family_fixed500.pdf'
plt.savefig(out1, bbox_inches='tight')
plt.savefig(out1.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
print(f'Saved: {out1}')
plt.close()

# ── FIGURE 2: % significant neurons by family ─────────────────────────────────
fig2, axes2 = plt.subplots(1, len(REGIONS), figsize=(6.5 * len(REGIONS), 5), squeeze=False)
x = np.arange(len(COMPARISONS))
w = 0.35

for ax, region in zip(axes2[0], REGIONS):
    for i, cond in enumerate(CONDITIONS):
        pcts = []
        for comp in COMPARISONS:
            sub = data[(data['region'] == region) & (data['condition'] == cond)]
            _, _, _, sig_raw, _ = vp_components(sub, comp)
            pcts.append(100 * sig_raw.mean())
        ax.bar(x + (i - 0.5) * w, pcts, width=w,
               label=cond, color=COND_COLORS[cond])
        for xi, pct in zip(x, pcts):
            ax.text(xi + (i - 0.5) * w, pct + 0.3, f'{pct:.1f}%',
                    ha='center', va='bottom', fontsize=7.5)

    ax.axhline(5, color='gray', linewidth=0.8, linestyle='--', label='5% chance')
    ax.set_xticks(x)
    ax.set_xticklabels([COMP_LABELS[c] for c in COMPARISONS], rotation=20, ha='right')
    ax.set_ylabel('% neurons (raw p<0.05)')
    ax.set_title(region)
    ax.legend(frameon=False, fontsize=9)

plt.suptitle('% significant neurons (raw p<0.05) by confound family controlled for\n'
             'gpt2-xl L36  |  Fixed 500ms window  |  xcirc null + block CV', y=1.04)
plt.tight_layout()
out2 = f'{FIG_DIR}/10_pct_sig_by_family_fixed500.pdf'
plt.savefig(out2, bbox_inches='tight')
plt.savefig(out2.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
print(f'Saved: {out2}')
plt.close()
