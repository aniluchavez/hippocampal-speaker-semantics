#!/usr/bin/env python3
"""Generate LaTeX tables for the gpt2-xl L48 fixed-window PC sweep:
train/test/null R2 + significance, and self/other beta correlation (r_cross)
vs its noise ceiling. Source data pickled by the interactive analysis in
scripts/plot_fixed_gpt2xl_L48_nullcheck.py / plot_fixed_gpt2xl_L48_rcross_vs_pc.py
sessions (recomputed here directly from the sbatch sweep outputs).
"""
import glob
import pickle

import numpy as np

GLM_XCIRC = ('/scratch/aniluchavez/ConvoDATAS/SemanticGLM/'
             'gpt2-xl_ctx200_fixed_selfm200_otherp100_len500_notebookexact_shuf_xcirc')
GLM_REL = ('/scratch/aniluchavez/ConvoDATAS/SemanticGLM/'
           'gpt2-xl_ctx200_fixed_selfm200_otherp100_len500_notebookexact_shuf_xcirc_r2only')
PC_COUNTS = [5, 10, 20, 50, 100, 200, 300, 500]
RESULTS_DIR = '/scratch/aniluchavez/hippocampal-speaker-semantics/results'


def mean_sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    return np.mean(x), (np.std(x, ddof=1) / np.sqrt(x.size) if x.size > 1 else 0.0)


def collect_rows():
    rows = []
    for pc in PC_COUNTS:
        files = sorted(glob.glob(f'{GLM_XCIRC}/pc{pc}/*_L48_sem.pkl'))
        per_pat = {c: {'train': [], 'test': [], 'null': [], 'sig': []} for c in ('self', 'other')}
        for f in files:
            obj = pickle.load(open(f, 'rb'))
            df = obj['df']
            pn = obj['perm_nulls']
            for cond in ('self', 'other'):
                sub = df[(df['region'] == 'hippocampus') & (df['condition'] == cond)]
                if sub.empty:
                    continue
                per_pat[cond]['train'].append(np.nanmean(sub['r2_train']))
                per_pat[cond]['test'].append(np.nanmean(sub['r2']))
                per_pat[cond]['sig'].append(100 * np.mean(sub['raw_significant']))
                d = pn[('hippocampus', cond)]
                ll_null, ll_perms = d['ll_null'], d['ll_perms']
                valid = ll_null < -0.1
                with np.errstate(invalid='ignore', divide='ignore'):
                    r2_perm = 1.0 - ll_perms[:, valid] / ll_null[None, valid]
                per_pat[cond]['null'].append(np.nanmedian(r2_perm))

        relfiles = sorted(glob.glob(f'{GLM_REL}/pc{pc}/*_sem.pkl'))
        rc_pat, ceil_pat = [], []
        for f in relfiles:
            obj = pickle.load(open(f, 'rb'))
            rel = obj.get('reliability', {}).get('hippocampus', [])
            rc = np.array([n['r_cross'] for n in rel], dtype=float)
            ce = np.array([n['ceil_median'] for n in rel], dtype=float)
            rc, ce = rc[np.isfinite(rc)], ce[np.isfinite(ce)]
            if rc.size:
                rc_pat.append(np.mean(rc))
            if ce.size:
                ceil_pat.append(np.mean(ce))

        row = {'pc': pc}
        for cond in ('self', 'other'):
            row[f'{cond}_train'] = mean_sem(per_pat[cond]['train'])
            row[f'{cond}_test'] = mean_sem(per_pat[cond]['test'])
            row[f'{cond}_null'] = mean_sem(per_pat[cond]['null'])
            row[f'{cond}_sig'] = mean_sem(per_pat[cond]['sig'])
        row['r_cross'] = mean_sem(rc_pat)
        row['ceiling'] = mean_sem(ceil_pat)
        rows.append(row)
    return rows


def fmt(v, d=4):
    m, s = v
    return f'{m:.{d}f} \\pm {s:.{d}f}'


def write_train_test_table(rows):
    lines = []
    lines.append(r'\begin{table}[ht]')
    lines.append(r'\centering')
    lines.append(
        r'\caption{Hippocampal encoding model (GPT2-XL L48, fixed windows; self: '
        r'$-200$ to $+300$~ms, other: $+100$ to $+600$~ms) train/test McFadden $R^2$ '
        r'versus a circular-shift (xcirc) permutation null, across PC count. Values are '
        r'per-patient means $\pm$ SEM across 15 patients. \% significant is the raw '
        r'(uncorrected) permutation-test rate per neuron, versus 5\% expected by chance.}')
    lines.append(r'\label{tab:fixed_gpt2xl_L48_pcsweep_train_test_null}')
    lines.append(r'\resizebox{\textwidth}{!}{%')
    lines.append(r'\begin{tabular}{r l r r r r}')
    lines.append(r'\toprule')
    lines.append(r'PCs & Condition & Train $R^2$ & Test $R^2$ & Null $R^2$ (xcirc) & \% significant \\')
    lines.append(r'\midrule')
    for r in rows:
        pc = r['pc']
        used = ' (used)' if pc == 100 else ''
        for i, cond in enumerate(('self', 'other')):
            pc_cell = f'{pc}{used}' if i == 0 else ''
            lines.append(
                f'{pc_cell} & {cond} & ${fmt(r[f"{cond}_train"])}$ & '
                f'${fmt(r[f"{cond}_test"])}$ & ${fmt(r[f"{cond}_null"])}$ & '
                f'${fmt(r[f"{cond}_sig"], 1)}\\%$ \\\\')
        if pc != rows[-1]['pc']:
            lines.append(r'\addlinespace')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}%')
    lines.append(r'}')
    lines.append(
        r'\begin{flushleft}\footnotesize Train/test $R^2$ and null $R^2$ are per-patient '
        r'means of per-neuron McFadden pseudo-$R^2$ (training-mean Poisson null as the '
        r'$R^2=0$ baseline). Null $R^2$ is the median $R^2$ achieved by models refit on '
        r'circularly time-shifted embeddings (xcirc, lag $>$ embedding autocorrelation '
        r'length, 100 shifts/neuron), scored against the same real held-out spikes. '
        r'\% significant is the fraction of neurons with raw permutation $p<0.05$ against '
        r'their own 100-shift null, averaged per patient then across patients.'
        r'\end{flushleft}')
    lines.append(r'\end{table}')
    out = f'{RESULTS_DIR}/fixed_gpt2xl_L48_pcsweep_train_test_null_table.tex'
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'wrote {out}')


def write_rcross_table(rows):
    lines = []
    lines.append(r'\begin{table}[ht]')
    lines.append(r'\centering')
    lines.append(
        r'\caption{Self/other $\beta$-vector correlation ($r_\mathrm{cross}$) versus its '
        r'split-half noise ceiling, across PC count (GPT2-XL L48, fixed windows, '
        r'hippocampus). Values are per-patient means $\pm$ SEM across 15 patients '
        r'($n_\mathrm{null}=n_\mathrm{half\_splits}=50$).}')
    lines.append(r'\label{tab:fixed_gpt2xl_L48_pcsweep_rcross}')
    lines.append(r'\resizebox{0.7\textwidth}{!}{%')
    lines.append(r'\begin{tabular}{r r r r}')
    lines.append(r'\toprule')
    lines.append(r'PCs & $r_\mathrm{cross}$ & Noise ceiling & Ratio \\')
    lines.append(r'\midrule')
    for r in rows:
        pc = r['pc']
        used = ' (used)' if pc == 100 else ''
        ratio = r['r_cross'][0] / r['ceiling'][0] * 100
        lines.append(
            f'{pc}{used} & ${fmt(r["r_cross"], 3)}$ & ${fmt(r["ceiling"], 3)}$ & '
            f'{ratio:.1f}\\% \\\\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}%')
    lines.append(r'}')
    lines.append(
        r'\begin{flushleft}\footnotesize $r_\mathrm{cross} = \mathrm{corr}(\beta_\mathrm{self}, '
        r'\beta_\mathrm{other})$ per neuron, fit on the full self/other data at each PC count. '
        r'Noise ceiling is $\sqrt{\mathrm{SB}(r_\mathrm{self})\cdot\mathrm{SB}(r_\mathrm{other})}$ '
        r'from within-condition split-half reliability (Spearman--Brown corrected). At every PC '
        r'count, the population-mean $r_\mathrm{cross}$ exceeds a per-neuron circular/row-permute '
        r'null at $p \le 1/(n_\mathrm{null}+1) \approx 0.02$ (the resolution floor for '
        r'$n_\mathrm{null}=50$).\end{flushleft}')
    lines.append(r'\end{table}')
    out = f'{RESULTS_DIR}/fixed_gpt2xl_L48_pcsweep_rcross_table.tex'
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'wrote {out}')


if __name__ == '__main__':
    rows = collect_rows()
    write_train_test_table(rows)
    write_rcross_table(rows)
