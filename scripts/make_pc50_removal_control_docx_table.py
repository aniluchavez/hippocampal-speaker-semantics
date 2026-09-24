#!/usr/bin/env python3
"""Make DOCX/TEX table for PC50 significant-neuron removal control."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import pandas as pd


PROJECT = Path("/scratch/aniluchavez/hippocampal-speaker-semantics")
RESULTS = PROJECT / "results"
CONVERTER = PROJECT / "scripts" / "tex_table_to_docx.py"


def num(x: float, digits: int = 3) -> str:
    if not math.isfinite(float(x)):
        return "---"
    return f"{float(x):.{digits}f}"


def mean_sem(mean: float, sem: float, digits: int = 3) -> str:
    if not math.isfinite(float(mean)):
        return "---"
    if not math.isfinite(float(sem)):
        return num(mean, digits)
    return f"${float(mean):.{digits}f} \\pm {float(sem):.{digits}f}$"


def pval(p: float) -> str:
    if not math.isfinite(float(p)):
        return "---"
    if float(p) < 0.0001:
        return "$<0.0001$"
    return f"{float(p):.4f}"


def write_tex(path: Path, caption: str, label: str, header: list[str], rows: list[list[str]], footnote: str) -> None:
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{" + "l" + "c" * (len(header) - 1) + "}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{flushleft}\footnotesize",
            footnote,
            r"\end{flushleft}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def convert(tex_path: Path) -> None:
    subprocess.run(
        [
            "/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3",
            str(CONVERTER),
            str(tex_path),
            str(tex_path.with_suffix(".docx")),
        ],
        check=True,
    )


def main() -> None:
    all_pc = pd.read_csv(
        RESULTS / "bert_l12_fixed_unbalanced_pc_sweep_rcross_above_null_no_fdr_summary.csv"
    )
    sig = pd.read_csv(RESULTS / "bert_l12_fixed_pc50_sigonly_rcross_summary.csv")
    nonsig = pd.read_csv(RESULTS / "bert_l12_fixed_pc50_nonsigonly_rcross_summary.csv")

    all50 = all_pc.loc[all_pc["pc"].eq(50)].iloc[0]
    sig50 = sig.iloc[0]
    nonsig50 = nonsig.iloc[0]

    rows = [
        [
            "All PC50 neurons",
            f"{int(all50['n_neurons'])}/{int(all50['n_neurons'])} (100.0\\%)",
            mean_sem(all50["patient_mean_obs_r_cross"], all50["patient_diff_sem"]),
            num(all50["patient_mean_null_mean"]),
            num(all50["patient_mean_diff_obs_minus_nullmean"]),
            f"{int(all50['n_sig_neurons_p05_no_fdr'])}/{int(all50['n_neurons'])} ({all50['pct_sig_neurons_p05_no_fdr']:.1f}\\%)",
            num(all50["patient_wilcoxon_W_greater"], 0),
            pval(all50["patient_wilcoxon_p_greater"]),
        ],
        [
            "Significant neurons only",
            f"{int(sig50['n_sig_neurons'])}/{int(sig50['n_total_neurons'])} ({sig50['pct_neurons_retained']:.1f}\\%)",
            mean_sem(sig50["patient_mean_r_cross_sig"], sig50["patient_sem_r_cross_sig"]),
            num(sig50["patient_mean_null_mean_sig"]),
            num(sig50["patient_mean_diff_obs_minus_null_sig"]),
            f"{int(sig50['n_patients_with_sig_neurons'])}/{int(sig50['n_patients_total'])}",
            num(sig50["wilcoxon_W_obs_minus_null_greater"], 0),
            pval(sig50["wilcoxon_p_obs_minus_null_greater"]),
        ],
        [
            "Nonsignificant neurons only",
            f"{int(nonsig50['n_nonsig_neurons'])}/{int(nonsig50['n_total_neurons'])} ({nonsig50['pct_neurons_retained']:.1f}\\%)",
            mean_sem(
                nonsig50["patient_mean_r_cross_nonsig"],
                nonsig50["patient_sem_r_cross_nonsig"],
            ),
            num(nonsig50["patient_mean_null_mean_nonsig"]),
            num(nonsig50["patient_mean_diff_obs_minus_null_nonsig"]),
            f"13/{int(nonsig50['n_patients_total'])}",
            num(nonsig50["wilcoxon_W_obs_minus_null_greater"], 0),
            pval(nonsig50["wilcoxon_p_obs_minus_null_greater"]),
        ],
    ]

    tex = RESULTS / "bert_l12_fixed_pc50_removal_control_table.tex"
    write_tex(
        tex,
        (
            "Population-level PC50 cross-condition reliability after removing "
            "individually significant neurons."
        ),
        "tab:bert_l12_fixed_pc50_removal_control",
        [
            "Neuron set",
            "Neurons retained",
            "$r_\\mathrm{cross}$",
            "Null mean",
            "$\\Delta$",
            "Patients/neurons $>$ null",
            "Wilcoxon $W$",
            "$p$",
        ],
        rows,
        (
            "Values are patient means $\\pm$ SEM. "
            "$\\Delta$ is patient-mean $r_\\mathrm{cross}$ minus the empirical-null mean. "
            "The all-neuron row reports the number of neurons individually significant above null at "
            "$p<0.05$ uncorrected. The significant-only row keeps only those individually significant neurons; "
            "the nonsignificant-only row removes them and recomputes the population effect. "
            "Patient tests are one-sided Wilcoxon signed-rank tests of $\\Delta>0$ across 15 patients. "
            "No FDR correction was applied. Fixed windows used BERT-base L12, 50 PCs, self onset$-200$ ms "
            "and other onset$+200$ ms, both 500 ms long."
        ),
    )
    convert(tex)
    print(tex)
    print(tex.with_suffix(".docx"))


if __name__ == "__main__":
    main()
