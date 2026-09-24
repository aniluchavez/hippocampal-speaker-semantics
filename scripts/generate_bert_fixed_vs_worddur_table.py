#!/usr/bin/env python3
"""BERT L12 counterpart of results/fixed_vs_worddur_encoding_matched_table.tex
(GPT-2 XL L48 version). Same window (self=[onset-200,+300], other=[onset+100,
+600]), PC=20, shuffled outer CV, matched xcirc/xshuffle null comparison.

Reads per-patient neuron-level pkls from the 4 semantic_glm.py --reliability
output folders (fixed/worddur x xcirc/xshuffle) and reports per-patient
mean +/- SEM train/test R2 and raw-significant-neuron percentage.
"""
import glob
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

GLM_ROOT = Path("/scratch/aniluchavez/ConvoDATAS/SemanticGLM")
RESULTS = Path("/scratch/aniluchavez/hippocampal-speaker-semantics/results")
LAYER = 12
PC = 20

FOLDERS = {
    ("fixed", "xcirc"): "bert-base_ctx200_fixed_selfm200_otherp100_len500_notebookexact_shuf_xcirc",
    ("fixed", "xshuffle"): "bert-base_ctx200_fixed_selfm200_otherp100_len500_notebookexact_shuf_xshuffle",
    ("worddur", "xcirc"): "bert-base_ctx200_worddur_exposure_a1e6_tinner_min20_nullinit_shuf_xcirc",
    ("worddur", "xshuffle"): "bert-base_ctx200_worddur_exposure_a1e6_tinner_min20_nullinit_shuf_xshuffle",
}

CONDITIONS = [("self", "Speaking (self)"), ("other", "Listening (other)")]
WINDOWS = [("fixed", "Fixed"), ("worddur", "Word duration")]


def mean_sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return np.mean(x), (np.std(x, ddof=1) / np.sqrt(x.size) if x.size > 1 else 0.0)


def per_patient_means(folder_name, cond):
    folder = GLM_ROOT / folder_name / f"pc{PC}"
    files = sorted(glob.glob(str(folder / f"PTY*_L{LAYER:02d}_sem.pkl")))
    train, test, sig = [], [], []
    for f in files:
        obj = pickle.load(open(f, "rb"))
        df = obj["df"]
        sub = df[(df["region"] == "hippocampus") & (df["condition"] == cond)]
        if sub.empty:
            continue
        train.append(np.nanmean(sub["r2_train"]))
        test.append(np.nanmean(sub["r2"]))
        sig.append(100 * np.mean(sub["raw_significant"]))
    return len(files), train, test, sig


def collect():
    rows = []
    for window_key, window_label in WINDOWS:
        xcirc_folder = FOLDERS[(window_key, "xcirc")]
        xshuffle_folder = FOLDERS[(window_key, "xshuffle")]
        for cond_key, cond_label in CONDITIONS:
            n_xcirc, train, test, sig_xcirc = per_patient_means(xcirc_folder, cond_key)
            n_xshuf, _, _, sig_xshuffle = per_patient_means(xshuffle_folder, cond_key)
            rows.append(
                {
                    "window": window_label,
                    "condition": cond_label,
                    "n_xcirc": n_xcirc,
                    "n_xshuffle": n_xshuf,
                    "train": mean_sem(train),
                    "test": mean_sem(test),
                    "sig_xcirc": mean_sem(sig_xcirc),
                    "sig_xshuffle": mean_sem(sig_xshuffle),
                }
            )
    return rows


def fmt(v, d=3):
    m, s = v
    return f"{m:.{d}f} \\pm {s:.{d}f}"


def write_tex(rows, out_path):
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{Overall semantic encoding strength in hippocampus for fixed "
        r"windows versus variable word-duration windows, 15 patients, BERT L12, "
        r"20 PCs, shuffled outer CV. Word-duration models use duration as a "
        r"Poisson exposure offset.}",
        r"\label{tab:fixed_vs_worddur_encoding_matched_bert}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Window & Condition & Train $R^2$ & Test $R^2$ & Sig.\ shuffle-CV + xcirc & Sig.\ shuffle-CV + xshuffle \\",
        r"\midrule",
    ]
    previous_window = None
    for row in rows:
        window = row["window"] if row["window"] != previous_window else ""
        if previous_window is not None and window:
            lines.append(r"\addlinespace")
        lines.append(
            f"{window} & {row['condition']} & "
            f"${fmt(row['train'])}$ & ${fmt(row['test'])}$ & "
            f"${fmt(row['sig_xcirc'], 1)}\\%$ & ${fmt(row['sig_xshuffle'], 1)}\\%$ \\\\"
        )
        previous_window = row["window"]
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{flushleft}\footnotesize",
            r"Values are per-patient means $\pm$ SEM. Train and test $R^2$ values "
            r"come from the real-data shuffled outer CV fits. Significance columns "
            r"report raw, uncorrected neuron-level $p<0.05$ under the same shuffled "
            r"outer CV, using either the calibrated circular-shift null (xcirc) or "
            r"a random row-shuffle feature null (xshuffle). This table uses matched "
            r"PC count and shuffled outer CV, matching the GPT-2 XL L48 version "
            r"(fixed\_vs\_worddur\_encoding\_matched\_table.tex).",
            r"\end{flushleft}",
            r"\end{table}",
            "",
        ]
    )
    out_path.write_text("\n".join(lines))


def main():
    rows = collect()
    for r in rows:
        print(r["window"], r["condition"], "n_xcirc=", r["n_xcirc"], "n_xshuffle=", r["n_xshuffle"])
    out_tex = RESULTS / "fixed_vs_worddur_encoding_matched_table_bert.tex"
    write_tex(rows, out_tex)
    print(f"wrote {out_tex}")

    out_csv = RESULTS / "fixed_vs_worddur_encoding_matched_table_bert.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
