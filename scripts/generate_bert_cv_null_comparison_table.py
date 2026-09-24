#!/usr/bin/env python3
"""Fixed-window table comparing CV method x null type: block CV with xcirc,
block CV with xshuffle, shuffle CV with xcirc, and shuffle CV with xshuffle. Same window
(self=[onset-200,+300], other=[onset+100,+600]), PC=20, hippocampus.

Reads per-patient neuron-level pkls from the semantic_glm.py output
folders and reports per-patient mean +/- SEM train/test R2 (over
raw-significant neurons only) and raw significant-neuron percentage.

Usage:
  python3 generate_bert_cv_null_comparison_table.py --model bert-base --layer 12
  python3 generate_bert_cv_null_comparison_table.py --model gpt2-xl --layer 48
"""
import argparse
import glob
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

GLM_ROOT = Path("/scratch/aniluchavez/ConvoDATAS/SemanticGLM")
RESULTS = Path("/scratch/aniluchavez/hippocampal-speaker-semantics/results")
PC = 20

ROWS = [
    ("block", "xcirc", "Block CV, xcirc null"),
    ("block", "xshuffle", "Block CV, xshuffle null"),
    ("shuffle", "xcirc", "Shuffle CV, xcirc null"),
    ("shuffle", "xshuffle", "Shuffle CV, xshuffle null"),
]
CONDITIONS = [("self", "Speaking (self)"), ("other", "Listening (other)")]


def folders_for(model):
    base = f"{model}_ctx200_fixed_selfm200_otherp100_len500_notebookexact"
    return {
        ("block", "xcirc"): f"{base}_xcirc",
        ("block", "xshuffle"): f"{base}_xshuffle",
        ("shuffle", "xcirc"): f"{base}_shuf_xcirc",
        ("shuffle", "xshuffle"): f"{base}_shuf_xshuffle",
    }


def mean_sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return np.mean(x), (np.std(x, ddof=1) / np.sqrt(x.size) if x.size > 1 else 0.0)


def per_patient_means(folder_name, cond, layer):
    folder = GLM_ROOT / folder_name / f"pc{PC}"
    files = sorted(glob.glob(str(folder / f"PTY*_L{layer:02d}_sem.pkl")))
    train, test, sig = [], [], []
    for f in files:
        obj = pickle.load(open(f, "rb"))
        df = obj["df"]
        sub = df[(df["region"] == "hippocampus") & (df["condition"] == cond)]
        if sub.empty:
            continue
        sig_sub = sub[sub["raw_significant"]]
        train.append(np.nanmean(sig_sub["r2_train"]) if not sig_sub.empty else np.nan)
        test.append(np.nanmean(sig_sub["r2"]) if not sig_sub.empty else np.nan)
        sig.append(100 * np.mean(sub["raw_significant"]))
    return len(files), train, test, sig


def collect(model, layer):
    folders = folders_for(model)
    rows = []
    for cv_key, perm_key, row_label in ROWS:
        folder = folders[(cv_key, perm_key)]
        for cond_key, cond_label in CONDITIONS:
            n, train, test, sig = per_patient_means(folder, cond_key, layer)
            rows.append(
                {
                    "row": row_label,
                    "condition": cond_label,
                    "n": n,
                    "train": mean_sem(train),
                    "test": mean_sem(test),
                    "sig": mean_sem(sig),
                }
            )
    return rows


def fmt(v, d=3):
    m, s = v
    return f"{m:.{d}f} \\pm {s:.{d}f}"


def write_tex(rows, out_path, model_label, layer):
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{Semantic encoding strength in hippocampus for fixed windows, "
        f"{model_label} L{layer}, 15 patients, 20 PCs, comparing outer CV method "
        r"and null type. Block CV uses contiguous temporal folds; shuffle CV "
        r"uses random KFold. The xcirc null is the calibrated circular-shift "
        r"null; xshuffle is a naive random row-shuffle null. Train/Test $R^2$ "
        r"are computed over raw-significant neurons only.}",
        r"\label{tab:" + model_label.lower().replace(" ", "").replace("-", "") + "_cv_null_comparison}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"CV / null & Condition & Train $R^2$ & Test $R^2$ & Sig.\ raw $p<0.05$ \\",
        r"\midrule",
    ]
    previous_row = None
    for row in rows:
        label = row["row"] if row["row"] != previous_row else ""
        if previous_row is not None and label:
            lines.append(r"\addlinespace")
        lines.append(
            f"{label} & {row['condition']} & "
            f"${fmt(row['train'])}$ & ${fmt(row['test'])}$ & "
            f"${fmt(row['sig'], 1)}\\%$ \\\\"
        )
        previous_row = row["row"]
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{flushleft}\footnotesize",
            r"Values are per-patient means $\pm$ SEM. Significance column reports "
            r"raw, uncorrected neuron-level $p<0.05$. Shuffle CV + xshuffle "
            r"combines the two least-conservative choices and should not be used "
            r"to judge real effect size. The shuffle CV + xcirc row isolates the "
            r"effect of random CV while retaining the calibrated circular-shift "
            r"null; block CV + xcirc is the most conservative reference.",
            r"\end{flushleft}",
            r"\end{table}",
            "",
        ]
    )
    out_path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--model_label", default=None)
    ap.add_argument("--out_stem", default=None)
    args = ap.parse_args()

    model_label = args.model_label or ("GPT-2 XL" if args.model == "gpt2-xl" else "BERT-base")
    default_stem = ("bert" if args.model == "bert-base" else args.model.replace("-", "_")) + "_cv_null_comparison_table"
    out_stem = args.out_stem or default_stem

    rows = collect(args.model, args.layer)
    for r in rows:
        print(r["row"], r["condition"], "n=", r["n"], "sig=", r["sig"])

    out_tex = RESULTS / f"{out_stem}.tex"
    write_tex(rows, out_tex, model_label, args.layer)
    print(f"wrote {out_tex}")

    out_csv = RESULTS / f"{out_stem}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
