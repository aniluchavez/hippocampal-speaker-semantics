#!/usr/bin/env python3
"""Build fixed-window r_cross table from semantic_glm reliability outputs."""

from __future__ import annotations

import argparse
import pickle
import subprocess
from pathlib import Path

import numpy as np


RESULT_ROOT = Path("/scratch/aniluchavez/ConvoDATAS/SemanticGLM")
PROJECT = Path("/scratch/aniluchavez/hippocampal-speaker-semantics")
RESULTS_DIR = PROJECT / "results"

WINDOWS = [
    ("fixed_selfm200_otherp100_len500", -200, 100),
    ("fixed_selfm300_otherp20_len500", -300, 20),
    ("fixed_selfm100_other0_len500", -100, 0),
    ("fixed_self0_other0_len500", 0, 0),
    ("fixed_self0_otherp200_len500", 0, 200),
    ("fixed_selfm300_otherp200_len500", -300, 200),
]


def folder(model: str, context: str, tag: str, pc: int) -> Path:
    return RESULT_ROOT / f"{model}{context}_{tag}_notebookexact_shuf_xcirc_r2only" / f"pc{pc}"


def mean_sem(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    sem = arr.std(ddof=1) / np.sqrt(arr.size) if arr.size > 1 else 0.0
    return float(arr.mean()), float(sem)


def collect_window(model: str, context: str, layer: int, pc: int, tag: str) -> dict:
    paths = sorted(folder(model, context, tag, pc).glob(f"*_L{layer:02d}_sem.pkl"))
    paths = [p for p in paths if not p.name.startswith("L")]
    if not paths:
        raise FileNotFoundError(f"No patient PKLs for {model}{context} {tag} L{layer} pc{pc}")

    patient_rcross = []
    patient_ceiling = []
    n_patients = 0
    n_neurons = 0
    for path in paths:
        with path.open("rb") as handle:
            obj = pickle.load(handle)
        rel = obj.get("reliability", {}).get("hippocampus")
        if not rel:
            raise KeyError(f"{path} lacks reliability['hippocampus']")
        rc = np.asarray([r["r_cross"] for r in rel], dtype=float)
        ce = np.asarray([r["ceil_mean"] for r in rel], dtype=float)
        patient_rcross.append(float(np.nanmean(rc)))
        patient_ceiling.append(float(np.nanmean(ce)))
        n_patients += 1
        n_neurons += len(rel)

    rc_m, rc_sem = mean_sem(patient_rcross)
    ce_m, ce_sem = mean_sem(patient_ceiling)
    return {
        "r_cross": (rc_m, rc_sem),
        "ceiling": (ce_m, ce_sem),
        "ratio": rc_m / ce_m if np.isfinite(rc_m) and np.isfinite(ce_m) and ce_m != 0 else np.nan,
        "n_patients": n_patients,
        "n_neurons": n_neurons,
    }


def fmt_pm(pair: tuple[float, float]) -> str:
    return f"{pair[0]:.3f} \\pm {pair[1]:.3f}"


def ms(x: int) -> str:
    return f"${x:+d}$ ms" if x != 0 else "$0$ ms"


def write_tex(args, rows: list[dict], out_tex: Path) -> None:
    display = args.display_model
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        (
            r"\caption{Self/other beta-vector correlation "
            r"($r_\mathrm{cross}$) across fixed-window placements in hippocampus, "
            f"{rows[0]['n_patients']} patients, {display} L{args.layer}, {args.pc} PCs. "
            r"Window length was held fixed at 500 ms; starts are relative to word onset.}"
        ),
        rf"\label{{tab:{args.label}}}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Speaking start & Listening start & $r_\mathrm{cross}$ & Noise ceiling & Ratio \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{ms(row['self_start'])} & {ms(row['other_start'])} & "
            f"${fmt_pm(row['r_cross'])}$ & ${fmt_pm(row['ceiling'])}$ & "
            f"{row['ratio']:.2f} \\\\"
        )
    rc_vals = [r["r_cross"][0] for r in rows]
    ratio_vals = [r["ratio"] for r in rows]
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{flushleft}\footnotesize",
        (
            r"Values are per-patient means $\pm$ SEM. "
            r"$r_\mathrm{cross}=\mathrm{corr}(\beta_\mathrm{self},\beta_\mathrm{other})$ "
            r"per neuron; the noise ceiling is "
            r"$\sqrt{\mathrm{SB}(r_\mathrm{self})\cdot\mathrm{SB}(r_\mathrm{other})}$ "
            r"from within-condition split-half reliability. "
            f"Across the tested timing range, $r_\\mathrm{{cross}}$ spans "
            f"{np.nanmin(rc_vals):.3f}--{np.nanmax(rc_vals):.3f} and the "
            f"ceiling-normalized ratio spans {np.nanmin(ratio_vals):.2f}--"
            f"{np.nanmax(ratio_vals):.2f}."
        ),
        r"\end{flushleft}",
        r"\end{table}",
    ])
    out_tex.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base")
    ap.add_argument("--context_tag", default="_ctx200")
    ap.add_argument("--display_model", default="BERT-base")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--pc", type=int, default=100)
    ap.add_argument("--label", default="bert_L12_window_shift_rcross")
    ap.add_argument("--output_stem", default="bert_L12_window_shift_rcross_table")
    ap.add_argument("--make_docx", action="store_true")
    args = ap.parse_args()

    rows = []
    for tag, self_start, other_start in WINDOWS:
        stats = collect_window(args.model, args.context_tag, args.layer, args.pc, tag)
        rows.append({"tag": tag, "self_start": self_start, "other_start": other_start, **stats})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_tex = RESULTS_DIR / f"{args.output_stem}.tex"
    write_tex(args, rows, out_tex)
    print(out_tex)

    if args.make_docx:
        out_docx = RESULTS_DIR / f"{args.output_stem}.docx"
        subprocess.run(
            [
                "/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3",
                str(PROJECT / "scripts" / "tex_table_to_docx.py"),
                str(out_tex),
                str(out_docx),
            ],
            check=True,
        )
        print(out_docx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
