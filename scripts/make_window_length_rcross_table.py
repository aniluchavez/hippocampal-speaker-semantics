#!/usr/bin/env python3
"""Build fixed-window-LENGTH r_cross table from semantic_glm reliability outputs.

Companion to make_window_shift_rcross_table.py: that script sweeps window
PLACEMENT at a fixed 500ms length; this one holds placement fixed at the
"bestfixed" self=-300ms/other=+20ms start and sweeps window LENGTH instead.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

from make_window_shift_rcross_table import collect_window, fmt_pm

PROJECT = Path("/scratch/aniluchavez/hippocampal-speaker-semantics")
RESULTS_DIR = PROJECT / "results"

SELF_START = -300
OTHER_START = 20
LENGTHS = [200, 300, 500, 700, 800]


def write_tex(args, rows: list[dict], out_tex: Path) -> None:
    display = args.display_model
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        (
            r"\caption{Self/other beta-vector correlation "
            r"($r_\mathrm{cross}$) across fixed window lengths in hippocampus, "
            f"{rows[0]['n_patients']} patients, {display} L{args.layer}, {args.pc} PCs. "
            r"Window placement was held fixed at speaking start $-300$ ms, "
            r"listening start $+20$ ms (relative to word onset); only the "
            r"shared window length varied.}"
        ),
        rf"\label{{tab:{args.label}}}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Window length & $r_\mathrm{cross}$ & Noise ceiling & Ratio \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"${row['length']}$ ms & "
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
            f"Across the tested lengths, $r_\\mathrm{{cross}}$ spans "
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
    ap.add_argument("--label", default="bert_L12_window_length_rcross")
    ap.add_argument("--output_stem", default="bert_L12_window_length_rcross_table")
    ap.add_argument("--make_docx", action="store_true")
    args = ap.parse_args()

    rows = []
    for length in LENGTHS:
        tag = f"fixed_selfm300_otherp20_len{length}"
        stats = collect_window(args.model, args.context_tag, args.layer, args.pc, tag)
        rows.append({"tag": tag, "length": length, **stats})

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
