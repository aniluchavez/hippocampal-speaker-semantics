#!/usr/bin/env python3
"""Broad fixed- or variable-window semantic-GLM screen.

Fixed mode uses the same fixed duration for self and other while allowing
different onset-relative starts. Variable mode allows separate onset-relative
starts and offset-relative ends. This is an R²-only discovery screen; run the
full permutation null only on the best window(s).
"""

from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
from pathlib import Path

import pandas as pd


PYTHON = Path("/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3")
PROJECT = Path("/scratch/aniluchavez/hippocampal-speaker-semantics")
RESULT_ROOT = Path("/scratch/aniluchavez/ConvoDATAS/SemanticGLM")
PLOTS_DIR = RESULT_ROOT / "plots"


def fmt_shift(value: int) -> str:
    if value < 0:
        return f"m{abs(value)}"
    if value == 0:
        return "0"
    return f"p{value}"


def tag_for(
    mode: str,
    self_start: int,
    other_start: int,
    *,
    length: int | None = None,
    self_end_shift: int | None = None,
    other_end_shift: int | None = None,
) -> str:
    if mode == "fixed":
        return (
            f"fixed_self{fmt_shift(self_start)}_other{fmt_shift(other_start)}"
            f"_len{length}"
        )
    return (
        f"varwin_self{fmt_shift(self_start)}tooffset{fmt_shift(self_end_shift or 0)}"
        f"_other{fmt_shift(other_start)}tooffset{fmt_shift(other_end_shift or 0)}"
    )


def run_cmd(cmd: list[str], *, cwd: Path, dry_run: bool = False) -> None:
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def result_pkl(tag: str, model: str, context_tag: str, layer: int, pc: int, patient: str) -> Path:
    folder = (
        RESULT_ROOT
        / f"{model}{context_tag}_{tag}_notebookexact_shuf_xcirc_r2only"
        / f"pc{pc}"
    )
    return folder / f"{patient}_L{layer:02d}_sem.pkl"


def summarize_one(
    pkl_path: Path,
    *,
    patient: str,
    mode: str,
    self_start: int,
    other_start: int,
    length: int | None,
    self_end_shift: int | None,
    other_end_shift: int | None,
    tag: str,
):
    if not pkl_path.exists():
        return []
    with pkl_path.open("rb") as f:
        obj = pickle.load(f)
    df = obj["df"] if isinstance(obj, dict) and "df" in obj else obj
    df.to_csv(pkl_path.with_name(pkl_path.name.replace("_sem.pkl", "_sem_results.csv")), index=False)
    rows = []
    for cond, g in df.groupby("condition"):
        rows.append(
            {
                "patient": patient,
                "window_mode": mode,
                "length_ms": length,
                "self_start_ms": self_start,
                "self_end_shift_ms": self_end_shift,
                "other_start_ms": other_start,
                "other_end_shift_ms": other_end_shift,
                "tag": tag,
                "condition": cond,
                "n_neurons": len(g),
                "median_r2": float(g["r2"].median()),
                "mean_r2": float(g["r2"].mean()),
                "n_r2_positive": int((g["r2"] > 0).sum()),
                "median_train_r2": float(g["r2_train"].median()),
                "mean_train_r2": float(g["r2_train"].mean()),
                "median_alpha": float(g["best_alpha"].median()),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="PTYEU_task147")
    ap.add_argument("--region", default="hippocampus")
    ap.add_argument("--model", default="gpt2-large")
    ap.add_argument("--context_tag", default="_ctx200")
    ap.add_argument("--layer", type=int, default=36)
    ap.add_argument("--pc", type=int, default=50)
    ap.add_argument("--window_mode", choices=["fixed", "varwin"], default="fixed")
    ap.add_argument("--lengths", default="200,300,500")
    ap.add_argument("--self_starts", default="-500,-400,-300,-200,-100,0")
    ap.add_argument("--other_starts", default="0,20,100,200")
    ap.add_argument(
        "--self_end_shifts",
        default="0",
        help="Variable mode only: comma-separated shifts relative to word offset",
    )
    ap.add_argument(
        "--other_end_shifts",
        default="100",
        help="Variable mode only: comma-separated shifts relative to word offset",
    )
    ap.add_argument(
        "--summary_path",
        type=Path,
        default=None,
        help="Optional output CSV path (useful for isolated SLURM-array tasks)",
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",") if x]
    self_starts = [int(x) for x in args.self_starts.split(",") if x]
    other_starts = [int(x) for x in args.other_starts.split(",") if x]
    self_end_shifts = [int(x) for x in args.self_end_shifts.split(",") if x]
    other_end_shifts = [int(x) for x in args.other_end_shifts.split(",") if x]

    if args.window_mode == "fixed":
        configs = [
            (length, self_start, other_start, None, None)
            for length in lengths
            for self_start in self_starts
            for other_start in other_starts
        ]
        summary_suffix = "fixed_same_length_broad_sweep_r2only.csv"
    else:
        configs = [
            (None, self_start, other_start, self_end, other_end)
            for self_start in self_starts
            for other_start in other_starts
            for self_end in self_end_shifts
            for other_end in other_end_shifts
        ]
        summary_suffix = "varwin_broad_sweep_r2only.csv"

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary_path or (
        PLOTS_DIR
        / f"{args.patient}_{args.model.replace('-', '')}_L{args.layer}_pc{args.pc}"
        f"_{summary_suffix}"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    total = len(configs)
    done = 0
    print(f"Broad {args.window_mode}-window sweep: {total} configs", flush=True)
    print(f"summary_path={summary_path}", flush=True)

    for length, self_start, other_start, self_end, other_end in configs:
        done += 1
        tag = tag_for(
            args.window_mode,
            self_start,
            other_start,
            length=length,
            self_end_shift=self_end,
            other_end_shift=other_end,
        )
        pkl_path = result_pkl(
            tag, args.model, args.context_tag, args.layer, args.pc, args.patient
        )
        if args.window_mode == "fixed":
            window_desc = f"length={length} self={self_start} other={other_start}"
        else:
            window_desc = (
                f"self=[onset{self_start:+d}, offset{self_end:+d}] "
                f"other=[onset{other_start:+d}, offset{other_end:+d}]"
            )
        print(f"\n[{done}/{total}] {window_desc} tag={tag}", flush=True)

        if args.force or not pkl_path.exists():
            gen_cmd = [
                str(PYTHON),
                "-u",
                "scripts/generate_spike_windows.py",
                "--mode",
                args.window_mode,
                "--target_shift",
                str(self_start),
                "--other_shift",
                str(other_start),
                "--out_tag",
                tag,
                "--patients",
                args.patient,
                "--clip_bounds",
            ]
            if args.window_mode == "fixed":
                gen_cmd.extend(
                    [
                        "--target_window_length",
                        str(length),
                        "--other_window_length",
                        str(length),
                    ]
                )
            else:
                gen_cmd.extend(
                    [
                        "--target_end_shift",
                        str(self_end),
                        "--other_end_shift",
                        str(other_end),
                    ]
                )
            if args.force:
                gen_cmd.append("--force")
            run_cmd(gen_cmd, cwd=PROJECT, dry_run=args.dry_run)

            glm_cmd = [
                str(PYTHON),
                "-u",
                "scripts/semantic_glm.py",
                "--layer",
                str(args.layer),
                "--patient",
                args.patient,
                "--region",
                args.region,
                "--window_type",
                args.window_mode,
                "--spike_tag",
                tag,
                "--model",
                args.model,
                "--context_tag",
                args.context_tag,
                "--n_components",
                str(args.pc),
                "--outer_cv",
                "shuffle",
                "--force_all",
                "--notebook_exact",
                "--r2_only",
            ]
            run_cmd(glm_cmd, cwd=PROJECT, dry_run=args.dry_run)
        else:
            print("  existing result found; summarizing", flush=True)

        rows = summarize_one(
            pkl_path,
            patient=args.patient,
            mode=args.window_mode,
            self_start=self_start,
            other_start=other_start,
            length=length,
            self_end_shift=self_end,
            other_end_shift=other_end,
            tag=tag,
        )
        all_rows.extend(rows)
        if all_rows:
            out = pd.DataFrame(all_rows)
            duplicate_cols = ["self_start_ms", "other_start_ms", "condition"]
            display_cols = ["self_start_ms", "other_start_ms", "median_r2"]
            if args.window_mode == "fixed":
                duplicate_cols.insert(0, "length_ms")
                display_cols.insert(0, "length_ms")
            else:
                duplicate_cols[1:1] = ["self_end_shift_ms", "other_end_shift_ms"]
                display_cols[1:1] = ["self_end_shift_ms", "other_end_shift_ms"]
            out = out.drop_duplicates(duplicate_cols, keep="last")
            out.to_csv(summary_path, index=False)
            best_self = (
                out[out["condition"].eq("self")]
                .sort_values("median_r2", ascending=False)
                .head(3)
            )
            best_other = (
                out[out["condition"].eq("other")]
                .sort_values("median_r2", ascending=False)
                .head(3)
            )
            print("  current top self:", flush=True)
            print(best_self[display_cols].to_string(index=False), flush=True)
            print("  current top other:", flush=True)
            print(best_other[display_cols].to_string(index=False), flush=True)

    print(f"\nDone. Summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
