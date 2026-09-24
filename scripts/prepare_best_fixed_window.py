#!/usr/bin/env python3
"""Select the best all-patient fixed windows and generate their spike counts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd


PYTHON = "/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3"
PROJECT = Path("/scratch/aniluchavez/hippocampal-speaker-semantics")


def fmt_shift(value: int) -> str:
    if value < 0:
        return f"m{abs(value)}"
    if value > 0:
        return f"p{value}"
    return "0"


def select_best(df: pd.DataFrame, condition: str) -> dict:
    if condition == "self":
        keys = ["length_ms", "self_start_ms"]
    else:
        keys = ["length_ms", "other_start_ms"]

    patient_level = (
        df.loc[df["condition"].eq(condition)]
        .groupby(["patient", *keys], as_index=False)
        .agg(patient_median_r2=("median_r2", "mean"))
    )
    across_patients = (
        patient_level.groupby(keys, as_index=False)
        .agg(
            median_patient_r2=("patient_median_r2", "median"),
            mean_patient_r2=("patient_median_r2", "mean"),
            n_patients=("patient", "nunique"),
        )
        .sort_values(
            ["median_patient_r2", "mean_patient_r2"],
            ascending=[False, False],
        )
    )
    return across_patients.iloc[0].to_dict()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "/scratch/aniluchavez/ConvoDATAS/SemanticGLM/plots/"
            "all15_gpt2large_L36_pc50_fixed_same_length_broad_sweep_r2only.csv"
        ),
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(
            "/scratch/aniluchavez/ConvoDATAS/SemanticGLM/plots/"
            "best_fixed_window_all15.json"
        ),
    )
    args = ap.parse_args()

    df = pd.read_csv(args.summary)
    if "patient" not in df:
        raise ValueError("Summary lacks a patient column")
    df["patient"] = df["patient"].fillna("PTYEU_task147")
    self_best = select_best(df, "self")
    other_best = select_best(df, "other")

    self_start = int(self_best["self_start_ms"])
    self_length = int(self_best["length_ms"])
    other_start = int(other_best["other_start_ms"])
    other_length = int(other_best["length_ms"])
    tag = (
        f"bestfixed_self{fmt_shift(self_start)}_len{self_length}"
        f"_other{fmt_shift(other_start)}_len{other_length}"
    )
    config = {
        "spike_tag": tag,
        "self_start_ms": self_start,
        "self_length_ms": self_length,
        "self_median_patient_r2": float(self_best["median_patient_r2"]),
        "other_start_ms": other_start,
        "other_length_ms": other_length,
        "other_median_patient_r2": float(other_best["median_patient_r2"]),
        "n_patients": int(
            min(self_best["n_patients"], other_best["n_patients"])
        ),
    }
    args.config.parent.mkdir(parents=True, exist_ok=True)
    args.config.write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2), flush=True)

    cmd = [
        PYTHON,
        "-u",
        "scripts/generate_spike_windows.py",
        "--mode",
        "fixed",
        "--target_shift",
        str(self_start),
        "--target_window_length",
        str(self_length),
        "--other_shift",
        str(other_start),
        "--other_window_length",
        str(other_length),
        "--out_tag",
        tag,
        "--clip_bounds",
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT, check=True)
    print(f"Prepared spike tag: {tag}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
