#!/usr/bin/env python3
"""Merge isolated CSV summaries produced by a window-sweep SLURM array."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_ids", nargs="+", help="One or more SLURM array job IDs")
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/scratch/aniluchavez/ConvoDATAS/SemanticGLM/plots/window_sweep_tasks"
        ),
    )
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    task_dirs = [args.root / str(job_id) for job_id in args.job_ids]
    files = [
        path
        for task_dir in task_dirs
        for path in sorted(task_dir.glob("*.csv"), key=lambda p: int(p.stem))
    ]
    if not files:
        raise SystemExit(f"No task summaries found in: {task_dirs}")

    frames = [pd.read_csv(path) for path in files]
    merged = pd.concat(frames, ignore_index=True)
    output = args.output or args.root / f"{'_'.join(args.job_ids)}_merged.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    print(f"Merged {len(files)} task files ({len(merged)} rows): {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
