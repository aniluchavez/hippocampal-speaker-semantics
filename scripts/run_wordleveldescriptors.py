#!/usr/bin/env python3
"""
Rerun the core descriptor analyses from notebooks/wordleveldescriptors.ipynb.

This version uses the current transcript source instead of the old local Mac paths:
  /scratch/aniluchavez/ConvoDATAS/Transcripts

Outputs are written under results/wordleveldescriptors by default.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PATIENTS = [
    "PTYEU_task147",
    "PTYEV_task37",
    "PTYEY_task86",
    "PTYEZ_task60",
    "PTYFA_task25",
    "PTYFC_task28",
    "PTYFF_task17",
    "PTYFG_task18",
    "PTYFI_task81",
    "PTYFK_task40",
    "PTYFM_task104",
    "PTYFP_task88",
    "PTYFR_task91",
    "PTYFS_task95",
    "PTYFU_task224",
]
NEW5 = {"PTYFM_task104", "PTYFP_task88", "PTYFR_task91", "PTYFS_task95", "PTYFU_task224"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transcripts-root",
        type=Path,
        default=Path("/scratch/aniluchavez/ConvoDATAS/Transcripts"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/scratch/aniluchavez/hippocampal-speaker-semantics/results/wordleveldescriptors"),
    )
    parser.add_argument("--patients", nargs="+", default=DEFAULT_PATIENTS)
    parser.add_argument("--target-speaker", default="Speaker1")
    return parser.parse_args()


def find_excel(root: Path, patient_id: str) -> Path:
    matches = sorted(
        p for p in root.glob(f"{patient_id}*.xlsx")
        if not p.name.startswith(("~$", "._")) and p.exists()
    )
    if not matches:
        raise FileNotFoundError(f"No transcript .xlsx found for {patient_id}: {root}")
    # Prefer the manually updated Newest transcript when both New and Newest exist.
    return sorted(matches, key=lambda p: ("Newest" not in p.name, p.name))[0]


def speaker_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if re.match(r"^Speaker\d+$", str(c))]


def normalize_token(token: object) -> str:
    text = str(token).strip().lower()
    text = re.sub(r"^[^\w']+|[^\w']+$", "", text)
    return text


def load_long_words(excel_path: Path, patient_id: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path)
    spk_cols = speaker_columns(df)
    if not spk_cols:
        raise ValueError(f"No Speaker# columns found in {excel_path}; columns={list(df.columns)}")

    id_vars = [c for c in ["onset", "offset", "Duration", "regress_dur"] if c in df.columns]
    long_df = df.melt(
        id_vars=id_vars,
        value_vars=spk_cols,
        var_name="speaker",
        value_name="word_token",
    )
    long_df = long_df.dropna(subset=["word_token"]).copy()
    long_df["word"] = long_df["word_token"]
    long_df["patient_id"] = patient_id
    long_df["is_new5"] = patient_id in NEW5
    long_df["source_excel"] = str(excel_path)
    if "Duration" in long_df.columns:
        long_df["Duration"] = pd.to_numeric(long_df["Duration"], errors="coerce")
    long_df["token_norm"] = long_df["word_token"].map(normalize_token)
    long_df = long_df[long_df["token_norm"] != ""].reset_index(drop=True)
    return long_df


def wpm_summary(words: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for patient_id, sub in words.groupby("patient_id", sort=False):
        dur = pd.to_numeric(sub["Duration"], errors="coerce").dropna()
        total_words = int(len(dur))
        total_dur_ms = float(dur.sum())
        mean_dur_ms = float(dur.mean()) if total_words else np.nan
        rows.append(
            {
                "patient_id": patient_id,
                "is_new5": patient_id in NEW5,
                "n_words": total_words,
                "total_dur_ms": total_dur_ms,
                "total_dur_min": total_dur_ms / 60000.0,
                "mean_dur_ms": mean_dur_ms,
                "median_dur_ms": float(dur.median()) if total_words else np.nan,
                "wpm_by_sum": total_words / (total_dur_ms / 60000.0) if total_dur_ms > 0 else np.nan,
                "wpm_by_mean": 60000.0 / mean_dur_ms if mean_dur_ms > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def vocab_summary(words: pd.DataFrame, target_speaker: str) -> tuple[pd.DataFrame, dict]:
    rows = []
    overall_self_vocab = set()
    overall_other_vocab = set()
    shared_counter = Counter()

    for patient_id, sub in words.groupby("patient_id", sort=False):
        self_tokens = sub.loc[sub["speaker"] == target_speaker, "token_norm"].tolist()
        other_tokens = sub.loc[sub["speaker"] != target_speaker, "token_norm"].tolist()
        unique_self = set(self_tokens)
        unique_other = set(other_tokens)
        shared = unique_self & unique_other
        overall_self_vocab.update(unique_self)
        overall_other_vocab.update(unique_other)
        shared_counter.update([t for t in self_tokens + other_tokens if t in shared])
        rows.append(
            {
                "patient_id": patient_id,
                "is_new5": patient_id in NEW5,
                "n_self_tokens": len(self_tokens),
                "n_other_tokens": len(other_tokens),
                "n_unique_self": len(unique_self),
                "n_unique_other": len(unique_other),
                "ttr_self": len(unique_self) / len(self_tokens) if self_tokens else np.nan,
                "ttr_other": len(unique_other) / len(other_tokens) if other_tokens else np.nan,
                "n_shared_unique": len(shared),
                "n_shared_occurrences": sum(1 for t in self_tokens + other_tokens if t in shared),
            }
        )

    overall = {
        "target_speaker": target_speaker,
        "n_patients": int(words["patient_id"].nunique()),
        "n_new5_patients": int(words.loc[words["is_new5"], "patient_id"].nunique()),
        "overall_unique_self": len(overall_self_vocab),
        "overall_unique_other": len(overall_other_vocab),
        "overall_shared_unique": len(overall_self_vocab & overall_other_vocab),
        "top_shared_tokens": shared_counter.most_common(50),
    }
    return pd.DataFrame(rows), overall


def turn_summary(words: pd.DataFrame, target_speaker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_turns = []
    for patient_id, sub in words.groupby("patient_id", sort=False):
        sort_cols = [c for c in ["onset", "offset"] if c in sub.columns]
        long_df = sub.sort_values(sort_cols).reset_index(drop=True) if sort_cols else sub.reset_index(drop=True)
        long_df["speaker_shift"] = long_df["speaker"] != long_df["speaker"].shift()
        long_df["turn_id"] = long_df["speaker_shift"].cumsum()
        grouped = (
            long_df.groupby("turn_id", as_index=False)
            .agg(speaker=("speaker", "first"), onset=("onset", "min"), n_words=("word", "count"))
        )
        grouped["patient_id"] = patient_id
        grouped["is_new5"] = patient_id in NEW5
        grouped["role"] = np.where(grouped["speaker"] == target_speaker, "self", "other")
        all_turns.append(grouped)

    turns = pd.concat(all_turns, ignore_index=True)
    role_summary = (
        turns.groupby(["role"], as_index=False)["n_words"]
        .agg(n_turns="count", mean_n_words="mean", median_n_words="median", std_n_words="std")
    )
    return turns, role_summary


def save_figures(words: pd.DataFrame, turns: pd.DataFrame, out_dir: Path) -> None:
    plt.rcParams["font.family"] = "Arial"

    durations = pd.to_numeric(words["Duration"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist(durations, bins=70, color="#64c6c2", edgecolor="black")
    ax.set_xlim(0, 1200)
    ax.set_xlabel("Duration (ms)", fontsize=20)
    ax.set_ylabel("Count", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "word_duration_histogram.png", dpi=300)
    fig.savefig(out_dir / "word_duration_histogram.eps", format="eps")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist(turns["n_words"], bins=100, edgecolor="black", color="#CC5500")
    ax.set_xlim(0, 80)
    ax.set_xlabel("Words per speaker turn", fontsize=20)
    ax.set_ylabel("Number of turns", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "all_patients_turn_histogram.png", dpi=300)
    fig.savefig(out_dir / "all_patients_turn_histogram.eps", format="eps")
    plt.close(fig)

    sorted_sizes = np.sort(turns["n_words"].to_numpy())
    cumulative = np.arange(1, len(sorted_sizes) + 1) / len(sorted_sizes)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(sorted_sizes, cumulative, color="#CC5500")
    ax.set_xlim(0, 125)
    ax.set_xlabel("Words per speaker turn", fontsize=20)
    ax.set_ylabel("Cumulative fraction of turns", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_dir / "all_patients_turn_cdf.png", dpi=300)
    fig.savefig(out_dir / "all_patients_turn_cdf.eps", format="eps")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    long_frames = []
    manifest = []
    for patient_id in args.patients:
        excel = find_excel(args.transcripts_root, patient_id)
        print(f"reading {patient_id}: {excel}", flush=True)
        long_df = load_long_words(excel, patient_id)
        long_frames.append(long_df)
        manifest.append(
            {
                "patient_id": patient_id,
                "is_new5": patient_id in NEW5,
                "source_excel": str(excel),
                "n_words": int(len(long_df)),
                "speakers": sorted(long_df["speaker"].unique().tolist()),
            }
        )

    words = pd.concat(long_frames, ignore_index=True)
    words.to_csv(args.out_dir / "all_patients_words_durations.csv", index=False)
    pd.DataFrame(manifest).to_csv(args.out_dir / "manifest.csv", index=False)

    wpm = wpm_summary(words)
    wpm.to_csv(args.out_dir / "per_patient_wpm_summary.csv", index=False)

    vocab, overall_vocab = vocab_summary(words, args.target_speaker)
    vocab.to_csv(args.out_dir / "per_patient_role_vocab_summary.csv", index=False)
    (args.out_dir / "overall_role_vocab_summary.json").write_text(
        json.dumps(overall_vocab, indent=2) + "\n"
    )

    turns, role_turns = turn_summary(words, args.target_speaker)
    turns.to_csv(args.out_dir / "all_patients_turn_summary.csv", index=False)
    role_turns.to_csv(args.out_dir / "turn_summary_by_role.csv", index=False)

    save_figures(words, turns, args.out_dir)

    print("\nDone.")
    print(f"patients: {words['patient_id'].nunique()} total, {words.loc[words['is_new5'], 'patient_id'].nunique()} new5")
    print(f"words: {len(words):,}; turns: {len(turns):,}")
    print(f"outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
