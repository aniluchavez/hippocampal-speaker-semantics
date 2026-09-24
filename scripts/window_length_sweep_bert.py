#!/usr/bin/env python3
"""Fixed-window LENGTH sweep (300/500/700ms) at the paper's chosen shifts
(self -150ms, other +200ms), using BERT-base-causal (ctx200, spktag) L12
embeddings, same lightweight PoissonRegressor ll_diff pipeline as the
word2vec version (window_length_sweep_word2vec.py). Reimplements the
build_XY row-alignment logic directly against the BERT embedding cache
array instead of round-tripping through the word2vec embedding CSV.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path("/scratch/aniluchavez/hippocampal-speaker-semantics")
CONVO_DIR = Path("/scratch/aniluchavez/ConvoDATAS")
WORD2VEC_ROOT = CONVO_DIR / "Word2VecFromEmbedCache_fasttext-wiki"
SPIKEWINDOWS_ROOT = CONVO_DIR / "SpikeWindows"
EMBED_CACHE_DIR = CONVO_DIR / "EmbedCache"
OUT_CSV = PROJECT_DIR / "results" / "window_length_sweep_bert_l12_llh.csv"

for module_dir in (PROJECT_DIR / "scripts", PROJECT_DIR / "utils"):
    sys.path.insert(0, str(module_dir))

from build_XY import clean_inputs  # noqa: E402
from sweep_regression import sweep_all_configs  # noqa: E402
from runningthroughwindows_fasttext_shift_sweep import (  # noqa: E402
    PATIENT_CONFIGS,
    alpha_grid_default,
)

LENGTHS = [300, 500, 700]
TARGET_SHIFT = -150
OTHER_SHIFT = 200
REGION = "hippocampus"
N_PCS = 100
N_SHUFFLES = 20
BERT_TAG = "bert-base-causal_ctx200spktag"
BERT_LAYER = 12


def spike_dir_for(patient: str, length: int) -> Path:
    tag = f"tshift{TARGET_SHIFT:+d}_tlen{length}_oshift{OTHER_SHIFT:+d}_olen{length}"
    return SPIKEWINDOWS_ROOT / f"output_{patient}_english_only_{tag}"


def load_spike_data(spk_dir: Path, region: str) -> dict:
    spike_data = {}
    for sub in sorted(spk_dir.iterdir()):
        if not sub.is_dir() or not sub.name.startswith("Speaker"):
            continue
        npy_path = sub / f"{region}_spike_counts.npy"
        if npy_path.exists():
            spike_data[sub.name] = {region: np.load(npy_path)}
    return spike_data


def build_cleanXY_bert(meta_csv_path, meta_xlsx_path, bert_emb, spike_data,
                        region="hippocampus", target_speaker="SPK1"):
    """Same row-alignment logic as build_XY.build_cleanX_from_spike_dict,
    but pulls the embedding vector for row i directly from `bert_emb`
    (n_words, dim) instead of parsing a CSV 'Embedding' column."""
    meta_df = pd.read_csv(meta_csv_path)
    dur_df = pd.read_excel(meta_xlsx_path)
    meta_df["regress_dur"] = dur_df["regress_dur"]

    data_self_X, data_self_Y, data_self_dur = [], [], []
    data_other_X, data_other_Y, data_other_dur = [], [], []
    speaker_counters = {spk: {reg: 0 for reg in spike_data[spk]} for spk in spike_data}

    for i, row in meta_df.iterrows():
        speaker_raw = str(row["Speaker"]).strip().upper().replace("SPK", "")
        try:
            speaker_folder = f"Speaker{int(speaker_raw)}"
        except ValueError:
            continue
        if speaker_folder not in spike_data or region not in spike_data[speaker_folder]:
            continue

        spikes = spike_data[speaker_folder][region]
        idx = speaker_counters[speaker_folder][region]
        if idx >= spikes.shape[0]:
            continue

        emb = bert_emb[i]
        if emb.ndim != 1 or emb.size == 0 or not np.all(np.isfinite(emb)):
            speaker_counters[speaker_folder][region] += 1
            continue

        dur = row["regress_dur"]
        spike_row = spikes[idx]

        if row["Speaker"] == target_speaker:
            data_self_X.append(emb); data_self_Y.append(spike_row); data_self_dur.append(dur)
        else:
            data_other_X.append(emb); data_other_Y.append(spike_row); data_other_dur.append(dur)

        speaker_counters[speaker_folder][region] += 1

    def finalize(Xs, Ys, Ds):
        if Xs:
            return np.vstack(Xs), np.vstack(Ys), np.array(Ds).reshape(-1, 1)
        return np.empty((0,)), np.empty((0,)), np.empty((0, 1))

    return finalize(data_self_X, data_self_Y, data_self_dur), finalize(data_other_X, data_other_Y, data_other_dur)


def main() -> None:
    rows = []
    for cfg in PATIENT_CONFIGS:
        patient_id = cfg["patient_id"]
        patient_prefix = cfg["patient_prefix"]
        region_ranges = cfg["region_ranges"]
        if REGION not in region_ranges:
            print(f"[SKIP] {patient_prefix}: no {REGION}", flush=True)
            continue

        emb_csv = WORD2VEC_ROOT / f"{patient_prefix}_words_word2vec" / f"{patient_prefix}_aligned_word2vec_embeddings.csv"
        bert_npy = EMBED_CACHE_DIR / f"{patient_prefix}_{BERT_TAG}_word_emb_layers.npy"
        if not emb_csv.exists() or not bert_npy.exists():
            print(f"[SKIP] {patient_prefix}: missing {emb_csv if not emb_csv.exists() else bert_npy}", flush=True)
            continue

        bert_emb_full = np.load(bert_npy, mmap_mode="r")
        n_rows_meta = len(pd.read_csv(emb_csv))
        if bert_emb_full.shape[1] != n_rows_meta:
            print(f"[SKIP] {patient_prefix}: bert rows {bert_emb_full.shape[1]} != meta rows {n_rows_meta}", flush=True)
            continue
        bert_emb = np.asarray(bert_emb_full[BERT_LAYER], dtype=np.float64)

        for length in LENGTHS:
            spk_dir = spike_dir_for(patient_id, length)
            xlsx_matches = sorted(spk_dir.glob("*_with_regress_dur.xlsx")) if spk_dir.exists() else []
            if not xlsx_matches:
                print(f"[SKIP] {patient_prefix} len={length}: no *_with_regress_dur.xlsx in {spk_dir}", flush=True)
                continue
            regress_xlsx = xlsx_matches[0]

            spike_data = load_spike_data(spk_dir, REGION)
            if not spike_data:
                print(f"[SKIP] {patient_prefix} len={length}: no spike npy found", flush=True)
                continue

            (x_self, y_self, dur_self), (x_other, y_other, dur_other) = build_cleanXY_bert(
                str(emb_csv), str(regress_xlsx), bert_emb, spike_data,
                region=REGION, target_speaker="SPK1",
            )
            x_self, y_self, dur_self = clean_inputs(x_self, y_self, dur_self)
            x_other, y_other, dur_other = clean_inputs(x_other, y_other, dur_other)

            alpha_grid = alpha_grid_default()

            for label, x, y, dur in [("self", x_self, y_self, dur_self), ("other", x_other, y_other, dur_other)]:
                if len(x) == 0 or y.ndim < 2 or y.shape[1] == 0:
                    print(f"{patient_prefix}/{label}/len{length}: empty X={x.shape} Y={y.shape}", flush=True)
                    continue
                max_pcs = min(x.shape[0], x.shape[1])
                n_pcs = min(N_PCS, max_pcs)
                out = sweep_all_configs(
                    X_embed=x, dur=dur, Y=y,
                    test_sizes=[None], n_pcs_list=[n_pcs], alpha_grid=alpha_grid,
                    n_shuffles=N_SHUFFLES, n_jobs=-1,
                )
                out["patient_id"] = patient_id
                out["patient_prefix"] = patient_prefix
                out["condition"] = label
                out["length_ms"] = length
                out["region"] = REGION
                rows.append(out)
                mean_ll_diff = out["ll_diff"].mean() if len(out) else float("nan")
                print(f"[DONE] {patient_prefix}/{label}/len{length}ms: "
                      f"n_neurons={len(out)} mean_ll_diff={mean_ll_diff:.3f}", flush=True)

    df_all = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(OUT_CSV, index=False)
    print(f"\n[SAVED] {OUT_CSV}  ({len(df_all)} rows)", flush=True)


if __name__ == "__main__":
    main()
