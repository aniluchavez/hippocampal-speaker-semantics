#!/usr/bin/env python3
"""Fixed-window LENGTH sweep (300/500/700ms) at the paper's chosen shifts
(self -150ms, other +200ms), using word2vec (fasttext-wiki) embeddings and
the same lightweight PoissonRegressor ll_diff pipeline used for the Fig1j
shift-sweep (sweep_regression.py). One row per (patient, condition, length,
neuron); ll_diff = LL(real model) - LL(mean over embedding-shuffle nulls).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path("/scratch/aniluchavez/hippocampal-speaker-semantics")
CONVO_DIR = Path("/scratch/aniluchavez/ConvoDATAS")
WORD2VEC_ROOT = CONVO_DIR / "Word2VecFromEmbedCache_fasttext-wiki"
SPIKEWINDOWS_ROOT = CONVO_DIR / "SpikeWindows"
OUT_CSV = PROJECT_DIR / "results" / "window_length_sweep_word2vec_llh.csv"

for module_dir in (PROJECT_DIR / "scripts", PROJECT_DIR / "utils"):
    sys.path.insert(0, str(module_dir))

from build_XY import build_cleanX_from_spike_dict, clean_inputs  # noqa: E402
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


def main() -> None:
    rows = []
    for cfg in PATIENT_CONFIGS:
        patient_id = cfg["patient_id"]  # e.g. ptYFS_task95
        patient_prefix = cfg["patient_prefix"]  # e.g. PTYFS_task95
        region_ranges = cfg["region_ranges"]
        if REGION not in region_ranges:
            print(f"[SKIP] {patient_prefix}: no {REGION}", flush=True)
            continue

        emb_csv = WORD2VEC_ROOT / f"{patient_prefix}_words_word2vec" / f"{patient_prefix}_aligned_word2vec_embeddings.csv"
        if not emb_csv.exists():
            print(f"[SKIP] {patient_prefix}: missing {emb_csv}", flush=True)
            continue

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

            (x_self, y_self, dur_self), (x_other, y_other, dur_other) = build_cleanX_from_spike_dict(
                str(emb_csv),
                str(regress_xlsx),
                spike_data,
                region=REGION,
                target_speaker="SPK1",
                embedding_col="Embedding",
                separate_self_other=True,
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
