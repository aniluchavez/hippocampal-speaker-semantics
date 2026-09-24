#!/usr/bin/env python3
"""
Run the runningthroughwindows shift-window sweep on the five added patients.

This is the notebook's smoother method: keep each spike window at word duration
(onset_to_offset), then sweep a temporal shift over that word-duration window.

The static word embeddings live in ConvoDATAS/EmbedCache as fasttext-wiki
arrays. This script creates the old word2vec-style CSV/XLSX inputs expected by
run_patient_pipeline/build_XY, then runs one patient or all patients.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path("/scratch/aniluchavez/hippocampal-speaker-semantics")
CONVO_DIR = Path("/scratch/aniluchavez/ConvoDATAS")
TRANSCRIPTS_DIR = CONVO_DIR / "Transcripts"
SPIKES_MAT_ROOT = CONVO_DIR / "SpikesMAT"
EMBED_CACHE_DIR = CONVO_DIR / "EmbedCache"

MODEL_TAG = "fasttext-wiki"
EMBED_LAYER = 0

WORD2VEC_ROOT = CONVO_DIR / "Word2VecFromEmbedCache_fasttext-wiki"
SPIKE_OUTPUT_ROOT = CONVO_DIR / "spikesforw2v_shift_windows_fasttext-wiki"
RESULTS_DIR = PROJECT_DIR / "results" / "runningthroughwindows_fasttext_wiki"

for module_dir in (PROJECT_DIR / "scripts", PROJECT_DIR / "utils"):
    sys.path.insert(0, str(module_dir))

from run_spike_pipeline import run_patient_pipeline  # noqa: E402
from build_XY import build_cleanX_from_spike_dict, clean_inputs  # noqa: E402
from sweep_regression import sweep_all_configs  # noqa: E402


ADDED_PATIENT_CONFIGS = [
    {
        "patient_id": "ptYFM_task104",
        "patient_prefix": "PTYFM_task104",
        "region_ranges": {"hippocampus": [(33, 48)]},
    },
    {
        "patient_id": "ptYFP_task88",
        "patient_prefix": "PTYFP_task88",
        "region_ranges": {"hippocampus": [(17, 24), (25, 32), (49, 56), (57, 64)]},
    },
    {
        "patient_id": "ptYFR_task91",
        "patient_prefix": "PTYFR_task91",
        "region_ranges": {"hippocampus": [(1, 16), (41, 56)]},
    },
    {
        "patient_id": "ptYFS_task95",
        "patient_prefix": "PTYFS_task95",
        "region_ranges": {"hippocampus": [(1, 24)]},
    },
    {
        "patient_id": "ptYFU_task224",
        "patient_prefix": "PTYFU_task224",
        "region_ranges": {"hippocampus": [(17, 32), (41, 56)]},
    },
]


ORIGINAL_PATIENT_CONFIGS = [
    {
        "patient_id": "ptYFC_task28",
        "patient_prefix": "PTYFC_task28",
        "region_ranges": {
            "hippocampus": [(1, 8), (33, 48)],
            "ACC": [(17, 32), (49, 64)],
        },
    },
    {
        "patient_id": "ptYFI_task81",
        "patient_prefix": "PTYFI_task81",
        "region_ranges": {
            "hippocampus": [(1, 8), (25, 40)],
            "ACC": [(9, 16)],
        },
    },
    {
        "patient_id": "ptYEU_task147",
        "patient_prefix": "PTYEU_task147",
        "region_ranges": {
            "hippocampus": [(1, 16), (25, 40)],
            "ACC": [(17, 24), (41, 48)],
        },
    },
    {
        "patient_id": "ptYFG_task18",
        "patient_prefix": "PTYFG_task18",
        "region_ranges": {
            "hippocampus": [(9, 16)],
            "ACC": [(25, 56)],
        },
    },
    {
        "patient_id": "ptYEZ_task60",
        "patient_prefix": "PTYEZ_task60",
        "region_ranges": {
            "hippocampus": [(1, 16)],
            "ACC": [(17, 24)],
        },
    },
    {
        "patient_id": "ptYFA_task25",
        "patient_prefix": "PTYFA_task25",
        "region_ranges": {
            "hippocampus": [(1, 16), (25, 40)],
            "ACC": [(17, 24)],
        },
    },
    {
        "patient_id": "ptYEV_task37",
        "patient_prefix": "PTYEV_task37",
        "region_ranges": {
            "hippocampus": [(1, 16), (25, 40)],
            "ACC": [(17, 24), (41, 48)],
        },
    },
    {
        "patient_id": "ptYFK_task40",
        "patient_prefix": "PTYFK_task40",
        "region_ranges": {
            "hippocampus": [(1, 16), (25, 40)],
            "ACC": [(49, 56)],
        },
    },
    {
        "patient_id": "ptYFF_task17",
        "patient_prefix": "PTYFF_task17",
        "region_ranges": {
            "hippocampus": [(9, 16), (25, 40)],
            "ACC": [(17, 24), (41, 48)],
        },
    },
    {
        "patient_id": "ptYEY_task86",
        "patient_prefix": "PTYEY_task86",
        "region_ranges": {
            "hippocampus": [(1, 16)],
        },
    },
]


PATIENT_CONFIGS = ORIGINAL_PATIENT_CONFIGS + ADDED_PATIENT_CONFIGS


def alpha_grid_default() -> np.ndarray:
    return np.unique(
        np.concatenate(
            [
                [0.1],
                np.logspace(0, 2.5, 10),
                [1000.0],
                np.logspace(-2, 4, 20),
            ]
        )
    )


def config_by_prefix(patient_prefix: str) -> dict:
    for cfg in PATIENT_CONFIGS:
        if cfg["patient_prefix"] == patient_prefix:
            return cfg
    known = ", ".join(cfg["patient_prefix"] for cfg in PATIENT_CONFIGS)
    raise ValueError(f"Unknown patient {patient_prefix!r}. Known: {known}")


def transcript_path(patient_prefix: str) -> Path:
    exact = TRANSCRIPTS_DIR / f"{patient_prefix}_filtered_used_rows_withNP_withClusterIDNew.xlsx"
    if exact.exists():
        return exact
    matches = sorted(TRANSCRIPTS_DIR.glob(f"{patient_prefix}_filtered_used_rows_withNP_withClusterID*.xlsx"))
    matches = [p for p in matches if not p.name.startswith(("._", "~$"))]
    if not matches:
        raise FileNotFoundError(f"No transcript Excel found for {patient_prefix} in {TRANSCRIPTS_DIR}")
    return matches[-1]


def embedding_cache_path(patient_prefix: str) -> Path:
    return EMBED_CACHE_DIR / f"{patient_prefix}_{MODEL_TAG}_word_emb_layers.npy"


def patient_spikes_dir(patient_id: str) -> Path:
    subfolder = patient_id.split("_")[0].replace("pt", "")
    return SPIKES_MAT_ROOT / subfolder


def word2vec_paths(patient_prefix: str, word2vec_root: Path = WORD2VEC_ROOT) -> dict[str, Path]:
    folder = Path(word2vec_root) / f"{patient_prefix}_words_word2vec"
    return {
        "word2vec_folder": folder,
        "embedding_csv": folder / f"{patient_prefix}_aligned_word2vec_embeddings.csv",
        "word2vec_excel": folder / f"{patient_prefix}_filtered_used_rows_word2vec.xlsx",
    }


def speaker_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if str(c).lower().startswith("speaker")]

    def key(col: str) -> int:
        digits = "".join(ch for ch in str(col) if ch.isdigit())
        return int(digits) if digits else 999

    return sorted(cols, key=key)


def word_and_speaker_from_row(row: pd.Series, spk_cols: list[str]) -> tuple[str, str]:
    for col in spk_cols:
        value = row.get(col)
        if pd.notna(value) and str(value).strip() not in ("", "nan", "xxx"):
            digits = "".join(ch for ch in str(col) if ch.isdigit())
            return str(value).strip(), f"SPK{int(digits)}"
    fallback = row.get("CleanedWord", row.get("AllWords", ""))
    return str(fallback).strip(), ""


def prepare_word2vec_compatible_inputs(
    patient_configs: list[dict],
    word2vec_root: Path = WORD2VEC_ROOT,
    overwrite: bool = False,
) -> pd.DataFrame:
    rows = []
    word2vec_root.mkdir(parents=True, exist_ok=True)

    for cfg in patient_configs:
        patient_prefix = cfg["patient_prefix"]
        xlsx_in = transcript_path(patient_prefix)
        npy_in = embedding_cache_path(patient_prefix)
        paths = word2vec_paths(patient_prefix, word2vec_root)
        paths["word2vec_folder"].mkdir(parents=True, exist_ok=True)

        if not npy_in.exists():
            raise FileNotFoundError(f"Missing embedding cache: {npy_in}")

        if paths["embedding_csv"].exists() and paths["word2vec_excel"].exists() and not overwrite:
            status = "exists"
        else:
            df = pd.read_excel(xlsx_in)
            df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
            df = df.dropna(subset=["onset"]).sort_values("onset").reset_index(drop=True)

            emb = np.load(npy_in, mmap_mode="r")[EMBED_LAYER].astype(np.float32)
            if emb.shape[0] != len(df):
                raise ValueError(
                    f"{patient_prefix}: embedding rows ({emb.shape[0]}) != transcript rows ({len(df)})"
                )

            spk_cols = speaker_columns(df)
            words, speakers = zip(*(word_and_speaker_from_row(row, spk_cols) for _, row in df.iterrows()))
            meta = pd.DataFrame(
                {
                    "Word": list(words),
                    "Speaker": list(speakers),
                    "RowIndex": np.arange(len(df), dtype=int),
                    "Embedding": [np.asarray(vec, dtype=float).tolist() for vec in emb],
                }
            )
            meta.to_csv(paths["embedding_csv"], index=False)
            df.to_excel(paths["word2vec_excel"], index=False)
            status = "created"

        rows.append(
            {
                "patient_prefix": patient_prefix,
                "status": status,
                "transcript": str(xlsx_in),
                "embedding_cache": str(npy_in),
                "embedding_csv": str(paths["embedding_csv"]),
                "word2vec_excel": str(paths["word2vec_excel"]),
            }
        )

    return pd.DataFrame(rows)


def preflight_patient_inputs(patient_configs: list[dict], word2vec_root: Path = WORD2VEC_ROOT) -> pd.DataFrame:
    rows = []
    for cfg in patient_configs:
        patient_id = cfg["patient_id"]
        patient_prefix = cfg["patient_prefix"]
        paths = word2vec_paths(patient_prefix, word2vec_root)
        mat_file = patient_spikes_dir(patient_id) / f"{patient_id}_new_spikes.mat"
        cache_file = embedding_cache_path(patient_prefix)
        row = {
            "patient_prefix": patient_prefix,
            "mat_file_exists": mat_file.exists(),
            "embedding_cache_exists": cache_file.exists(),
            "embedding_csv_exists": paths["embedding_csv"].exists(),
            "word2vec_excel_exists": paths["word2vec_excel"].exists(),
            "mat_file": str(mat_file),
            "embedding_cache": str(cache_file),
            "embedding_csv": str(paths["embedding_csv"]),
            "word2vec_excel": str(paths["word2vec_excel"]),
        }
        row["ready"] = (
            row["mat_file_exists"]
            and row["embedding_cache_exists"]
            and row["embedding_csv_exists"]
            and row["word2vec_excel_exists"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def raise_for_missing_inputs(patient_id: str, patient_prefix: str, word2vec_root: Path) -> tuple[Path, dict[str, Path]]:
    mat_file = patient_spikes_dir(patient_id) / f"{patient_id}_new_spikes.mat"
    paths = word2vec_paths(patient_prefix, word2vec_root)
    required = [mat_file, embedding_cache_path(patient_prefix), paths["embedding_csv"], paths["word2vec_excel"]]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        msg = "Missing required input files:\n" + "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(msg)
    return mat_file, paths


def sweep_shifts_over_patient(
    patient_id: str,
    patient_prefix: str,
    regions: list[str],
    region_ranges: dict,
    shift_range: range,
    test_sizes: list[float | None],
    n_pcs_list: list[int],
    alpha_grid: np.ndarray,
    n_shuffles: int,
    n_jobs: int,
    word2vec_root: Path,
    spike_base_dir: Path,
    results_dir: Path,
    overwrite_results: bool = False,
) -> pd.DataFrame:
    results_dir.mkdir(parents=True, exist_ok=True)
    spike_base_dir.mkdir(parents=True, exist_ok=True)

    output_csv = results_dir / f"regression_results_{patient_prefix}_{MODEL_TAG}_ALLREGIONS_shifts.csv"
    if output_csv.exists() and not overwrite_results:
        print(f"[SKIP] Existing result: {output_csv}", flush=True)
        return pd.read_csv(output_csv)

    mat_file, paths = raise_for_missing_inputs(patient_id, patient_prefix, word2vec_root)
    mat_base = mat_file.parent
    embedding_csv = paths["embedding_csv"]
    base_excel = paths["word2vec_excel"]

    df_words = pd.read_excel(base_excel)
    active_speaker_columns = [
        col for col in df_words.columns if str(col).lower().startswith("speaker") and df_words[col].notna().any()
    ]
    if not active_speaker_columns:
        raise ValueError(f"No non-empty Speaker* columns found in {base_excel}")

    print(f"{patient_prefix}: speaker columns {active_speaker_columns}", flush=True)
    window_sweep_config = {spk: {"mode": "onset_to_offset"} for spk in active_speaker_columns}

    all_results = []
    for shift in shift_range:
        print(f"\n=== {patient_prefix} shift {shift:+} ms ===", flush=True)
        output_suffix = f"shift{shift:+d}ms".replace("+", "p").replace("-", "m")

        spike_data, _, meta_xlsx = run_patient_pipeline(
            patient_id=patient_id,
            patient_prefix=patient_prefix,
            mat_base=str(mat_base),
            excel_base=str(word2vec_root),
            region_ranges=region_ranges,
            speaker_window_modes=window_sweep_config,
            output_suffix=output_suffix,
            spike_base_dir=str(spike_base_dir),
            use_sliding_window=True,
            shift_ms=shift,
            return_spike_data=True,
        )

        for region in regions:
            print(f"{patient_prefix}: region {region}", flush=True)
            (x_self, y_self, dur_self), (x_other, y_other, dur_other) = build_cleanX_from_spike_dict(
                str(embedding_csv),
                meta_xlsx,
                spike_data,
                region=region,
                target_speaker="SPK1",
                embedding_col="Embedding",
                separate_self_other=True,
            )

            x_self, y_self, dur_self = clean_inputs(x_self, y_self, dur_self)
            x_other, y_other, dur_other = clean_inputs(x_other, y_other, dur_other)

            def run_condition(x, y, dur, label):
                if len(x) == 0 or y.ndim < 2 or y.shape[1] == 0:
                    print(f"{patient_prefix}/{region}: skipping {label} X={x.shape} Y={y.shape}", flush=True)
                    return pd.DataFrame()
                max_pcs = min(x.shape[0], x.shape[1])
                effective_n_pcs = [min(n_pcs, max_pcs) for n_pcs in n_pcs_list]
                if effective_n_pcs != n_pcs_list:
                    print(
                        f"{patient_prefix}/{region}/{label}: capped n_pcs {n_pcs_list} -> "
                        f"{effective_n_pcs} for X={x.shape}",
                        flush=True,
                    )
                out = sweep_all_configs(
                    X_embed=x,
                    dur=dur,
                    Y=y,
                    test_sizes=test_sizes,
                    n_pcs_list=effective_n_pcs,
                    alpha_grid=alpha_grid,
                    n_shuffles=n_shuffles,
                    n_jobs=n_jobs,
                )
                out["patient_prefix"] = patient_prefix
                out["model_tag"] = MODEL_TAG
                out["condition"] = label
                out["shift_ms"] = shift
                out["region"] = region
                return out

            all_results.append(run_condition(x_self, y_self, dur_self, "self"))
            all_results.append(run_condition(x_other, y_other, dur_other, "other"))

    nonempty = [df for df in all_results if not df.empty]
    df_all = pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()
    df_all.to_csv(output_csv, index=False)
    print(f"[DONE] Saved {output_csv}", flush=True)
    return df_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", default="all", help="Patient prefix, e.g. PTYFM_task104, or 'all'.")
    parser.add_argument("--region", default="hippocampus", help="Region to run; default hippocampus.")
    parser.add_argument("--shift-start", type=int, default=-1000)
    parser.add_argument("--shift-stop", type=int, default=1000)
    parser.add_argument("--shift-step", type=int, default=10)
    parser.add_argument("--n-pcs", type=int, default=100)
    parser.add_argument("--test-size", default="none", help="'none' for full-data fit, or float like 0.3.")
    parser.add_argument("--n-shuffles", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--spike-output-root", type=Path, default=SPIKE_OUTPUT_ROOT)
    parser.add_argument("--word2vec-root", type=Path, default=WORD2VEC_ROOT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite-inputs", action="store_true")
    parser.add_argument("--overwrite-results", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = PATIENT_CONFIGS if args.patient == "all" else [config_by_prefix(args.patient)]

    print("[INFO] Preparing word2vec-style inputs", flush=True)
    prepared = prepare_word2vec_compatible_inputs(configs, args.word2vec_root, overwrite=args.overwrite_inputs)
    print(prepared.to_string(index=False), flush=True)

    preflight = preflight_patient_inputs(configs, args.word2vec_root)
    print("\n[INFO] Preflight", flush=True)
    print(preflight[["patient_prefix", "mat_file_exists", "embedding_cache_exists", "embedding_csv_exists", "word2vec_excel_exists", "ready"]].to_string(index=False), flush=True)
    if not preflight["ready"].all():
        raise SystemExit("Some inputs are missing; stopping before sweep.")

    if args.prepare_only:
        print("[DONE] prepare-only complete", flush=True)
        return

    test_size = None if str(args.test_size).lower() in {"none", "null", "full"} else float(args.test_size)
    shift_range = range(args.shift_start, args.shift_stop + 1, args.shift_step)
    alpha_grid = alpha_grid_default()

    for cfg in configs:
        region_ranges = cfg["region_ranges"]
        if args.region not in region_ranges:
            print(f"[SKIP] {cfg['patient_prefix']}: region {args.region!r} not present", flush=True)
            continue
        sweep_shifts_over_patient(
            patient_id=cfg["patient_id"],
            patient_prefix=cfg["patient_prefix"],
            regions=[args.region],
            region_ranges={args.region: region_ranges[args.region]},
            shift_range=shift_range,
            test_sizes=[test_size],
            n_pcs_list=[args.n_pcs],
            alpha_grid=alpha_grid,
            n_shuffles=args.n_shuffles,
            n_jobs=args.n_jobs,
            word2vec_root=args.word2vec_root,
            spike_base_dir=args.spike_output_root,
            results_dir=args.results_dir,
            overwrite_results=args.overwrite_results,
        )


if __name__ == "__main__":
    main()
