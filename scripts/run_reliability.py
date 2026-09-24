#!/usr/bin/env python3
"""
Standalone reliability runner — safe to run under nohup/screen.

Loads pre-computed beta results (ALL_CONDITIONS_RESULTS.pkl) for each patient,
re-builds X/Y from raw data, and runs beta reliability analysis.
Skips patients that already have *_RELIABILITY_RESULTS.pkl files.

Usage:
    cd /scratch/aniluchavez/hippocampal-speaker-semantics
    nohup python3 -u scripts/run_reliability.py > logs/reliability_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    echo "PID: $!"
"""

import os
import sys
import pickle
import warnings
import importlib.util

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = "/scratch/aniluchavez/hippocampal-speaker-semantics"
EMBED_CACHE_DIR = "/scratch/aniluchavez/ConvoDATAS/EmbedCache"
BERT_EMBED_DIR  = "/scratch/aniluchavez/ConvoDATAS/BERTEmbeds"
SPIKE_MAT_ROOT  = "/scratch/aniluchavez/ConvoDATAS/SpikesMAT"
SPIKE_WINDOW_OUTPUT_ROOT = "/scratch/aniluchavez/ConvoDATAS/SpikeWindows"

RESULTS_ROOT = "/projects/bhayden/anilu/Language_docs/RegressionRESULTSLLAMA31_L18/allwords"
if RESULTS_ROOT.startswith("/projects") and not os.path.isdir("/projects"):
    RESULTS_ROOT = "/scratch/aniluchavez/RegressionRESULTSLLAMA31_L18/allwords"

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── config ─────────────────────────────────────────────────────────────────────
MODEL_TAG        = "llama-3.1-8b"
LLAMA_LAYER      = 18
N_COMPONENTS     = 30
TARGET_SPEAKER   = "SPK1"
ALPHAS           = np.logspace(-3, 2, 10)
SPEAKERS         = [f"Speaker{i}" for i in range(1, 13)]

N_JOBS_RELIABILITY  = 10
N_NULLS_RELIABILITY = 100

SPIKE_SAMPLE_RATE = 1000
SPIKE_VALUE_MODE  = "counts"
SPIKE_WINDOW_PRESETS = {
    "self":  {"speaker": "Speaker1", "start_offset_ms": -150, "window_length_ms": 500},
    "other": {"start_offset_ms": 200, "window_length_ms": 500},
}

PATIENTS = [
    {"patient_ID": "PTYEU_task147", "patient": "ptYEU_task147",
     "region_ranges": {"hippocampus": [(1,16),(25,40)], "ACC": [(17,24),(41,48)]}},
    {"patient_ID": "PTYFF_task17",  "patient": "ptYFF_task17",
     "region_ranges": {"hippocampus": [(9,16),(25,40)], "ACC": [(17,24),(41,48)]}},
    {"patient_ID": "PTYFG_task18",  "patient": "ptYFG_task18",
     "region_ranges": {"hippocampus": [(9,16)],         "ACC": [(25,56)]}},
    {"patient_ID": "PTYFI_task81",  "patient": "ptYFI_task81",
     "region_ranges": {"hippocampus": [(1,8),(25,40)],  "ACC": [(9,16)]}},
    {"patient_ID": "PTYFA_task25",  "patient": "ptYFA_task25",
     "region_ranges": {"hippocampus": [(1,16),(25,40)], "ACC": [(17,24)]}},
    {"patient_ID": "PTYFK_task40",  "patient": "ptYFK_task40",
     "region_ranges": {"hippocampus": [(1,16),(25,40)], "ACC": [(49,56)]}},
    {"patient_ID": "PTYEY_task86",  "patient": "ptYEY_task86",
     "region_ranges": {"hippocampus": [(1,16)]}},
    {"patient_ID": "PTYEV_task37",  "patient": "ptYEV_task37",
     "region_ranges": {"hippocampus": [(1,16),(25,40)], "ACC": [(17,24),(41,48)]}},
    {"patient_ID": "PTYEZ_task60",  "patient": "ptYEZ_task60",
     "region_ranges": {"hippocampus": [(1,16)],         "ACC": [(17,24)]}},
    {"patient_ID": "PTYFC_task28",  "patient": "ptYFC_task28",
     "region_ranges": {"hippocampus": [(1,8),(33,48)],  "ACC": [(17,32),(49,64)]}},
]

# ── reliability module loader ───────────────────────────────────────────────────
def load_reliability_module():
    path = os.path.join(PROJECT_ROOT, "neural_encoding", "reliability.py")
    name = "nn_reliability"
    if name in sys.modules:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── data helpers (copied verbatim from notebook) ───────────────────────────────

def load_and_reduce_llama_embeddings(patient_ID, n_components=N_COMPONENTS):
    bert_csv = os.path.join(
        BERT_EMBED_DIR, f"{patient_ID}_words_english_only",
        f"{patient_ID}_aligned_embeddings_withNP.csv"
    )
    df_metadata = pd.read_csv(bert_csv)[["Word", "Speaker", "RowIndex"]].reset_index(drop=True)
    npy_path = os.path.join(EMBED_CACHE_DIR, f"{patient_ID}_{MODEL_TAG}_word_emb_layers.npy")
    arr = np.load(npy_path)
    emb = arr[LLAMA_LAYER].astype(np.float32)
    if emb.shape[0] != len(df_metadata):
        raise ValueError(f"{patient_ID}: embedding rows ({emb.shape[0]}) != metadata rows ({len(df_metadata)})")
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(emb)
    pcs_df = pd.DataFrame(pcs, columns=[f"PC{i+1}" for i in range(n_components)])
    print(f"  {patient_ID}: Llama layer {LLAMA_LAYER}, emb {emb.shape} → PCA {pcs.shape}", flush=True)
    return df_metadata, pcs_df


def build_feature_matrix_core(pcs_df, durations):
    pcs = pcs_df.values
    dur = durations.values.reshape(-1, 1)
    X = np.hstack([pcs, dur, pcs * dur])
    return StandardScaler().fit_transform(X)


def get_spike_window_output_dir(patient):
    self_cfg = SPIKE_WINDOW_PRESETS["self"]
    other_cfg = SPIKE_WINDOW_PRESETS["other"]
    tag = (
        f"tshift{self_cfg['start_offset_ms']:+d}_tlen{self_cfg['window_length_ms']}"
        f"_oshift{other_cfg['start_offset_ms']:+d}_olen{other_cfg['window_length_ms']}"
    )
    return os.path.join(SPIKE_WINDOW_OUTPUT_ROOT, f"output_{patient}_english_only_{tag}")


def get_spike_duration_file(patient):
    return os.path.join(get_spike_window_output_dir(patient), f"{patient}_with_regress_dur.xlsx")


def get_self_and_other_features(patient_ID, duration_file, n_components=N_COMPONENTS, target_speaker=TARGET_SPEAKER):
    df_metadata, pcs_df = load_and_reduce_llama_embeddings(patient_ID, n_components)
    if duration_file.endswith(".xlsx"):
        dur_df = pd.read_excel(duration_file)
    else:
        dur_df = pd.read_csv(duration_file)
    dur_df = dur_df.reset_index(drop=True)
    df_metadata["regress_dur"] = dur_df["regress_dur"]

    mask_self  = df_metadata["Speaker"] == target_speaker
    mask_other = df_metadata["Speaker"] != target_speaker

    X_self  = build_feature_matrix_core(
        pcs_df[mask_self].reset_index(drop=True),
        df_metadata.loc[mask_self,  "regress_dur"].reset_index(drop=True))
    X_other = build_feature_matrix_core(
        pcs_df[mask_other].reset_index(drop=True),
        df_metadata.loc[mask_other, "regress_dur"].reset_index(drop=True))
    print(f"  X_self: {X_self.shape}  X_other: {X_other.shape}", flush=True)
    return X_self, X_other, df_metadata


def load_spike_data(speakers, regions, self_base_dir):
    spike_data = {}
    for speaker in speakers:
        try:
            all_folders = os.listdir(self_base_dir)
        except FileNotFoundError:
            continue
        folder = next((f for f in all_folders if f.strip().lower() == speaker.strip().lower()), None)
        if not folder:
            continue
        speaker_path = os.path.join(self_base_dir, folder)
        spike_data[speaker] = {}
        for region in regions:
            candidates = [
                f for f in os.listdir(speaker_path)
                if f.lower().startswith(region.lower()) and f.endswith(".npy")
            ]
            if candidates:
                spike_data[speaker][region] = np.load(os.path.join(speaker_path, candidates[0]))
                print(f"  Loaded {speaker}-{region}: {spike_data[speaker][region].shape}", flush=True)
    return spike_data


def build_Y_matrices(df_metadata, spike_data, regions, target_speaker=TARGET_SPEAKER):
    speaker_tag_map = {
        f"SPK{spk.replace('Speaker', '').strip()}": spk.strip()
        for spk in spike_data.keys()
    }
    region_Ys = {}
    for region in regions:
        Y_self, Y_other = [], []
        counters = {spk: 0 for spk in spike_data}
        for _, row in df_metadata.iterrows():
            spk_tag = row["Speaker"]
            spk_name = speaker_tag_map.get(spk_tag)
            if spk_name is None or region not in spike_data.get(spk_name, {}):
                continue
            idx = counters[spk_name]
            if idx >= spike_data[spk_name][region].shape[0]:
                continue
            row_data = spike_data[spk_name][region][idx, :]
            if spk_tag == target_speaker:
                Y_self.append(row_data)
            else:
                Y_other.append(row_data)
            counters[spk_name] += 1
        print(f"  {region}: Y_self={len(Y_self)}  Y_other={len(Y_other)}", flush=True)
        region_Ys[region] = {
            "self":  np.vstack(Y_self)  if Y_self  else np.empty((0,)),
            "other": np.vstack(Y_other) if Y_other else np.empty((0,)),
        }
    return region_Ys


def clean_XY(X, Y, label=""):
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"{label}: X rows ({X.shape[0]}) != Y rows ({Y.shape[0]})")
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1))
    return X[mask], Y[mask], mask


# ── main ────────────────────────────────────────────────────────────────────────

def reliability_complete(patient_ID, regions):
    patient_dir = os.path.join(RESULTS_ROOT, patient_ID)
    for region in regions:
        p = os.path.join(patient_dir, f"{region.upper()}_RELIABILITY_RESULTS.pkl")
        if not os.path.exists(p):
            return False
    return True


def main():
    print(f"RESULTS_ROOT: {RESULTS_ROOT}", flush=True)
    rel_mod = load_reliability_module()
    run_beta_reliability_all_neurons = rel_mod.run_beta_reliability_all_neurons
    ReliabilityConfig = rel_mod.ReliabilityConfig

    cfg_rel = ReliabilityConfig(
        n_null=N_NULLS_RELIABILITY,
        alphas=tuple(float(a) for a in ALPHAS),
        n_jobs=N_JOBS_RELIABILITY,
    )

    for cfg in PATIENTS:
        patient_ID    = cfg["patient_ID"]
        patient       = cfg["patient"]
        region_ranges = cfg["region_ranges"]
        regions       = list(region_ranges.keys())

        print(f"\n{'='*55}", flush=True)
        print(f"Patient: {patient_ID}  |  regions: {regions}", flush=True)
        print(f"{'='*55}", flush=True)

        if reliability_complete(patient_ID, regions):
            print("  Skipping — all reliability files already exist", flush=True)
            continue

        regression_pkl = os.path.join(RESULTS_ROOT, patient_ID, "ALL_CONDITIONS_RESULTS.pkl")
        if not os.path.exists(regression_pkl):
            print(f"  Skipping — no regression results at {regression_pkl}", flush=True)
            continue

        spike_base_dir = get_spike_window_output_dir(patient)
        duration_file  = get_spike_duration_file(patient)
        if not os.path.isdir(spike_base_dir) or not os.path.exists(duration_file):
            print(f"  Skipping — cached spike windows not found: {spike_base_dir}", flush=True)
            continue

        try:
            X_self, X_other, df_metadata = get_self_and_other_features(
                patient_ID, duration_file, n_components=N_COMPONENTS, target_speaker=TARGET_SPEAKER
            )
            spike_data = load_spike_data(SPEAKERS, regions, spike_base_dir)
            if not spike_data:
                print("  Skipping — no spikes loaded", flush=True)
                continue
            region_Ys = build_Y_matrices(df_metadata, spike_data, regions)

            X_dict, Y_dict = {}, {}
            for region in regions:
                for cond in ["self", "other"]:
                    key = f"{region}_{cond}"
                    Y = region_Ys.get(region, {}).get(cond)
                    X = X_self if cond == "self" else X_other
                    if Y is None or Y.ndim < 2 or Y.shape[0] == 0:
                        continue
                    try:
                        Xc, Yc, _ = clean_XY(X, Y, label=key)
                    except ValueError as e:
                        print(f"  {key}: {e}", flush=True)
                        continue
                    if Xc.shape[0] > 0:
                        X_dict[key] = Xc
                        Y_dict[key] = Yc

            patient_dir = os.path.join(RESULTS_ROOT, patient_ID)
            for region in regions:
                ks, ko = f"{region}_self", f"{region}_other"
                if ks not in X_dict or ko not in X_dict:
                    print(f"  {region}: missing self/other data; skipping", flush=True)
                    continue
                rel_path = os.path.join(patient_dir, f"{region.upper()}_RELIABILITY_RESULTS.pkl")
                if os.path.exists(rel_path):
                    print(f"  {region}: already done, skipping", flush=True)
                    continue
                print(f"  Running reliability: {region}", flush=True)
                rel = run_beta_reliability_all_neurons(
                    X_self=X_dict[ks], X_other=X_dict[ko],
                    Y_self=Y_dict[ks], Y_other=Y_dict[ko],
                    cfg=cfg_rel,
                    verbose=True,
                )
                with open(rel_path, "wb") as f:
                    pickle.dump(rel, f)
                print(f"  Saved: {rel_path}", flush=True)

        except Exception as e:
            print(f"  ERROR for {patient_ID}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
