#!/usr/bin/env python3
"""Speaker-vs-speaker r_cross: for patients with two Listening-condition speakers who
each contribute a substantial number of words, treat each speaker's words as a separate
condition (playing the role self/other normally play) and compute r_cross between the
two SPEAKERS' beta vectors -- i.e. does hippocampal beta-weight encoding during
listening generalize across different specific talkers, or is it talker-specific?

Reuses neural_encoding/reliability.py's run_beta_reliability_all_neurons exactly as
semantic_glm.py's --reliability path does for self/other, just with two listening
speakers instead of self/other.

Model: BERT (bert-base_ctx200), layer 12, PC=30, fixed window
(spike_tag fixed_selfm200_otherp100_len500 -- both speakers use the "other" timing
window since both are Listening-condition talkers).

Usage:
    python3 -u scripts/speaker_pair_r_cross.py [patient_id ...]
    (no args -> run all patients in PATIENTS)
"""
import os
import sys
import pickle
import importlib.util as ilu

import numpy as np
import pandas as pd

PROJECT = "/scratch/aniluchavez/hippocampal-speaker-semantics"
sys.path.insert(0, PROJECT)

_rel_spec = ilu.spec_from_file_location(
    "nn_reliability", os.path.join(PROJECT, "neural_encoding", "reliability.py")
)
_rel = ilu.module_from_spec(_rel_spec)
sys.modules["nn_reliability"] = _rel
_rel_spec.loader.exec_module(_rel)

EMBED_DIR = "/scratch/aniluchavez/ConvoDATAS/EmbedCache"
SPIKE_ROOT = "/scratch/aniluchavez/ConvoDATAS/SpikeWindows"
SPIKE_TAG = "fixed_selfm200_otherp100_len500"
MODEL_TAG = "bert-base"
CONTEXT_TAG = "_ctx200"
LAYER = 12
N_COMPONENTS = 30
REGION = "hippocampus"
MIN_SPIKES = 0
RESULTS_DIR = os.path.join(PROJECT, "results")

# patient -> (speakerA, speakerB): the two other-speakers with the most words,
# picked from word counts only (no transcript content inspected).
PATIENTS = {
    "ptYFK_task40":  ("Speaker2", "Speaker4"),
    "ptYFG_task18":  ("Speaker2", "Speaker3"),
    "ptYFF_task17":  ("Speaker2", "Speaker3"),
    "ptYFC_task28":  ("Speaker2", "Speaker5"),
    "ptYFA_task25":  ("Speaker3", "Speaker2"),
    "ptYEY_task86":  ("Speaker4", "Speaker2"),
    "ptYEU_task147": ("Speaker5", "Speaker3"),
    "ptYFM_task104": ("Speaker2", "Speaker4"),
    "ptYFP_task88":  ("Speaker2", "Speaker3"),
    "ptYFR_task91":  ("Speaker2", "Speaker3"),
    "ptYFS_task95":  ("Speaker3", "Speaker5"),
    "ptYFU_task224": ("Speaker2", "Speaker3"),
}


def load_speaker_word_mask(spike_dir: str, speaker: str) -> np.ndarray:
    """Raw per-speaker-column membership (matches how each speaker's own
    spike matrix was generated: one row per non-null entry in that speaker's
    column), not a "first non-empty column wins" resolution across speakers
    -- the latter under-counts rows when two speaker columns briefly overlap
    (cross-talk), causing a length mismatch against that speaker's own
    pre-computed spike matrix."""
    cands = [f for f in os.listdir(spike_dir) if f.endswith("_with_regress_dur.xlsx")]
    tx = pd.read_excel(os.path.join(spike_dir, cands[0]))

    def _nn(v):
        return pd.notna(v) and str(v).strip() not in ("", "nan")

    return np.array([_nn(v) for v in tx[speaker]], dtype=bool)


def load_spike_matrix(spike_dir: str, speaker: str, region: str):
    spk_dir = os.path.join(spike_dir, speaker)
    if not os.path.isdir(spk_dir):
        return None
    cands = [
        f for f in os.listdir(spk_dir)
        if f.lower().startswith(region.lower()) and f.endswith("_spike_counts.npy")
    ]
    return np.load(os.path.join(spk_dir, cands[0])) if cands else None


def run_patient(patient: str, spkA: str, spkB: str) -> dict | None:
    patient_ID = "PT" + patient[2:]  # e.g. "ptYFS_task95" -> "PTYFS_task95"
    spike_dir = os.path.join(SPIKE_ROOT, f"output_{patient}_english_only_{SPIKE_TAG}")
    npy_path = os.path.join(EMBED_DIR, f"{patient_ID}_{MODEL_TAG}{CONTEXT_TAG}_word_emb_layers.npy")
    if not os.path.exists(npy_path) or not os.path.isdir(spike_dir):
        print(f"{patient}: SKIP -- missing embeddings or spike dir", flush=True)
        return None

    mask_a = load_speaker_word_mask(spike_dir, spkA)
    mask_b = load_speaker_word_mask(spike_dir, spkB)
    Y_a = load_spike_matrix(spike_dir, spkA, REGION)
    Y_b = load_spike_matrix(spike_dir, spkB, REGION)
    if Y_a is None or Y_b is None:
        print(f"{patient}: SKIP -- missing spike matrix for {spkA}/{spkB}", flush=True)
        return None
    if Y_a.shape[0] != mask_a.sum() or Y_b.shape[0] != mask_b.sum():
        print(f"{patient}: SKIP -- row mismatch", flush=True)
        return None
    if Y_a.shape[1] != Y_b.shape[1]:
        print(f"{patient}: SKIP -- neuron count mismatch", flush=True)
        return None

    spike_ok = (Y_a.sum(0) >= MIN_SPIKES) & (Y_b.sum(0) >= MIN_SPIKES)
    n_m = int(spike_ok.sum())
    if n_m == 0:
        print(f"{patient}: SKIP -- 0 neurons pass spike filter", flush=True)
        return None

    X_layer_raw = np.load(npy_path, mmap_mode="r")[LAYER].astype(np.float64)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X_pca = PCA(n_components=N_COMPONENTS).fit_transform(X_layer_raw)
    X_a = StandardScaler().fit_transform(X_pca[mask_a])
    X_b = StandardScaler().fit_transform(X_pca[mask_b])
    Y_a_f = Y_a[:, spike_ok].astype(np.float64)
    Y_b_f = Y_b[:, spike_ok].astype(np.float64)

    print(f"{patient}: {spkA}={X_a.shape[0]}w  {spkB}={X_b.shape[0]}w  neurons={n_m}", flush=True)

    cfg = _rel.ReliabilityConfig(n_null=100, n_half_splits=100, n_jobs=12, random_state=2026)
    res = _rel.run_beta_reliability_all_neurons(
        X_self=X_a, X_other=X_b, Y_self=Y_a_f, Y_other=Y_b_f, cfg=cfg, verbose=False,
    )
    r_cross_vals = [r["r_cross"] for r in res if np.isfinite(r.get("r_cross", np.nan))]
    med = float(np.median(r_cross_vals)) if r_cross_vals else float("nan")
    print(
        f"  -> n_neurons_valid={len(r_cross_vals)}/{n_m}  "
        f"median r_cross({spkA},{spkB})={med:.4f}",
        flush=True,
    )

    out_path = os.path.join(
        RESULTS_DIR, f"speaker_pair_rcross_{patient}_bert_L12_pc30.pkl"
    )
    with open(out_path, "wb") as f:
        pickle.dump(
            {"patient": patient, "speakerA": spkA, "speakerB": spkB,
             "n_neurons": n_m, "reliability": res},
            f,
        )

    return {
        "patient": patient, "speakerA": spkA, "speakerB": spkB,
        "n_words_A": int(X_a.shape[0]), "n_words_B": int(X_b.shape[0]),
        "n_neurons": n_m, "n_neurons_valid": len(r_cross_vals),
        "median_r_cross": med,
    }


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(PATIENTS.keys())
    rows = []
    for patient in targets:
        spkA, spkB = PATIENTS[patient]
        row = run_patient(patient, spkA, spkB)
        if row is not None:
            rows.append(row)

    if not rows:
        print("No results.")
        return

    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, "speaker_pair_rcross_bert_L12_pc30_summary.csv")
    if os.path.exists(out_csv):
        prior = pd.read_csv(out_csv)
        prior = prior[~prior["patient"].isin(df["patient"])]
        df = pd.concat([prior, df], ignore_index=True)
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
