#!/usr/bin/env python3
"""Three pairwise r_cross comparisons per patient: Self-vs-SpeakerA,
Self-vs-SpeakerB, and SpeakerA-vs-SpeakerB, where SpeakerA/SpeakerB are each
patient's top-2 non-Speaker1 (Listening) speakers by word count -- the same
pairing used in speaker_pair_r_cross.py, not literally "Speaker2"/"Speaker3".

Same model/window/PC as speaker_pair_r_cross.py and self_other_r_cross_pc200.py
(BERT L12, PC=30, fixed_selfm200_otherp100_len500), so directly comparable to
both of those runs. PCA is fit once per patient and reused for all 3 pairs.
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
N_COMPONENTS = 200
REGION = "hippocampus"
MIN_SPIKES = 0
RESULTS_DIR = os.path.join(PROJECT, "results")

# patient -> (speakerA, speakerB): same top-2 other-speakers used in
# speaker_pair_r_cross.py (picked by word count, not fixed column number).
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
    # Speaker3 has only 31 words (< PC=200 threshold), so only the
    # Self-vs-Speaker2 pair will actually run; the other two auto-skip.
    "ptYEZ_task60":  ("Speaker2", "Speaker3"),
}


def load_speaker_word_mask(spike_dir: str, speaker: str) -> np.ndarray:
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


def fit_pair(patient, X_pca, masks, Ys, spkA, spkB):
    Y_a, Y_b = Ys[spkA], Ys[spkB]
    mask_a, mask_b = masks[spkA], masks[spkB]
    if Y_a is None or Y_b is None:
        print(f"{patient} {spkA}-{spkB}: SKIP -- missing spike matrix", flush=True)
        return None
    if Y_a.shape[0] != mask_a.sum() or Y_b.shape[0] != mask_b.sum():
        print(f"{patient} {spkA}-{spkB}: SKIP -- row mismatch", flush=True)
        return None
    if Y_a.shape[1] != Y_b.shape[1]:
        print(f"{patient} {spkA}-{spkB}: SKIP -- neuron count mismatch", flush=True)
        return None
    if Y_a.shape[0] <= N_COMPONENTS or Y_b.shape[0] <= N_COMPONENTS:
        print(f"{patient} {spkA}-{spkB}: SKIP -- fewer words ({Y_a.shape[0]}/{Y_b.shape[0]}) "
              f"than PCs ({N_COMPONENTS})", flush=True)
        return None

    spike_ok = (Y_a.sum(0) >= MIN_SPIKES) & (Y_b.sum(0) >= MIN_SPIKES)
    n_m = int(spike_ok.sum())
    if n_m == 0:
        print(f"{patient} {spkA}-{spkB}: SKIP -- 0 neurons pass spike filter", flush=True)
        return None

    from sklearn.preprocessing import StandardScaler
    X_a = StandardScaler().fit_transform(X_pca[mask_a])
    X_b = StandardScaler().fit_transform(X_pca[mask_b])
    Y_a_f = Y_a[:, spike_ok].astype(np.float64)
    Y_b_f = Y_b[:, spike_ok].astype(np.float64)

    print(f"{patient} {spkA}-{spkB}: {spkA}={X_a.shape[0]}w  {spkB}={X_b.shape[0]}w  neurons={n_m}", flush=True)

    cfg = _rel.ReliabilityConfig(n_null=100, n_half_splits=100, n_jobs=12, random_state=2026)
    res = _rel.run_beta_reliability_all_neurons(
        X_self=X_a, X_other=X_b, Y_self=Y_a_f, Y_other=Y_b_f, cfg=cfg, verbose=False,
    )
    r_cross_vals = [r["r_cross"] for r in res if np.isfinite(r.get("r_cross", np.nan))]
    med = float(np.median(r_cross_vals)) if r_cross_vals else float("nan")
    print(f"  -> n_neurons_valid={len(r_cross_vals)}/{n_m}  median r_cross({spkA},{spkB})={med:.4f}", flush=True)

    pair_tag = f"{spkA}_{spkB}"
    out_path = os.path.join(RESULTS_DIR, f"threeway_rcross_{patient}_{pair_tag}_bert_L12_pc200.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(
            {"patient": patient, "speakerA": spkA, "speakerB": spkB,
             "n_neurons": n_m, "reliability": res},
            f,
        )
    return {
        "patient": patient, "pair": pair_tag,
        "n_words_A": int(X_a.shape[0]), "n_words_B": int(X_b.shape[0]),
        "n_neurons": n_m, "n_neurons_valid": len(r_cross_vals), "median_r_cross": med,
    }


def run_patient(patient: str) -> list[dict]:
    spkA, spkB = PATIENTS[patient]
    patient_ID = "PT" + patient[2:]
    spike_dir = os.path.join(SPIKE_ROOT, f"output_{patient}_english_only_{SPIKE_TAG}")
    npy_path = os.path.join(EMBED_DIR, f"{patient_ID}_{MODEL_TAG}{CONTEXT_TAG}_word_emb_layers.npy")
    if not os.path.exists(npy_path) or not os.path.isdir(spike_dir):
        print(f"{patient}: SKIP -- missing embeddings or spike dir", flush=True)
        return []

    masks, Ys = {}, {}
    for spk in ("Speaker1", spkA, spkB):
        masks[spk] = load_speaker_word_mask(spike_dir, spk)
        Ys[spk] = load_spike_matrix(spike_dir, spk, REGION)

    X_layer_raw = np.load(npy_path, mmap_mode="r")[LAYER].astype(np.float64)
    from sklearn.decomposition import PCA
    X_pca = PCA(n_components=N_COMPONENTS).fit_transform(X_layer_raw)

    rows = []
    for pA, pB in [("Speaker1", spkA), ("Speaker1", spkB), (spkA, spkB)]:
        row = fit_pair(patient, X_pca, masks, Ys, pA, pB)
        if row is not None:
            rows.append(row)
    return rows


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(PATIENTS.keys())
    rows = []
    for patient in targets:
        rows.extend(run_patient(patient))

    if not rows:
        print("No results.")
        return

    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, "threeway_rcross_bert_L12_pc200_summary.csv")
    if os.path.exists(out_csv):
        prior = pd.read_csv(out_csv)
        prior = prior[~((prior["patient"].isin(df["patient"])) & (prior["pair"].isin(df["pair"])))]
        df = pd.concat([prior, df], ignore_index=True)
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
