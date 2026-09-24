#!/usr/bin/env python3
"""Three pairwise r_cross comparisons per patient: Self-vs-SpeakerA,
Self-vs-SpeakerB, and SpeakerA-vs-SpeakerB, where SpeakerA/SpeakerB are each
patient's top-2 non-Speaker1 (Listening) speakers by word count.

Same as three_way_r_cross_pc10_globalbalance_spktag.py (speaker-tagged bidirectional
BERT embeddings, bert-base-bidirmax-spktag) but swaps the balancing scheme back to
the per-pair 2.5:1 max-ratio cap (matching the reconstructed old BERTBETAS
methodology) instead of the global per-patient 1:1 balance.

- Window: self = onset-200ms, len 500ms; all other speakers = onset+200ms, len 500ms
  (spike tag fixed_selfm200_otherp200_len500).
- Balancing: per pair, whenever max(n_a, n_b) / min(n_a, n_b) > 2.5, the larger side
  is randomly subsampled (seed=42) down to exactly 2.5x the smaller side's count.
- No word-count-vs-PC skip: every patient/pair is fit regardless of how it compares
  to N_COMPONENTS.
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
SPIKE_TAG = "tshift-150_tlen500_oshift+200_olen500"
MODEL_TAG = "bert-base-bidirmax-spktag"
CONTEXT_TAG = ""
LAYER = 12
N_COMPONENTS = 10
REGION = "hippocampus"
MIN_SPIKES = 0
MAX_RATIO = 2.5
DOWNSAMPLE_SEED = 42
RESULTS_DIR = os.path.join(PROJECT, "results")

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


def ratio_cap_rows(n_a: int, n_b: int, max_ratio: float, seed: int):
    """Return (rows_a, rows_b) row-index arrays into the per-speaker matrices,
    subsampling whichever side exceeds max_ratio x the other side's count."""
    rng = np.random.RandomState(seed)
    rows_a = np.arange(n_a)
    rows_b = np.arange(n_b)
    if n_b > 0 and n_a > max_ratio * n_b:
        keep_n = int(round(max_ratio * n_b))
        rows_a = np.sort(rng.choice(n_a, size=keep_n, replace=False))
    elif n_a > 0 and n_b > max_ratio * n_a:
        keep_n = int(round(max_ratio * n_a))
        rows_b = np.sort(rng.choice(n_b, size=keep_n, replace=False))
    return rows_a, rows_b


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

    spike_ok = (Y_a.sum(0) >= MIN_SPIKES) & (Y_b.sum(0) >= MIN_SPIKES)
    n_m = int(spike_ok.sum())
    if n_m == 0:
        print(f"{patient} {spkA}-{spkB}: SKIP -- 0 neurons pass spike filter", flush=True)
        return None

    true_idx_a = np.flatnonzero(mask_a)
    true_idx_b = np.flatnonzero(mask_b)
    n_a_orig, n_b_orig = len(true_idx_a), len(true_idx_b)
    rows_a, rows_b = ratio_cap_rows(n_a_orig, n_b_orig, MAX_RATIO, DOWNSAMPLE_SEED)
    downsample_mode = "cap" if (len(rows_a) < n_a_orig or len(rows_b) < n_b_orig) else "none"

    kept_true_idx_a = true_idx_a[rows_a]
    kept_true_idx_b = true_idx_b[rows_b]
    mask_a_bal = np.zeros_like(mask_a)
    mask_a_bal[kept_true_idx_a] = True
    mask_b_bal = np.zeros_like(mask_b)
    mask_b_bal[kept_true_idx_b] = True

    from sklearn.preprocessing import StandardScaler
    X_a = StandardScaler().fit_transform(X_pca[mask_a_bal])
    X_b = StandardScaler().fit_transform(X_pca[mask_b_bal])
    Y_a_f = Y_a[rows_a][:, spike_ok].astype(np.float64)
    Y_b_f = Y_b[rows_b][:, spike_ok].astype(np.float64)

    print(f"{patient} {spkA}-{spkB}: {spkA}={X_a.shape[0]}w(orig {n_a_orig})  "
          f"{spkB}={X_b.shape[0]}w(orig {n_b_orig})  neurons={n_m}  "
          f"downsample_mode={downsample_mode}", flush=True)

    cfg = _rel.ReliabilityConfig(n_null=100, n_half_splits=100, n_jobs=12, random_state=2026)
    res = _rel.run_beta_reliability_all_neurons(
        X_self=X_a, X_other=X_b, Y_self=Y_a_f, Y_other=Y_b_f, cfg=cfg, verbose=False,
    )
    r_cross_vals = [r["r_cross"] for r in res if np.isfinite(r.get("r_cross", np.nan))]
    med = float(np.median(r_cross_vals)) if r_cross_vals else float("nan")
    print(f"  -> n_neurons_valid={len(r_cross_vals)}/{n_m}  median r_cross({spkA},{spkB})={med:.4f}", flush=True)

    pair_tag = f"{spkA}_{spkB}"
    out_path = os.path.join(RESULTS_DIR, f"threeway_rcross_cap25_spktag_truewindow_{patient}_{pair_tag}_bert_L12_pc10.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(
            {"patient": patient, "speakerA": spkA, "speakerB": spkB,
             "n_neurons": n_m, "reliability": res,
             "downsample_info": {
                 "mode": downsample_mode, "n_self_orig": n_a_orig, "n_other_orig": n_b_orig,
                 "n_self_kept": len(rows_a), "n_other_kept": len(rows_b),
                 "max_ratio": MAX_RATIO, "downsample_seed": DOWNSAMPLE_SEED,
             }},
            f,
        )
    return {
        "patient": patient, "pair": pair_tag,
        "n_words_A": int(X_a.shape[0]), "n_words_B": int(X_b.shape[0]),
        "n_words_A_orig": n_a_orig, "n_words_B_orig": n_b_orig,
        "downsample_mode": downsample_mode,
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
        mask = load_speaker_word_mask(spike_dir, spk)
        Y = load_spike_matrix(spike_dir, spk, REGION)
        if Y is not None and Y.shape[0] == mask.sum():
            # Drop individual word-rows with NaN spike counts (e.g. window extends
            # before recording start) rather than letting one bad row zero out
            # every neuron via the downstream sum() filter.
            bad_rows = np.isnan(Y).any(axis=1)
            if bad_rows.any():
                true_idx = np.flatnonzero(mask)
                new_mask = np.zeros_like(mask)
                new_mask[true_idx[~bad_rows]] = True
                print(f"{patient} {spk}: dropping {bad_rows.sum()} NaN word-row(s) "
                      f"({Y.shape[0]} -> {(~bad_rows).sum()})", flush=True)
                mask = new_mask
                Y = Y[~bad_rows]
        masks[spk] = mask
        Ys[spk] = Y

    X_raw = np.load(npy_path, mmap_mode="r")
    # bidirmax-spktag embeddings are (n_words, embed_dim) -- no per-layer stack,
    # unlike the ctx200 variant's (n_layers, n_words, embed_dim). Only index by
    # LAYER when a layer dimension is actually present.
    X_layer_raw = (X_raw[LAYER] if X_raw.ndim == 3 else X_raw).astype(np.float64)
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
    out_csv = os.path.join(RESULTS_DIR, "threeway_rcross_cap25_spktag_truewindow_bert_L12_pc10_summary.csv")
    if os.path.exists(out_csv):
        prior = pd.read_csv(out_csv)
        prior = prior[~((prior["patient"].isin(df["patient"])) & (prior["pair"].isin(df["pair"])))]
        df = pd.concat([prior, df], ignore_index=True)
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
