#!/usr/bin/env python3
"""Parameterized version of three_way_r_cross_pc30_globalbalance_spktag_truewindow.py
for sweeping PC count and/or spike window. Global per-patient balancing (all three
speakers capped to shared N), speaker-tagged bidirectional BERT embeddings.

Usage:
    three_way_r_cross_globalbalance_spktag_sweep.py --spike_tag TAG --n_components N --out_tag LABEL [patients...]

--out_tag sets the output filename suffix, e.g. --out_tag pc50_trueWindow ->
threeway_rcross_globalbal_spktag_sweep_pc50_trueWindow_bert_L12_summary.csv
"""
import os
import sys
import argparse
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
MODEL_TAG = "bert-base-bidirmax-spktag"
CONTEXT_TAG = ""
LAYER = 12
REGION = "hippocampus"
MIN_SPIKES = 0
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


def fit_pair(patient, X_pca, masks_bal, Ys_bal, spkA, spkB, out_tag, n_components):
    Y_a, Y_b = Ys_bal[spkA], Ys_bal[spkB]
    mask_a, mask_b = masks_bal[spkA], masks_bal[spkB]
    if Y_a is None or Y_b is None:
        print(f"{patient} {spkA}-{spkB}: SKIP -- missing spike matrix", flush=True)
        return None
    if Y_a.shape[1] != Y_b.shape[1]:
        print(f"{patient} {spkA}-{spkB}: SKIP -- neuron count mismatch", flush=True)
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
    out_path = os.path.join(RESULTS_DIR, f"threeway_rcross_sweep_{out_tag}_{patient}_{pair_tag}.pkl")
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


def run_patient(patient: str, spike_tag: str, n_components: int, out_tag: str) -> list[dict]:
    spkA, spkB = PATIENTS[patient]
    patient_ID = "PT" + patient[2:]
    spike_dir = os.path.join(SPIKE_ROOT, f"output_{patient}_english_only_{spike_tag}")
    npy_path = os.path.join(EMBED_DIR, f"{patient_ID}_{MODEL_TAG}{CONTEXT_TAG}_word_emb_layers.npy")
    if not os.path.exists(npy_path) or not os.path.isdir(spike_dir):
        print(f"{patient}: SKIP -- missing embeddings or spike dir ({spike_dir})", flush=True)
        return []

    masks, Ys = {}, {}
    for spk in ("Speaker1", spkA, spkB):
        mask = load_speaker_word_mask(spike_dir, spk)
        Y = load_spike_matrix(spike_dir, spk, REGION)
        if Y is None or Y.shape[0] != mask.sum():
            print(f"{patient}: SKIP -- missing/mismatched spike matrix for {spk}", flush=True)
            return []
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

    n_self = int(masks["Speaker1"].sum())
    n_a = int(masks[spkA].sum())
    n_b = int(masks[spkB].sum())
    N = min(n_self, n_a, n_b)

    rng = np.random.RandomState(DOWNSAMPLE_SEED)
    masks_bal, Ys_bal = {}, {}
    for spk in ("Speaker1", spkA, spkB):
        true_idx = np.flatnonzero(masks[spk])
        n_spk = len(true_idx)
        rows = np.arange(n_spk) if n_spk == N else np.sort(rng.choice(n_spk, size=N, replace=False))
        kept_true_idx = true_idx[rows]
        m_bal = np.zeros_like(masks[spk])
        m_bal[kept_true_idx] = True
        masks_bal[spk] = m_bal
        Ys_bal[spk] = Ys[spk][rows]

    print(f"{patient}: n_self={n_self}  n_{spkA}={n_a}  n_{spkB}={n_b}  -> global N={N}", flush=True)

    X_raw = np.load(npy_path, mmap_mode="r")
    X_layer_raw = (X_raw[LAYER] if X_raw.ndim == 3 else X_raw).astype(np.float64)
    from sklearn.decomposition import PCA
    X_pca = PCA(n_components=n_components).fit_transform(X_layer_raw)

    rows = []
    for pA, pB in [("Speaker1", spkA), ("Speaker1", spkB), (spkA, spkB)]:
        row = fit_pair(patient, X_pca, masks_bal, Ys_bal, pA, pB, out_tag, n_components)
        if row is not None:
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spike_tag", required=True)
    ap.add_argument("--n_components", type=int, required=True)
    ap.add_argument("--out_tag", required=True)
    ap.add_argument("patients", nargs="*")
    args = ap.parse_args()

    targets = args.patients if args.patients else list(PATIENTS.keys())
    rows = []
    for patient in targets:
        rows.extend(run_patient(patient, args.spike_tag, args.n_components, args.out_tag))

    if not rows:
        print("No results.")
        return

    df = pd.DataFrame(rows)
    df["spike_tag"] = args.spike_tag
    df["n_components"] = args.n_components
    out_csv = os.path.join(RESULTS_DIR, f"threeway_rcross_sweep_{args.out_tag}_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
