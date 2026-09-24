#!/usr/bin/env python3
"""Self vs other r_cross, matched exactly to speaker_pair_r_cross.py's setup
(BERT L12, PC=30, fixed window self=[-200,+300]/other=[+100,+600], spike_tag
fixed_selfm200_otherp100_len500) for a clean comparison against the
speaker-vs-speaker r_cross run.

"other" pools all non-Speaker1 speakers (the standard self/other split used
throughout this project), unlike speaker_pair_r_cross.py which isolates two
individual listening speakers.
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

PATIENTS = [
    "ptYEU_task147", "ptYFF_task17", "ptYFG_task18", "ptYFI_task81",
    "ptYFA_task25", "ptYFK_task40", "ptYEY_task86", "ptYEV_task37",
    "ptYEZ_task60", "ptYFC_task28", "ptYFM_task104", "ptYFP_task88",
    "ptYFR_task91", "ptYFS_task95", "ptYFU_task224",
]


def load_speaker_assignment(spike_dir: str):
    cands = [f for f in os.listdir(spike_dir) if f.endswith("_with_regress_dur.xlsx")]
    tx = pd.read_excel(os.path.join(spike_dir, cands[0]))
    spk_cols = sorted(
        [c for c in tx.columns if str(c).startswith("Speaker")],
        key=lambda c: int(c.replace("Speaker", "").strip())
        if c.replace("Speaker", "").strip().isdigit() else 999,
    )

    def _nn(v):
        return pd.notna(v) and str(v).strip() not in ("", "nan")

    dir_membership = {col: np.array([_nn(v) for v in tx[col]], dtype=bool) for col in spk_cols}
    n = len(tx)
    assign = np.array([None] * n, dtype=object)
    for i in range(n):
        for col in spk_cols:
            if dir_membership[col][i]:
                assign[i] = col
                break
    mask_self = assign == "Speaker1"
    mask_other = np.array([(a is not None and a != "Speaker1") for a in assign], dtype=bool)
    return assign, mask_self, mask_other, dir_membership


def load_spike_matrix(spike_dir: str, speaker: str, region: str):
    spk_dir = os.path.join(spike_dir, speaker)
    if not os.path.isdir(spk_dir):
        return None
    cands = [
        f for f in os.listdir(spk_dir)
        if f.lower().startswith(region.lower()) and f.endswith("_spike_counts.npy")
    ]
    return np.load(os.path.join(spk_dir, cands[0])) if cands else None


def load_other_ordered(spike_dir, region, assign, dir_membership):
    other_spks = sorted([c for c in dir_membership if c != "Speaker1"])
    mats = {s: load_spike_matrix(spike_dir, s, region) for s in other_spks}
    mats = {s: m for s, m in mats.items() if m is not None}
    if not mats:
        return None
    dir_pos = {s: 0 for s in mats}
    rows = []
    for i, spk in enumerate(assign):
        if spk is None or spk == "Speaker1":
            continue
        for s in mats:
            if dir_membership[s][i]:
                if spk == s:
                    rows.append(mats[s][dir_pos[s]])
                dir_pos[s] += 1
    return np.vstack(rows) if rows else None


def run_patient(patient: str):
    patient_ID = "PT" + patient[2:]
    spike_dir = os.path.join(SPIKE_ROOT, f"output_{patient}_english_only_{SPIKE_TAG}")
    npy_path = os.path.join(EMBED_DIR, f"{patient_ID}_{MODEL_TAG}{CONTEXT_TAG}_word_emb_layers.npy")
    if not os.path.exists(npy_path) or not os.path.isdir(spike_dir):
        print(f"{patient}: SKIP -- missing embeddings or spike dir", flush=True)
        return None

    assign, mask_self, mask_other, dir_membership = load_speaker_assignment(spike_dir)
    Y_self = load_spike_matrix(spike_dir, "Speaker1", REGION)
    Y_other = load_other_ordered(spike_dir, REGION, assign, dir_membership)
    if Y_self is None or Y_other is None:
        print(f"{patient}: SKIP -- missing self or other spike matrix", flush=True)
        return None
    if Y_self.shape[0] != mask_self.sum() or Y_other.shape[0] != mask_other.sum():
        print(f"{patient}: SKIP -- row mismatch", flush=True)
        return None
    if Y_self.shape[1] != Y_other.shape[1]:
        print(f"{patient}: SKIP -- neuron count mismatch", flush=True)
        return None

    spike_ok = (Y_self.sum(0) >= MIN_SPIKES) & (Y_other.sum(0) >= MIN_SPIKES)
    n_m = int(spike_ok.sum())
    if n_m == 0:
        print(f"{patient}: SKIP -- 0 neurons pass spike filter", flush=True)
        return None

    X_layer_raw = np.load(npy_path, mmap_mode="r")[LAYER].astype(np.float64)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X_pca = PCA(n_components=N_COMPONENTS).fit_transform(X_layer_raw)
    X_self = StandardScaler().fit_transform(X_pca[mask_self])
    X_other = StandardScaler().fit_transform(X_pca[mask_other])
    Y_self_f = Y_self[:, spike_ok].astype(np.float64)
    Y_other_f = Y_other[:, spike_ok].astype(np.float64)

    print(f"{patient}: self={X_self.shape[0]}w  other={X_other.shape[0]}w  neurons={n_m}", flush=True)

    cfg = _rel.ReliabilityConfig(n_null=100, n_half_splits=100, n_jobs=12, random_state=2026)
    res = _rel.run_beta_reliability_all_neurons(
        X_self=X_self, X_other=X_other, Y_self=Y_self_f, Y_other=Y_other_f, cfg=cfg, verbose=False,
    )
    r_cross_vals = [r["r_cross"] for r in res if np.isfinite(r.get("r_cross", np.nan))]
    med = float(np.median(r_cross_vals)) if r_cross_vals else float("nan")
    print(f"  -> n_neurons_valid={len(r_cross_vals)}/{n_m}  median r_cross(self,other)={med:.4f}", flush=True)

    out_path = os.path.join(RESULTS_DIR, f"self_other_rcross_{patient}_bert_L12_pc30.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"patient": patient, "n_neurons": n_m, "reliability": res}, f)

    return {
        "patient": patient, "n_words_self": int(X_self.shape[0]), "n_words_other": int(X_other.shape[0]),
        "n_neurons": n_m, "n_neurons_valid": len(r_cross_vals), "median_r_cross": med,
    }


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else PATIENTS
    rows = []
    for patient in targets:
        row = run_patient(patient)
        if row is not None:
            rows.append(row)

    if not rows:
        print("No results.")
        return

    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, "self_other_rcross_bert_L12_pc30_summary.csv")
    if os.path.exists(out_csv):
        prior = pd.read_csv(out_csv)
        prior = prior[~prior["patient"].isin(df["patient"])]
        df = pd.concat([prior, df], ignore_index=True)
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
