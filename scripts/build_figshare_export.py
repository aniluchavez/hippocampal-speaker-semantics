"""
build_figshare_export.py
=========================
Assemble a self-contained figshare export: for each patient, hippocampus-only
spike-count matrices for spoken (self) and heard (other) words, paired with
the corresponding row-aligned embeddings (BERT bidirmax, BERT bidirmax
speaker-tagged, GPT2-xl ctx200 layer 48).

Window convention: self = [onset-150ms, onset+350ms], other = [onset+200ms,
onset+700ms], both 500ms (SpikeWindows tag tshift-150_tlen500_oshift+200_olen500).

Alignment logic (assign/order words to self vs. other, and order "other"
across multiple speaker directories) is copied from semantic_glm.py's
load_speaker_assignment / load_spike_matrix / load_condition_ordered — same
functions the main GLM pipeline uses, so this export is guaranteed consistent
with every existing result built on this window.

Every patient is hard-validated before saving:
  - spike row count == mask.sum() for that condition
  - embedding row count (post self/other split) == spike row count
  - neuron count consistent across the two conditions
Any patient that fails is skipped with a printed reason, never silently
misaligned.

Output files are named by an anonymous subject_id (sub-01..sub-NN), assigned
via an unseeded shuffle at runtime. This mapping back to patient_ID/task is
intentionally never printed or written anywhere (not to disk, not to stdout
alongside the subject_id) — by design there is no reversible key. Re-running
this script reassigns subject codes fresh (existing output files with old
patient_ID-based names should be deleted first, since nothing here removes
stale files).

Usage:
    cd /scratch/aniluchavez/hippocampal-speaker-semantics
    python3 -u scripts/build_figshare_export.py
"""

import os
import random
import numpy as np
import pandas as pd

SPIKE_ROOT = "/scratch/aniluchavez/ConvoDATAS/SpikeWindows"
EMBED_DIR  = "/scratch/aniluchavez/ConvoDATAS/EmbedCache"
OUT_DIR    = "/scratch/aniluchavez/hippocampal-speaker-semantics/figshare"
WINDOW_TAG = "tshift-150_tlen500_oshift+200_olen500"
REGION     = "hippocampus"

EMBED_SPECS = {
    "bert":         dict(tag="bert-base-bidirmax",         layer=None),
    "bert_spktag":  dict(tag="bert-base-bidirmax-spktag",  layer=None),
    "gpt2":         dict(tag="gpt2-xl_ctx200",              layer=48),
}

PATIENTS = [
    {"patient_ID": "PTYEU_task147", "patient": "ptYEU_task147"},
    {"patient_ID": "PTYEV_task37",  "patient": "ptYEV_task37"},
    {"patient_ID": "PTYEY_task86",  "patient": "ptYEY_task86"},
    {"patient_ID": "PTYEZ_task60",  "patient": "ptYEZ_task60"},
    {"patient_ID": "PTYFA_task25",  "patient": "ptYFA_task25"},
    {"patient_ID": "PTYFC_task28",  "patient": "ptYFC_task28"},
    {"patient_ID": "PTYFF_task17",  "patient": "ptYFF_task17"},
    {"patient_ID": "PTYFG_task18",  "patient": "ptYFG_task18"},
    {"patient_ID": "PTYFI_task81",  "patient": "ptYFI_task81"},
    {"patient_ID": "PTYFK_task40",  "patient": "ptYFK_task40"},
    {"patient_ID": "PTYFM_task104", "patient": "ptYFM_task104"},
    {"patient_ID": "PTYFP_task88",  "patient": "ptYFP_task88"},
    {"patient_ID": "PTYFR_task91",  "patient": "ptYFR_task91"},
    {"patient_ID": "PTYFS_task95",  "patient": "ptYFS_task95"},
    {"patient_ID": "PTYFU_task224", "patient": "ptYFU_task224"},
]


def find_spike_dir(patient):
    d = os.path.join(SPIKE_ROOT, f"output_{patient}_english_only_{WINDOW_TAG}")
    return d if os.path.isdir(d) else None


def load_spike_matrix(spike_dir, speaker, region):
    spk_dir = os.path.join(spike_dir, speaker)
    if not os.path.isdir(spk_dir):
        return None
    cands = [f for f in os.listdir(spk_dir)
              if f.lower().startswith(region.lower()) and f.endswith("_spike_counts.npy")]
    return np.load(os.path.join(spk_dir, cands[0])) if cands else None


def load_speaker_assignment(spike_dir):
    cands = [f for f in os.listdir(spike_dir) if f.endswith("_with_regress_dur.xlsx")]
    if not cands:
        raise FileNotFoundError(f"No _with_regress_dur.xlsx in {spike_dir}")
    tx = pd.read_excel(os.path.join(spike_dir, cands[0]))
    spk_cols = sorted(
        [c for c in tx.columns if str(c).startswith("Speaker")],
        key=lambda c: int(c.replace("Speaker", "").strip())
                      if c.replace("Speaker", "").strip().isdigit() else 999,
    )
    def _nn(val):
        return pd.notna(val) and str(val).strip() not in ("", "nan")
    dir_membership = {
        col: np.array([_nn(v) for v in tx[col]], dtype=bool) for col in spk_cols
    }
    n = len(tx)
    assign = np.array([None] * n, dtype=object)
    for i in range(n):
        for col in spk_cols:
            if dir_membership[col][i]:
                assign[i] = col; break
    mask_self  = assign == "Speaker1"
    mask_other = np.array([(a is not None and a != "Speaker1") for a in assign], dtype=bool)
    return assign, mask_self, mask_other, dir_membership, n


def load_condition_ordered(spike_dir, cond, region, spk_assignment, dir_membership):
    if cond == "self":
        return load_spike_matrix(spike_dir, "Speaker1", region)
    other_spks = sorted([c for c in dir_membership if c != "Speaker1"])
    mats = {s: load_spike_matrix(spike_dir, s, region) for s in other_spks}
    mats = {s: m for s, m in mats.items() if m is not None}
    if not mats:
        return None
    dir_pos = {s: 0 for s in mats}
    rows = []
    for i, spk in enumerate(spk_assignment):
        if spk is None or spk == "Speaker1":
            continue
        for s in mats:
            if dir_membership[s][i]:
                if spk == s:
                    rows.append(mats[s][dir_pos[s]])
                dir_pos[s] += 1
    return np.vstack(rows) if rows else None


def load_embedding_full(patient_ID, spec):
    path = os.path.join(EMBED_DIR, f"{patient_ID}_{spec['tag']}_word_emb_layers.npy")
    if not os.path.exists(path):
        return None
    arr = np.load(path, mmap_mode="r")
    if arr.ndim == 3:
        if spec["layer"] is None:
            raise ValueError(f"{path} is 3D but no layer specified")
        return np.asarray(arr[spec["layer"]], dtype=np.float32)
    elif arr.ndim == 2:
        return np.asarray(arr, dtype=np.float32)
    raise ValueError(f"Unexpected embedding shape for {path}: {arr.shape}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest_rows = []

    # Unseeded shuffle: output subject codes carry no reconstructable link back
    # to patient_ID/task order, and nothing below ever prints or saves the
    # pid <-> subject_id correspondence together, so no mapping persists
    # anywhere (by design — see docstring).
    subject_ids = [f"sub-{i:02d}" for i in range(1, len(PATIENTS) + 1)]
    random.shuffle(subject_ids)

    for cfg, subject_id in zip(PATIENTS, subject_ids):
        pid, pat = cfg["patient_ID"], cfg["patient"]
        print(f"\n{'='*60}\n  processing next patient...", flush=True)

        spike_dir = find_spike_dir(pat)
        if spike_dir is None:
            print("  SKIP — spike dir not found"); continue

        try:
            spk_assignment, mask_self, mask_other, dir_membership, n_words = \
                load_speaker_assignment(spike_dir)
        except FileNotFoundError as e:
            print(f"  SKIP — {e}"); continue

        n_self, n_other = int(mask_self.sum()), int(mask_other.sum())

        Y_self  = load_condition_ordered(spike_dir, "self",  REGION, spk_assignment, dir_membership)
        Y_other = load_condition_ordered(spike_dir, "other", REGION, spk_assignment, dir_membership)

        if Y_self is None or Y_other is None:
            print(f"  SKIP — missing spike matrix (self={Y_self is not None}, other={Y_other is not None})")
            continue
        if Y_self.shape[0] != n_self:
            print(f"  SKIP — self spike rows ({Y_self.shape[0]}) != mask_self.sum() ({n_self})"); continue
        if Y_other.shape[0] != n_other:
            print(f"  SKIP — other spike rows ({Y_other.shape[0]}) != mask_other.sum() ({n_other})"); continue
        if Y_self.shape[1] != Y_other.shape[1]:
            print(f"  SKIP — neuron count mismatch self={Y_self.shape[1]} other={Y_other.shape[1]}"); continue
        n_neurons = Y_self.shape[1]

        # A handful of rows in the raw SpikeWindows source are all-NaN for a given
        # word (window couldn't be computed, e.g. too close to a recording edge).
        # Drop those rows from spikes AND every embedding for that condition, in
        # lockstep, so nothing NaN reaches the saved files and row alignment is
        # preserved across all four categories.
        valid_self  = ~np.isnan(Y_self).any(axis=1)
        valid_other = ~np.isnan(Y_other).any(axis=1)
        n_dropped_self, n_dropped_other = int((~valid_self).sum()), int((~valid_other).sum())
        if n_dropped_self or n_dropped_other:
            print(f"  Dropping NaN rows: spoken {n_dropped_self}/{n_self}, heard {n_dropped_other}/{n_other}")
        Y_self, Y_other = Y_self[valid_self], Y_other[valid_other]
        n_self_valid, n_other_valid = int(valid_self.sum()), int(valid_other.sum())

        embeds_self, embeds_other = {}, {}
        ok = True
        for name, spec in EMBED_SPECS.items():
            X_full = load_embedding_full(pid, spec)
            if X_full is None:
                print(f"  SKIP — embedding '{name}' missing"); ok = False; break
            if X_full.shape[0] != n_words:
                print(f"  SKIP — embedding '{name}' rows ({X_full.shape[0]}) != transcript rows ({n_words})")
                ok = False; break
            X_self, X_other = X_full[mask_self][valid_self], X_full[mask_other][valid_other]
            if X_self.shape[0] != n_self_valid or X_other.shape[0] != n_other_valid:
                print(f"  SKIP — embedding '{name}' self/other row split mismatch"); ok = False; break
            embeds_self[name], embeds_other[name] = X_self, X_other
        if not ok:
            continue

        if np.isnan(Y_self).any() or np.isnan(Y_other).any():
            print("  SKIP — NaN survived filtering (unexpected)"); continue
        if (Y_self < 0).any() or (Y_other < 0).any():
            print("  SKIP — negative spike count (unexpected)"); continue
        n_self, n_other = n_self_valid, n_other_valid

        categories = {
            "spikes":        ("spike_counts", Y_self,               Y_other),
            "bert":          ("embeddings",   embeds_self["bert"],         embeds_other["bert"]),
            "bert_spktag":   ("embeddings",   embeds_self["bert_spktag"],  embeds_other["bert_spktag"]),
            "gpt2":          ("embeddings",   embeds_self["gpt2"],         embeds_other["gpt2"]),
        }
        for cat_name, (key, arr_spoken, arr_heard) in categories.items():
            cat_dir = os.path.join(OUT_DIR, cat_name)
            os.makedirs(cat_dir, exist_ok=True)
            np.savez_compressed(os.path.join(cat_dir, f"{subject_id}_hippocampus_spoken.npz"), **{key: arr_spoken})
            np.savez_compressed(os.path.join(cat_dir, f"{subject_id}_hippocampus_heard.npz"),  **{key: arr_heard})

        print(f"  OK  n_neurons={n_neurons}  spoken={n_self}  heard={n_other}")
        print(f"  Saved: {OUT_DIR}/{{spikes,bert,bert_spktag,gpt2}}/{subject_id}_hippocampus_{{spoken,heard}}.npz")

        manifest_rows.append(dict(
            subject_id=subject_id, region=REGION, n_neurons=n_neurons,
            n_words_spoken=n_self, n_words_heard=n_other,
            bert_dim=embeds_self["bert"].shape[1],
            bert_spktag_dim=embeds_self["bert_spktag"].shape[1],
            gpt2_dim=embeds_self["gpt2"].shape[1],
        ))

    manifest = pd.DataFrame(manifest_rows).sort_values("subject_id").reset_index(drop=True)
    manifest_path = os.path.join(OUT_DIR, "manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    print(f"\n{'='*60}\nManifest ({len(manifest)}/{len(PATIENTS)} subjects): {manifest_path}")
    print(manifest.to_string(index=False))
    print("\nNo patient_ID <-> subject_id mapping was printed or saved anywhere in this run.")


if __name__ == "__main__":
    main()
