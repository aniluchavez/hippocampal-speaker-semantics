"""
compare_collab_embeddings.py
============================
Compare our gpt2-large ctx200 embeddings against the collaborator's
word_embeddings_gpt2.npy for every overlapping patient.

Our format:   EmbedCache/{pid}_gpt2-large_ctx200_word_emb_layers.npy
              shape (37, n_words, 1280)  — (n_layers, n_words, embed_dim)
              last layer = [36, :, :]

Collab format: /mnt/labworlds/.../Y{XY}/embeddings/word_embeddings_gpt2.npy
              shape (n_words, 37, 1280)  — (n_words, n_layers, embed_dim)
              last layer = [:, -1, :]

Usage:
    cd /scratch/aniluchavez/hippocampal-speaker-semantics
    python3 -u scripts/compare_collab_embeddings.py
"""

import os
import numpy as np

CACHE_DIR   = "/scratch/aniluchavez/ConvoDATAS/EmbedCache"
COLLAB_DIR  = "/mnt/labworlds/Hayden/Hayden_Lab/speech_247/anilu_comparison"

# Full patient list (same as extract_embeddings_ctx200.py)
ALL_PATIENTS = [
    "PTYEU_task147", "PTYEV_task37", "PTYEY_task86", "PTYEZ_task60",
    "PTYFA_task25",  "PTYFC_task28", "PTYFF_task17", "PTYFG_task18",
    "PTYFI_task81",  "PTYFK_task40", "PTYFM_task104","PTYFP_task88",
    "PTYFR_task91",  "PTYFS_task95", "PTYFU_task224",
]

def collab_id(pid):
    """PTYEY_task86 -> YEY"""
    return "Y" + pid[3:5]


def cosine_sim_matrix(A, B):
    """Row-wise cosine similarity between two (n, d) arrays."""
    A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return (A_n * B_n).sum(axis=1)


for pid in ALL_PATIENTS:
    cid = collab_id(pid)

    our_path    = os.path.join(CACHE_DIR, f"{pid}_gpt2-large_ctx200_word_emb_layers.npy")
    collab_path = os.path.join(COLLAB_DIR, cid, "embeddings", "word_embeddings_gpt2.npy")

    if not os.path.exists(our_path):
        print(f"{pid}: SKIP — our embedding not found (run extract_embeddings_ctx200.py first)")
        continue
    if not os.path.exists(collab_path):
        print(f"{pid}: SKIP — collaborator embedding not found at {collab_path}")
        continue

    our   = np.load(our_path,    mmap_mode="r")   # (37, n_words, 1280)
    collab = np.load(collab_path, mmap_mode="r")  # (n_words, 37, 1280)

    our_last    = np.array(our[36, :, :],    dtype=np.float32)   # (n_words, 1280)
    collab_last = np.array(collab[:, -1, :], dtype=np.float32)   # (n_words, 1280)

    n_ours   = our_last.shape[0]
    n_collab = collab_last.shape[0]

    if n_ours != n_collab:
        print(f"{pid}: word count mismatch — ours={n_ours}, collab={n_collab}")
        # compare on the shorter prefix so we can still see similarity
        n = min(n_ours, n_collab)
        our_last    = our_last[:n]
        collab_last = collab_last[:n]
    else:
        n = n_ours

    cos  = cosine_sim_matrix(our_last, collab_last)
    diff = np.abs(our_last - collab_last)

    print(f"\n{'='*60}")
    print(f"Patient : {pid}  <->  {cid}  (n_words={n_ours}, collab={n_collab})")
    print(f"  Cosine sim  — mean={cos.mean():.6f}  min={cos.min():.6f}  "
          f"median={np.median(cos):.6f}")
    print(f"  Abs diff    — mean={diff.mean():.6f}  max={diff.max():.6f}  "
          f"p99={np.percentile(diff, 99):.6f}")
    print(f"  Max element — ours={np.abs(our_last).max():.4f}  "
          f"collab={np.abs(collab_last).max():.4f}")

    # Flag if they look identical vs just correlated
    if diff.max() < 1e-4:
        print("  >>> MATCH: arrays are numerically identical (max diff < 1e-4)")
    elif cos.mean() > 0.999:
        print("  >>> CLOSE: very high cosine similarity but not identical")
    else:
        print("  >>> DIFFER: embeddings are not the same — likely different "
              "context/tokenizer settings")

print("\nDone.")
