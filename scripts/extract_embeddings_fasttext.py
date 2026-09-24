"""
Extract static (non-contextual) fastText word embeddings for the ctx200 patient
set — a baseline to contrast against the contextual transformer embeddings
(BERT/GPT2/LLaMA) used elsewhere in this project.

Model: fastText wiki-news-300d-1M (1M-word vocab, 300-dim, trained on
Wikipedia 2017 + UMBC + statmt.org news), downloaded from fastText's site.

Each word gets exactly one vector regardless of context (a simple vocab
lookup), saved with shape (1, n_words, 300) so it's a drop-in match for
semantic_glm.py's `np.load(npy_path, mmap_mode='r')[LAYER]` loader at LAYER=0.

OOV handling: try exact match, then lowercase; if still missing, use the
zero vector (counted and reported per patient).

Usage:
  cd /scratch/aniluchavez/hippocampal-speaker-semantics
  python3 -u scripts/extract_embeddings_fasttext.py
"""

import os, re, string, time
import numpy as np
import pandas as pd

# Some patients' CleanedWord still carries raw punctuation/capitalization
# (trailing commas, "?", "...", capitalized sentence-starts) and contractions
# fastText's vocab tokenizes as separate sub-tokens (e.g. "'s", "'t" exist,
# "it's" doesn't). Expand common contractions to their two-word form so both
# halves resolve; fall back to the pre-apostrophe stem for possessives/names.
_CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "can not", "couldn't": "could not", "won't": "will not",
    "wouldn't": "would not", "shouldn't": "should not", "mightn't": "might not",
    "mustn't": "must not", "isn't": "is not", "aren't": "are not",
    "wasn't": "was not", "weren't": "were not", "hasn't": "has not",
    "haven't": "have not", "hadn't": "had not",
    "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is",
    "it's": "it is", "we're": "we are", "they're": "they are",
    "that's": "that is", "there's": "there is", "here's": "here is",
    "what's": "what is", "who's": "who is", "let's": "let us",
    "i've": "i have", "you've": "you have", "we've": "we have",
    "they've": "they have", "i'll": "i will", "you'll": "you will",
    "he'll": "he will", "she'll": "she will", "we'll": "we will",
    "they'll": "they will", "i'd": "i would", "you'd": "you would",
    "he'd": "he would", "she'd": "she would", "we'd": "we would",
    "they'd": "they would",
}

TRANSCRIPT_DIR = "/scratch/aniluchavez/ConvoDATAS/Transcripts"
EMBED_DIR      = "/scratch/aniluchavez/ConvoDATAS/EmbedCache"
VEC_PATH       = "/scratch/aniluchavez/ConvoDATAS/StaticEmbeds/wiki-news-300d-1M.vec"
KV_CACHE_PATH  = "/scratch/aniluchavez/ConvoDATAS/StaticEmbeds/wiki-news-300d-1M.kv"
MODEL_TAG      = "fasttext-wiki"

ALL_PATIENTS = [
    "PTYEU_task147", "PTYEV_task37", "PTYEY_task86", "PTYEZ_task60",
    "PTYFA_task25",  "PTYFC_task28", "PTYFF_task17", "PTYFG_task18",
    "PTYFI_task81",  "PTYFK_task40", "PTYFM_task104","PTYFP_task88",
    "PTYFR_task91",  "PTYFS_task95", "PTYFU_task224",
]


def load_words(pid):
    xls = os.path.join(TRANSCRIPT_DIR,
                       f"{pid}_filtered_used_rows_withNP_withClusterIDNew.xlsx")
    if not os.path.exists(xls):
        print(f"  SKIP — transcript not found: {xls}")
        return None
    xl    = pd.ExcelFile(xls)
    sheet = "Sheet1" if "Sheet1" in xl.sheet_names else xl.sheet_names[0]
    df    = xl.parse(sheet, keep_default_na=False)
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df = df.dropna(subset=["onset"]).sort_values("onset").reset_index(drop=True)
    spk_cols = sorted([c for c in df.columns if c.lower().startswith("speaker")])
    if "CleanedWord" not in df.columns:
        df["CleanedWord"] = df[spk_cols].apply(
            lambda row: next(
                (str(v).strip() for v in row
                 if str(v).strip() not in ("", "nan", "xxx")), ""),
            axis=1)
    return df["CleanedWord"].astype(str).str.strip().tolist()


def load_fasttext_kv():
    from gensim.models import KeyedVectors
    if os.path.exists(KV_CACHE_PATH):
        print(f"  Loading cached binary KeyedVectors: {KV_CACHE_PATH}", flush=True)
        return KeyedVectors.load(KV_CACHE_PATH)
    print(f"  Parsing text vectors (slow, ~3-4 min): {VEC_PATH}", flush=True)
    t0 = time.time()
    kv = KeyedVectors.load_word2vec_format(VEC_PATH, binary=False)
    print(f"  Parsed {len(kv)} vectors in {time.time()-t0:.0f}s", flush=True)
    kv.save(KV_CACHE_PATH)
    print(f"  Cached binary copy → {KV_CACHE_PATH}", flush=True)
    return kv


def _lookup(kv, w):
    if w in kv.key_to_index:
        return kv[w]
    lw = w.lower()
    if lw in kv.key_to_index:
        return kv[lw]
    return None


def word_vector(kv, word, dim):
    v = _lookup(kv, word)
    if v is not None:
        return v, True

    stripped = word.strip(string.punctuation)
    v = _lookup(kv, stripped)
    if v is not None:
        return v, True

    lw = stripped.lower()
    if lw in _CONTRACTIONS:
        parts = _CONTRACTIONS[lw].split()
        vecs = [_lookup(kv, p) for p in parts]
        vecs = [v for v in vecs if v is not None]
        if vecs:
            return np.mean(vecs, axis=0), True

    if "'" in stripped:
        stem = stripped.split("'")[0]
        v = _lookup(kv, stem)
        if v is not None:
            return v, True

    return np.zeros(dim, dtype=np.float32), False


def main():
    kv  = load_fasttext_kv()
    dim = kv.vector_size

    for pid in ALL_PATIENTS:
        out_path = os.path.join(EMBED_DIR, f"{pid}_{MODEL_TAG}_word_emb_layers.npy")
        if os.path.exists(out_path):
            print(f"[SKIP] {pid} — already exists", flush=True)
            continue

        words = load_words(pid)
        if words is None:
            continue

        n_words = len(words)
        emb = np.zeros((1, n_words, dim), dtype=np.float32)
        n_oov = 0
        for i, w in enumerate(words):
            vec, found = word_vector(kv, w, dim)
            emb[0, i] = vec
            if not found:
                n_oov += 1

        np.save(out_path, emb)
        print(f"  {pid}: {n_words} words, {n_oov} OOV ({100*n_oov/n_words:.1f}%)  "
              f"→ {out_path}", flush=True)

    print("\nAll patients done.", flush=True)


if __name__ == "__main__":
    main()
