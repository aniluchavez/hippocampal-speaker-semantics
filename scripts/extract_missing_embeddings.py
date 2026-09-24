"""
extract_missing_embeddings.py
=============================
Generate EmbedCache .npy files for any model/patient combinations
that are missing. Safe to re-run — skips already-cached files.

Saves (n_layers, n_words, embed_dim) float32 arrays to EmbedCache,
identical format to existing files.

Usage:
    cd /scratch/aniluchavez/hippocampal-speaker-semantics
    conda run -n gpt2_embed python3 -u scripts/extract_missing_embeddings.py \
        > /tmp/extract_embeddings.log 2>&1
"""

import os, gc, math
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE",  "1")
os.environ.setdefault("HF_HUB_OFFLINE",        "1")

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

CACHE_DIR   = "/scratch/aniluchavez/ConvoDATAS/EmbedCache"
TRANSCRIPT_DIR = "/scratch/aniluchavez/ConvoDATAS/Transcripts"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    p = torch.cuda.get_device_properties(0)
    print(f"  {p.name}  {p.total_memory//1024**2} MB VRAM")

# ── Models to generate ────────────────────────────────────────────────────────
# Add/remove entries as needed. Cache-hit check means already-done combos skip.
MODELS = [
    {"tag": "gpt2",        "hf_id": "gpt2",         "max_ctx": 1024, "special_tokens": False},
    {"tag": "gpt2-medium", "hf_id": "gpt2-medium",  "max_ctx": 1024, "special_tokens": False},
    {"tag": "gpt2-xl",     "hf_id": "openai-community/gpt2-xl", "max_ctx": 1024, "special_tokens": False},
]

# All 15 patients — script will skip any that already have a cached file
ALL_PATIENTS = [
    "PTYEU_task147", "PTYEV_task37", "PTYEY_task86", "PTYEZ_task60",
    "PTYFA_task25",  "PTYFC_task28", "PTYFF_task17", "PTYFG_task18",
    "PTYFI_task81",  "PTYFK_task40", "PTYFM_task104","PTYFP_task88",
    "PTYFR_task91",  "PTYFS_task95", "PTYFU_task224",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_words(pid):
    """Load all words in temporal order from the patient transcript xlsx."""
    xls = os.path.join(TRANSCRIPT_DIR,
                       f"{pid}_filtered_used_rows_withNP_withClusterIDNew.xlsx")
    if not os.path.exists(xls):
        print(f"  SKIP — transcript not found: {xls}")
        return None
    xl  = pd.ExcelFile(xls)
    sheet = "Sheet1" if "Sheet1" in xl.sheet_names else xl.sheet_names[0]
    df  = xl.parse(sheet, keep_default_na=False)
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df = df.dropna(subset=["onset"]).sort_values("onset").reset_index(drop=True)
    spk_cols = sorted([c for c in df.columns if c.lower().startswith("speaker")])
    if "CleanedWord" not in df.columns:
        df["CleanedWord"] = df[spk_cols].apply(
            lambda row: next((str(v).strip() for v in row
                              if str(v).strip() not in ("", "nan", "xxx")), ""),
            axis=1)
    return df["CleanedWord"].astype(str).str.strip().tolist()


def get_n_layers_embed_dim(model):
    cfg = model.config
    embed_dim = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    probe = torch.zeros((1, 2), dtype=torch.long,
                        device=next(model.parameters()).device)
    with torch.no_grad():
        out = model(probe, output_hidden_states=True)
    n_layers = sum(1 for hs in out.hidden_states if hs.shape[-1] == embed_dim)
    return n_layers, embed_dim


@torch.no_grad()
def embed_patient(all_words, tokenizer, model, n_layers, embed_dim,
                  max_ctx, special_tokens):
    """
    Sliding-window tokenization → pool to word level → shape (n_layers, n_words, embed_dim).
    """
    unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0
    all_tok_ids, word_ranges = [], []
    for w in all_words:
        toks = tokenizer.encode(" " + w, add_special_tokens=False) or [unk_id]
        word_ranges.append((len(all_tok_ids), len(all_tok_ids) + len(toks)))
        all_tok_ids.extend(toks)

    n_tokens = len(all_tok_ids)
    content  = max_ctx - (2 if special_tokens else 0)
    stride   = content // 2

    all_hidden = np.zeros((n_layers, n_tokens, embed_dim), dtype=np.float32)
    counts     = np.zeros(n_tokens, dtype=np.float32)

    for cs in range(0, n_tokens, stride):
        ce       = min(n_tokens, cs + content)
        chunk    = all_tok_ids[cs:ce]
        ids      = ([tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]
                    if special_tokens else chunk)
        toks_t   = torch.tensor([ids], device=device)
        out      = model(toks_t, attention_mask=torch.ones_like(toks_t),
                         output_hidden_states=True)
        hs_list  = [hs for hs in out.hidden_states if hs.shape[-1] == embed_dim]
        for li, hs in enumerate(hs_list):
            h = hs.squeeze(0).float().cpu().numpy()
            if special_tokens:
                h = h[1:-1]
            all_hidden[li, cs:ce] += h
        counts[cs:ce] += 1
        del out, toks_t; torch.cuda.empty_cache()

    all_hidden /= np.maximum(counts, 1)[np.newaxis, :, np.newaxis]

    word_emb = np.zeros((n_layers, len(all_words), embed_dim), dtype=np.float32)
    for wi, (ts, te) in enumerate(word_ranges):
        if te > ts:
            word_emb[:, wi] = all_hidden[:, ts:te].mean(axis=1)

    return word_emb


# ── Main loop ─────────────────────────────────────────────────────────────────

for model_cfg in MODELS:
    tag, hf_id = model_cfg["tag"], model_cfg["hf_id"]
    max_ctx, special_tokens = model_cfg["max_ctx"], model_cfg["special_tokens"]

    missing = [p for p in ALL_PATIENTS
               if not os.path.exists(os.path.join(CACHE_DIR,
                                                   f"{p}_{tag}_word_emb_layers.npy"))]
    if not missing:
        print(f"\n{tag}: all {len(ALL_PATIENTS)} patients already cached — skipping")
        continue

    print(f"\n{'#'*70}")
    print(f"MODEL: {tag}  ({hf_id})")
    print(f"Missing: {len(missing)} patients: {missing}")
    print(f"{'#'*70}")

    print(f"Loading {tag}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    lm = AutoModel.from_pretrained(hf_id).to(device)
    lm.eval()
    n_layers, embed_dim = get_n_layers_embed_dim(lm)
    print(f"  n_layers={n_layers}  embed_dim={embed_dim}", flush=True)

    for pid in missing:
        cache = os.path.join(CACHE_DIR, f"{pid}_{tag}_word_emb_layers.npy")
        print(f"\n  {pid}...", flush=True)
        words = load_words(pid)
        if words is None:
            continue
        print(f"    {len(words)} words", flush=True)
        word_emb = embed_patient(words, tokenizer, lm,
                                 n_layers, embed_dim, max_ctx, special_tokens)
        np.save(cache, word_emb)
        print(f"    saved {cache}  shape={word_emb.shape}", flush=True)

    del lm, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n{tag} done.")

print("\nAll done.")
