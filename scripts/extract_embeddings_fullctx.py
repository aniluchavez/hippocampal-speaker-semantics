"""
extract_embeddings_fullctx.py
=============================
Extract word-level embeddings with FULL causal context (no word-count cap).

For each word at position t:
  - Context: words[0 : t+1]  (all preceding words + current word)
  - Tokenize with RIGHT-padding, LEFT-truncation so:
      * Real tokens sit at natural absolute positions (0, 1, 2, ...)
        which matters for models with absolute pos embeddings (GPT-2, OPT, BERT)
      * When context exceeds model max_length, oldest tokens are dropped
  - Extract hidden state at the FIRST subword of the current word per layer
    (= real_len - n_trailing_special - n_subwords_of_word_t)
  - Save (n_layers, n_words, embed_dim) float32

Key differences from extract_embeddings_ctx200.py:
  - No 200-word cap — each word sees all preceding words
  - RIGHT-padding (not left) so absolute positions are correct for GPT-2 family
  - Extract at FIRST subword of current word (not last), which aligns with
    word onset — the most natural choice for neural encoding analyses

Usage:
    cd /scratch/aniluchavez/hippocampal-speaker-semantics
    nohup conda run -n gpt2_embed --no-capture-output \\
        python3 -u scripts/extract_embeddings_fullctx.py \\
        > /tmp/embed_fullctx.log 2>&1 &
"""

import os, gc
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE",  "1")
os.environ.setdefault("HF_HUB_OFFLINE",        "1")

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, BertConfig, BertModel

CACHE_DIR      = "/scratch/aniluchavez/ConvoDATAS/EmbedCache"
TRANSCRIPT_DIR = "/scratch/aniluchavez/ConvoDATAS/Transcripts"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    p = torch.cuda.get_device_properties(0)
    print(f"  {p.name}  {p.total_memory // 1024**2} MB VRAM")

# ── Model definitions ─────────────────────────────────────────────────────────
# max_tokens: hard cap on sequence length (overrides tokenizer's model_max_length,
#             which is unreliable for some models like OPT)
# batch_size: forward passes per GPU call — smaller for large models / long contexts
# causal:     load BERT with is_decoder=True for causal self-attention
# bfloat16:   use bfloat16 weights (large models only)
MODELS = [
    # {"tag": "gpt2-large",      "hf_id": "gpt2-large",                       "max_tokens": 1024, "batch_size": 16},
    # {"tag": "gpt2",            "hf_id": "gpt2",                             "max_tokens": 1024, "batch_size": 32},
    # {"tag": "gpt2-medium",     "hf_id": "gpt2-medium",                      "max_tokens": 1024, "batch_size": 32},
    # {"tag": "gpt2-xl",         "hf_id": "openai-community/gpt2-xl",         "max_tokens": 1024, "batch_size":  8},
    # {"tag": "bert-base-causal","hf_id": "bert-base-uncased", "causal": True, "max_tokens":  512, "batch_size": 32},
    {"tag": "bert-base",       "hf_id": "bert-base-uncased",                "max_tokens":  512, "batch_size": 32},
    # {"tag": "opt-350m",        "hf_id": "facebook/opt-350m",                "max_tokens": 2048, "batch_size": 16},
    # {"tag": "llama-3.1-8b",    "hf_id": "meta-llama/Meta-Llama-3.1-8B",
    #                             "bfloat16": True, "max_tokens": 8192, "batch_size": 2},
    # {"tag": "llama-2-7b",      "hf_id": "meta-llama/Llama-2-7b-hf",
    #                             "bfloat16": True, "max_tokens": 4096, "batch_size": 2},
]

ALL_PATIENTS = [
    "PTYEU_task147", "PTYEV_task37", "PTYEY_task86", "PTYEZ_task60",
    "PTYFA_task25",  "PTYFC_task28", "PTYFF_task17", "PTYFG_task18",
    "PTYFI_task81",  "PTYFK_task40", "PTYFM_task104","PTYFP_task88",
    "PTYFR_task91",  "PTYFS_task95", "PTYFU_task224",
]

# ── Transcript loader ─────────────────────────────────────────────────────────

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


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(cfg):
    hf_id  = cfg["hf_id"]
    causal = cfg.get("causal", False)
    bf16   = cfg.get("bfloat16", False)
    dtype_kw = {"torch_dtype": torch.bfloat16} if bf16 else {}

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    tokenizer.padding_side   = "right"   # real tokens at natural absolute positions
    tokenizer.truncation_side = "left"   # drop oldest tokens when context is too long
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if causal:
        bert_cfg = BertConfig.from_pretrained(hf_id)
        bert_cfg.is_decoder = True
        model = BertModel.from_pretrained(hf_id, config=bert_cfg, **dtype_kw)
    else:
        model = AutoModel.from_pretrained(hf_id, **dtype_kw)

    model = model.to(device)
    model.eval()
    return tokenizer, model


def get_n_layers_embed_dim(model):
    cfg      = model.config
    embed_dim = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    probe    = torch.zeros((1, 2), dtype=torch.long,
                           device=next(model.parameters()).device)
    with torch.no_grad():
        out = model(probe, output_hidden_states=True)
    n_layers = sum(1 for hs in out.hidden_states if hs.shape[-1] == embed_dim)
    return n_layers, embed_dim


# ── Embedding extraction ──────────────────────────────────────────────────────

def _detect_trailing_special(tokenizer):
    """Return 1 if the tokenizer appends a trailing special token (e.g. BERT's [SEP])."""
    probe_ids = tokenizer("test", add_special_tokens=True)["input_ids"]
    if len(probe_ids) < 2:
        return 0
    last_id = probe_ids[-1]
    special_ids = {tokenizer.sep_token_id, tokenizer.eos_token_id} - {None}
    # GPT-2 sets eos == pad but does NOT append it to sequences — exclude that case
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id == tokenizer.bos_token_id:
        special_ids.discard(tokenizer.eos_token_id)
    return 1 if last_id in special_ids else 0


@torch.no_grad()
def embed_patient(words, tokenizer, model, n_layers, embed_dim, max_tokens, batch_size,
                  trailing_special):
    """
    For each word t: context = words[0:t+1] (full preceding + current word).
    Right-pad to longest in batch, left-truncate to max_tokens.
    Extract at FIRST subword of the current word:
      first_idx = real_len - trailing_special - n_subwords_of_word_t

    n_subwords_of_word_t is computed by tokenizing the word with its natural
    leading space (as it appears in context), without special tokens.

    Returns (n_layers, n_words, embed_dim) float32.
    """
    n_words  = len(words)
    word_emb = np.zeros((n_layers, n_words, embed_dim), dtype=np.float32)

    for batch_start in range(0, n_words, batch_size):
        batch_idx = list(range(batch_start, min(batch_start + batch_size, n_words)))

        context_strings = [" ".join(words[0 : t + 1]) for t in batch_idx]

        tokens = tokenizer(
            context_strings,
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens,
            padding=True,           # right-pad to longest in batch
        ).to(device)

        real_lens = tokens["attention_mask"].sum(dim=1).tolist()

        # Tokenize each word with its natural leading space (no special tokens)
        # to count how many subword tokens it produces
        word_strs   = [(" " if t > 0 else "") + words[t] for t in batch_idx]
        word_n_toks = [
            len(tokenizer(w, add_special_tokens=False)["input_ids"])
            for w in word_strs
        ]
        first_idxs = [
            max(0, int(rl) - trailing_special - wn)
            for rl, wn in zip(real_lens, word_n_toks)
        ]

        out = model(**tokens, output_hidden_states=True)
        hs_list = [hs for hs in out.hidden_states if hs.shape[-1] == embed_dim]

        for li, hs in enumerate(hs_list):
            for j, (t, idx) in enumerate(zip(batch_idx, first_idxs)):
                word_emb[li, t] = hs[j, idx, :].float().cpu().numpy()

        del out, tokens
        torch.cuda.empty_cache()

    return word_emb


# ── Main loop ─────────────────────────────────────────────────────────────────

for model_cfg in MODELS:
    tag        = model_cfg["tag"]
    max_tokens = model_cfg["max_tokens"]
    batch_size = model_cfg["batch_size"]

    missing = [p for p in ALL_PATIENTS
               if not os.path.exists(
                   os.path.join(CACHE_DIR, f"{p}_{tag}_fullctx_word_emb_layers.npy"))]

    if not missing:
        print(f"\n{tag}: all patients cached — skipping")
        continue

    print(f"\n{'#'*70}")
    print(f"MODEL: {tag}  ({model_cfg['hf_id']})"
          + ("  [CAUSAL]" if model_cfg.get("causal") else ""))
    print(f"max_tokens={max_tokens}  batch_size={batch_size}")
    print(f"Missing: {len(missing)} patients")
    print(f"{'#'*70}")

    print("Loading model...", flush=True)
    tokenizer, lm = load_model(model_cfg)
    n_layers, embed_dim = get_n_layers_embed_dim(lm)
    trailing_special = _detect_trailing_special(tokenizer)
    print(f"  n_layers={n_layers}  embed_dim={embed_dim}  "
          f"trailing_special={trailing_special}", flush=True)

    for pid in missing:
        cache = os.path.join(CACHE_DIR, f"{pid}_{tag}_fullctx_word_emb_layers.npy")
        print(f"\n  {pid}...", flush=True)
        words = load_words(pid)
        if words is None:
            continue
        print(f"    {len(words)} words, full context (max {max_tokens} tokens)", flush=True)
        emb = embed_patient(words, tokenizer, lm, n_layers, embed_dim,
                            max_tokens, batch_size, trailing_special)
        np.save(cache, emb)
        print(f"    saved {cache}  shape={emb.shape}", flush=True)

    del lm, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n{tag} done.")

print("\nAll done.")
