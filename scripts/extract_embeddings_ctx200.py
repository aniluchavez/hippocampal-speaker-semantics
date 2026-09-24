"""
extract_embeddings_ctx200.py
============================
Extract word-level embeddings using a causal context window.

For each word at position t:
  - Build context: all_words[max(0, t-CONTEXT_WORDS) : t+1]
    (up to CONTEXT_WORDS preceding words + the current word)
  - Tokenize with RIGHT-padding, LEFT-truncation so real tokens sit at
    natural absolute positions (matters for GPT-2/OPT/BERT's absolute
    position embeddings)
  - Run model → extract hidden state at the FIRST subword of the current
    word, for ALL hidden-state layers
    (= real_len - n_trailing_special - n_subwords_of_word_t)
  - Save (n_layers, n_words, embed_dim) float32 to EmbedCache

By default this gives every word a fixed causal context of ≤200 words, consistently
across all models.  Models with large context limits (LLaMA, 8192 tokens)
previously saw the entire conversation in one pass — this forces a controlled
200-word window so comparisons across model families are apples-to-apples.
Extracting at the FIRST subword (not the last) aligns with word onset — the
most natural choice for neural encoding analyses. Same convention as
extract_embeddings_fullctx.py.

Causal BERT (bert-base-causal):
  bert-base-uncased loaded with is_decoder=True enables causal self-attention.
  Same weights as standard BERT, same architecture, but each token only attends
  to previous tokens.  Allows direct bidirectional vs unidirectional comparison.

Usage:
    cd /scratch/aniluchavez/hippocampal-speaker-semantics
    nohup conda run -n gpt2_embed --no-capture-output \
        python3 -u scripts/extract_embeddings_ctx200.py \
            --context_words 200 --max_tokens 512 \
        > /tmp/embed_ctx200.log 2>&1 &
"""

import os, gc, argparse
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE",  "1")
os.environ.setdefault("HF_HUB_OFFLINE",        "1")

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, BertConfig, BertModel

CACHE_DIR      = "/scratch/aniluchavez/ConvoDATAS/EmbedCache"
TRANSCRIPT_DIR = "/scratch/aniluchavez/ConvoDATAS/Transcripts"
BATCH_SIZE     = 32   # context strings per GPU forward pass

def _cli_args():
    p = argparse.ArgumentParser()
    p.add_argument("--patients", type=str, default=None,
                   help="Comma-separated patient_IDs to (re)extract; all if omitted "
                        "(e.g. for regenerating a single patient after a transcript fix)")
    p.add_argument("--models", type=str, default=None,
                   help="Comma-separated model tags to run; all if omitted")
    p.add_argument("--context_words", type=int, default=200,
                   help="Number of preceding words to include before the current word")
    p.add_argument("--context_tag", type=str, default=None,
                   help="Output cache context tag. Defaults to _ctx{context_words}.")
    p.add_argument("--max_tokens", type=int, default=512,
                   help="Tokenizer max_length cap. Original ctx200 cache used 512; "
                        "GPT-2 XL can use 1024 for larger context-window sweeps.")
    p.add_argument("--overwrite", action="store_true",
                   help="Regenerate caches even if output files already exist.")
    return p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    p = torch.cuda.get_device_properties(0)
    print(f"  {p.name}  {p.total_memory // 1024**2} MB VRAM")

# ── Model definitions ─────────────────────────────────────────────────────────
# causal=True  → load with is_decoder=True (causal BERT)
# bfloat16     → use bfloat16 weights (large models only)
MODELS = [
    {"tag": "gpt2",            "hf_id": "gpt2"},
    {"tag": "gpt2-medium",     "hf_id": "gpt2-medium"},
    {"tag": "gpt2-large",      "hf_id": "gpt2-large"},
    {"tag": "gpt2-xl",         "hf_id": "openai-community/gpt2-xl"},
    {"tag": "bert-base",        "hf_id": "bert-base-uncased"},
    {"tag": "bert-base-causal", "hf_id": "bert-base-uncased", "causal": True},
    {"tag": "llama-3.1-8b",    "hf_id": "meta-llama/Meta-Llama-3.1-8B",   "bfloat16": True},
    {"tag": "llama-2-7b",      "hf_id": "meta-llama/Llama-2-7b-hf",       "bfloat16": True},
    {"tag": "opt-350m",        "hf_id": "facebook/opt-350m"},
    {"tag": "roberta-base",    "hf_id": "roberta-base"},
    {"tag": "gemma-2-9b",      "hf_id": "google/gemma-2-9b",              "bfloat16": True},
    {"tag": "mistral-7b",      "hf_id": "mistralai/Mistral-7B-v0.3",      "bfloat16": True},
]

ALL_PATIENTS = [
    "PTYEU_task147", "PTYEV_task37", "PTYEY_task86", "PTYEZ_task60",
    "PTYFA_task25",  "PTYFC_task28", "PTYFF_task17", "PTYFG_task18",
    "PTYFI_task81",  "PTYFK_task40", "PTYFM_task104","PTYFP_task88",
    "PTYFR_task91",  "PTYFS_task95", "PTYFU_task224",
]

# ── Transcript loader ─────────────────────────────────────────────────────────

def load_words(pid):
    # Same fallback chain as run_geometry_analysis.build_paths(): PTYFA_task25's
    # Transcripts-dir file is a broken symlink to a nonexistent "...ClusterIDNew.xlsx";
    # its real file is BERTEmbeds/..._words_english_only/..._ClusterID.xlsx (no "New").
    bertembeds_dir = "/scratch/aniluchavez/ConvoDATAS/BERTEmbeds"
    candidates = [
        os.path.join(TRANSCRIPT_DIR, f"{pid}_filtered_used_rows_withNP_withClusterIDNew.xlsx"),
        os.path.join(bertembeds_dir, f"{pid}_words_english_only",
                     f"{pid}_filtered_used_rows_withNP_withClusterIDNew.xlsx"),
        os.path.join(bertembeds_dir, f"{pid}_words_english_only",
                     f"{pid}_filtered_used_rows_withNP_withClusterID.xlsx"),
    ]
    xls = next((p for p in candidates if os.path.exists(p)), None)
    if xls is None:
        print(f"  SKIP — transcript not found in any candidate location for {pid}")
        return None
    xl    = pd.ExcelFile(xls)
    sheet = "Sheet1" if "Sheet1" in xl.sheet_names else xl.sheet_names[0]
    df    = xl.parse(sheet, keep_default_na=False)
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df = df.dropna(subset=["onset"]).sort_values("onset").reset_index(drop=True)
    spk_cols = sorted([c for c in df.columns if c.lower().startswith("speaker")])
    if "CleanedWord" not in df.columns:
        raw = df[spk_cols].apply(
            lambda row: next(
                (str(v).strip() for v in row
                 if str(v).strip() not in ("", "nan", "xxx")), ""),
            axis=1)
        # Match the original CleanedWord convention: lowercase, strip all
        # non-alphanumeric characters (verified against PTYEU_task147:
        # CollapsedWord -> CleanedWord is exactly this transform, 0 mismatches).
        df["CleanedWord"] = raw.str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    return df["CleanedWord"].astype(str).str.strip().tolist()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(cfg):
    hf_id    = cfg["hf_id"]
    causal   = cfg.get("causal", False)
    bf16     = cfg.get("bfloat16", False)
    dtype_kw = {"torch_dtype": torch.bfloat16} if bf16 else {}

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    # Right-pad: real tokens sit at natural absolute positions 0,1,2,...
    # which is required for GPT-2's absolute positional embeddings to be correct.
    tokenizer.padding_side    = "right"
    tokenizer.truncation_side = "left"   # drop oldest tokens if context > max_length
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if causal:
        # Causal BERT: same weights, causal attention mask
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
    """Return 1 if tokenizer appends a trailing special token (e.g. BERT's [SEP])."""
    probe_ids = tokenizer("test", add_special_tokens=True)["input_ids"]
    if len(probe_ids) < 2:
        return 0
    last_id   = probe_ids[-1]
    special_ids = {tokenizer.sep_token_id, tokenizer.eos_token_id} - {None}
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id == tokenizer.bos_token_id:
        special_ids.discard(tokenizer.eos_token_id)
    return 1 if last_id in special_ids else 0


@torch.no_grad()
def embed_patient(words, tokenizer, model, n_layers, embed_dim, trailing_special,
                  context_words, max_tokens):
    """
    For each word t: context = words[max(0, t-context_words) : t+1].
    Right-pad to longest in batch, left-truncate to max_tokens.
    Extract at FIRST subword of the current word:
      first_idx = real_len - trailing_special - n_subwords_of_word_t

    Returns (n_layers, n_words, embed_dim) float32 numpy array.
    """
    n_words  = len(words)
    word_emb = np.zeros((n_layers, n_words, embed_dim), dtype=np.float32)

    for batch_start in range(0, n_words, BATCH_SIZE):
        batch_idx = list(range(batch_start, min(batch_start + BATCH_SIZE, n_words)))

        context_strings = []
        for t in batch_idx:
            ctx_start = max(0, t - context_words)
            context_strings.append(" ".join(words[ctx_start : t + 1]))

        tokens = tokenizer(
            context_strings,
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens,
            padding=True,     # right-pad to longest in batch
        ).to(device)

        real_lens = tokens["attention_mask"].sum(dim=1).tolist()

        # Tokenize each word with its natural leading space to count its subwords
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

def main():
    args = _cli_args()
    patients = args.patients.split(",") if args.patients else ALL_PATIENTS
    model_tags = set(args.models.split(",")) if args.models else None
    models = [m for m in MODELS if model_tags is None or m["tag"] in model_tags]
    context_tag = args.context_tag or f"_ctx{args.context_words}"

    for model_cfg in models:
        tag = model_cfg["tag"]

        missing = [p for p in patients
                   if args.overwrite or not os.path.exists(
                       os.path.join(CACHE_DIR, f"{p}_{tag}{context_tag}_word_emb_layers.npy"))]

        if not missing:
            print(f"\n{tag}: all requested patients cached — skipping")
            continue

        print(f"\n{'#'*70}")
        print(f"MODEL: {tag}  ({model_cfg['hf_id']})"
              + ("  [CAUSAL]" if model_cfg.get("causal") else ""))
        print(f"context_words={args.context_words}  context_tag={context_tag}  "
              f"max_tokens={args.max_tokens}")
        print(f"Missing: {len(missing)} patients")
        print(f"{'#'*70}")

        print("Loading model...", flush=True)
        tokenizer, lm = load_model(model_cfg)
        n_layers, embed_dim = get_n_layers_embed_dim(lm)
        trailing_special = _detect_trailing_special(tokenizer)
        print(f"  n_layers={n_layers}  embed_dim={embed_dim}  "
              f"trailing_special={trailing_special}", flush=True)

        for pid in missing:
            cache = os.path.join(CACHE_DIR, f"{pid}_{tag}{context_tag}_word_emb_layers.npy")
            print(f"\n  {pid}...", flush=True)
            words = load_words(pid)
            if words is None:
                continue
            print(f"    {len(words)} words, context≤{args.context_words} words, "
                  f"token cap≤{args.max_tokens}", flush=True)
            emb = embed_patient(words, tokenizer, lm, n_layers, embed_dim,
                                trailing_special, args.context_words, args.max_tokens)
            np.save(cache, emb)
            print(f"    saved {cache}  shape={emb.shape}", flush=True)

        del lm, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"\n{tag} done.")

    print("\nAll done.")


if __name__ == "__main__":
    main()
