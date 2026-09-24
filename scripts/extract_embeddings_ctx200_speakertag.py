"""
extract_embeddings_ctx200_speakertag.py
========================================
Same as extract_embeddings_ctx200.py (causal 200-word context window,
extraction at the FIRST subword of the current word), but each turn in the
context is prefixed with a per-speaker tag, e.g.:

    [SPEAKER1] You know what I put one against the shelving in the back .
    [SPEAKER3] You might remember that too ...

Tags are inserted once per turn (when the speaker changes), not before every
word, to keep the 200-word context readable as a transcript and to avoid
eating the token budget with repeated tags.

Speaker labels come straight from the transcript's Speaker1..SpeakerN columns
(same column-priority convention as load_words in extract_embeddings_ctx200.py
and load_speaker_assignment in semantic_glm.py: first non-empty column wins
on overlap rows).

Currently scoped to bert-base-causal only — gpt2-xl and llama-3.1-8b already
have ctx200 (untagged) embeddings cached and don't need a speaker-tagged
variant for this round.

Usage:
    cd /scratch/aniluchavez/hippocampal-speaker-semantics
    nohup conda run -n gpt2_embed --no-capture-output \
        python3 -u scripts/extract_embeddings_ctx200_speakertag.py \
        > /tmp/embed_ctx200_speakertag.log 2>&1 &
"""

import os, gc
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE",  "1")
os.environ.setdefault("HF_HUB_OFFLINE",        "1")

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, BertConfig, BertModel

CACHE_DIR      = "/scratch/aniluchavez/ConvoDATAS/EmbedCache"
TRANSCRIPT_DIR = "/scratch/aniluchavez/ConvoDATAS/Transcripts"
CONTEXT_WORDS  = 200
BATCH_SIZE     = 32   # context strings per GPU forward pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    p = torch.cuda.get_device_properties(0)
    print(f"  {p.name}  {p.total_memory // 1024**2} MB VRAM")

# ── Model definitions ─────────────────────────────────────────────────────────
MODELS = [
    {"tag": "bert-base-causal", "hf_id": "bert-base-uncased", "causal": True},
]

ALL_PATIENTS = [
    "PTYEU_task147", "PTYEV_task37", "PTYEY_task86", "PTYEZ_task60",
    "PTYFA_task25",  "PTYFC_task28", "PTYFF_task17", "PTYFG_task18",
    "PTYFI_task81",  "PTYFK_task40", "PTYFM_task104","PTYFP_task88",
    "PTYFR_task91",  "PTYFS_task95", "PTYFU_task224",
]

# ── Transcript loader ─────────────────────────────────────────────────────────

def load_words_and_speakers(pid):
    """Return (words, speakers) — speakers[i] is the column name (e.g. 'Speaker1')
    that word i was assigned to, using first-non-empty-column priority on
    overlap rows (same convention as load_speaker_assignment in semantic_glm.py)."""
    xls = os.path.join(TRANSCRIPT_DIR,
                       f"{pid}_filtered_used_rows_withNP_withClusterIDNew.xlsx")
    if not os.path.exists(xls):
        print(f"  SKIP — transcript not found: {xls}")
        return None, None
    xl    = pd.ExcelFile(xls)
    sheet = "Sheet1" if "Sheet1" in xl.sheet_names else xl.sheet_names[0]
    df    = xl.parse(sheet, keep_default_na=False)
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df = df.dropna(subset=["onset"]).sort_values("onset").reset_index(drop=True)
    spk_cols = sorted(
        [c for c in df.columns if c.lower().startswith("speaker")],
        key=lambda c: int("".join(filter(str.isdigit, c)) or 999),
    )

    def _nn(v):
        return str(v).strip() not in ("", "nan", "xxx")

    def _row_word_and_speaker(row):
        for col in spk_cols:
            v = row[col]
            if _nn(v):
                return str(v).strip(), col
        return "", None

    pairs   = df[spk_cols].apply(_row_word_and_speaker, axis=1)
    words   = [p[0] for p in pairs]
    speakers = [p[1] for p in pairs]
    return words, speakers


def build_tagged_context(words, speakers, ctx_start, t):
    """Join words[ctx_start:t+1] into one string, inserting a [SPEAKERn] tag
    each time the speaker changes (including at the start of the window)."""
    parts    = []
    prev_spk = None
    for i in range(ctx_start, t + 1):
        spk = speakers[i]
        if spk != prev_spk:
            parts.append(f"[{spk.upper()}]" if spk else "[UNKNOWN]")
            prev_spk = spk
        parts.append(words[i])
    return " ".join(parts)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(cfg):
    hf_id  = cfg["hf_id"]
    causal = cfg.get("causal", False)
    bf16   = cfg.get("bfloat16", False)
    dtype_kw = {"torch_dtype": torch.bfloat16} if bf16 else {}

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    tokenizer.padding_side    = "right"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if causal:
        bert_cfg = BertConfig.from_pretrained(hf_id)
        bert_cfg.is_decoder = True
        model = BertModel.from_pretrained(hf_id, config=bert_cfg, **dtype_kw)
    else:
        from transformers import AutoModel
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
    probe_ids = tokenizer("test", add_special_tokens=True)["input_ids"]
    if len(probe_ids) < 2:
        return 0
    last_id   = probe_ids[-1]
    special_ids = {tokenizer.sep_token_id, tokenizer.eos_token_id} - {None}
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id == tokenizer.bos_token_id:
        special_ids.discard(tokenizer.eos_token_id)
    return 1 if last_id in special_ids else 0


@torch.no_grad()
def embed_patient(words, speakers, tokenizer, model, n_layers, embed_dim, trailing_special):
    """
    For each word t: context = words[max(0, t-200):t+1] with [SPEAKERn] tags
    inserted at turn boundaries.  Right-pad to longest in batch, left-truncate
    to 512 tokens (safety cap — also bert-base's absolute position-embedding
    limit).  Extract at FIRST subword of the current word, same as
    extract_embeddings_ctx200.py.  The current word is always preceded by a
    tag or another word (never the literal start of the string), so it always
    has a natural leading space — unlike the untagged variant's t==0 special
    case.

    Returns (n_layers, n_words, embed_dim) float32 numpy array.
    """
    n_words  = len(words)
    word_emb = np.zeros((n_layers, n_words, embed_dim), dtype=np.float32)

    for batch_start in range(0, n_words, BATCH_SIZE):
        batch_idx = list(range(batch_start, min(batch_start + BATCH_SIZE, n_words)))

        context_strings = []
        for t in batch_idx:
            ctx_start = max(0, t - CONTEXT_WORDS)
            context_strings.append(build_tagged_context(words, speakers, ctx_start, t))

        tokens = tokenizer(
            context_strings,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(device)

        real_lens = tokens["attention_mask"].sum(dim=1).tolist()

        word_strs   = [" " + words[t] for t in batch_idx]
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
    for model_cfg in MODELS:
        tag = model_cfg["tag"]

        missing = [p for p in ALL_PATIENTS
                   if not os.path.exists(
                       os.path.join(CACHE_DIR, f"{p}_{tag}_ctx200spktag_word_emb_layers.npy"))]

        if not missing:
            print(f"\n{tag}: all patients cached — skipping")
            continue

        print(f"\n{'#'*70}")
        print(f"MODEL: {tag}  ({model_cfg['hf_id']})"
              + ("  [CAUSAL]" if model_cfg.get("causal") else ""))
        print(f"Missing: {len(missing)} patients")
        print(f"{'#'*70}")

        print("Loading model...", flush=True)
        tokenizer, lm = load_model(model_cfg)
        n_layers, embed_dim = get_n_layers_embed_dim(lm)
        trailing_special = _detect_trailing_special(tokenizer)
        print(f"  n_layers={n_layers}  embed_dim={embed_dim}  "
              f"trailing_special={trailing_special}", flush=True)

        for pid in missing:
            cache = os.path.join(CACHE_DIR, f"{pid}_{tag}_ctx200spktag_word_emb_layers.npy")
            print(f"\n  {pid}...", flush=True)
            words, speakers = load_words_and_speakers(pid)
            if words is None:
                continue
            print(f"    {len(words)} words, context≤{CONTEXT_WORDS} words, speaker-tagged", flush=True)
            emb = embed_patient(words, speakers, tokenizer, lm, n_layers, embed_dim, trailing_special)
            np.save(cache, emb)
            print(f"    saved {cache}  shape={emb.shape}", flush=True)

        del lm, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"\n{tag} done.")

    print("\nAll done.")


if __name__ == "__main__":
    main()
