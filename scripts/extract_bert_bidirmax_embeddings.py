#!/usr/bin/env python3
"""Extract original-style bidirectional BERT embeddings with overlapping windows."""

import argparse
import os

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

TRANSCRIPT_DIR = "/scratch/aniluchavez/ConvoDATAS/Transcripts"
CACHE_DIR = "/scratch/aniluchavez/ConvoDATAS/EmbedCache"
MAX_TOKENS = 512
CONTENT_TOKENS = MAX_TOKENS - 2
STRIDE = CONTENT_TOKENS // 2


def load_words_and_speakers(patient):
    suffix = "Newest" if patient == "PTYFA_task25" else "New"
    path = os.path.join(
        TRANSCRIPT_DIR,
        f"{patient}_filtered_used_rows_withNP_withClusterID{suffix}.xlsx",
    )
    df = pd.read_excel(path, keep_default_na=False)
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df = df.dropna(subset=["onset"]).sort_values("onset").reset_index(drop=True)
    speaker_columns = sorted(
        c for c in df.columns if c.lower().startswith("speaker")
    )

    def word_and_speaker(row):
        for column in speaker_columns:
            value = str(row[column]).strip()
            if value not in ("", "nan", "xxx"):
                return value, f"SPK{column.replace('Speaker', '').strip()}"
        return "", ""

    pairs = df.apply(word_and_speaker, axis=1)
    return [p[0] for p in pairs], [p[1] for p in pairs]


def flatten_tokens(words, speakers, tokenizer, speaker_tags):
    unknown = tokenizer.unk_token_id or 0
    token_ids, word_ranges = [], []
    for word, speaker in zip(words, speakers):
        if speaker_tags:
            token_ids.extend(
                tokenizer.encode(speaker, add_special_tokens=False) or [unknown]
            )
        pieces = tokenizer.encode(
            (" " if token_ids else "") + str(word), add_special_tokens=False
        ) or [unknown]
        start = len(token_ids)
        token_ids.extend(pieces)
        word_ranges.append((start, len(token_ids)))
    return token_ids, word_ranges


@torch.no_grad()
def extract_variant(words, speakers, tokenizer, model, device, speaker_tags):
    token_ids, word_ranges = flatten_tokens(
        words, speakers, tokenizer, speaker_tags
    )
    sums = np.zeros((len(token_ids), model.config.hidden_size), dtype=np.float32)
    counts = np.zeros(len(token_ids), dtype=np.float32)

    for start in range(0, len(token_ids), STRIDE):
        end = min(start + CONTENT_TOKENS, len(token_ids))
        chunk = token_ids[start:end]
        ids = [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        output = model(input_ids, attention_mask=torch.ones_like(input_ids))
        hidden = output.last_hidden_state[0, 1:-1].float().cpu().numpy()
        sums[start:end] += hidden
        counts[start:end] += 1
        if end == len(token_ids):
            break

    sums /= np.maximum(counts, 1)[:, None]
    embeddings = np.zeros(
        (len(word_ranges), model.config.hidden_size), dtype=np.float32
    )
    for index, (start, end) in enumerate(word_ranges):
        embeddings[index] = sums[start:end].mean(axis=0)
    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").to(device).eval()
    words, speakers = load_words_and_speakers(args.patient)

    variants = [
        ("bert-base-bidirmax", False),
        ("bert-base-bidirmax-spktag", True),
    ]
    for tag, speaker_tags in variants:
        output = os.path.join(
            CACHE_DIR, f"{args.patient}_{tag}_word_emb_layers.npy"
        )
        embeddings = extract_variant(
            words, speakers, tokenizer, model, device, speaker_tags
        )
        np.save(output, embeddings)
        print(
            f"{args.patient} {tag}: {embeddings.shape} -> {output}",
            flush=True,
        )


if __name__ == "__main__":
    main()
