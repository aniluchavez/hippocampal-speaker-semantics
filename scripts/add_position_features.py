"""
Patch existing ControlFeatures CSVs with serial_position and sent_position,
without re-running the expensive audio (F0/RMS/spectral flux) extraction.

serial_position : word index since session start (0, 1, 2, ...)
sent_position   : 0-indexed position of the word within its spaCy-detected
                   sentence (resets at sentence boundaries within a turn)

Usage:
  cd /scratch/aniluchavez/hippocampal-speaker-semantics
  python3 -u scripts/add_position_features.py
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import spacy

TRANSCRIPT_DIR = "/scratch/aniluchavez/ConvoDATAS/Transcripts"
CONTROL_DIR    = "/scratch/aniluchavez/ConvoDATAS/ControlFeatures"

nlp = spacy.load("en_core_web_sm")


def _token_depth(tok):
    d, t = 0, tok
    while t.head.i != t.i:
        t = t.head
        d += 1
    return d


def get_sent_position(collapsed_words, speakers):
    n = len(collapsed_words)
    sent_pos = np.zeros(n, dtype=float)

    i = 0
    while i < n:
        spk = speakers[i]
        j = i
        while j < n and speakers[j] == spk:
            j += 1

        turn_tokens = [str(w) for w in collapsed_words[i:j]]
        doc = nlp(" ".join(turn_tokens))
        word_toks = [tok for tok in doc if not tok.is_punct]

        sent_counter = {}
        for k, tok in zip(range(len(turn_tokens)), word_toks):
            sent_id = tok.sent.start
            pos = sent_counter.get(sent_id, 0)
            sent_pos[i + k] = pos
            sent_counter[sent_id] = pos + 1

        i = j

    return sent_pos


def row_speaker(row, speaker_cols):
    for c in speaker_cols:
        if pd.notna(row[c]) and str(row[c]).strip():
            return c
    return "unknown"


transcript_files = sorted(f for f in glob.glob(os.path.join(TRANSCRIPT_DIR, "*.xlsx"))
                          if not os.path.basename(f).startswith(("._", "~$")))

for tx_path in transcript_files:
    basename   = os.path.basename(tx_path)
    patient_ID = re.sub(r"_filtered_used_rows_withNP_withClusterIDNew(?:est)?\.xlsx$", "", basename)

    ctrl_path = os.path.join(CONTROL_DIR, f"{patient_ID}_control_features.csv")
    if not os.path.exists(ctrl_path):
        print(f"[SKIP] {patient_ID} — no control_features.csv")
        continue

    ctrl = pd.read_csv(ctrl_path)
    if "serial_position" in ctrl.columns and "sent_position" in ctrl.columns:
        print(f"[SKIP] {patient_ID} — already patched")
        continue

    print(f"[{patient_ID}] loading transcript...")
    tx = pd.read_excel(tx_path)
    speaker_cols = [c for c in tx.columns if c.startswith("Speaker")]
    if "CollapsedWord" not in tx.columns:
        def _collapsed(row):
            for c in speaker_cols:
                v = row[c]
                if pd.notna(v) and str(v).strip():
                    return str(v).strip()
            return ""
        tx["CollapsedWord"] = tx.apply(_collapsed, axis=1)

    if len(tx) != len(ctrl):
        print(f"[{patient_ID}] ROW MISMATCH transcript={len(tx)} control={len(ctrl)} — skip")
        continue

    speakers  = tx.apply(lambda r: row_speaker(r, speaker_cols), axis=1).tolist()
    collapsed = tx["CollapsedWord"].fillna("").tolist()

    print(f"[{patient_ID}] dep parse for sent_position...")
    ctrl["serial_position"] = np.arange(len(ctrl), dtype=float)
    ctrl["sent_position"]   = get_sent_position(collapsed, speakers)

    ctrl.to_csv(ctrl_path, index=False)
    print(f"[{patient_ID}] patched → {ctrl_path}")

print("\nDone.")
