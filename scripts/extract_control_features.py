"""
Extract lexical, syntactic, and acoustic control features aligned to word onsets.

Sources:
  Transcripts : /scratch/aniluchavez/ConvoDATAS/Transcripts/  (CollapsedWord has punctuation)
  Audio       : /scratch/aniluchavez/ConvoDATAS/Audio/
  Word/POS    : /scratch/aniluchavez/ConvoDATAS/BERTEmbeds/   (_withPOS.xlsx, for POS + word_clean)

Output: /scratch/aniluchavez/ConvoDATAS/ControlFeatures/{patient_ID}_control_features.csv
  One row per word (same order as transcript), columns:
    onset, offset, Duration, word_clean, POS, POS_ID
    log_word_freq, word_length, local_count, local_rate
    serial_position, sent_position
    dep_depth, dep_children
    speaking_rate, f0_mean, f0_std, rms_mean
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import spacy
import parselmouth
import librosa
from wordfreq import zipf_frequency

_args = argparse.ArgumentParser()
_args.add_argument("--patient", type=str, default=None,
                    help="Run only this patient_ID (regenerates even if output exists); runs all missing if omitted")
PATIENT_FILTER = _args.parse_args().patient

# ─── CONFIG ──────────────────────────────────────────────────────────────────

TRANSCRIPT_DIR = "/scratch/aniluchavez/ConvoDATAS/Transcripts"
AUDIO_DIR      = "/scratch/aniluchavez/ConvoDATAS/Audio"
BERT_EMBED_DIR = "/scratch/aniluchavez/ConvoDATAS/BERTEmbeds"
OUT_DIR        = "/scratch/aniluchavez/ConvoDATAS/ControlFeatures"

SPEAKING_RATE_WINDOW = 5   # words on each side for local rate estimate

os.makedirs(OUT_DIR, exist_ok=True)

# ─── LOAD SPACY ──────────────────────────────────────────────────────────────

nlp = spacy.load("en_core_web_sm")

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _token_depth(tok):
    d, t = 0, tok
    while t.head.i != t.i:   # compare by index — Token objects may not be identical objects
        t = t.head
        d += 1
    return d


def get_dep_features(collapsed_words, speakers):
    """
    Parse each contiguous same-speaker turn.
    CollapsedWord has punctuation attached to words (e.g. "back.", "right?"),
    so joining with spaces gives natural text spaCy can parse correctly.
    Each source row = one word, matching one non-punct spaCy token.

    Also returns sent_position: 0-indexed position of the word within its
    spaCy-detected sentence (resets at sentence boundaries within a turn).
    """
    n = len(collapsed_words)
    depths     = np.zeros(n, dtype=float)
    n_children = np.zeros(n, dtype=float)
    dep_labels = np.full(n, "", dtype=object)
    sent_pos   = np.zeros(n, dtype=float)

    i = 0
    while i < n:
        spk = speakers[i]
        j = i
        while j < n and speakers[j] == spk:
            j += 1

        turn_tokens = [str(w) for w in collapsed_words[i:j]]
        doc = nlp(" ".join(turn_tokens))

        # spaCy splits off trailing punct — filter to word tokens only
        word_toks = [tok for tok in doc if not tok.is_punct]

        sent_counter = {}
        for k, tok in zip(range(len(turn_tokens)), word_toks):
            depths[i + k]     = _token_depth(tok)
            n_children[i + k] = len(list(tok.children))
            dep_labels[i + k] = tok.dep_
            sent_id  = tok.sent.start
            pos      = sent_counter.get(sent_id, 0)
            sent_pos[i + k] = pos
            sent_counter[sent_id] = pos + 1

        i = j

    return depths, n_children, dep_labels, sent_pos


def get_audio_features(audio_path, onsets_ms, offsets_ms):
    """
    Per-word acoustic features: pitch (f0_mean, f0_std), envelope (rms_mean),
    and spectral flux (mean frame-to-frame spectral change).
    onsets/offsets in milliseconds.
    """
    # load via librosa for spectral flux (needed as numpy array)
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    # parselmouth for pitch
    snd         = parselmouth.Sound(audio_path)
    pitch       = snd.to_pitch(pitch_floor=75, pitch_ceiling=600)
    pitch_times = pitch.xs()
    pitch_f0    = pitch.selected_array["frequency"]   # 0 = unvoiced

    # precompute full-file STFT for spectral flux
    hop_length  = 512
    S           = np.abs(librosa.stft(y, hop_length=hop_length))
    flux_frames = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))  # (n_frames-1,)
    flux_times  = librosa.frames_to_time(np.arange(1, S.shape[1]), sr=sr,
                                          hop_length=hop_length)

    f0_mean      = np.full(len(onsets_ms), np.nan)
    f0_std       = np.full(len(onsets_ms), np.nan)
    rms          = np.full(len(onsets_ms), np.nan)
    spectral_flux = np.full(len(onsets_ms), np.nan)

    for i, (t0, t1) in enumerate(zip(onsets_ms / 1000.0, offsets_ms / 1000.0)):
        if t1 <= t0:
            continue
        # envelope (RMS)
        try:
            chunk  = snd.extract_part(from_time=t0, to_time=t1, preserve_times=False)
            rms[i] = np.sqrt(np.mean(chunk.values ** 2))
        except Exception:
            pass
        # pitch
        try:
            mask   = (pitch_times >= t0) & (pitch_times <= t1)
            voiced = pitch_f0[mask]
            voiced = voiced[voiced > 0]
            if len(voiced):
                f0_mean[i] = np.mean(voiced)
                f0_std[i]  = np.std(voiced)
        except Exception:
            pass
        # spectral flux
        try:
            mask = (flux_times >= t0) & (flux_times <= t1)
            if mask.any():
                spectral_flux[i] = np.mean(flux_frames[mask])
        except Exception:
            pass

    return {"f0_mean": f0_mean, "f0_std": f0_std,
            "rms_mean": rms, "spectral_flux": spectral_flux}


def speaking_rate(onsets_ms, offsets_ms, window=SPEAKING_RATE_WINDOW):
    n     = len(onsets_ms)
    rates = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n - 1, i + window)
        span_s = (offsets_ms[hi] - onsets_ms[lo]) / 1000.0
        if span_s > 0:
            rates[i] = (hi - lo + 1) / span_s
    return rates


def find_audio(patient_code):
    """Glob Audio dir for a file containing the patient code (e.g. PTYEU)."""
    matches = glob.glob(os.path.join(AUDIO_DIR, f"*{patient_code}*"))
    return matches[0] if matches else None


def row_speaker(row, speaker_cols):
    for c in speaker_cols:
        if pd.notna(row[c]) and str(row[c]).strip():
            return c
    return "unknown"


# ─── MAIN PER-PATIENT LOOP ───────────────────────────────────────────────────

transcript_files = sorted(f for f in glob.glob(os.path.join(TRANSCRIPT_DIR, "*.xlsx"))
                          if not os.path.basename(f).startswith(("._", "~$")))

for tx_path in transcript_files:
    basename   = os.path.basename(tx_path)
    patient_ID = re.sub(r"_filtered_used_rows_withNP_withClusterIDNew(?:est)?\.xlsx$", "", basename)
    patient_code = patient_ID[:5]   # e.g. PTYEU

    if PATIENT_FILTER and patient_ID != PATIENT_FILTER:
        continue

    out_path = os.path.join(OUT_DIR, f"{patient_ID}_control_features.csv")
    if os.path.exists(out_path) and not PATIENT_FILTER:
        print(f"[SKIP] {patient_ID}")
        continue

    print(f"\n[{patient_ID}] loading transcript...")
    tx = pd.read_excel(tx_path)

    # derive CollapsedWord if missing (words are stored in Speaker columns)
    speaker_cols = [c for c in tx.columns if c.startswith("Speaker")]
    if "CollapsedWord" not in tx.columns:
        def _collapsed(row):
            for c in speaker_cols:
                v = row[c]
                if pd.notna(v) and str(v).strip():
                    return str(v).strip()
            return ""
        tx["CollapsedWord"] = tx.apply(_collapsed, axis=1)

    # load withPOS file for word_clean + POS columns
    pos_candidates = glob.glob(os.path.join(
        BERT_EMBED_DIR, f"{patient_ID}_words_english_only",
        f"*_withNP_withClusterIDNew_withPOS.xlsx"
    ))
    if pos_candidates:
        pos_df = pd.read_excel(pos_candidates[0])
        tx["word_clean"] = pos_df["word_clean"].values
        tx["POS"]        = pos_df["POS"].values
        tx["POS_ID"]     = pos_df["POS_ID"].values
    elif "CleanedWord" in tx.columns:
        tx["word_clean"] = tx["CleanedWord"]
        tx["POS"]        = np.nan
        tx["POS_ID"]     = np.nan
    elif "word" in tx.columns:
        tx["word_clean"] = tx["word"].str.lower()
        tx["POS"]        = np.nan
        tx["POS_ID"]     = np.nan
    else:
        tx["word_clean"] = tx["CollapsedWord"].str.lower()
        tx["POS"]        = np.nan
        tx["POS_ID"]     = np.nan

    speakers  = tx.apply(lambda r: row_speaker(r, speaker_cols), axis=1).tolist()
    words     = tx["word_clean"].fillna("").tolist()
    collapsed = tx["CollapsedWord"].fillna("").tolist()

    # ── lexical ──────────────────────────────────────────────────────────────
    tx["log_word_freq"] = [zipf_frequency(w, "en") for w in words]
    tx["word_length"]   = [len(w) for w in words]

    seen = {}
    local_count = np.zeros(len(words), dtype=int)
    local_rate  = np.zeros(len(words), dtype=float)
    for i, w in enumerate(words):
        local_count[i] = seen.get(w, 0)
        local_rate[i]  = local_count[i] / i if i > 0 else 0.0
        seen[w]        = seen.get(w, 0) + 1
    tx["local_count"] = local_count
    tx["local_rate"]  = local_rate

    # ── position ────────────────────────────────────────────────────────────
    tx["serial_position"] = np.arange(len(words), dtype=float)

    # ── syntactic (uses CollapsedWord with punctuation for better parsing) ────
    print(f"[{patient_ID}] dep parse...")
    depths, n_children, dep_labels, sent_pos = get_dep_features(collapsed, speakers)
    tx["dep_depth"]      = depths
    tx["dep_children"]   = n_children
    tx["dep_label"]      = dep_labels
    tx["sent_position"]  = sent_pos

    # ── acoustic ─────────────────────────────────────────────────────────────
    onsets  = tx["onset"].values.astype(float)
    offsets = tx["offset"].values.astype(float)
    tx["speaking_rate"] = speaking_rate(onsets, offsets)

    audio_path = find_audio(patient_code)
    if audio_path:
        print(f"[{patient_ID}] extracting F0 + RMS from {os.path.basename(audio_path)}...")
        acoustic = get_audio_features(audio_path, onsets, offsets)
        for k, v in acoustic.items():
            tx[k] = v
    else:
        print(f"[{patient_ID}] no audio found — skipping F0/RMS")
        tx["f0_mean"]  = np.nan
        tx["f0_std"]   = np.nan
        tx["rms_mean"] = np.nan

    # ── save ─────────────────────────────────────────────────────────────────
    feature_cols = [
        "onset", "offset", "Duration", "word_clean", "POS", "POS_ID",
        "log_word_freq", "word_length", "local_count", "local_rate",
        "serial_position", "sent_position",
        "dep_depth", "dep_children", "dep_label",
        "speaking_rate", "f0_mean", "f0_std", "rms_mean", "spectral_flux",
    ]
    tx[feature_cols].to_csv(out_path, index=False)
    print(f"[{patient_ID}] saved → {out_path}")

print("\nDone.")
