"""
Compute GPT-2 surprisal for patients that have no GPT2Embeds CSV.
Reads words directly from Transcripts Excel files, calls the same
get_llm_surprisal.py used in the geometry paper, and saves output in
the identical format as existing surprisal CSVs.

Usage:
    cd /scratch/aniluchavez/hippocampal-speaker-semantics
    conda run -n gpt2_embed --no-capture-output python3 -u scripts/run_surprisal_new_patients.py
"""

import os, sys, subprocess, glob, re
import numpy as np
import pandas as pd

TRANSCRIPT_DIR = "/scratch/aniluchavez/ConvoDATAS/Transcripts"
OUT_DIR        = "/scratch/aniluchavez/ConvoDATAS/Surprisal"
TMP_DIR        = "/scratch/aniluchavez/ConvoDATAS/Surprisal/tmp"
SCRIPT         = "/scratch/aniluchavez/llm_surprisal/get_llm_surprisal.py"
MODEL          = "gpt2"
MODE           = "word"

EXISTING = set()  # process all patients

os.makedirs(TMP_DIR, exist_ok=True)


def get_words_and_speakers(tx_path):
    """Return (words, speakers) lists from a Transcripts Excel file."""
    df = pd.read_excel(tx_path)
    speaker_cols = [c for c in df.columns if c.startswith("Speaker")]

    # derive word per row
    if "CleanedWord" in df.columns:
        words = df["CleanedWord"].fillna("").astype(str).tolist()
    elif "word" in df.columns:
        words = df["word"].fillna("").str.lower().astype(str).tolist()
    elif "CollapsedWord" in df.columns:
        words = df["CollapsedWord"].fillna("").str.lower().astype(str).tolist()
    else:
        # fall back to whichever Speaker column is non-null
        def row_word(row):
            for c in speaker_cols:
                v = row[c]
                if pd.notna(v) and str(v).strip():
                    return str(v).strip().lower()
            return ""
        words = df.apply(row_word, axis=1).tolist()

    # derive speaker label per row
    def row_speaker(row):
        for i, c in enumerate(speaker_cols, 1):
            if pd.notna(row[c]) and str(row[c]).strip():
                return f"SPK{i}"
        return "SPK0"
    speakers = df.apply(row_speaker, axis=1).tolist()

    words = [w if isinstance(w, str) else "" for w in words]
    return words, speakers


# patterns that indicate unintelligible / non-word tokens
_UNINTELLIGIBLE = re.compile(
    r'^[xX\*\?\_\#]{2,}$'          # XXX, xxx, ***, ???, ___
    r'|^\[.*\]$'                    # [inaudible], [unclear]
    r'|^\(.*\)$'                    # (inaudible)
    r'|^<.*>$',                     # <unk>, <noise>
    re.IGNORECASE
)

def is_unintelligible(w):
    return bool(_UNINTELLIGIBLE.match(w.strip())) if w.strip() else True


def build_article_file(words, path):
    """Replace unintelligible tokens with 'the' so GPT-2 gets a neutral common word."""
    cleaned = []
    for w in words:
        if is_unintelligible(w):
            cleaned.append("the")
        else:
            # collapse internal spaces so the join doesn't create phantom tokens
            cleaned.append(w.replace(" ", "-"))
    with open(path, "w") as f:
        f.write("!ARTICLE\n")
        f.write(" ".join(cleaned) + "\n")


def parse_surprisal_output(raw):
    lines = raw.strip().splitlines()
    assert lines[0].strip() == "word llmsurp", f"Unexpected header: {lines[0]}"
    return [float(line.split()[1]) for line in lines[1:] if len(line.split()) == 2]


# ── main loop ─────────────────────────────────────────────────────────────────

tx_files = sorted(
    f for f in glob.glob(os.path.join(TRANSCRIPT_DIR, "*.xlsx"))
    if not os.path.basename(f).startswith(("._", "~$"))
)

for tx_path in tx_files:
    basename   = os.path.basename(tx_path)
    patient_ID = basename.replace("_filtered_used_rows_withNP_withClusterIDNew.xlsx", "")

    if patient_ID in EXISTING:
        print(f"[SKIP] {patient_ID} (already exists)")
        continue

    print(f"\n=== {patient_ID} ===")

    words, speakers = get_words_and_speakers(tx_path)
    print(f"  {len(words)} words")

    article_path = os.path.join(TMP_DIR, f"{patient_ID}_input.txt")
    build_article_file(words, article_path)

    cmd = [sys.executable, SCRIPT, article_path, MODEL, MODE]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR:\n{result.stderr[-2000:]}")
        continue

    surprisals = parse_surprisal_output(result.stdout)

    if len(surprisals) != len(words):
        print(f"  MISMATCH: {len(surprisals)} surprisal values for {len(words)} words — skipping")
        continue

    surp_array = [np.nan if is_unintelligible(w) else s
                  for w, s in zip(words, surprisals)]
    out_df = pd.DataFrame({
        "Word":      words,
        "Speaker":   speakers,
        "RowIndex":  range(len(words)),
        "surprisal": surp_array,
    })
    out_path = os.path.join(OUT_DIR, f"{patient_ID}_surprisal.csv")
    out_df.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}  (mean surprisal={out_df['surprisal'].mean():.3f})")

print("\nDone.")
