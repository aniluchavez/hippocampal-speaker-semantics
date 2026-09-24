"""
Sanity-check the control feature CSVs against the source transcripts.
Prints a per-patient report; no word content is displayed.
"""

import os, glob
import numpy as np
import pandas as pd

TRANSCRIPT_DIR = "/scratch/aniluchavez/ConvoDATAS/Transcripts"
OUT_DIR        = "/scratch/aniluchavez/ConvoDATAS/ControlFeatures"

EXPECTED_RANGES = {
    "log_word_freq":  (0, 8),       # zipf scale
    "word_length":    (0, 30),
    "local_count":    (0, np.inf),
    "local_rate":     (0, 1),
    "dep_depth":      (0, 20),
    "dep_children":   (0, 20),
    "speaking_rate":  (0.1, 20),    # words/sec
    "f0_mean":        (50, 700),    # Hz, voiced frames only
    "f0_std":         (0, 200),
    "rms_mean":       (0, np.inf),
}

feat_files = sorted(glob.glob(os.path.join(OUT_DIR, "*_control_features.csv")))

if not feat_files:
    print("No output files found yet.")
else:
    for fpath in feat_files:
        patient_ID = os.path.basename(fpath).replace("_control_features.csv", "")
        tx_path    = os.path.join(TRANSCRIPT_DIR,
                        f"{patient_ID}_filtered_used_rows_withNP_withClusterIDNew.xlsx")

        df  = pd.read_csv(fpath)
        n   = len(df)

        # row count match
        if os.path.exists(tx_path):
            tx_n = len(pd.read_excel(tx_path))
            row_ok = "OK" if n == tx_n else f"MISMATCH (feat={n}, tx={tx_n})"
        else:
            row_ok = f"{n} rows (no transcript to compare)"

        print(f"\n{'='*55}")
        print(f"  {patient_ID}")
        print(f"  rows: {row_ok}")

        # NaN counts
        nan_cols = {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()}
        if nan_cols:
            for c, cnt in nan_cols.items():
                pct = cnt / n * 100
                note = " (expected — no audio)" if c in ("f0_mean","f0_std","rms_mean") else ""
                print(f"  NaN  {c}: {cnt}/{n} ({pct:.0f}%){note}")
        else:
            print("  NaN: none")

        # range checks
        for col, (lo, hi) in EXPECTED_RANGES.items():
            if col not in df.columns:
                print(f"  MISSING column: {col}")
                continue
            s = df[col].dropna()
            if len(s) == 0:
                continue
            out_of_range = ((s < lo) | (s > hi)).sum()
            if out_of_range:
                print(f"  RANGE WARN {col}: {out_of_range} values outside [{lo}, {hi}]"
                      f"  (min={s.min():.3f}, max={s.max():.3f})")

        # quick stats (no word content)
        print(f"  dep_depth   mean={df['dep_depth'].mean():.2f}  "
              f"zeros={( df['dep_depth']==0).sum()} (roots)")
        print(f"  log_freq    mean={df['log_word_freq'].mean():.2f}  "
              f"zeros={(df['log_word_freq']==0).sum()} (unknown words)")
        print(f"  speaking_rate mean={df['speaking_rate'].mean():.2f} words/sec")
        if df["f0_mean"].notna().any():
            print(f"  f0_mean     mean={df['f0_mean'].mean():.1f} Hz  "
                  f"voiced={(df['f0_mean'].notna()).sum()}/{n} words")
            print(f"  rms_mean    mean={df['rms_mean'].mean():.4f}")

print("\nDone.")
