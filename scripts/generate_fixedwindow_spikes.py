"""
generate_fixedwindow_spikes.py
===============================
Regenerate SpikeWindows using the FIXED 500ms-window convention (the original
"tshift-150_tlen500_oshift+200_olen500" tag), reading word events directly
from the canonical Transcripts/ file — same approach as
generate_varwindow_spikes.py, just for the fixed-window convention instead
of the variable-duration one.

Window definition (reverse-engineered from the existing regress_pre_onset /
regress_post_offset columns of an unaffected patient — see chat history):
  Self  (Speaker1): [onset - 150ms,  onset - 150ms + 500ms]  → fixed 500ms
  Other (Speaker2+):[onset + 200ms,  onset + 200ms + 500ms]  → fixed 500ms

This is what `target_vs_other_fixed_window_from_ref` mode in
neural_encoding/spike_processing.py computes.

Output directory tag: tshift-150_tlen500_oshift+200_olen500
  (same tag find_spike_dir() in semantic_glm.py builds by default when
  window_type == "fixed")

Each speaker subdir gets:
  {region}_spike_counts.npy  —  (n_words_for_speaker, n_neurons)  spike counts

The xlsx gets:
  regress_dur          — window duration in ms (constant: 500)
  regress_pre_onset    — window start time (ms)
  regress_post_offset  — window end time (ms)
  word_dur             — acoustic word duration (offset - onset)

Reading the word list straight from the current Transcripts/ file (rather
than a separately-maintained legacy file) means if words are ever dropped
from the transcript (e.g. non-speech "?" placeholders), the regenerated
spike windows automatically drop in lockstep — no separate file can drift
out of alignment with the embeddings, which read from the same source.

Usage:
    cd /scratch/aniluchavez/hippocampal-speaker-semantics
    python3 -u scripts/generate_fixedwindow_spikes.py [--patients PID1,PID2] [--out-tag TAG] [--force]
"""

import os, sys, argparse
import numpy as np
import pandas as pd

PROJECT = "/scratch/aniluchavez/hippocampal-speaker-semantics"
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "spike_processing",
    os.path.join(PROJECT, "neural_encoding", "spike_processing.py"))
_sp = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_sp)

load_mat_data                       = _sp.load_mat_data
get_cells_by_region                 = _sp.get_cells_by_region
extract_speaker_events              = _sp.extract_speaker_events
annotate_word_and_regress_durations = _sp.annotate_word_and_regress_durations

# ── PATHS ─────────────────────────────────────────────────────────────────────

MAT_ROOT        = "/scratch/aniluchavez/ConvoDATAS/SpikesMAT"
TRANSCRIPT_ROOT = "/scratch/aniluchavez/ConvoDATAS/Transcripts"
OUT_ROOT        = "/scratch/aniluchavez/ConvoDATAS/SpikeWindows"

DEFAULT_OUT_TAG = "tshift-150_tlen500_oshift+200_olen500"

# ── WINDOW CONFIG ─────────────────────────────────────────────────────────────
# Self:  [onset + target_shift, onset + target_shift + target_window_length]
# Other: [onset + other_shift,  onset + other_shift  + other_window_length]
# Defaults match the original "tshift-150_tlen500_oshift+200_olen500" tag;
# override via --target-shift/--other-shift/--target-len/--other-len.

WINDOW_CFG_DEFAULT = dict(
    mode                 = "target_vs_other_fixed_window_from_ref",
    speaker_of_interest  = "Speaker1",
    target_ref_point      = "onset",
    target_shift          = -150,
    target_window_length  = 500,
    other_ref_point       = "onset",
    other_shift           = 200,
    other_window_length   = 500,
)

# ── PATIENTS ──────────────────────────────────────────────────────────────────
# (region_ranges copied from generate_varwindow_spikes.py — same patients, same
# channel/region mapping; only the window convention differs.)

PATIENTS = [
    {"patient_ID": "PTYEU_task147", "patient": "ptYEU_task147", "mat_subdir": "YEU",
     "region_ranges": {"hippocampus": [(1,16),(25,40)], "ACC": [(17,24),(41,48)]}},
    {"patient_ID": "PTYEV_task37",  "patient": "ptYEV_task37",  "mat_subdir": "YEV",
     "region_ranges": {"hippocampus": [(1,16),(25,40)], "ACC": [(17,24),(41,48)]}},
    {"patient_ID": "PTYEY_task86",  "patient": "ptYEY_task86",  "mat_subdir": "YEY",
     "region_ranges": {"hippocampus": [(1,16)]}},
    {"patient_ID": "PTYEZ_task60",  "patient": "ptYEZ_task60",  "mat_subdir": "YEZ",
     "region_ranges": {"hippocampus": [(1,16)],         "ACC": [(17,24)]}},
    {"patient_ID": "PTYFA_task25",  "patient": "ptYFA_task25",  "mat_subdir": "YFA",
     "region_ranges": {"hippocampus": [(1,16),(25,40)], "ACC": [(17,24)]}},
    {"patient_ID": "PTYFC_task28",  "patient": "ptYFC_task28",  "mat_subdir": "YFC",
     "region_ranges": {"hippocampus": [(1,8),(33,48)],  "ACC": [(17,32),(49,64)]}},
    {"patient_ID": "PTYFF_task17",  "patient": "ptYFF_task17",  "mat_subdir": "YFF",
     "region_ranges": {"hippocampus": [(9,16),(25,40)], "ACC": [(17,24),(41,48)]}},
    {"patient_ID": "PTYFG_task18",  "patient": "ptYFG_task18",  "mat_subdir": "YFG",
     "region_ranges": {"hippocampus": [(9,16)],         "ACC": [(25,56)]}},
    {"patient_ID": "PTYFI_task81",  "patient": "ptYFI_task81",  "mat_subdir": "YFI",
     "region_ranges": {"hippocampus": [(1,8),(25,40)],  "ACC": [(9,16)]}},
    {"patient_ID": "PTYFK_task40",  "patient": "ptYFK_task40",  "mat_subdir": "YFK",
     "region_ranges": {"hippocampus": [(1,16),(25,40)], "ACC": [(49,56)]}},
    {"patient_ID": "PTYFM_task104", "patient": "ptYFM_task104", "mat_subdir": "YFM",
     "region_ranges": {"hippocampus": [(33,48)]}},
    {"patient_ID": "PTYFP_task88",  "patient": "ptYFP_task88",  "mat_subdir": "YFP",
     "region_ranges": {"hippocampus": [(17,24),(25,32),(49,56),(57,64)]}},
    {"patient_ID": "PTYFR_task91",  "patient": "ptYFR_task91",  "mat_subdir": "YFR",
     "region_ranges": {"hippocampus": [(1,16),(41,56)]}},
    {"patient_ID": "PTYFS_task95",  "patient": "ptYFS_task95",  "mat_subdir": "YFS",
     "region_ranges": {"hippocampus": [(1,24)]}},
    {"patient_ID": "PTYFU_task224", "patient": "ptYFU_task224", "mat_subdir": "YFU",
     "region_ranges": {"hippocampus": [(17,32),(41,56)]}},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=str, default=None,
                     help="Comma-separated patient_IDs to (re)generate; all if omitted")
    ap.add_argument("--out-tag", type=str, default=None,
                     help="Output directory tag suffix (default: auto-built from the shift/len "
                          "args, matching semantic_glm.py's fixed-window naming convention)")
    ap.add_argument("--target-shift", type=int, default=WINDOW_CFG_DEFAULT["target_shift"],
                     help="Self-condition window start, ms relative to word onset")
    ap.add_argument("--target-len", type=int, default=WINDOW_CFG_DEFAULT["target_window_length"],
                     help="Self-condition window length, ms")
    ap.add_argument("--other-shift", type=int, default=WINDOW_CFG_DEFAULT["other_shift"],
                     help="Other-condition window start, ms relative to word onset")
    ap.add_argument("--other-len", type=int, default=WINDOW_CFG_DEFAULT["other_window_length"],
                     help="Other-condition window length, ms")
    ap.add_argument("--force", action="store_true",
                     help="Regenerate even if output xlsx already exists")
    args = ap.parse_args()

    WINDOW_CFG = dict(WINDOW_CFG_DEFAULT)
    WINDOW_CFG["target_shift"] = args.target_shift
    WINDOW_CFG["target_window_length"] = args.target_len
    WINDOW_CFG["other_shift"] = args.other_shift
    WINDOW_CFG["other_window_length"] = args.other_len

    if args.out_tag is None:
        args.out_tag = (f"tshift{args.target_shift:+d}_tlen{args.target_len}"
                         f"_oshift{args.other_shift:+d}_olen{args.other_len}")

    wanted = set(args.patients.split(",")) if args.patients else None
    patients = [p for p in PATIENTS if wanted is None or p["patient_ID"] in wanted]

    for cfg in patients:
        pid    = cfg["patient_ID"]
        pat    = cfg["patient"]
        subdir = cfg["mat_subdir"]

        out_dir = os.path.join(OUT_ROOT, f"output_{pat}_english_only_{args.out_tag}")
        xlsx_out = os.path.join(out_dir, f"{pid}_with_regress_dur.xlsx")
        if os.path.exists(xlsx_out) and not args.force:
            print(f"[SKIP] {pid} — already done (use --force to regenerate)")
            continue

        print(f"\n{'='*60}\n  {pid}", flush=True)

        mat_path = os.path.join(MAT_ROOT, subdir, f"{pat}_new_spikes.mat")
        if not os.path.exists(mat_path):
            print(f"  SKIP — mat not found: {mat_path}")
            continue

        xlsx_src = os.path.join(TRANSCRIPT_ROOT,
                                f"{pid}_filtered_used_rows_withNP_withClusterIDNew.xlsx")
        if not os.path.exists(xlsx_src):
            # "...New.xlsx" symlink can be broken (e.g. PTYFA_task25) -- fall back
            # to "...Newest.xlsx" if that's a real file.
            xlsx_fallback = os.path.join(TRANSCRIPT_ROOT,
                                f"{pid}_filtered_used_rows_withNP_withClusterIDNewest.xlsx")
            if os.path.exists(xlsx_fallback):
                print(f"  NOTE: ...New.xlsx missing/broken, using ...Newest.xlsx instead")
                xlsx_src = xlsx_fallback
            else:
                print(f"  SKIP — transcript not found: {xlsx_src}")
                continue

        print("  Loading .mat...", flush=True)
        spikes, qual, chan = load_mat_data(mat_path)
        region_cells = get_cells_by_region(chan, qual, cfg["region_ranges"])
        for reg, idxs in region_cells.items():
            print(f"    {reg}: {len(idxs)} neurons", flush=True)

        print("  Extracting speaker events...", flush=True)
        speaker_events = extract_speaker_events(xlsx_src, **WINDOW_CFG)

        os.makedirs(out_dir, exist_ok=True)

        for speaker, ev_arr in speaker_events.items():
            spk_dir = os.path.join(out_dir, speaker)
            os.makedirs(spk_dir, exist_ok=True)
            n_words = ev_arr.shape[0]
            for region, neuron_idxs in region_cells.items():
                if not neuron_idxs:
                    continue
                T = spikes.shape[0]
                reg_spikes = spikes[:, neuron_idxs]
                counts = np.zeros((n_words, len(neuron_idxs)), dtype=np.float32)
                pre_onsets   = ev_arr[:, 1]
                post_offsets = ev_arr[:, 2]
                for wi in range(n_words):
                    s = int(round(pre_onsets[wi]))
                    e = int(round(post_offsets[wi]))
                    s = max(0, s); e = min(T, e)
                    if e > s:
                        counts[wi] = reg_spikes[s:e].sum(axis=0)
                out_path = os.path.join(spk_dir, f"{region}_spike_counts.npy")
                np.save(out_path, counts)
                print(f"    {speaker}/{region}: {counts.shape}  "
                      f"mean_dur={float((post_offsets-pre_onsets).mean()):.0f}ms",
                      flush=True)

        df = pd.read_excel(xlsx_src, keep_default_na=False)
        df_out = annotate_word_and_regress_durations(df, speaker_events=speaker_events,
                                                      add_bounds_cols=True)
        df_out.to_excel(xlsx_out, index=False)
        print(f"  Saved: {xlsx_out}", flush=True)

        dur_col = df_out["regress_dur"].dropna()
        print(f"  regress_dur: min={dur_col.min():.0f}ms  "
              f"max={dur_col.max():.0f}ms  mean={dur_col.mean():.0f}ms  "
              f"std={dur_col.std():.0f}ms", flush=True)

    print("\nAll done.")


if __name__ == "__main__":
    main()
