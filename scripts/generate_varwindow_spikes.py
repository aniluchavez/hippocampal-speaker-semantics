"""
generate_varwindow_spikes.py
============================
Regenerate SpikeWindows using variable-duration windows anchored to word OFFSET.

Window definition:
  Self  (Speaker1): [onset - 150ms,  offset + 500ms]  → δ_i = word_dur + 650ms
  Other (Speaker2+):[onset + 200ms,  offset + 500ms]  → δ_i = word_dur + 300ms

This matches the old appwindow_BERT_spikes_202507/dur_150_200_500 convention
and the collaborators' formulation where δ_i is used as a Poisson exposure offset.

Output directory tag: varwin_m150p500_p200p500
  (variable window, self: -150ms pre-onset + 500ms post-offset,
                    other: +200ms post-onset + 500ms post-offset)

Each speaker subdir gets:
  {region}_spike_counts.npy  —  (n_words_for_speaker, n_neurons)  int spike counts

The xlsx gets:
  regress_dur          — actual window duration in ms (variable across words)
  regress_pre_onset    — window start time (ms)
  regress_post_offset  — window end time (ms)
  word_dur             — acoustic word duration (offset - onset)

Usage:
    cd /scratch/aniluchavez/hippocampal-speaker-semantics
    nohup python3 -u scripts/generate_varwindow_spikes.py \\
        > /tmp/gen_varwin.log 2>&1 &
"""

import os, sys
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

load_mat_data                   = _sp.load_mat_data
get_cells_by_region             = _sp.get_cells_by_region
extract_speaker_events          = _sp.extract_speaker_events
annotate_word_and_regress_durations = _sp.annotate_word_and_regress_durations

# ── PATHS ─────────────────────────────────────────────────────────────────────

MAT_ROOT        = "/scratch/aniluchavez/ConvoDATAS/SpikesMAT"
TRANSCRIPT_ROOT = "/scratch/aniluchavez/ConvoDATAS/Transcripts"
OUT_ROOT        = "/scratch/aniluchavez/ConvoDATAS/SpikeWindows"

WINDOW_TAG = "varwin_m150p500_p200p500"

# ── WINDOW CONFIG ─────────────────────────────────────────────────────────────
# mode="target_vs_other_custom_bounds":
#   pre_onset  = {start_ref} + start_shift
#   post_offset= {end_ref}   + end_shift
#
# Self:  [onset - 150ms, offset + 500ms]
# Other: [onset + 200ms, offset + 500ms]

WINDOW_CFG = dict(
    mode                  = "target_vs_other_custom_bounds",
    speaker_of_interest   = "Speaker1",
    target_start_ref      = "onset",
    target_start_shift    = -150,
    target_end_ref        = "offset",
    target_end_shift      = 500,
    other_start_ref       = "onset",
    other_start_shift     = 200,
    other_end_ref         = "offset",
    other_end_shift       = 500,
)

# ── PATIENTS ──────────────────────────────────────────────────────────────────

PATIENTS = [
    {"patient_ID": "PTYEU_task147", "patient": "ptYEU_task147",
     "mat_subdir": "YEU",
     "region_ranges": {"hippocampus": [(1,16),(25,40)], "ACC": [(17,24),(41,48)]}},
    {"patient_ID": "PTYEV_task37",  "patient": "ptYEV_task37",
     "mat_subdir": "YEV",
     "region_ranges": {"hippocampus": [(1,16),(25,40)], "ACC": [(17,24),(41,48)]}},
    {"patient_ID": "PTYEY_task86",  "patient": "ptYEY_task86",
     "mat_subdir": "YEY",
     "region_ranges": {"hippocampus": [(1,16)]}},
    {"patient_ID": "PTYEZ_task60",  "patient": "ptYEZ_task60",
     "mat_subdir": "YEZ",
     "region_ranges": {"hippocampus": [(1,16)],         "ACC": [(17,24)]}},
    {"patient_ID": "PTYFA_task25",  "patient": "ptYFA_task25",
     "mat_subdir": "YFA",
     "region_ranges": {"hippocampus": [(1,16),(25,40)], "ACC": [(17,24)]}},
    {"patient_ID": "PTYFC_task28",  "patient": "ptYFC_task28",
     "mat_subdir": "YFC",
     "region_ranges": {"hippocampus": [(1,8),(33,48)],  "ACC": [(17,32),(49,64)]}},
    {"patient_ID": "PTYFF_task17",  "patient": "ptYFF_task17",
     "mat_subdir": "YFF",
     "region_ranges": {"hippocampus": [(9,16),(25,40)], "ACC": [(17,24),(41,48)]}},
    {"patient_ID": "PTYFG_task18",  "patient": "ptYFG_task18",
     "mat_subdir": "YFG",
     "region_ranges": {"hippocampus": [(9,16)],         "ACC": [(25,56)]}},
    {"patient_ID": "PTYFI_task81",  "patient": "ptYFI_task81",
     "mat_subdir": "YFI",
     "region_ranges": {"hippocampus": [(1,8),(25,40)],  "ACC": [(9,16)]}},
    {"patient_ID": "PTYFK_task40",  "patient": "ptYFK_task40",
     "mat_subdir": "YFK",
     "region_ranges": {"hippocampus": [(1,16),(25,40)], "ACC": [(49,56)]}},
    {"patient_ID": "PTYFM_task104", "patient": "ptYFM_task104",
     "mat_subdir": "YFM",
     "region_ranges": {"hippocampus": [(33,48)]}},
    {"patient_ID": "PTYFP_task88",  "patient": "ptYFP_task88",
     "mat_subdir": "YFP",
     "region_ranges": {"hippocampus": [(17,24),(25,32),(49,56),(57,64)]}},
    {"patient_ID": "PTYFR_task91",  "patient": "ptYFR_task91",
     "mat_subdir": "YFR",
     "region_ranges": {"hippocampus": [(1,16),(41,56)]}},
    {"patient_ID": "PTYFS_task95",  "patient": "ptYFS_task95",
     "mat_subdir": "YFS",
     "region_ranges": {"hippocampus": [(1,24)]}},
    {"patient_ID": "PTYFU_task224", "patient": "ptYFU_task224",
     "mat_subdir": "YFU",
     "region_ranges": {"hippocampus": [(17,32),(41,56)]}},
]

# ── MAIN ──────────────────────────────────────────────────────────────────────

for cfg in PATIENTS:
    pid    = cfg["patient_ID"]
    pat    = cfg["patient"]
    subdir = cfg["mat_subdir"]

    out_dir = os.path.join(OUT_ROOT, f"output_{pat}_english_only_{WINDOW_TAG}")

    # Check if already done (all speaker dirs present + xlsx)
    xlsx_out = os.path.join(out_dir, f"{pid}_with_regress_dur.xlsx")
    if os.path.exists(xlsx_out):
        print(f"[SKIP] {pid} — already done")
        continue

    print(f"\n{'='*60}\n  {pid}", flush=True)

    # ── Locate files ──────────────────────────────────────────────
    mat_path = os.path.join(MAT_ROOT, subdir, f"{pat}_new_spikes.mat")
    if not os.path.exists(mat_path):
        print(f"  SKIP — mat not found: {mat_path}")
        continue

    # Transcript xlsx (use the canonical Transcripts dir)
    xlsx_src = os.path.join(TRANSCRIPT_ROOT,
                            f"{pid}_filtered_used_rows_withNP_withClusterIDNew.xlsx")
    if not os.path.exists(xlsx_src):
        print(f"  SKIP — transcript not found: {xlsx_src}")
        continue

    # ── Load spikes ───────────────────────────────────────────────
    print("  Loading .mat...", flush=True)
    spikes, qual, chan = load_mat_data(mat_path)
    region_cells = get_cells_by_region(chan, qual, cfg["region_ranges"])
    for reg, idxs in region_cells.items():
        print(f"    {reg}: {len(idxs)} neurons", flush=True)

    # ── Extract variable-duration word events ─────────────────────
    print("  Extracting speaker events...", flush=True)
    speaker_events = extract_speaker_events(xlsx_src, **WINDOW_CFG)

    # ── Compute and save spike counts ─────────────────────────────
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
            pre_onsets  = ev_arr[:, 1]
            post_offsets = ev_arr[:, 2]
            for wi in range(n_words):
                s = int(round(pre_onsets[wi]))
                e = int(round(post_offsets[wi])) + 1
                s = max(0, s); e = min(T, e)
                if e > s:
                    counts[wi] = reg_spikes[s:e].sum(axis=0)
            out_path = os.path.join(spk_dir, f"{region}_spike_counts.npy")
            np.save(out_path, counts)
            print(f"    {speaker}/{region}: {counts.shape}  "
                  f"mean_dur={float((post_offsets-pre_onsets).mean()):.0f}ms",
                  flush=True)

    # ── Annotate xlsx with regress_dur ────────────────────────────
    df = pd.read_excel(xlsx_src, keep_default_na=False)
    df_out = annotate_word_and_regress_durations(df, speaker_events=speaker_events,
                                                  add_bounds_cols=True)
    df_out.to_excel(xlsx_out, index=False)
    print(f"  Saved: {xlsx_out}", flush=True)

    # Sanity check: confirm regress_dur is now variable
    dur_col = df_out["regress_dur"].dropna()
    print(f"  regress_dur: min={dur_col.min():.0f}ms  "
          f"max={dur_col.max():.0f}ms  mean={dur_col.mean():.0f}ms  "
          f"std={dur_col.std():.0f}ms", flush=True)

print("\nAll done.")
