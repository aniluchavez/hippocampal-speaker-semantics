"""
generate_spike_windows.py
==========================
Consolidated spike-window generator — replaces generate_fixedwindow_spikes.py,
generate_varwindow_spikes.py, and generate_worddur_spikes.py, which had each
independently hardcoded one window convention, duplicated the same 15-patient
region/mat-subdir table, and (twice) accumulated the same Transcripts/spike-dir
alignment bug when a transcript was corrected after spike windows were built.

ALWAYS reads word events (onset, offset, speaker columns) directly from the
canonical Transcripts/{pid}_filtered_used_rows_withNP_withClusterIDNew.xlsx —
never from a separately-maintained file — so spike windows can never drift out
of alignment with whatever words are currently in the transcript (and
therefore the embeddings, which read from the same source).

Modes
-----
fixed    Self / other windows are FIXED-LENGTH, anchored to word onset:
           self  window = [onset + target_shift,  + target_window_length]
           other window = [onset + other_shift,   + other_window_length]
         Default params (-150/500, +200/500) reproduce the original
         "tshift-150_tlen500_oshift+200_olen500" convention.
         Window-end index is EXCLUSIVE (matches that convention's existing data).

varwin   Self / other windows are VARIABLE-LENGTH, start anchored to onset,
         end anchored to offset:
           self  window = [onset + target_shift,  offset + target_end_shift]
           other window = [onset + other_shift,   offset + other_end_shift]
         Default params (-150/+500, +200/+500) reproduce the original
         "varwin_m150p500_p200p500" convention.
         Window-end index is INCLUSIVE (matches that convention's existing data;
         yes, this differs from "fixed" mode — that inconsistency predates this
         script and is preserved intentionally for backward compatibility).

worddur  Window is the literal acoustic word span [onset, offset] — no shift
         params. word_dur = offset - onset is saved as regress_dur, used as a
         Poisson exposure offset downstream. Reproduces the original "worddur"
         convention, but reads straight from Transcripts/ instead of depending
         on a separately-generated varwin xlsx as its word-list source.

Each speaker subdir gets:
  {region}_spike_counts.npy        — (n_words_for_speaker, n_neurons)

The xlsx gets:
  word_dur, regress_dur, regress_pre_onset, regress_post_offset

Usage
-----
  cd /scratch/aniluchavez/hippocampal-speaker-semantics

  # Reproduce existing defaults for one patient (e.g. after a transcript fix):
  python3 -u scripts/generate_spike_windows.py --mode fixed   --patients PTYFU_task224 --force
  python3 -u scripts/generate_spike_windows.py --mode varwin  --patients PTYFU_task224 --force
  python3 -u scripts/generate_spike_windows.py --mode worddur --patients PTYFU_task224 --force

  # A custom shifted fixed window (e.g. 300ms starting 100ms before onset):
  python3 -u scripts/generate_spike_windows.py --mode fixed --target_shift -100 \\
      --target_window_length 300 --other_shift 100 --other_window_length 300 \\
      --out_tag tshift-100_tlen300_oshift+100_olen300

  # All patients, all defaults (initial / full regeneration):
  python3 -u scripts/generate_spike_windows.py --mode fixed
  python3 -u scripts/generate_spike_windows.py --mode varwin
  python3 -u scripts/generate_spike_windows.py --mode worddur
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

# ── PATIENTS ──────────────────────────────────────────────────────────────────
# Single canonical table (previously duplicated across all three generators).

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


def default_out_tag(args):
    if args.out_tag:
        return args.out_tag
    if args.mode == "fixed":
        return (f"tshift{args.target_shift:+d}_tlen{args.target_window_length}_"
                 f"oshift{args.other_shift:+d}_olen{args.other_window_length}")
    if args.mode == "varwin":
        return (f"varwin_m{abs(args.target_shift)}p{args.target_end_shift}_"
                 f"p{args.other_shift}p{args.other_end_shift}")
    return "worddur"


def build_speaker_events(xlsx_src, args):
    """fixed/varwin modes: delegate to spike_processing.extract_speaker_events."""
    if args.mode == "fixed":
        return extract_speaker_events(
            xlsx_src,
            mode="target_vs_other_fixed_window_from_ref",
            speaker_of_interest="Speaker1",
            target_ref_point=args.target_ref_point, target_shift=args.target_shift,
            target_window_length=args.target_window_length,
            other_ref_point="onset", other_shift=args.other_shift,
            other_window_length=args.other_window_length,
        )
    elif args.mode == "varwin":
        return extract_speaker_events(
            xlsx_src,
            mode="target_vs_other_custom_bounds",
            speaker_of_interest="Speaker1",
            target_start_ref="onset", target_start_shift=args.target_shift,
            target_end_ref="offset",  target_end_shift=args.target_end_shift,
            other_start_ref="onset",  other_start_shift=args.other_shift,
            other_end_ref="offset",   other_end_shift=args.other_end_shift,
        )
    raise ValueError(f"build_speaker_events not used for mode={args.mode}")


def clip_speaker_events_to_recording(speaker_events, n_timepoints, *, min_window_ms=1):
    """Make event bounds match the countable recording interval.

    This is useful for custom variable windows such as [onset - 100, offset],
    where early words can start before t=0, and for short words where an
    offset-relative endpoint can fall before a shifted start.  We preserve row
    counts/alignment and minimally expand invalid windows to min_window_ms.
    """
    clipped = {}
    upper = float(max(n_timepoints - 1, 0))
    min_window_ms = float(min_window_ms)
    for speaker, ev_arr in speaker_events.items():
        ev = np.asarray(ev_arr, dtype=float).copy()
        if ev.size == 0:
            clipped[speaker] = ev
            continue
        ev[:, 1] = np.clip(ev[:, 1], 0.0, upper)
        ev[:, 2] = np.clip(ev[:, 2], 0.0, upper)
        too_short = ev[:, 2] <= (ev[:, 1] + min_window_ms)
        ev[too_short, 2] = np.minimum(ev[too_short, 1] + min_window_ms, upper)
        still_too_short = ev[:, 2] <= ev[:, 1]
        ev[still_too_short, 1] = np.maximum(ev[still_too_short, 2] - min_window_ms, 0.0)
        ev[:, 3] = np.round(ev[:, 2] - ev[:, 1])
        clipped[speaker] = ev
    return clipped


def count_spikes_for_events(spikes_region, pre_onsets, post_offsets, end_inclusive,
                             truncate=False):
    """Sum spikes in [pre_onset, post_offset] per event.

    end_inclusive controls whether post_offset's millisecond is included (+1).
    truncate controls int(x) vs int(round(x)) for converting ms floats to
    indices. Both flags exist only to exactly match each convention's
    pre-existing real data (validated against unaffected patients before this
    script touched anything real):
      fixed   -> round, exclusive end
      varwin  -> round, inclusive end
      worddur -> truncate, inclusive end
    """
    T = spikes_region.shape[0]
    n_events = len(pre_onsets)
    n_neurons = spikes_region.shape[1]
    counts = np.zeros((n_events, n_neurons), dtype=np.float32)
    to_idx = (lambda x: int(x)) if truncate else (lambda x: int(round(x)))
    for i in range(n_events):
        s = to_idx(pre_onsets[i])
        e = to_idx(post_offsets[i]) + (1 if end_inclusive else 0)
        s = max(0, s); e = min(T, e)
        if e > s:
            counts[i] = spikes_region[s:e].sum(axis=0)
    return counts


def run_fixed_or_varwin(cfg, args, out_dir, xlsx_src):
    pid = cfg["patient_ID"]
    mat_path = os.path.join(MAT_ROOT, cfg["mat_subdir"], f"{cfg['patient']}_new_spikes.mat")
    if not os.path.exists(mat_path):
        print(f"  SKIP — mat not found: {mat_path}"); return

    print("  Loading .mat...", flush=True)
    spikes, qual, chan = load_mat_data(mat_path)
    region_cells = get_cells_by_region(chan, qual, cfg["region_ranges"])
    for reg, idxs in region_cells.items():
        print(f"    {reg}: {len(idxs)} neurons", flush=True)

    print("  Extracting speaker events...", flush=True)
    speaker_events = build_speaker_events(xlsx_src, args)
    end_inclusive = (args.mode == "varwin")
    if args.clip_bounds:
        speaker_events = clip_speaker_events_to_recording(
            speaker_events, spikes.shape[0], min_window_ms=args.min_window_ms)

    os.makedirs(out_dir, exist_ok=True)
    for speaker, ev_arr in speaker_events.items():
        spk_dir = os.path.join(out_dir, speaker)
        os.makedirs(spk_dir, exist_ok=True)
        pre_onsets, post_offsets = ev_arr[:, 1], ev_arr[:, 2]
        for region, neuron_idxs in region_cells.items():
            if not neuron_idxs:
                continue
            counts = count_spikes_for_events(
                spikes[:, neuron_idxs], pre_onsets, post_offsets, end_inclusive)
            out_path = os.path.join(spk_dir, f"{region}_spike_counts.npy")
            np.save(out_path, counts)
            print(f"    {speaker}/{region}: {counts.shape}  "
                  f"mean_dur={float((post_offsets-pre_onsets).mean()):.0f}ms", flush=True)

    df = pd.read_excel(xlsx_src, keep_default_na=False)
    df_out = annotate_word_and_regress_durations(df, speaker_events=speaker_events,
                                                  add_bounds_cols=True)
    xlsx_out = os.path.join(out_dir, f"{pid}_with_regress_dur.xlsx")
    df_out.to_excel(xlsx_out, index=False)
    print(f"  Saved: {xlsx_out}", flush=True)
    dur_col = df_out["regress_dur"].dropna()
    print(f"  regress_dur: min={dur_col.min():.0f}ms  max={dur_col.max():.0f}ms  "
          f"mean={dur_col.mean():.0f}ms  std={dur_col.std():.0f}ms", flush=True)


def run_worddur(cfg, args, out_dir, xlsx_src):
    pid = cfg["patient_ID"]
    mat_path = os.path.join(MAT_ROOT, cfg["mat_subdir"], f"{cfg['patient']}_new_spikes.mat")
    if not os.path.exists(mat_path):
        print(f"  SKIP — mat not found: {mat_path}"); return

    print("  Loading .mat...", flush=True)
    spikes, qual, chan = load_mat_data(mat_path)
    region_cells = get_cells_by_region(chan, qual, cfg["region_ranges"])
    for reg, idxs in region_cells.items():
        print(f"    {reg}: {len(idxs)} neurons", flush=True)

    df = pd.read_excel(xlsx_src, keep_default_na=False)
    spk_cols = sorted(
        [c for c in df.columns if str(c).startswith("Speaker")],
        key=lambda c: int(c.replace("Speaker", "").strip())
                      if c.replace("Speaker", "").strip().isdigit() else 999,
    )

    def _nn(v):
        return str(v).strip() not in ("", "nan", "xxx")

    dir_membership = {c: np.array([_nn(v) for v in df[c]], dtype=bool) for c in spk_cols}
    onset_ms  = pd.to_numeric(df["onset"],  errors="coerce").values
    offset_ms = pd.to_numeric(df["offset"], errors="coerce").values

    os.makedirs(out_dir, exist_ok=True)
    speaker_events = {}
    for speaker in spk_cols:
        spk_mask = dir_membership[speaker]
        if spk_mask.sum() == 0:
            continue
        spk_dir = os.path.join(out_dir, speaker)
        os.makedirs(spk_dir, exist_ok=True)
        pre_onsets, post_offsets = onset_ms[spk_mask], offset_ms[spk_mask]
        speaker_events[speaker] = np.vstack([
            pre_onsets, pre_onsets, post_offsets, post_offsets - pre_onsets]).T
        for region, neuron_idxs in region_cells.items():
            if not neuron_idxs:
                continue
            counts = count_spikes_for_events(
                spikes[:, neuron_idxs], pre_onsets, post_offsets,
                end_inclusive=True, truncate=True)
            out_path = os.path.join(spk_dir, f"{region}_worddur_spike_counts.npy")
            np.save(out_path, counts)
            print(f"    {speaker}/{region}: {counts.shape}  "
                  f"mean={counts.mean():.3f}", flush=True)

    df_out = annotate_word_and_regress_durations(df, speaker_events=speaker_events,
                                                  add_bounds_cols=True)
    xlsx_out = os.path.join(out_dir, f"{pid}_with_regress_dur.xlsx")
    df_out.to_excel(xlsx_out, index=False)
    print(f"  Saved: {xlsx_out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fixed", "varwin", "worddur"], default="fixed")
    ap.add_argument("--target_shift", type=int, default=-150)
    ap.add_argument("--target_ref_point", choices=["onset", "offset"], default="onset")
    ap.add_argument("--target_window_length", type=int, default=500)
    ap.add_argument("--target_end_shift", type=int, default=500)
    ap.add_argument("--other_shift", type=int, default=200)
    ap.add_argument("--other_window_length", type=int, default=500)
    ap.add_argument("--other_end_shift", type=int, default=500)
    ap.add_argument("--clip_bounds", action="store_true",
                     help="Clip event bounds to recording time and enforce a positive window length")
    ap.add_argument("--min_window_ms", type=float, default=1.0,
                     help="Minimum window length after clipping when --clip_bounds is set")
    ap.add_argument("--out_tag", type=str, default=None)
    ap.add_argument("--patients", type=str, default=None,
                     help="Comma-separated patient_IDs; all if omitted")
    ap.add_argument("--force", action="store_true",
                     help="Regenerate even if output already exists")
    args = ap.parse_args()

    out_tag = default_out_tag(args)
    wanted = set(args.patients.split(",")) if args.patients else None
    patients = [p for p in PATIENTS if wanted is None or p["patient_ID"] in wanted]

    print(f"mode={args.mode}  out_tag={out_tag}  patients={len(patients)}", flush=True)

    for cfg in patients:
        pid = cfg["patient_ID"]
        out_dir = os.path.join(OUT_ROOT, f"output_{cfg['patient']}_english_only_{out_tag}")
        xlsx_out = os.path.join(out_dir, f"{pid}_with_regress_dur.xlsx")
        if os.path.exists(xlsx_out) and not args.force:
            print(f"[SKIP] {pid} — already done (use --force to regenerate)")
            continue

        transcript_names = [
            f"{pid}_filtered_used_rows_withNP_withClusterIDNew.xlsx",
            f"{pid}_filtered_used_rows_withNP_withClusterIDNewest.xlsx",
        ]
        xlsx_src = next(
            (
                os.path.join(TRANSCRIPT_ROOT, name)
                for name in transcript_names
                if os.path.exists(os.path.join(TRANSCRIPT_ROOT, name))
            ),
            os.path.join(TRANSCRIPT_ROOT, transcript_names[0]),
        )
        if not os.path.exists(xlsx_src):
            print(f"  SKIP — transcript not found: {xlsx_src}")
            continue

        print(f"\n{'='*60}\n  {pid}", flush=True)
        if args.mode == "worddur":
            run_worddur(cfg, args, out_dir, xlsx_src)
        else:
            run_fixed_or_varwin(cfg, args, out_dir, xlsx_src)

    print("\nAll done.")


if __name__ == "__main__":
    main()
