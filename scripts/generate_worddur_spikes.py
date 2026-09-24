"""
Extract word-duration-only spike counts from new_spikes.mat for all patients.

For each patient + region:
  1. Load new_spikes.mat (CSC sparse: rows=units, cols=time_bins in ms)
  2. Select units whose channel (chan) falls in region_ranges
  3. Load the varwin Excel (has onset, offset, Duration, speaker columns)
  4. For each word: count spikes in [onset_ms, offset_ms] for each selected unit
  5. Save per-speaker: {region}_worddur_spike_counts.npy  (n_words × n_units)

Output directory mirrors the varwin structure:
  SPIKE_ROOT/output_{patient}_english_only_worddur/{speaker}/{region}_worddur_spike_counts.npy

Usage:
  cd /scratch/aniluchavez/hippocampal-speaker-semantics
  python3 -u scripts/generate_worddur_spikes.py
"""

import os, time, argparse
import numpy as np
import pandas as pd
import h5py
import scipy.sparse as sp

MAT_DIR    = "/scratch/aniluchavez/ConvoDATAS/SpikesMAT"
VARWIN_ROOT = "/scratch/aniluchavez/ConvoDATAS/SpikeWindows"
OUT_ROOT    = "/scratch/aniluchavez/ConvoDATAS/SpikeWindows"

PATIENTS = [
    {"patient_ID": "PTYEU_task147", "patient": "ptYEU_task147",
     "region_ranges": {"hippocampus": [(1,16),(25,40)], "ACC": [(17,24),(41,48)]}},
    {"patient_ID": "PTYFF_task17",  "patient": "ptYFF_task17",
     "region_ranges": {"hippocampus": [(9,16),(25,40)], "ACC": [(17,24),(41,48)]}},
    {"patient_ID": "PTYFG_task18",  "patient": "ptYFG_task18",
     "region_ranges": {"hippocampus": [(9,16)],         "ACC": [(25,56)]}},
    {"patient_ID": "PTYFI_task81",  "patient": "ptYFI_task81",
     "region_ranges": {"hippocampus": [(1,8),(25,40)],  "ACC": [(9,16)]}},
    {"patient_ID": "PTYFA_task25",  "patient": "ptYFA_task25",
     "region_ranges": {"hippocampus": [(1,16),(25,40)], "ACC": [(17,24)]}},
    {"patient_ID": "PTYFK_task40",  "patient": "ptYFK_task40",
     "region_ranges": {"hippocampus": [(1,16),(25,40)], "ACC": [(49,56)]}},
    {"patient_ID": "PTYEY_task86",  "patient": "ptYEY_task86",
     "region_ranges": {"hippocampus": [(1,16)]}},
    {"patient_ID": "PTYEV_task37",  "patient": "ptYEV_task37",
     "region_ranges": {"hippocampus": [(1,16),(25,40)], "ACC": [(17,24),(41,48)]}},
    {"patient_ID": "PTYEZ_task60",  "patient": "ptYEZ_task60",
     "region_ranges": {"hippocampus": [(1,16)],         "ACC": [(17,24)]}},
    {"patient_ID": "PTYFC_task28",  "patient": "ptYFC_task28",
     "region_ranges": {"hippocampus": [(1,8),(33,48)],  "ACC": [(17,32),(49,64)]}},
    {"patient_ID": "PTYFM_task104", "patient": "ptYFM_task104",
     "region_ranges": {"hippocampus": [(33,48)]}},
    {"patient_ID": "PTYFP_task88",  "patient": "ptYFP_task88",
     "region_ranges": {"hippocampus": [(17,24),(25,32),(49,56),(57,64)]}},
    {"patient_ID": "PTYFR_task91",  "patient": "ptYFR_task91",
     "region_ranges": {"hippocampus": [(1,16),(41,56)]}},
    {"patient_ID": "PTYFS_task95",  "patient": "ptYFS_task95",
     "region_ranges": {"hippocampus": [(1,24)]}},
    {"patient_ID": "PTYFU_task224", "patient": "ptYFU_task224",
     "region_ranges": {"hippocampus": [(17,32),(41,56)]}},
]

# patient_code → mat subdirectory name (YEU, YFF, etc.)
_CODE_MAP = {
    "PTYEU_task147": "YEU", "PTYFF_task17": "YFF", "PTYFG_task18": "YFG",
    "PTYFI_task81":  "YFI", "PTYFA_task25": "YFA", "PTYFK_task40": "YFK",
    "PTYEY_task86":  "YEY", "PTYEV_task37": "YEV", "PTYEZ_task60": "YEZ",
    "PTYFC_task28":  "YFC", "PTYFM_task104":"YFM", "PTYFP_task88": "YFP",
    "PTYFR_task91":  "YFR", "PTYFS_task95": "YFS", "PTYFU_task224":"YFU",
}


def load_sparse_mat(mat_path):
    """Load new_spikes.mat CSC matrix. Returns (csc_matrix, chan_array).
    Handles two layouts: {'chan','qual','spikes','waveform'} and
    {'channelIds','clusterIds','spikes','waveforms'}.
    """
    with h5py.File(mat_path, "r") as f:
        # Channel label key varies by patient
        if "chan" in f:
            chan = f["chan"][:].flatten().astype(int)
        elif "channelIds" in f:
            chan = f["channelIds"][:].flatten().astype(int)
        else:
            raise KeyError(f"No channel key in {mat_path}; keys={list(f.keys())}")
        data  = f["spikes"]["data"][:]
        ir    = f["spikes"]["ir"][:]
        jc    = f["spikes"]["jc"][:]
    n_units = len(chan)
    n_bins  = len(jc) - 1
    mat = sp.csc_matrix((data.astype(np.uint8), ir.astype(np.int32), jc.astype(np.int32)),
                        shape=(n_units, n_bins))
    return mat, chan


def region_unit_mask(chan, region_ranges):
    """Boolean mask of units whose channel falls in any of the range tuples."""
    mask = np.zeros(len(chan), dtype=bool)
    for lo, hi in region_ranges:
        mask |= (chan >= lo) & (chan <= hi)
    return mask


def count_word_spikes(mat_region, onsets_ms, offsets_ms):
    """
    mat_region : csc_matrix (n_region_units, n_bins)
    onsets_ms  : (n_words,) int array, onset in ms
    offsets_ms : (n_words,) int array, offset in ms (inclusive)

    Returns spike_counts (n_words, n_region_units) as float32.
    """
    n_words = len(onsets_ms)
    n_units = mat_region.shape[0]
    out = np.zeros((n_words, n_units), dtype=np.float32)
    for i in range(n_words):
        t0 = int(onsets_ms[i])
        t1 = int(offsets_ms[i]) + 1          # +1 because slice excludes end
        t0 = max(t0, 0)
        t1 = min(t1, mat_region.shape[1])
        if t1 > t0:
            out[i] = np.asarray(mat_region[:, t0:t1].sum(axis=1)).flatten()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=str, default=None,
                     help="Comma-separated patient_IDs to (re)generate; all if omitted")
    args = ap.parse_args()
    wanted = set(args.patients.split(",")) if args.patients else None
    patients = [p for p in PATIENTS if wanted is None or p["patient_ID"] in wanted]

    for cfg in patients:
        pid    = cfg["patient_ID"]
        pat    = cfg["patient"]
        code   = _CODE_MAP[pid]
        t_pat  = time.time()

        # Locate new_spikes.mat
        mat_path_cands = [
            os.path.join(MAT_DIR, code, f"{pat}_new_spikes.mat"),
        ]
        mat_path = next((p for p in mat_path_cands if os.path.exists(p)), None)
        if mat_path is None:
            print(f"[SKIP] {pid} — no new_spikes.mat in {MAT_DIR}/{code}/")
            continue

        # Locate varwin Excel (source of word timestamps + speaker assignment)
        varwin_dir = os.path.join(VARWIN_ROOT,
                                  f"output_{pat}_english_only_varwin_m150p500_p200p500")
        if not os.path.isdir(varwin_dir):
            print(f"[SKIP] {pid} — no varwin dir at {varwin_dir}")
            continue
        xlsx_cands = [f for f in os.listdir(varwin_dir) if f.endswith("_with_regress_dur.xlsx")]
        if not xlsx_cands:
            print(f"[SKIP] {pid} — no _with_regress_dur.xlsx in {varwin_dir}")
            continue
        tx = pd.read_excel(os.path.join(varwin_dir, xlsx_cands[0]))

        # Parse speaker columns
        spk_cols = sorted(
            [c for c in tx.columns if str(c).startswith("Speaker")],
            key=lambda c: int(c.replace("Speaker","").strip())
                          if c.replace("Speaker","").strip().isdigit() else 999,
        )
        def _nn(v):
            return pd.notna(v) and str(v).strip() not in ("", "nan")
        n_total = len(tx)
        dir_membership = {col: np.array([_nn(v) for v in tx[col]], dtype=bool)
                          for col in spk_cols}
        assign = np.array([None] * n_total, dtype=object)
        for i in range(n_total):
            for col in spk_cols:
                if dir_membership[col][i]:
                    assign[i] = col; break

        # Global word onset/offset (ms)
        onset_ms  = tx["onset"].values.astype(np.float64)
        offset_ms = tx["offset"].values.astype(np.float64)

        print(f"\n{'='*60}\n  {pid}  ({n_total} words, {len(spk_cols)} speakers)", flush=True)

        # Load sparse spiketrain
        print(f"  Loading {mat_path.split('/')[-1]}...", flush=True)
        t0 = time.time()
        mat, chan = load_sparse_mat(mat_path)
        print(f"  mat shape: {mat.shape}  ({time.time()-t0:.1f}s)", flush=True)

        # Output directory
        out_dir = os.path.join(OUT_ROOT,
                               f"output_{pat}_english_only_worddur")
        os.makedirs(out_dir, exist_ok=True)

        # Copy the Excel to the output dir so semantic_glm.py can find it
        import shutil
        shutil.copy(os.path.join(varwin_dir, xlsx_cands[0]),
                    os.path.join(out_dir, xlsx_cands[0]))

        for region, ranges in cfg["region_ranges"].items():
            umask = region_unit_mask(chan, ranges)
            n_units = umask.sum()
            if n_units == 0:
                print(f"  {region}: 0 units — skip")
                continue

            mat_region = mat[umask, :]     # (n_units, n_bins)

            # Per-speaker extraction
            for spk in spk_cols:
                spk_mask = dir_membership[spk]    # (n_total,) bool
                n_words_spk = spk_mask.sum()
                if n_words_spk == 0:
                    continue

                spk_onsets  = onset_ms[spk_mask]
                spk_offsets = offset_ms[spk_mask]

                t_spk = time.time()
                counts = count_word_spikes(mat_region, spk_onsets, spk_offsets)
                print(f"  {region}/{spk}: {counts.shape}  "
                      f"mean={counts.mean():.3f}  ({time.time()-t_spk:.1f}s)", flush=True)

                # Save
                spk_out = os.path.join(out_dir, spk)
                os.makedirs(spk_out, exist_ok=True)
                np.save(os.path.join(spk_out, f"{region}_worddur_spike_counts.npy"), counts)

        print(f"  {pid} done  ({time.time()-t_pat:.1f}s total)", flush=True)

    print("\nAll patients done.", flush=True)


if __name__ == "__main__":
    main()
