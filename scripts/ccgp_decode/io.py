import os
import re
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def parse_patient_tag_flexible(patient_tag: str):
    """
    Accepts:
      PTYFF_task17
      PTYFF_task017
    Returns:
      short_id = YFF
      task_int = 17
      task_unpadded = 'task17'
      task_padded3  = 'task017'
      canonical_patient_tag = 'PTYFF_task17'
    """
    s = patient_tag.upper().strip()
    m = re.match(r"^PT([A-Z]{3})_TASK0*(\d+)$", s)
    if not m:
        raise ValueError(f"Bad patient_tag: {patient_tag} (expected like 'PTYFF_task17' or 'PTYFF_task017')")
    short_id = m.group(1)
    task_int = int(m.group(2))
    task_unpadded = f"task{task_int}"
    task_padded3 = f"task{task_int:03d}"
    canonical = f"PT{short_id}_{task_unpadded}"
    return short_id, task_int, task_unpadded, task_padded3, canonical


def build_minimal_paths(patient_tag: str, spikes_root: str, excel_root: str) -> Tuple[str, str, str]:
    short_id, task_int, task_unpadded, task_padded3, canonical = parse_patient_tag_flexible(patient_tag)

    spikes_candidates = [
        os.path.join(spikes_root, short_id, f"pt{short_id}_{task_unpadded}_new_spikes.mat"),
        os.path.join(spikes_root, short_id, f"pt{short_id}_{task_padded3}_new_spikes.mat"),
    ]
    spikes_mat_path = next((p for p in spikes_candidates if os.path.exists(p)), spikes_candidates[0])

    folder_candidates = [
        f"{canonical}_words_english_only",
        f"{patient_tag}_words_english_only",
    ]
    file_candidates = [
        f"{canonical}_filtered_used_rows_withNP_withClusterIDNew.xlsx",
        f"{patient_tag}_filtered_used_rows_withNP_withClusterIDNew.xlsx",
    ]

    excel_path = None
    for fold in folder_candidates:
        for fn in file_candidates:
            p = os.path.join(excel_root, fold, fn)
            if os.path.exists(p):
                excel_path = p
                break
        if excel_path:
            break

    if excel_path is None:
        excel_path = os.path.join(
            excel_root,
            f"{canonical}_words_english_only",
            f"{canonical}_filtered_used_rows_withNP_withClusterIDNew.xlsx",
        )

    return spikes_mat_path, excel_path, canonical


def load_one_patient_ccgp_inputs_word_order(
    patient_tag: str,
    region: str,
    region_ranges: dict,
    spikes_root: str,
    excel_root: str,
    speaker_of_interest: str = "Speaker1",
    mode: str = "target_vs_other_fixed_window_from_ref",
    extract_kwargs: Optional[Dict[str, Any]] = None,
    accepted_qual=(4, 5),
    label_col: str = "FinalClusterID",
    verbose: bool = True,
):
    """
    Returns:
      Y_self, y_self, Y_other, y_other, spk_kept
    where:
      self = rows spoken by speaker_of_interest (e.g., Speaker1)
      other = rows spoken by anyone else
    Critically: rows are constructed in the *Excel row order*.
    """
    if extract_kwargs is None:
        extract_kwargs = {}

    # local import so package doesn't hard-depend during import
    from spike_processing_utils_FIXED_durations_gaussian import (
        load_mat_data, get_cells_by_region, extract_speaker_events, compute_spike_sums
    )

    spikes_mat_path, excel_path, _ = build_minimal_paths(
        patient_tag=patient_tag,
        spikes_root=spikes_root,
        excel_root=excel_root,
    )

    if not os.path.exists(spikes_mat_path):
        raise FileNotFoundError(f"{patient_tag}: missing spikes mat: {spikes_mat_path}")
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"{patient_tag}: missing excel: {excel_path}")

    df = pd.read_excel(excel_path, sheet_name="Sheet1", keep_default_na=False)

    if label_col not in df.columns:
        raise KeyError(f"{patient_tag}: Excel missing label column '{label_col}'")

    y_all = pd.to_numeric(df[label_col], errors="coerce").to_numpy()

    speaker_cols = [c for c in df.columns if str(c).lower().startswith("speaker")]
    if len(speaker_cols) == 0:
        raise KeyError(f"{patient_tag}: no Speaker* columns in excel")

    def present(v):
        s = str(v).strip().lower()
        return s not in ("", "xxx", "nan", "none")

    row_speaker = []
    for _, r in df.iterrows():
        pres = [c for c in speaker_cols if present(r.get(c, ""))]
        row_speaker.append(pres[0] if pres else None)

    speaker_events = extract_speaker_events(
        excel_path,
        speaker_of_interest=speaker_of_interest,
        mode=mode,
        **extract_kwargs
    )

    spikes, qual, chan = load_mat_data(spikes_mat_path)
    region_cells = get_cells_by_region(chan, qual, region_ranges, accepted_qual=accepted_qual)
    if region not in region_cells or len(region_cells[region]) == 0:
        raise ValueError(f"{patient_tag}: no neurons for region={region}")

    Ys_by_spk = {}
    for spk_col, ev in speaker_events.items():
        out = compute_spike_sums(
            spikes,
            {region: region_cells[region]},
            ev,
            mode="explicit_event_bounds"
        )
        Ys_by_spk[spk_col] = out[region]

    counters = {k: 0 for k in Ys_by_spk.keys()}
    Y_rows = []
    for spk in row_speaker:
        if spk is None or spk not in Ys_by_spk:
            Y_rows.append(None)
            continue
        i = counters[spk]
        counters[spk] += 1
        Y_rows.append(Ys_by_spk[spk][i, :] if i < Ys_by_spk[spk].shape[0] else None)

    keep_mask = np.array([r is not None for r in Y_rows], dtype=bool)
    if keep_mask.sum() == 0:
        raise RuntimeError(f"{patient_tag}: no rows were constructed (check speaker columns / events)")

    Y_all = np.vstack([r for r in Y_rows if r is not None])
    y_kept = y_all[keep_mask]
    spk_kept = np.array([s for s, k in zip(row_speaker, keep_mask) if k], dtype=object)

    mask_valid = np.isfinite(y_kept) & (~np.isnan(Y_all).any(axis=1))
    Y_all = Y_all[mask_valid]
    y_kept = y_kept[mask_valid].astype(int)
    spk_kept = spk_kept[mask_valid]

    soi = speaker_of_interest.lower()
    mask_self = np.array([str(s).lower() == soi for s in spk_kept], dtype=bool)

    Y_self, y_self = Y_all[mask_self], y_kept[mask_self]
    Y_other, y_other = Y_all[~mask_self], y_kept[~mask_self]

    if verbose:
        print(f"{patient_tag} | {region} | neurons={Y_all.shape[1]}")
        print(f"  speaker_of_interest={speaker_of_interest} -> self={len(y_self)} other={len(y_other)}")
        print(f"  classes_total={len(np.unique(y_kept))} | kept_rows={len(y_kept)}/{len(df)}")

    return Y_self, y_self, Y_other, y_other, spk_kept


def load_multi_patient_ccgp_inputs_word_order(
    patient_tags,
    region,
    region_ranges_by_patient,
    spikes_root,
    excel_root,
    speaker_of_interest="Speaker1",
    mode="target_vs_other_fixed_window_from_ref",
    extract_kwargs=None,
    verbose=True,
):
    out = {}
    for pid in patient_tags:
        if pid not in region_ranges_by_patient:
            out[pid] = {"ok": False, "error": "missing_channel_ranges"}
            if verbose:
                print(f"⚠️ {pid}: missing channel ranges")
            continue
        try:
            Ys, ys, Yo, yo, spk_kept = load_one_patient_ccgp_inputs_word_order(
                patient_tag=pid,
                region=region,
                region_ranges=region_ranges_by_patient[pid],
                spikes_root=spikes_root,
                excel_root=excel_root,
                speaker_of_interest=speaker_of_interest,
                mode=mode,
                extract_kwargs=extract_kwargs,
                verbose=verbose,
            )
            out[pid] = {
                "ok": True,
                "Y_self": Ys, "y_self": ys,
                "Y_other": Yo, "y_other": yo,
                "spk_kept": spk_kept,
            }
        except Exception as e:
            out[pid] = {"ok": False, "error": str(e)}
            if verbose:
                print(f"❌ {pid} failed: {e}")
    return out