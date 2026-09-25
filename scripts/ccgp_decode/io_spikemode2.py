import os
import re
import ast
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def parse_patient_tag_flexible(patient_tag: str) -> Tuple[str, int, str, str, str]:
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
        raise ValueError(
            f"Bad patient_tag: {patient_tag} (expected like 'PTYFF_task17' or 'PTYFF_task017')"
        )
    short_id = m.group(1)
    task_int = int(m.group(2))
    task_unpadded = f"task{task_int}"
    task_padded3 = f"task{task_int:03d}"
    canonical = f"PT{short_id}_{task_unpadded}"
    return short_id, task_int, task_unpadded, task_padded3, canonical


def build_minimal_paths(patient_tag: str, spikes_root: str, excel_root: str) -> Tuple[str, str, str]:
    short_id, _, task_unpadded, task_padded3, canonical = parse_patient_tag_flexible(patient_tag)

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


def _infer_row_speaker(df: pd.DataFrame) -> np.ndarray:
    speaker_cols = [c for c in df.columns if str(c).lower().startswith("speaker")]
    if len(speaker_cols) == 0:
        raise KeyError("no Speaker* columns in excel")

    def present(v: Any) -> bool:
        s = str(v).strip().lower()
        return s not in ("", "xxx", "nan", "none")

    row_speaker = []
    for _, r in df.iterrows():
        pres = [c for c in speaker_cols if present(r.get(c, ""))]
        row_speaker.append(pres[0] if pres else None)

    return np.asarray(row_speaker, dtype=object)

_WORD_CLEAN_RE = re.compile(r"[^\w']+", re.UNICODE)  # keep letters/numbers/_ and apostrophes

def _normalize_word(w: Any) -> str:
    """
    Lowercase + strip punctuation-ish.
    Keeps apostrophes so "don't" stays "don't".
    """
    if w is None:
        return ""
    s = str(w).strip().lower()
    if s in ("", "xxx", "nan", "none"):
        return ""
    s = _WORD_CLEAN_RE.sub("", s)   # remove punctuation/spaces
    return s

def _row_word_from_speaker_col(df: pd.DataFrame, row_idx: int, spk_col: Any) -> str:
    """
    Given which Speaker* column was active for this row, return that word (raw).
    """
    if spk_col is None:
        return ""
    try:
        return df.loc[row_idx, spk_col]
    except Exception:
        return ""
    
def _resolve_embeddings_path(patient_tag: str, embeddings_root: str) -> str:
    """
    embeddings_root/
      <patient_tag>_words_english_only/
        <patient_tag>_aligned_embeddings_withNP.csv
    Also tries canonical tag folder/filename.
    """
    _, _, _, _, canonical = parse_patient_tag_flexible(patient_tag)

    folder_candidates = [
        f"{canonical}_words_english_only",
        f"{patient_tag}_words_english_only",
    ]
    file_candidates = [
        f"{canonical}_aligned_embeddings_withNP.csv",
        f"{patient_tag}_aligned_embeddings_withNP.csv",
    ]

    for fold in folder_candidates:
        for fn in file_candidates:
            p = os.path.join(embeddings_root, fold, fn)
            if os.path.exists(p):
                return p

    # fallback default
    return os.path.join(
        embeddings_root,
        f"{canonical}_words_english_only",
        f"{canonical}_aligned_embeddings_withNP.csv",
    )


def _load_embeddings_matrix_aligned(
    emb_csv_path: str,
    row_indices: np.ndarray,
    *,
    excel_n_rows: Optional[int] = None,
    verify_words: bool = True,
    excel_df: Optional[pd.DataFrame] = None,
    excel_row_speaker: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Load embeddings CSV and return X aligned to the provided Excel row indices.

    IMPORTANT: We assume the embeddings CSV is 1:1 row-aligned with the Excel:
      - embeddings CSV row i corresponds to Excel row i (after your filtering pipeline).

    We do **NOT** use RowIndex for alignment (it can refer to the original transcript index and contain gaps).
    RowIndex (if present) is treated as metadata only.

    Expects at minimum:
      - Embedding (stringified list)

    Optional sanity checks (recommended):
      - if excel_n_rows is provided, require len(embeddings_csv) == excel_n_rows
      - if verify_words=True and excel_df provided, compare normalized word strings for a sample
    """
    if not os.path.exists(emb_csv_path):
        raise FileNotFoundError(f"Missing embeddings csv: {emb_csv_path}")

    edf = pd.read_csv(emb_csv_path)

    if "Embedding" not in edf.columns:
        raise KeyError(
            f"Embeddings CSV must contain 'Embedding'. Got columns: {list(edf.columns)}"
        )

    if excel_n_rows is not None and len(edf) != int(excel_n_rows):
        raise ValueError(
            f"Embeddings/Excel row mismatch: embeddings rows={len(edf)} != excel rows={int(excel_n_rows)}. "
            "This pipeline assumes a 1:1 row map."
        )

    # Optional check: compare words for a small sample (helps catch misaligned files)
    if verify_words and (excel_df is not None) and ("Word" in edf.columns) and (excel_row_speaker is not None):
        # sample indices evenly spaced across the sheet
        n = len(edf)
        if n > 0:
            sample_idx = np.unique(np.linspace(0, min(n - 1, (excel_n_rows or n) - 1), num=min(25, n), dtype=int))
            mism = 0
            for i in sample_idx:
                spk_col = excel_row_speaker[i] if i < len(excel_row_speaker) else None
                w_excel = _normalize_word(_row_word_from_speaker_col(excel_df, int(i), spk_col))
                w_emb = _normalize_word(edf.iloc[int(i)]["Word"]) if int(i) < len(edf) else ""
                # only count if both non-empty
                if w_excel and w_emb and (w_excel != w_emb):
                    mism += 1
            if mism > 0:
                raise ValueError(
                    f"Embeddings/Excel word mismatch on sample check: {mism}/{len(sample_idx)} sampled rows differ. "
                    "Likely the wrong embeddings CSV for this Excel."
                )

    # Parse all embeddings in row order once (n_rows x D)
    vecs = []
    for i, s in enumerate(edf["Embedding"].tolist()):
        try:
            v = np.asarray(ast.literal_eval(s), dtype=float)
        except Exception as e:
            raise ValueError(f"Failed parsing Embedding at CSV row {i}: {e}")
        vecs.append(v)

    X_excel = np.vstack(vecs)
    return X_excel[row_indices.astype(int)]



def make_pc1_labels(
    pc1_scores: np.ndarray,
    mode: str = "median",      # "median" | "quantile" | "topbottom"
    q: float = 0.3,            # used for topbottom
    n_bins: int = 4,           # used for quantile
):
    """
    Returns:
      labels: int array same length as pc1_scores
      keep: boolean mask (only differs for topbottom)
      info: dict with thresholds used
    """
    x = np.asarray(pc1_scores, dtype=float)

    if mode == "median":
        thr = float(np.nanmedian(x))
        labels = (x > thr).astype(int)
        keep = np.isfinite(x)
        return labels, keep, {"mode": "median", "thr": thr}

    if mode == "quantile":
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.quantile(x[np.isfinite(x)], qs)
        edges[0] -= 1e-12
        edges[-1] += 1e-12
        labels = np.digitize(x, edges[1:-1], right=True).astype(int)
        keep = np.isfinite(x)
        return labels, keep, {"mode": "quantile", "edges": edges.tolist(), "n_bins": int(n_bins)}

    if mode == "topbottom":
        lo = float(np.quantile(x[np.isfinite(x)], q))
        hi = float(np.quantile(x[np.isfinite(x)], 1 - q))
        keep = (x <= lo) | (x >= hi)
        labels = np.full_like(x, fill_value=-1, dtype=int)
        labels[x <= lo] = 0
        labels[x >= hi] = 1
        return labels, keep, {"mode": "topbottom", "q": float(q), "lo": lo, "hi": hi}

    raise ValueError(f"Unknown mode={mode}")


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
    spike_mode: str = "counts",  # "counts" | "rates_mean" | "rates_binned"
    binSize_ms: int = 20,
    gaussian_sigma_ms: Optional[float] = None,
    gaussian_preserve_sum: bool = True,
    flatten_time: bool = True,
    # --- embeddings / PCA ---
    embeddings_root: Optional[str] = None,
    embeddings_csv: Optional[str] = None,
    return_X: bool = False,
    pca_components: Optional[int] = None,
    pca_random_state: int = 0,
    keep_only_shared_words: bool = False,
    shared_word_min_count: int = 1,
    pca_on_shared_words: bool = True,
    # --- PC1 label creation ---
    return_pc1_labels: bool = False,
    pc1_label_mode: str = "median",  # "median" | "quantile" | "topbottom"
    pc1_label_q: float = 0.3,
    pc1_label_n_bins: int = 4,
):
    """
    Returns (base):
      Y_self, y_self, Y_other, y_other, spk_kept

    Optional returns:
      - if return_X or pca_components is not None:
          ... , X_self, X_other, spk_kept
      - if return_pc1_labels:
          ... , ypc1_self, ypc1_other, pc1_info, ...
        (requires embeddings to be loaded and PCA to be applied)

    self  = rows spoken by speaker_of_interest (e.g., Speaker1)
    other = rows spoken by anyone else

    Critically: rows are constructed in the *Excel row order*.
    """
    if extract_kwargs is None:
        extract_kwargs = {}

    try:
        from spike_processing_utils_FIXED_durations_gaussian_spikemode import (
            load_mat_data,
            get_cells_by_region,
            compute_spike_sums,
            compute_spike_features,
            extract_speaker_events,
        )
    except Exception:
        from spike_processing_utils_FIXED_durations_gaussian_spikemode import (  # type: ignore
            load_mat_data,
            get_cells_by_region,
            compute_spike_sums,
            compute_spike_features,
            extract_speaker_events,
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
    row_speaker = _infer_row_speaker(df)

    # Extract explicit event bounds per speaker column
    speaker_events = extract_speaker_events(
        excel_path,
        speaker_of_interest=speaker_of_interest,
        mode=mode,
        **extract_kwargs,
    )

    spikes, qual, chan = load_mat_data(spikes_mat_path)
    region_cells = get_cells_by_region(chan, qual, region_ranges, accepted_qual=accepted_qual)
    if region not in region_cells or len(region_cells[region]) == 0:
        raise ValueError(f"{patient_tag}: no neurons for region={region}")

    # For rates_binned we must enforce fixed feature width across all speakers/conditions
    win_ms = 500
    tw = extract_kwargs.get("target_window_length", None)
    ow = extract_kwargs.get("other_window_length", None)
    cand = [v for v in (tw, ow) if v is not None]
    if len(cand) > 0:
        win_ms = int(max(cand))
    fixed_n_bins = int(win_ms // int(binSize_ms)) if int(binSize_ms) > 0 else 0

    Ys_by_spk: Dict[str, np.ndarray] = {}
    for spk_col, ev in speaker_events.items():
        if spike_mode == "counts":
            out = compute_spike_sums(
                spikes,
                {region: region_cells[region]},
                ev,
                mode="explicit_event_bounds",
            )
            Ys_by_spk[spk_col] = out[region]

        elif spike_mode == "rates_mean":
            out = compute_spike_features(
                spikes,
                {region: region_cells[region]},
                ev,
                mode="explicit_event_bounds",
                feature_mode="rates_mean",
            )
            Ys_by_spk[spk_col] = out[region]

        elif spike_mode == "rates_binned":
            if fixed_n_bins <= 0:
                raise ValueError(
                    f"{patient_tag}: rates_binned requires positive fixed_n_bins; "
                    f"got win_ms={win_ms}, binSize_ms={binSize_ms}"
                )
            out = compute_spike_features(
                spikes,
                {region: region_cells[region]},
                ev,
                mode="explicit_event_bounds",
                feature_mode="rates_binned",
                binSize=binSize_ms,
                gaussian_sigma_ms=gaussian_sigma_ms,
                gaussian_preserve_sum=gaussian_preserve_sum,
                flatten_time=flatten_time,
                fixed_n_bins=fixed_n_bins,
                pad_value=0.0,
            )
            Ys_by_spk[spk_col] = out[region]

        else:
            raise ValueError(
                f"Unknown spike_mode: {spike_mode}. Use 'counts', 'rates_mean', or 'rates_binned'."
            )

    # Reconstruct rows in Excel order (each Speaker column has its own event counter)
    counters = {k: 0 for k in Ys_by_spk.keys()}
    Y_rows = []
    for spk in row_speaker:
        if spk is None or spk not in Ys_by_spk:
            Y_rows.append(None)
            continue
        i = counters[spk]
        counters[spk] += 1
        Yspk = Ys_by_spk[spk]
        Y_rows.append(Yspk[i, :] if i < Yspk.shape[0] else None)

    keep_mask = np.array([r is not None for r in Y_rows], dtype=bool)
    if keep_mask.sum() == 0:
        raise RuntimeError(f"{patient_tag}: no rows were constructed (check speaker columns / events)")

    # Excel row indices for the constructed rows
    kept_excel_row_index = np.where(keep_mask)[0].astype(int)

    Y_all = np.vstack([r for r in Y_rows if r is not None])
    y_kept = y_all[keep_mask]
    spk_kept = np.array([s for s, k in zip(row_speaker, keep_mask) if k], dtype=object)

    # ----------------- validity mask (safe for int/float) -----------------
    mask_valid = np.isfinite(y_kept) & np.isfinite(Y_all).all(axis=1)

    kept_excel_row_index = kept_excel_row_index[mask_valid]
    Y_all = Y_all[mask_valid]
    y_kept = y_kept[mask_valid].astype(int)
    spk_kept = spk_kept[mask_valid]

    soi = speaker_of_interest.lower()
    mask_self = np.array([str(s).lower() == soi for s in spk_kept], dtype=bool)

    # ----------------- shared-words filter (optional / PCA-safe) -----------------
    # If you want PCA/PC labels defined on words shared across self/other, enable pca_on_shared_words.
    # This filters Y/y/spk (and later X) to the shared-vocab subset *before* loading embeddings + PCA.
    force_shared_for_pca = bool(pca_on_shared_words) and ((pca_components is not None) or bool(return_pc1_labels))
    do_shared_filter = bool(keep_only_shared_words) or force_shared_for_pca

    if do_shared_filter:
        # For each kept row, grab the word from the active speaker column for that row
        # NOTE: kept_excel_row_index gives the original Excel row index
        words_raw = np.array(
            [
                _row_word_from_speaker_col(df, int(ridx), spk_col)
                for ridx, spk_col in zip(kept_excel_row_index, spk_kept)
            ],
            dtype=object,
        )
        words_norm = np.array([_normalize_word(w) for w in words_raw], dtype=object)

        # drop empties
        nonempty = np.array([w != "" for w in words_norm], dtype=bool)

        # counts within each condition
        w_self = words_norm[mask_self & nonempty]
        w_other = words_norm[(~mask_self) & nonempty]

        from collections import Counter
        c_self = Counter(w_self.tolist())
        c_other = Counter(w_other.tolist())

        # intersection with optional min-count requirement
        shared_vocab = {
            w for w in c_self.keys()
            if (w in c_other) and (c_self[w] >= shared_word_min_count) and (c_other[w] >= shared_word_min_count)
        }

        shared_mask = nonempty & np.array([w in shared_vocab for w in words_norm], dtype=bool)

        # Apply to everything aligned in "kept-row space"
        kept_excel_row_index = kept_excel_row_index[shared_mask]
        Y_all = Y_all[shared_mask]
        y_kept = y_kept[shared_mask]
        spk_kept = spk_kept[shared_mask]

        # recompute self mask after filtering
        mask_self = np.array([str(s).lower() == soi for s in spk_kept], dtype=bool)

    # ----------------- embeddings load (+ optional PCA) -----------------

    X_self = X_other = None
    X_all = None
    pc1_info = None
    ypc1_self = ypc1_other = None

    need_embeddings = return_X or (pca_components is not None) or return_pc1_labels
    if need_embeddings:
        # Resolve embeddings csv path
        if embeddings_csv is None:
            if embeddings_root is None:
                raise ValueError("To load embeddings, pass embeddings_root=... or embeddings_csv=...")
            emb_csv_path = _resolve_embeddings_path(patient_tag, embeddings_root)
        else:
            emb_csv_path = embeddings_csv

        # Load X in the same row order as Y_all/y_kept (i.e., kept Excel rows)
        X_all = _load_embeddings_matrix_aligned(
            emb_csv_path,
            kept_excel_row_index,
            excel_n_rows=len(df),
            verify_words=True,
            excel_df=df,
            excel_row_speaker=row_speaker,
        )

        # If you want PC1 labels, we REQUIRE PCA so PC1 is well-defined
        if return_pc1_labels and pca_components is None:
            raise ValueError("return_pc1_labels=True requires pca_components to be set (so PC1 exists).")

        if pca_components is not None:
            if int(pca_components) <= 0:
                raise ValueError(f"pca_components must be positive. Got {pca_components}")
            pca = PCA(n_components=int(pca_components), random_state=int(pca_random_state))
            X_all = pca.fit_transform(X_all)

        X_self = X_all[mask_self]
        X_other = X_all[~mask_self]

        # ----------------- PC1 labels (in kept-row space) -----------------
        if return_pc1_labels:
            pc1_scores = X_all[:, 0]  # PC1
            pc1_labels_all, pc1_keep, pc1_info = make_pc1_labels(
                pc1_scores,
                mode=pc1_label_mode,
                q=pc1_label_q,
                n_bins=pc1_label_n_bins,
            )

            # If mode=topbottom, we need to drop middle rows consistently across everything
            if not np.all(pc1_keep):
                kept_excel_row_index = kept_excel_row_index[pc1_keep]
                Y_all = Y_all[pc1_keep]
                y_kept = y_kept[pc1_keep]
                spk_kept = spk_kept[pc1_keep]
                X_all = X_all[pc1_keep]

                # recompute self mask after filtering
                mask_self = np.array([str(s).lower() == soi for s in spk_kept], dtype=bool)

                X_self = X_all[mask_self]
                X_other = X_all[~mask_self]

                # also recompute Y splits below
            # After any optional pc1_keep filtering, slice labels in the current kept-row space
            if np.all(pc1_keep):
                labels_now = pc1_labels_all
            else:
                labels_now = make_pc1_labels(
                    X_all[:, 0], mode=pc1_label_mode, q=pc1_label_q, n_bins=pc1_label_n_bins
                )[0]
            ypc1_self = labels_now[mask_self]
            ypc1_other = labels_now[~mask_self]

    # ----------------- final splits -----------------
    Y_self, y_self = Y_all[mask_self], y_kept[mask_self]
    Y_other, y_other = Y_all[~mask_self], y_kept[~mask_self]

    if verbose:
        print(f"{patient_tag} | {region} | features={Y_all.shape[1]}")
        print(f"  speaker_of_interest={speaker_of_interest} -> self={len(y_self)} other={len(y_other)}")
        print(f"  classes_total={len(np.unique(y_kept))} | kept_rows={len(y_kept)}/{len(df)}")
        if return_pc1_labels:
            u = np.unique(np.concatenate([ypc1_self, ypc1_other])) if ypc1_self is not None else []
            print(f"  pc1_labels={pc1_label_mode} | unique={list(map(int, u))} | info={pc1_info}")

    # ----------------- return packing -----------------
    if return_pc1_labels and (return_X or (pca_components is not None)):
        return (
            Y_self, y_self,
            Y_other, y_other,
            ypc1_self, ypc1_other, pc1_info,
            X_self, X_other,
            spk_kept,
        )

    if return_pc1_labels:
        return (
            Y_self, y_self,
            Y_other, y_other,
            ypc1_self, ypc1_other, pc1_info,
            spk_kept,
        )

    if return_X or (pca_components is not None):
        return Y_self, y_self, Y_other, y_other, X_self, X_other, spk_kept

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
    spike_mode="counts",  # "counts" | "rates_mean" | "rates_binned"
    binSize_ms=20,
    gaussian_sigma_ms=None,
    gaussian_preserve_sum=True,
    flatten_time=True,
    # --- embeddings / PCA ---
    embeddings_root: Optional[str] = None,
    embeddings_csv_by_patient: Optional[Dict[str, str]] = None,
    return_X: bool = False,
    pca_components: Optional[int] = None,
    pca_random_state: int = 0,
    # --- shared-words / PCA-on-shared ---
    keep_only_shared_words: bool = False,
    shared_word_min_count: int = 1,
    pca_on_shared_words: bool = True,
    # --- PC1 labels ---
    return_pc1_labels: bool = False,
    pc1_label_mode: str = "median",
    pc1_label_q: float = 0.3,
    pc1_label_n_bins: int = 4,
):
    out = {}

    for pid in patient_tags:
        if pid not in region_ranges_by_patient:
            out[pid] = {"ok": False, "error": "missing_channel_ranges"}
            if verbose:
                print(f"⚠️ {pid}: missing channel ranges")
            continue

        try:
            emb_csv = None
            if embeddings_csv_by_patient is not None:
                emb_csv = embeddings_csv_by_patient.get(pid, None)

            res = load_one_patient_ccgp_inputs_word_order(
                patient_tag=pid,
                region=region,
                region_ranges=region_ranges_by_patient[pid],
                spikes_root=spikes_root,
                excel_root=excel_root,
                speaker_of_interest=speaker_of_interest,
                mode=mode,
                extract_kwargs=extract_kwargs,
                verbose=verbose,
                spike_mode=spike_mode,
                binSize_ms=binSize_ms,
                gaussian_sigma_ms=gaussian_sigma_ms,
                gaussian_preserve_sum=gaussian_preserve_sum,
                flatten_time=flatten_time,
                embeddings_root=embeddings_root,
                embeddings_csv=emb_csv,
                return_X=return_X,
                pca_components=pca_components,
                pca_random_state=pca_random_state,
                return_pc1_labels=return_pc1_labels,
                pc1_label_mode=pc1_label_mode,
                pc1_label_q=pc1_label_q,
                pc1_label_n_bins=pc1_label_n_bins,
            )

            # unpack based on requested outputs
            if return_pc1_labels and (return_X or (pca_components is not None)):
                Ys, ys, Yo, yo, ypc1s, ypc1o, pc1_info, Xs, Xo, spk_kept = res
                out[pid] = {
                    "ok": True,
                    "Y_self": Ys, "y_self": ys,
                    "Y_other": Yo, "y_other": yo,
                    "y_pc1_self": ypc1s, "y_pc1_other": ypc1o,
                    "pc1_info": pc1_info,
                    "X_self": Xs, "X_other": Xo,
                    "spk_kept": spk_kept,
                }
            elif return_pc1_labels:
                Ys, ys, Yo, yo, ypc1s, ypc1o, pc1_info, spk_kept = res
                out[pid] = {
                    "ok": True,
                    "Y_self": Ys, "y_self": ys,
                    "Y_other": Yo, "y_other": yo,
                    "y_pc1_self": ypc1s, "y_pc1_other": ypc1o,
                    "pc1_info": pc1_info,
                    "spk_kept": spk_kept,
                }
            elif return_X or (pca_components is not None):
                Ys, ys, Yo, yo, Xs, Xo, spk_kept = res
                out[pid] = {
                    "ok": True,
                    "Y_self": Ys, "y_self": ys,
                    "Y_other": Yo, "y_other": yo,
                    "X_self": Xs, "X_other": Xo,
                    "spk_kept": spk_kept,
                }
            else:
                Ys, ys, Yo, yo, spk_kept = res
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


def load_one_patient_ccgp_inputs_word_order_from_preloaded(
    *,
    patient_tag: str,
    region: str,
    excel_path: str,
    df: pd.DataFrame,
    y_all: np.ndarray,
    row_speaker: np.ndarray,
    spikes: np.ndarray,
    region_cells: dict,   # {region: np.array(cell_idx)}
    speaker_of_interest: str = "Speaker1",
    mode: str = "target_vs_other_fixed_window_from_ref",
    extract_kwargs=None,
    label_col: str = "FinalClusterID",
    verbose: bool = False,
    spike_mode: str = "counts",  # "counts" | "rates_mean" | "rates_binned"
    binSize_ms: int = 20,
    gaussian_sigma_ms=None,
    gaussian_preserve_sum: bool = True,
    flatten_time: bool = True,
):
    """
    Same outputs as load_one_patient_ccgp_inputs_word_order:
      Y_self, y_self, Y_other, y_other, spk_kept
    but uses preloaded df/spikes to avoid expensive I/O.
    """
    import numpy as np

    if extract_kwargs is None:
        extract_kwargs = {}

    from spike_processing_utils_FIXED_durations_gaussian_spikemode import (
        compute_spike_sums,
        compute_spike_features,
        extract_speaker_events,
    )

    # Extract explicit event bounds per speaker column
    speaker_events = extract_speaker_events(
        excel_path,
        speaker_of_interest=speaker_of_interest,
        mode=mode,
        **extract_kwargs,
    )

    # ---- the "helpful glue": fixed_n_bins for rates_binned ----
    win_ms = 500
    tw = extract_kwargs.get("target_window_length", None)
    ow = extract_kwargs.get("other_window_length", None)
    cand = [v for v in (tw, ow) if v is not None]
    if len(cand) > 0:
        win_ms = int(max(cand))
    fixed_n_bins = int(win_ms // int(binSize_ms)) if int(binSize_ms) > 0 else 0

    Ys_by_spk = {}
    for spk_col, ev in speaker_events.items():
        if spike_mode == "counts":
            out = compute_spike_sums(
                spikes,
                {region: region_cells[region]},
                ev,
                mode="explicit_event_bounds",
            )
            Ys_by_spk[spk_col] = out[region]

        elif spike_mode == "rates_mean":
            out = compute_spike_features(
                spikes,
                {region: region_cells[region]},
                ev,
                mode="explicit_event_bounds",
                feature_mode="rates_mean",
            )
            Ys_by_spk[spk_col] = out[region]

        elif spike_mode == "rates_binned":
            if fixed_n_bins <= 0:
                raise ValueError(
                    f"{patient_tag}: rates_binned requires positive fixed_n_bins; "
                    f"got win_ms={win_ms}, binSize_ms={binSize_ms}"
                )
            out = compute_spike_features(
                spikes,
                {region: region_cells[region]},
                ev,
                mode="explicit_event_bounds",
                feature_mode="rates_binned",
                binSize=binSize_ms,
                gaussian_sigma_ms=gaussian_sigma_ms,
                gaussian_preserve_sum=gaussian_preserve_sum,
                flatten_time=flatten_time,
                fixed_n_bins=fixed_n_bins,
                pad_value=0.0,
            )
            Ys_by_spk[spk_col] = out[region]

        else:
            raise ValueError(f"Unknown spike_mode: {spike_mode}")

    # Reconstruct rows in Excel order
    counters = {k: 0 for k in Ys_by_spk.keys()}
    Y_rows = []
    for spk in row_speaker:
        if spk is None or spk not in Ys_by_spk:
            Y_rows.append(None)
            continue
        i = counters[spk]
        counters[spk] += 1
        Yspk = Ys_by_spk[spk]
        Y_rows.append(Yspk[i, :] if i < Yspk.shape[0] else None)

    keep_mask = np.array([r is not None for r in Y_rows], dtype=bool)
    if keep_mask.sum() == 0:
        raise RuntimeError(f"{patient_tag}: no rows were constructed")

    Y_all = np.vstack([r for r in Y_rows if r is not None])
    y_kept = y_all[keep_mask]
    spk_kept = np.array([s for s, k in zip(row_speaker, keep_mask) if k], dtype=object)

    mask_valid = np.isfinite(y_kept) & np.isfinite(Y_all).all(axis=1)
    Y_all = Y_all[mask_valid]
    y_kept = y_kept[mask_valid].astype(int)
    spk_kept = spk_kept[mask_valid]

    soi = speaker_of_interest.lower()
    mask_self = np.array([str(s).lower() == soi for s in spk_kept], dtype=bool)

    Y_self, y_self = Y_all[mask_self], y_kept[mask_self]
    Y_other, y_other = Y_all[~mask_self], y_kept[~mask_self]

    return Y_self, y_self, Y_other, y_other, spk_kept