#!/usr/bin/env python3
"""
Build the Fig 5C source-data pairs (word-word cosine-distance RDM, self vs.
other), for patient PTYFF_task17, replicating the exact pipeline used for
Fig 5B (patient PTYEU_task147) in notebooks/RSA.ipynb: fixed onset window,
self shift=-150ms/len=500ms, other shift=+200ms/len=500ms (pooled across
non-Speaker1 speakers), hippocampus region, word-level FR averaged across
repeats, cosine distance between word vectors, upper-triangle pairs.

Usage: python3 -u scripts/build_fig5c_rdm_data.py --out OUT.csv
"""
import argparse
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

sys.path.insert(0, "/scratch/aniluchavez/hippocampal-speaker-semantics/legacy")
from spike_processing_utils import load_mat_data, get_cells_by_region, extract_speaker_events, compute_spike_sums

EXCEL_PATH = "/scratch/aniluchavez/ConvoDATAS/BERTEmbeds/PTYFF_task17_words_english_only/PTYFF_task17_filtered_used_rows_withNP_withClusterIDNew_withPOS.xlsx"
MAT_PATH = "/scratch/aniluchavez/ConvoDATAS/SpikesMAT/YFF/ptYFF_task17_new_spikes.mat"
REGION_RANGES = {"hippocampus": [(1, 16), (25, 40)]}
REGION_NAME = "hippocampus"

import re
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def _cell_to_first_token(x):
    if pd.isna(x):
        return np.nan
    toks = TOKEN_RE.findall(str(x).strip().lower())
    return toks[0] if toks else np.nan


def _get_speaker_cols(df):
    cols = [c for c in df.columns if re.fullmatch(r"Speaker\d+", str(c))]
    return sorted(cols, key=lambda x: int(x.replace("Speaker", "")))


def _words_in_extraction_order(df, speaker_col):
    tmp = df.copy()
    tmp[speaker_col] = tmp[speaker_col].astype(str).str.strip()
    spk_df = tmp[tmp[speaker_col] != ""]
    return [_cell_to_first_token(x) for x in spk_df[speaker_col].values]


def _shared_word_set(df, speaker1_col="Speaker1", min_repeats_each_side=1):
    speaker_cols = _get_speaker_cols(df)
    other_cols = [c for c in speaker_cols if c != speaker1_col]
    spk1_words = [w for w in _words_in_extraction_order(df, speaker1_col) if pd.notna(w)]
    other_words = []
    for spk in other_cols:
        other_words.extend([w for w in _words_in_extraction_order(df, spk) if pd.notna(w)])
    c1, co = Counter(spk1_words), Counter(other_words)
    shared = set(c1.keys()) & set(co.keys())
    shared = {w for w in shared if (c1[w] >= min_repeats_each_side and co[w] >= min_repeats_each_side)}
    return shared, speaker_cols


def make_shared_counts(words_self, words_other):
    c_self, c_other = Counter(words_self), Counter(words_other)
    shared = sorted(set(c_self.keys()) & set(c_other.keys()))
    counts_df = pd.DataFrame({
        "word": shared,
        "n_self": [c_self[w] for w in shared],
        "n_other": [c_other[w] for w in shared],
        "n_total": [c_self[w] + c_other[w] for w in shared],
    }).sort_values(["n_total", "word"], ascending=[False, True]).reset_index(drop=True)
    return shared, counts_df


def build_neuron_by_sharedword_matrices_fast(
    excel_path, mat_path, region_ranges, region_name,
    speaker1_col="Speaker1", window_spec_self=None, window_spec_other=None,
    average_repeats=True, min_repeats_each_side=1, bin_mode="explicit_event_bounds",
):
    df = pd.read_excel(excel_path, keep_default_na=False)
    shared, speaker_cols = _shared_word_set(df, speaker1_col=speaker1_col, min_repeats_each_side=min_repeats_each_side)
    other_cols = [c for c in speaker_cols if c != speaker1_col]

    spikes, qual, chan = load_mat_data(mat_path)
    region_cells_all = get_cells_by_region(chan, qual, region_ranges)
    region_cells = {region_name: region_cells_all[region_name]}
    n_neurons = len(region_cells_all[region_name])
    neuron_ids = list(range(n_neurons))

    events_self_all = extract_speaker_events(excel_path, **window_spec_self)
    events_other_all = extract_speaker_events(excel_path, **window_spec_other)

    ev_self = events_self_all.get(speaker1_col)
    words_self_full = _words_in_extraction_order(df, speaker1_col)[: ev_self.shape[0]]
    mask_self = np.array([(w in shared) for w in words_self_full], dtype=bool)
    ev_self = ev_self[mask_self]
    words_self = np.array(words_self_full, dtype=object)[mask_self]

    ev_other_list, words_other_list = [], []
    for spk in other_cols:
        ev = events_other_all.get(spk)
        if ev is None or len(ev) == 0:
            continue
        words_full = _words_in_extraction_order(df, spk)[: ev.shape[0]]
        mask = np.array([(w in shared) for w in words_full], dtype=bool)
        if np.any(mask):
            ev_other_list.append(ev[mask])
            words_other_list.append(np.array(words_full, dtype=object)[mask])

    ev_other = np.vstack(ev_other_list)
    words_other = np.concatenate(words_other_list)

    shared_words_present, shared_counts_df = make_shared_counts(words_self.tolist(), words_other.tolist())

    def compute_fr(ev):
        sums = compute_spike_sums(spikes, region_cells, ev, mode=bin_mode)
        mat = sums[region_name]
        win_ms = ev[:, 2] - ev[:, 1]
        win_s = np.where(win_ms > 0, win_ms / 1000.0, np.nan)
        return mat / win_s[:, None]

    fr_self_events = compute_fr(ev_self)
    fr_other_events = compute_fr(ev_other)

    col_labels = shared_words_present
    FR_self = np.full((n_neurons, len(col_labels)), np.nan)
    FR_other = np.full((n_neurons, len(col_labels)), np.nan)
    for j, w in enumerate(col_labels):
        ii = np.where(words_self == w)[0]
        jj = np.where(words_other == w)[0]
        if ii.size:
            FR_self[:, j] = np.nanmean(fr_self_events[ii, :], axis=0)
        if jj.size:
            FR_other[:, j] = np.nanmean(fr_other_events[jj, :], axis=0)

    return FR_self, FR_other, col_labels, neuron_ids, shared_counts_df


def word_word_cosine_distance(FR, col_labels, fill="col_mean"):
    X = FR.astype(float).copy()
    if np.isnan(X).any():
        col_means = np.nanmean(X, axis=0, keepdims=True)
        X = np.where(np.isfinite(X), X, col_means)
    D = cosine_distances(X.T)
    return pd.DataFrame(D, index=col_labels, columns=col_labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/fig5c_rdm_scatter_YFF.csv")
    args = ap.parse_args()

    window_self = dict(
        speaker_of_interest="Speaker1", mode="target_vs_other_fixed_window_from_ref",
        target_ref_point="onset", target_shift=-150, target_window_length=500,
        other_ref_point="onset", other_shift=200, other_window_length=500,
    )
    window_other = dict(window_self)

    FR_self, FR_other, col_labels, neuron_ids, shared_counts_df = build_neuron_by_sharedword_matrices_fast(
        excel_path=EXCEL_PATH, mat_path=MAT_PATH, region_ranges=REGION_RANGES, region_name=REGION_NAME,
        average_repeats=True, window_spec_self=window_self, window_spec_other=window_other,
    )
    print("FR_self shape:", FR_self.shape, "n shared words:", len(col_labels))

    D_self = word_word_cosine_distance(FR_self, col_labels)
    D_other = word_word_cosine_distance(FR_other, col_labels)

    iu = np.triu_indices(len(col_labels), k=1)
    v_self = D_self.values[iu]
    v_other = D_other.values[iu]
    mask = np.isfinite(v_self) & np.isfinite(v_other)
    v_self, v_other = v_self[mask], v_other[mask]

    from scipy.stats import pearsonr, spearmanr
    r_p, p_p = pearsonr(v_self, v_other)
    r_s, p_s = spearmanr(v_self, v_other)
    print(f"n pairs: {len(v_self)}")
    print(f"Pearson r = {r_p:.3f}, p = {p_p:.2e}")
    print(f"Spearman rho = {r_s:.3f}, p = {p_s:.2e}")

    out = pd.DataFrame({
        "pair_index": np.arange(1, len(v_self) + 1),
        "v_self_speaking_word_word_distance": v_self,
        "v_other_listening_word_word_distance": v_other,
    })
    out.to_csv(args.out, index=False)
    print("saved", args.out)


if __name__ == "__main__":
    main()
