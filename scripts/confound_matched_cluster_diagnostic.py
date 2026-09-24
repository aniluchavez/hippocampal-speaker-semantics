"""
Diagnostic: how tight are the k-means clusters used in
confound_matched_perm_prototype.py, on EACH individual confound feature
(not just the combined 14-D distance)? Tests the curse-of-dimensionality
hypothesis for why the confound-matched substitution null gave much higher
% significant than xcirc even at fine cluster granularity (N_CLUSTERS=50).

For each feature, computes the within-cluster std (mean over clusters,
weighted by cluster size) as a fraction of the overall (standardized) std.
A ratio near 0 means that feature is tightly matched within clusters; a
ratio near 1 means clustering did nothing for that feature (no better than
unclustered/random pairing).

Usage:
  cd /scratch/aniluchavez/hippocampal-speaker-semantics
  python3 -u scripts/confound_matched_cluster_diagnostic.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

PATIENT_ID = "PTYEU_task147"
CONTROL_DIR   = "/scratch/aniluchavez/ConvoDATAS/ControlFeatures"
SURPRISAL_DIR = "/scratch/aniluchavez/ConvoDATAS/Surprisal"
SPIKE_ROOT    = "/scratch/aniluchavez/ConvoDATAS/SpikeWindows"
WINDOW_TYPE   = "worddur"
REGION        = "hippocampus"

FEATURE_GROUPS = {
    "lexical":   ["log_word_freq", "word_length", "local_count", "local_rate", "surprisal"],
    "syntactic": ["dep_depth", "dep_children", "sent_position", "serial_position"],
    "acoustic":  ["f0_mean", "f0_std", "rms_mean", "spectral_flux", "speaking_rate"],
}
CONTROL_FEATURES = sum(FEATURE_GROUPS.values(), [])
N_CLUSTERS_SWEEP = [10, 20, 30, 50]
SEED = 0


def find_spike_dir(patient):
    d = os.path.join(SPIKE_ROOT, f"output_{patient}_english_only_{WINDOW_TYPE}")
    return d if os.path.isdir(d) else None


def load_speaker_assignment(spike_dir):
    cands = [f for f in os.listdir(spike_dir) if f.endswith("_with_regress_dur.xlsx")]
    tx = pd.read_excel(os.path.join(spike_dir, cands[0]))
    spk_cols = sorted(
        [c for c in tx.columns if str(c).startswith("Speaker")],
        key=lambda c: int(c.replace("Speaker", "").strip())
                      if c.replace("Speaker", "").strip().isdigit() else 999,
    )
    def _nn(val):
        return pd.notna(val) and str(val).strip() not in ("", "nan")
    dir_membership = {col: np.array([_nn(v) for v in tx[col]], dtype=bool) for col in spk_cols}
    n = len(tx)
    assign = np.array([None] * n, dtype=object)
    for i in range(n):
        for col in spk_cols:
            if dir_membership[col][i]:
                assign[i] = col
                break
    mask_self = assign == "Speaker1"
    return mask_self


def load_spike_matrix(spike_dir, speaker, region):
    spk_dir = os.path.join(spike_dir, speaker)
    cands = [f for f in os.listdir(spk_dir)
             if f.lower().startswith(region.lower()) and f.endswith(".npy")]
    return np.load(os.path.join(spk_dir, cands[0])) if cands else None


def load_control_features(patient_ID):
    ctrl = pd.read_csv(os.path.join(CONTROL_DIR, f"{patient_ID}_control_features.csv"))
    surp_path = os.path.join(SURPRISAL_DIR, f"{patient_ID}_surprisal.csv")
    if os.path.exists(surp_path) and "surprisal" not in ctrl.columns:
        surp = pd.read_csv(surp_path)
        if len(surp) == len(ctrl):
            ctrl["surprisal"] = surp["surprisal"].values
    return ctrl


spike_dir = find_spike_dir("ptYEU_task147")
mask_self = load_speaker_assignment(spike_dir)
ctrl = load_control_features(PATIENT_ID)

Y_mat = load_spike_matrix(spike_dir, "Speaker1", REGION)
valid = ~np.isnan(Y_mat).any(axis=1)

ctrl_self_valid = ctrl[mask_self][valid]
ctrl_cols = [c for c in CONTROL_FEATURES if c in ctrl_self_valid.columns]
X_raw = ctrl_self_valid[ctrl_cols].values.astype(np.float64)
col_means = np.nanmean(X_raw, axis=0)
col_means = np.where(np.isnan(col_means), 0.0, col_means)
nan_mask = np.isnan(X_raw)
X_raw[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
X_sc = StandardScaler().fit_transform(X_raw)
n_w = X_sc.shape[0]
print(f"n_words={n_w}  features={ctrl_cols}\n")


def within_cluster_std_ratio(X_sc, cluster_id):
    """For each feature, mean within-cluster std (size-weighted) / overall std."""
    overall_std = X_sc.std(axis=0)
    ratios = np.zeros(X_sc.shape[1])
    for j in range(X_sc.shape[1]):
        ws, sizes = [], []
        for c in np.unique(cluster_id):
            members = X_sc[cluster_id == c, j]
            if len(members) > 1:
                ws.append(members.std())
                sizes.append(len(members))
        ws, sizes = np.array(ws), np.array(sizes)
        within = np.average(ws, weights=sizes) if len(ws) else np.nan
        ratios[j] = within / (overall_std[j] + 1e-10)
    return ratios


print(f"{'feature':16s}  " + "  ".join(f"k={k:<4d}" for k in N_CLUSTERS_SWEEP))
all_ratios = {}
for nc in N_CLUSTERS_SWEEP:
    km = KMeans(n_clusters=nc, random_state=SEED, n_init=10).fit(X_sc)
    all_ratios[nc] = within_cluster_std_ratio(X_sc, km.labels_)

for j, feat in enumerate(ctrl_cols):
    row = "  ".join(f"{all_ratios[nc][j]:.3f} " for nc in N_CLUSTERS_SWEEP)
    print(f"{feat:16s}  {row}")

print(f"\n{'MEAN over features':16s}  " +
      "  ".join(f"{np.mean(all_ratios[nc]):.3f} " for nc in N_CLUSTERS_SWEEP))

# Reference: what ratio would a clean univariate decile-bin (10% of n_w per
# bin) achieve on a single feature, for comparison (matches stratified_perm
# style binning on ONE variable at a time)?
print("\n--- reference: univariate decile-bin within-bin std ratio (single feature at a time) ---")
ref_ratios = []
for j, feat in enumerate(ctrl_cols):
    bin_id = pd.qcut(X_sc[:, j], 10, labels=False, duplicates="drop")
    r = within_cluster_std_ratio(X_sc[:, [j]], bin_id)[0]
    ref_ratios.append(r)
    print(f"  {feat:16s}  {r:.3f}")
print(f"  {'MEAN':16s}  {np.mean(ref_ratios):.3f}")
