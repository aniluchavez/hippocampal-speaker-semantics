"""
Prototype: confound-matched word-substitution null vs. model-comparison (xcirc)
for testing whether semantic encoding survives controlling for ALL confounds
jointly (lexical + syntactic + acoustic, the 14 features in
variance_partitioning.FEATURE_GROUPS).

Generalizes the single-confound frequency-stratified permutation
(stratified_perm_prototype.py) to the full multivariate confound space:
instead of binning on one variable (e.g. word frequency deciles), cluster
words (k-means) on the combined standardized lexical+syntactic+acoustic
feature vector, then permute semantic PCA embeddings WITHIN each cluster.
Each word's embedding only ever gets swapped with another word that has
similar lexical, syntactic, AND acoustic features simultaneously — i.e. a
"confound-matched word substitution" null. The real confound feature values
are left untouched; only the semantic block is permuted, so the "controls"
model fit is reused unperturbed (same refit-only-affected-models trick as
the xcirc null in variance_partitioning.py).

One patient, one region/condition (PTYEU_task147, hippocampus, self) — directly
comparable to the existing xcirc all_controls result for the same patient
already computed in the full 15-patient family-decomposition VP run.

Usage:
  cd /scratch/aniluchavez/hippocampal-speaker-semantics
  python3 -u scripts/confound_matched_perm_prototype.py
"""

import os, time, pickle, warnings
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ── CONFIG ────────────────────────────────────────────────────────────────────
PATIENT_ID   = "PTYEU_task147"
PATIENT      = "ptYEU_task147"
REGION       = "hippocampus"
COND         = "self"
LAYER        = 36
N_COMPONENTS = 100
N_PERM       = 50
N_CLUSTERS_SWEEP = [10, 20, 30, 50]
SEED         = 0

ALPHAS     = np.logspace(-2, 4, 20)
N_OUTER    = 5
N_INNER    = 3
FDR_ALPHA  = 0.05
MIN_SPIKES = 5
ETA_CLIP   = 20.0
LBFGS_ITER = 20

EMBED_DIR   = "/scratch/aniluchavez/ConvoDATAS/EmbedCache"
CONTROL_DIR = "/scratch/aniluchavez/ConvoDATAS/ControlFeatures"
SURPRISAL_DIR = "/scratch/aniluchavez/ConvoDATAS/Surprisal"
SPIKE_ROOT  = "/scratch/aniluchavez/ConvoDATAS/SpikeWindows"
MODEL_TAG, CONTEXT_TAG, WINDOW_TYPE = "gpt2-xl", "_ctx200", "worddur"

FEATURE_GROUPS = {
    "lexical":   ["log_word_freq", "word_length", "local_count", "local_rate", "surprisal"],
    "syntactic": ["dep_depth", "dep_children", "sent_position", "serial_position"],
    "acoustic":  ["f0_mean", "f0_std", "rms_mean", "spectral_flux", "speaking_rate"],
}
CONTROL_FEATURES = sum(FEATURE_GROUPS.values(), [])

VP_PKL = ("/scratch/aniluchavez/ConvoDATAS/VPResults/gpt2-xl_ctx200_worddur_xcirc/"
          f"pc100/{PATIENT_ID}_L{LAYER:02d}_VP.pkl")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={DEVICE}", flush=True)

# ── GPU UTILITIES (copied from variance_partitioning.py) ─────────────────────

@torch.no_grad()
def gpu_pca(X_np, n_components):
    X = torch.tensor(X_np, dtype=torch.float32, device=DEVICE)
    X -= X.mean(0)
    k  = min(n_components + 10, min(X.shape))
    Y  = X @ torch.randn(X.shape[1], k, device=DEVICE)
    for _ in range(2):
        Y = X @ (X.T @ Y)
    Q, _ = torch.linalg.qr(Y)
    _, _, Vt = torch.linalg.svd(Q.T @ X, full_matrices=False)
    return (X @ Vt[:n_components].T).cpu().numpy().astype(np.float32)


def gpu_fit(X_t, Y_t, alpha):
    k, n, p = X_t.shape
    m  = Y_t.shape[2]
    n_eff = float(Y_t.sum().clamp(min=1).item())
    Xi = torch.cat([torch.ones(k, n, 1, device=DEVICE), X_t], dim=-1)
    beta = torch.zeros(k, m, p + 1, device=DEVICE, requires_grad=True)
    opt  = torch.optim.LBFGS([beta], max_iter=LBFGS_ITER, line_search_fn='strong_wolfe')
    a = alpha.to(DEVICE).view(k, 1, 1) if isinstance(alpha, torch.Tensor) else float(alpha)
    def closure():
        opt.zero_grad()
        eta  = torch.clamp(torch.bmm(Xi, beta.transpose(1, 2)), -ETA_CLIP, ETA_CLIP)
        loss = ((torch.exp(eta) - Y_t * eta).sum()
                + (a * beta[:, :, 1:].pow(2)).sum()) / n_eff
        loss.backward()
        return loss
    opt.step(closure)
    b = beta.detach()
    return b[:, :, 1:], b[:, :, :1]


def block_splits(n, k):
    b = n // k
    for i in range(k):
        s, e = i * b, (i * b + b if i < k - 1 else n)
        mask = np.zeros(n, bool); mask[s:e] = True
        yield ~mask, mask


def fdr_bh(pvals):
    n = len(pvals)
    order = np.argsort(pvals)
    adj   = np.minimum(1.0, pvals[order] * n / np.arange(1, n + 1))
    adj   = np.minimum.accumulate(adj[::-1])[::-1]
    out   = np.empty(n); out[order] = adj
    return out, out < FDR_ALPHA


def fit_one_model_ll(X_sc, Y_v):
    n_w = Y_v.shape[0]
    n_m = Y_v.shape[1]
    N_A = len(ALPHAS)
    alpha_t = torch.tensor(ALPHAS, dtype=torch.float32)
    p = X_sc.shape[1]
    ll = np.zeros(n_m, np.float64)

    for tr_m, te_m in block_splits(n_w, N_OUTER):
        n_tr = int(tr_m.sum())
        Y_tr_t = torch.tensor(Y_v[tr_m], device=DEVICE)
        Y_te_t = torch.tensor(Y_v[te_m], device=DEVICE)
        X_tr = X_sc[tr_m]

        inner_ll = np.zeros((N_A, n_m), np.float64)
        for itr_m, iva_m in block_splits(n_tr, N_INNER):
            Xii = torch.tensor(X_tr[itr_m], device=DEVICE).unsqueeze(0).expand(N_A,-1,-1).contiguous()
            Yii = Y_tr_t[itr_m].unsqueeze(0).expand(N_A,-1,-1).contiguous()
            bf, bi = gpu_fit(Xii, Yii, alpha_t)
            with torch.no_grad():
                Xiv = torch.tensor(X_tr[iva_m], device=DEVICE).unsqueeze(0).expand(N_A,-1,-1).contiguous()
                Yiv = Y_tr_t[iva_m].unsqueeze(0).expand(N_A,-1,-1).contiguous()
                eta_iv = torch.clamp(torch.bmm(Xiv, bf.transpose(1,2)) + bi.transpose(1,2),
                                     -ETA_CLIP, ETA_CLIP)
                inner_ll += (Yiv * eta_iv - torch.exp(eta_iv)).sum(1).cpu().numpy()
        best_ai = np.argmax(inner_ll, axis=0)

        X_tr_t = torch.tensor(X_tr[None], device=DEVICE)
        beta_f = torch.zeros(1, n_m, p, device=DEVICE)
        bias_f = torch.zeros(1, n_m, 1, device=DEVICE)
        for ai in np.unique(best_ai):
            ids = np.where(best_ai == ai)[0]
            bf2, bi2 = gpu_fit(X_tr_t, Y_tr_t[None, :, ids], float(ALPHAS[ai]))
            beta_f[0, ids] = bf2[0]; bias_f[0, ids] = bi2[0]

        X_te_t = torch.tensor(X_sc[te_m][None], device=DEVICE)
        with torch.no_grad():
            eta_te_t = torch.clamp(torch.bmm(X_te_t, beta_f.transpose(1,2)) + bias_f.transpose(1,2),
                                    -ETA_CLIP, ETA_CLIP)
            ll_fold = (Y_te_t * eta_te_t[0] - torch.exp(eta_te_t[0])).sum(0).cpu().numpy()
        ll += ll_fold
        del X_tr_t, X_te_t, beta_f, bias_f, eta_te_t, Y_tr_t, Y_te_t
        torch.cuda.empty_cache()

    return ll

# ── DATA LOADING (copied from variance_partitioning.py) ──────────────────────

def find_spike_dir(patient):
    d = os.path.join(SPIKE_ROOT, f"output_{patient}_english_only_{WINDOW_TYPE}")
    return d if os.path.isdir(d) else None


def load_spike_matrix(spike_dir, speaker, region):
    spk_dir = os.path.join(spike_dir, speaker)
    if not os.path.isdir(spk_dir):
        return None
    cands = [f for f in os.listdir(spk_dir)
             if f.lower().startswith(region.lower()) and f.endswith(".npy")]
    return np.load(os.path.join(spk_dir, cands[0])) if cands else None


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
    return assign, mask_self, dir_membership


def load_control_features(patient_ID):
    ctrl = pd.read_csv(os.path.join(CONTROL_DIR, f"{patient_ID}_control_features.csv"))
    surp_path = os.path.join(SURPRISAL_DIR, f"{patient_ID}_surprisal.csv")
    if os.path.exists(surp_path) and "surprisal" not in ctrl.columns:
        surp = pd.read_csv(surp_path)
        if len(surp) == len(ctrl):
            ctrl["surprisal"] = surp["surprisal"].values
    return ctrl


# ── MAIN ──────────────────────────────────────────────────────────────────────

spike_dir = find_spike_dir(PATIENT)
spk_assignment, mask_self, dir_membership = load_speaker_assignment(spike_dir)
ctrl = load_control_features(PATIENT_ID)

npy_path = os.path.join(EMBED_DIR, f"{PATIENT_ID}_{MODEL_TAG}{CONTEXT_TAG}_word_emb_layers.npy")
raw_layer = np.load(npy_path, mmap_mode='r')[LAYER].astype(np.float32)
t0 = time.time()
X_pca_all = gpu_pca(raw_layer, N_COMPONENTS)
print(f"PCA done {X_pca_all.shape} ({time.time()-t0:.1f}s)", flush=True)

Y_mat = load_spike_matrix(spike_dir, "Speaker1", REGION)
valid = ~np.isnan(Y_mat).any(axis=1)
Y_v_raw = Y_mat[valid].astype(np.float32)
spike_ok = Y_v_raw.sum(0) >= MIN_SPIKES
Y_v = Y_v_raw[:, spike_ok]
n_w, n_m = Y_v.shape
print(f"{REGION}/{COND}: {n_w}w x {n_m}n", flush=True)

X_sem_sc = StandardScaler().fit_transform(X_pca_all[mask_self][valid]).astype(np.float32)

ctrl_self_valid = ctrl[mask_self][valid]
ctrl_cols = [c for c in CONTROL_FEATURES if c in ctrl_self_valid.columns]
missing = set(CONTROL_FEATURES) - set(ctrl_cols)
if missing:
    print(f"  warning: missing control columns {missing}", flush=True)
X_ctrl_raw = ctrl_self_valid[ctrl_cols].values.astype(np.float64)
col_means = np.nanmean(X_ctrl_raw, axis=0)
col_means = np.where(np.isnan(col_means), 0.0, col_means)
nan_mask = np.isnan(X_ctrl_raw)
X_ctrl_raw[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
X_ctrl_raw = X_ctrl_raw.astype(np.float32)
X_ctrl_sc  = StandardScaler().fit_transform(X_ctrl_raw).astype(np.float32)
X_full_sc  = np.hstack([X_sem_sc, X_ctrl_sc]).astype(np.float32)

# ── Real fits: semantic, controls, full ───────────────────────────────────────
ll_null = np.zeros(n_m, np.float64)
ll_sat  = np.zeros(n_m, np.float64)
for tr_m, te_m in block_splits(n_w, N_OUTER):
    mean_tr = Y_v[tr_m].mean(0).clip(1e-10)
    ll_null += (Y_v[te_m] * np.log(mean_tr) - mean_tr).sum(0)
    Y_te_np = Y_v[te_m]
    ll_sat_fold = (Y_te_np * np.log(Y_te_np.clip(1e-10)) - Y_te_np).sum(0)
    ll_sat += np.where(Y_te_np.sum(0) > 0, ll_sat_fold, 0.0)
d_null = ll_sat - ll_null

def r2_from_ll(ll):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d_null > 0.1, (ll - ll_null) / d_null, np.nan)

t0 = time.time()
r2_sem_real   = r2_from_ll(fit_one_model_ll(X_sem_sc, Y_v))
r2_ctrl_real  = r2_from_ll(fit_one_model_ll(X_ctrl_sc, Y_v))
r2_full_real  = r2_from_ll(fit_one_model_ll(X_full_sc, Y_v))
unique_sem_real = r2_full_real - r2_ctrl_real
print(f"real fits done ({time.time()-t0:.1f}s)  "
      f"median R²_sem={np.nanmedian(r2_sem_real):.4f}  "
      f"median R²_ctrl={np.nanmedian(r2_ctrl_real):.4f}  "
      f"median unique_sem={np.nanmedian(unique_sem_real):.4f}", flush=True)

# ── Confound-matched permutation — SWEEP cluster count ────────────────────────
# K-means clusters on the combined standardized lexical+syntactic+acoustic
# feature vector (14 dims). Within each cluster, words are permuted among
# each other — so each word's semantic embedding only ever gets swapped with
# another word that's simultaneously similar in frequency, length, dep depth,
# sentence position, pitch, speaking rate, etc. Same refit-only-the-touched-
# model trick as xcirc: controls model stays fixed at its real fit.
sweep_results = {}

for nc in N_CLUSTERS_SWEEP:
    km = KMeans(n_clusters=nc, random_state=SEED, n_init=10).fit(X_ctrl_sc)
    cluster_id = km.labels_
    cluster_sizes = np.bincount(cluster_id)
    print(f"\n--- N_CLUSTERS={nc}  (sizes min={cluster_sizes.min()} "
          f"max={cluster_sizes.max()} median={int(np.median(cluster_sizes))}) ---", flush=True)

    rng = np.random.default_rng(SEED)
    perm_unique_sem = np.full((N_PERM, n_m), np.nan)
    t_perm = time.time()
    for pi in range(N_PERM):
        idx = np.arange(n_w)
        for c in np.unique(cluster_id):
            members = idx[cluster_id == c]
            if len(members) > 1:
                idx[cluster_id == c] = rng.permutation(members)
        X_sem_perm  = X_sem_sc[idx]
        X_full_perm = np.hstack([X_sem_perm, X_ctrl_sc]).astype(np.float32)
        r2_full_perm = r2_from_ll(fit_one_model_ll(X_full_perm, Y_v))
        perm_unique_sem[pi] = r2_full_perm - r2_ctrl_real
        if (pi + 1) % 10 == 0:
            print(f"  perm {pi+1}/{N_PERM}  {time.time()-t_perm:.0f}s", flush=True)

    p_vals = np.array([
        np.nanmean(perm_unique_sem[:, m] >= unique_sem_real[m])
        if not np.isnan(unique_sem_real[m]) else 1.0
        for m in range(n_m)
    ])
    p_fdr, sig = fdr_bh(p_vals)
    sig = sig & (unique_sem_real > 0)
    print(f"  N_CLUSTERS={nc}: {sig.sum()}/{n_m} sig ({100*sig.sum()/n_m:.1f}%)  "
          f"({time.time()-t_perm:.0f}s)", flush=True)
    sweep_results[nc] = {"p_vals": p_vals, "p_fdr": p_fdr, "significant": sig}

print("\n=== CLUSTER GRANULARITY SWEEP SUMMARY (confound-matched substitution null) ===")
for nc in N_CLUSTERS_SWEEP:
    sig = sweep_results[nc]["significant"]
    print(f"  N_CLUSTERS={nc:4d}  (~{n_w//nc}w/cluster):  {sig.sum()}/{n_m} sig ({100*sig.sum()/n_m:.1f}%)")

# ── Compare against the existing xcirc all_controls result for this patient ──
if os.path.exists(VP_PKL):
    vp = pickle.load(open(VP_PKL, "rb"))
    vp_sub = vp[(vp["region"] == REGION) & (vp["condition"] == COND)]
    if not vp_sub.empty:
        n_xcirc = len(vp_sub)
        sig_xcirc = vp_sub["significant"].sum()
        print(f"\n=== xcirc (model-comparison) for the SAME patient/region/condition ===")
        print(f"  xcirc:  {sig_xcirc}/{n_xcirc} sig ({100*sig_xcirc/n_xcirc:.1f}%)  "
              f"median unique_sem={vp_sub['unique_semantic'].median():.4f}")
else:
    print(f"\n(no existing xcirc VP result found at {VP_PKL} for comparison)")

out = pd.DataFrame({
    "neuron_idx": np.arange(n_m),
    "r2_semantic": r2_sem_real, "r2_controls": r2_ctrl_real, "r2_full": r2_full_real,
    "unique_semantic": unique_sem_real,
})
for nc in N_CLUSTERS_SWEEP:
    out[f"p_perm_confmatch_c{nc}"]      = sweep_results[nc]["p_vals"]
    out[f"p_fdr_confmatch_c{nc}"]       = sweep_results[nc]["p_fdr"]
    out[f"significant_confmatch_c{nc}"] = sweep_results[nc]["significant"]
out_path = "/scratch/aniluchavez/ConvoDATAS/VPResults/confound_matched_perm_prototype_PTYEU_hippo_self.pkl"
with open(out_path, "wb") as f:
    pickle.dump(out, f)
print(f"\nSaved -> {out_path}")
