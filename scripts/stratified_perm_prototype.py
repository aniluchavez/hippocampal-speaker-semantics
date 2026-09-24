"""
Prototype: confound-stratified permutation null vs. model-comparison (variance
partitioning) for testing whether semantic encoding survives controlling for
word frequency.

Instead of fitting a separate lexical-only model and subtracting R², this
null shuffles the semantic PCA embeddings WITHIN word-frequency bins (deciles
of log_word_freq) and refits the semantic-only model. The null never needs a
parametric frequency model — frequency-driven signal is preserved by
construction because each shuffled word is swapped only with another word of
similar frequency. Real semantic R² is compared to this null distribution.

One patient, one region/condition (fast — for comparing against the existing
variance-partitioning xcirc result already computed for the same patient).

Usage:
  cd /scratch/aniluchavez/hippocampal-speaker-semantics
  python3 -u scripts/stratified_perm_prototype.py
"""

import os, time, pickle, warnings
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

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
N_BINS       = 10
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
SPIKE_ROOT  = "/scratch/aniluchavez/ConvoDATAS/SpikeWindows"
MODEL_TAG, CONTEXT_TAG, WINDOW_TYPE = "gpt2-xl", "_ctx200", "worddur"

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
log_freq = ctrl["log_word_freq"].values[mask_self][valid]

# ── Real fit: semantic-only model ─────────────────────────────────────────────
ll_null = np.zeros(n_m, np.float64)
ll_sat  = np.zeros(n_m, np.float64)
for tr_m, te_m in block_splits(n_w, N_OUTER):
    mean_tr = Y_v[tr_m].mean(0).clip(1e-10)
    ll_null += (Y_v[te_m] * np.log(mean_tr) - mean_tr).sum(0)
    Y_te_np = Y_v[te_m]
    ll_sat_fold = (Y_te_np * np.log(Y_te_np.clip(1e-10)) - Y_te_np).sum(0)
    ll_sat += np.where(Y_te_np.sum(0) > 0, ll_sat_fold, 0.0)
d_null = ll_sat - ll_null

t0 = time.time()
ll_sem_real = fit_one_model_ll(X_sem_sc, Y_v)
with np.errstate(divide="ignore", invalid="ignore"):
    r2_sem_real = np.where(d_null > 0.1, (ll_sem_real - ll_null) / d_null, np.nan)
print(f"real semantic-only CV done ({time.time()-t0:.1f}s)  median R²={np.nanmedian(r2_sem_real):.4f}", flush=True)

# ── Frequency-stratified permutation null — SWEEP bin granularity ────────────
# Finer bins = tighter frequency matching = less room for residual within-bin
# frequency signal to leak through the null uncorrected.
BIN_SWEEP = [10, 20, 30, 50, 100]
sweep_results = {}

for nb in BIN_SWEEP:
    bin_id = pd.qcut(log_freq, nb, labels=False, duplicates="drop")
    actual_nb = len(np.unique(bin_id))
    bin_counts = np.bincount(bin_id)
    print(f"\n--- N_BINS={nb} (actual={actual_nb}, sizes min={bin_counts.min()} "
          f"max={bin_counts.max()} median={int(np.median(bin_counts))}) ---", flush=True)

    rng = np.random.default_rng(SEED)
    perm_r2 = np.full((N_PERM, n_m), np.nan)
    t_perm = time.time()
    for pi in range(N_PERM):
        idx = np.arange(n_w)
        for b in np.unique(bin_id):
            members = idx[bin_id == b]
            idx[bin_id == b] = rng.permutation(members)
        X_sem_perm = X_sem_sc[idx]
        ll_perm = fit_one_model_ll(X_sem_perm, Y_v)
        with np.errstate(divide="ignore", invalid="ignore"):
            perm_r2[pi] = np.where(d_null > 0.1, (ll_perm - ll_null) / d_null, np.nan)
        if (pi + 1) % 25 == 0:
            print(f"  perm {pi+1}/{N_PERM}  {time.time()-t_perm:.0f}s", flush=True)

    p_vals = np.array([
        np.nanmean(perm_r2[:, m] >= r2_sem_real[m]) if not np.isnan(r2_sem_real[m]) else 1.0
        for m in range(n_m)
    ])
    p_fdr, sig = fdr_bh(p_vals)
    sig = sig & (r2_sem_real > 0)
    print(f"  N_BINS={nb}: {sig.sum()}/{n_m} sig ({100*sig.sum()/n_m:.1f}%)  ({time.time()-t_perm:.0f}s)", flush=True)
    sweep_results[nb] = {"p_vals": p_vals, "p_fdr": p_fdr, "significant": sig}

print("\n=== BIN GRANULARITY SWEEP SUMMARY ===")
for nb in BIN_SWEEP:
    sig = sweep_results[nb]["significant"]
    print(f"  N_BINS={nb:4d}  (~{n_w//nb}w/bin):  {sig.sum()}/{n_m} sig ({100*sig.sum()/n_m:.1f}%)")

out = pd.DataFrame({"neuron_idx": np.arange(n_m), "r2_semantic": r2_sem_real})
for nb in BIN_SWEEP:
    out[f"p_perm_freqstrat_b{nb}"]      = sweep_results[nb]["p_vals"]
    out[f"p_fdr_freqstrat_b{nb}"]       = sweep_results[nb]["p_fdr"]
    out[f"significant_freqstrat_b{nb}"] = sweep_results[nb]["significant"]
out_path = "/scratch/aniluchavez/ConvoDATAS/VPResults/stratified_perm_prototype_PTYEU_hippo_self.pkl"
with open(out_path, "wb") as f:
    pickle.dump(out, f)
print(f"Saved -> {out_path}")

# ── APPLES-TO-APPLES: model-comparison restricted to frequency ONLY ──────────
# Same null mechanism (xcirc circular-shift of X_sem) as the lexical/syntactic/
# acoustic family comparisons, but baseline = frequency alone (1 feature), to
# isolate scope from null-construction-method when compared to the
# freq-stratified result above.

def compute_tau_embed(X_raw, thresh=0.05):
    n_words = X_raw.shape[0]
    max_lag = min(n_words // 4, 500)
    X = X_raw.astype(np.float64); X -= X.mean(0)
    v = X[0].copy(); v /= np.linalg.norm(v) + 1e-10
    for _ in range(20):
        v = X.T @ (X @ v); v /= np.linalg.norm(v) + 1e-10
    pc1 = X @ v; pc1 = (pc1 - pc1.mean()) / (pc1.std() + 1e-10)
    acf = np.array([np.dot(pc1[:n_words-lag], pc1[lag:]) / (n_words-lag)
                    for lag in range(1, max_lag+1)])
    below = np.where(np.abs(acf) < thresh)[0]
    return float(below[0]+1) if len(below) > 0 else float(max_lag)


X_freq_sc  = StandardScaler().fit_transform(log_freq.reshape(-1, 1)).astype(np.float32)
X_semfreq_sc = np.hstack([X_sem_sc, X_freq_sc]).astype(np.float32)

t0 = time.time()
ll_freq_real    = fit_one_model_ll(X_freq_sc, Y_v)
ll_semfreq_real = fit_one_model_ll(X_semfreq_sc, Y_v)
with np.errstate(divide="ignore", invalid="ignore"):
    r2_freq_real    = np.where(d_null > 0.1, (ll_freq_real    - ll_null) / d_null, np.nan)
    r2_semfreq_real = np.where(d_null > 0.1, (ll_semfreq_real - ll_null) / d_null, np.nan)
unique_sem_freq_real = r2_semfreq_real - r2_freq_real
print(f"\nfreq-only & sem+freq CV done ({time.time()-t0:.1f}s)  "
      f"median R²_freq={np.nanmedian(r2_freq_real):.4f}  "
      f"median unique_sem|freq={np.nanmedian(unique_sem_freq_real):.4f}", flush=True)

tau = compute_tau_embed(X_sem_sc)
min_lag = max(int(np.ceil(tau)), n_w // (N_OUTER * 2))
min_lag = min(min_lag, n_w // 2 - 1)
print(f"tau={tau:.0f}w  min_lag={min_lag}w", flush=True)

rng2 = np.random.default_rng(SEED)
perm_usem_freq = np.full((N_PERM, n_m), np.nan)
t_perm = time.time()
for pi in range(N_PERM):
    lag = int(rng2.integers(min_lag, n_w - min_lag))
    X_sem_perm     = np.roll(X_sem_sc, lag, axis=0)
    X_semfreq_perm = np.hstack([X_sem_perm, X_freq_sc]).astype(np.float32)
    ll_p = fit_one_model_ll(X_semfreq_perm, Y_v)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2_p = np.where(d_null > 0.1, (ll_p - ll_null) / d_null, np.nan)
    perm_usem_freq[pi] = r2_p - r2_freq_real
    if (pi + 1) % 10 == 0:
        print(f"  perm {pi+1}/{N_PERM}  {time.time()-t_perm:.0f}s", flush=True)
print(f"perms done ({time.time()-t_perm:.1f}s)", flush=True)

p_vals2 = np.array([
    np.nanmean(perm_usem_freq[:, m] >= unique_sem_freq_real[m])
    if not np.isnan(unique_sem_freq_real[m]) else 1.0
    for m in range(n_m)
])
p_fdr2, sig2 = fdr_bh(p_vals2)
sig2 = sig2 & (unique_sem_freq_real > 0)

print(f"\n=== model-comparison vs freq-only (circular-shift null): "
      f"{sig2.sum()}/{n_m} sig ({100*sig2.sum()/n_m:.1f}%) ===")

out2 = pd.DataFrame({
    "neuron_idx": np.arange(n_m),
    "unique_semantic_freqonly": unique_sem_freq_real,
    "p_perm_freqonly_mc": p_vals2,
    "p_fdr_freqonly_mc": p_fdr2,
    "significant_freqonly_mc": sig2,
})
out_path2 = "/scratch/aniluchavez/ConvoDATAS/VPResults/freqonly_modelcomp_prototype_PTYEU_hippo_self.pkl"
with open(out_path2, "wb") as f:
    pickle.dump(out2, f)
print(f"Saved -> {out_path2}")
