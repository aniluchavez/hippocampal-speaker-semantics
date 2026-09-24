"""
Corrected symmetric calibration check for unique_controls.

The first null_calibration_check.py run tested unique_controls's calibration
on Y_sim ~ Poisson(lambda_ctrl) -- but that dataset has a real, strong TRUE
controls effect baked in by construction (its rate function IS the controls-
only fit), so "82.9% significant for unique_controls" on that data was the
CORRECT detection of a real effect, not a calibration failure. That dataset
is only a valid zero-ground-truth test for the SEMANTIC direction.

This script builds the actually-symmetric counterpart: Y_sim2 ~
Poisson(lambda_sem), using a SEMANTIC-only fitted rate, so the ground truth
now has ZERO true controls effect (by construction) and a real true semantic
effect. On this dataset:
  - unique_semantic should come out HIGH (power check -- xcirc should detect
    the real effect we built in)
  - unique_controls should come out ~5% / raw p-values ~Uniform(0,1)
    (the actual calibration check for this direction)

One patient, one region/condition (PTYEU_task147, hippocampus, self), same
as every other check this session.

Usage:
  cd /scratch/aniluchavez/hippocampal-speaker-semantics
  python3 -u scripts/null_calibration_check_symmetric.py
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
N_COMPONENTS = 50
N_PERM_XCIRC = 50
SEED         = 1   # different seed from the first check -- independent draw

ALPHAS       = np.logspace(-2, 4, 20)
N_OUTER      = 5
N_INNER      = 3
FDR_ALPHA    = 0.05
MIN_SPIKES   = 5
ETA_CLIP     = 20.0
LBFGS_ITER   = 20
EPS          = 1e-6

EMBED_DIR     = "/scratch/aniluchavez/ConvoDATAS/EmbedCache"
CONTROL_DIR   = "/scratch/aniluchavez/ConvoDATAS/ControlFeatures"
SURPRISAL_DIR = "/scratch/aniluchavez/ConvoDATAS/Surprisal"
SPIKE_ROOT    = "/scratch/aniluchavez/ConvoDATAS/SpikeWindows"
MODEL_TAG, CONTEXT_TAG, WINDOW_TYPE = "gpt2-xl", "_ctx200", "worddur"

FEATURE_GROUPS = {
    "lexical":   ["log_word_freq", "word_length", "local_count", "local_rate", "surprisal"],
    "syntactic": ["dep_depth", "dep_children", "sent_position", "serial_position"],
    "acoustic":  ["f0_mean", "f0_std", "rms_mean", "spectral_flux", "speaking_rate"],
}
CONTROL_FEATURES = sum(FEATURE_GROUPS.values(), [])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={DEVICE}", flush=True)

# ── GPU UTILITIES ─────────────────────────────────────────────────────────────

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
    n_w, n_m = Y_v.shape
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


def fit_poisson_held_out_lambda(X_sc, Y_v):
    n_w, n_m = Y_v.shape
    N_A = len(ALPHAS)
    alpha_t = torch.tensor(ALPHAS, dtype=torch.float32)
    p = X_sc.shape[1]
    lam_hat = np.zeros((n_w, n_m), np.float64)
    for tr_m, te_m in block_splits(n_w, N_OUTER):
        n_tr = int(tr_m.sum())
        Y_tr_t = torch.tensor(Y_v[tr_m], device=DEVICE)
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
            lam_hat[te_m] = torch.exp(eta_te_t)[0].cpu().numpy()
        del X_tr_t, X_te_t, beta_f, bias_f, eta_te_t, Y_tr_t
        torch.cuda.empty_cache()
    return lam_hat


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

# ── DATA LOADING ──────────────────────────────────────────────────────────────

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
    return mask_self


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
mask_self = load_speaker_assignment(spike_dir)
ctrl = load_control_features(PATIENT_ID)

npy_path = os.path.join(EMBED_DIR, f"{PATIENT_ID}_{MODEL_TAG}{CONTEXT_TAG}_word_emb_layers.npy")
raw_layer = np.load(npy_path, mmap_mode='r')[LAYER].astype(np.float32)
X_pca_all = gpu_pca(raw_layer, N_COMPONENTS)

Y_mat = load_spike_matrix(spike_dir, "Speaker1", REGION)
valid = ~np.isnan(Y_mat).any(axis=1)
Y_v_raw = Y_mat[valid].astype(np.float32)
spike_ok = Y_v_raw.sum(0) >= MIN_SPIKES
Y_v_real = Y_v_raw[:, spike_ok]
n_w, n_m = Y_v_real.shape
print(f"{REGION}/{COND}: {n_w}w x {n_m}n", flush=True)

X_sem_sc = StandardScaler().fit_transform(X_pca_all[mask_self][valid]).astype(np.float32)
ctrl_self_valid = ctrl[mask_self][valid]
ctrl_cols = [c for c in CONTROL_FEATURES if c in ctrl_self_valid.columns]
X_ctrl_raw = ctrl_self_valid[ctrl_cols].values.astype(np.float64)
col_means = np.nanmean(X_ctrl_raw, axis=0)
col_means = np.where(np.isnan(col_means), 0.0, col_means)
nan_mask = np.isnan(X_ctrl_raw)
X_ctrl_raw[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
X_ctrl_sc = StandardScaler().fit_transform(X_ctrl_raw.astype(np.float32)).astype(np.float32)

# ── Build the CORRECTED known-null simulated dataset: zero TRUE controls effect
t0 = time.time()
lam_sem_real = fit_poisson_held_out_lambda(X_sem_sc, Y_v_real).clip(EPS)
print(f"real semantic-only Poisson fit done ({time.time()-t0:.1f}s)  "
      f"median lambda={np.median(lam_sem_real):.4f}", flush=True)

rng_sim = np.random.default_rng(SEED)
Y_v = rng_sim.poisson(lam_sem_real).astype(np.float32)
print(f"simulated Y_sim2 ~ Poisson(lambda_sem_real)  "
      f"(mean count: sim={Y_v.mean():.3f} vs real={Y_v_real.mean():.3f})  "
      f"-- ground truth: ZERO true CONTROLS effect, real true SEMANTIC effect", flush=True)

X_full_sc = np.hstack([X_sem_sc, X_ctrl_sc]).astype(np.float32)
ll_sem_real = fit_one_model_ll(X_sem_sc, Y_v)
ll_ctrl_real = fit_one_model_ll(X_ctrl_sc, Y_v)
ll_full_real = fit_one_model_ll(X_full_sc, Y_v)

ll_null = np.zeros(n_m, np.float64); ll_sat = np.zeros(n_m, np.float64)
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

r2_sem_real  = r2_from_ll(ll_sem_real)
r2_ctrl_real = r2_from_ll(ll_ctrl_real)
r2_full_real = r2_from_ll(ll_full_real)
unique_sem_real  = r2_full_real - r2_ctrl_real
unique_ctrl_real = r2_full_real - r2_sem_real

results = {}

# ── DIRECTION 1: unique_semantic (POWER check -- real effect should be detected)
print("\n" + "="*60 + "\nDIRECTION 1: unique_semantic (power check, real effect present)", flush=True)
tau_sem = compute_tau_embed(X_sem_sc)
min_lag_sem = max(int(np.ceil(tau_sem)), n_w // (N_OUTER * 2))
min_lag_sem = min(min_lag_sem, n_w // 2 - 1)
print(f"tau_sem={tau_sem:.0f}w  min_lag_sem={min_lag_sem}w", flush=True)

t0 = time.time()
rng1 = np.random.default_rng(SEED)
perm_usem = np.full((N_PERM_XCIRC, n_m), np.nan)
for pi in range(N_PERM_XCIRC):
    lag = int(rng1.integers(min_lag_sem, n_w - min_lag_sem))
    X_full_perm = np.hstack([np.roll(X_sem_sc, lag, axis=0), X_ctrl_sc]).astype(np.float32)
    r2_full_perm = r2_from_ll(fit_one_model_ll(X_full_perm, Y_v))
    perm_usem[pi] = r2_full_perm - r2_ctrl_real
    if (pi + 1) % 10 == 0:
        print(f"  perm {pi+1}/{N_PERM_XCIRC}  {time.time()-t0:.0f}s", flush=True)
p_vals_sem = np.array([np.nanmean(perm_usem[:, m] >= unique_sem_real[m])
                        if not np.isnan(unique_sem_real[m]) else 1.0 for m in range(n_m)])
_, sig_sem = fdr_bh(p_vals_sem)
sig_sem = sig_sem & (unique_sem_real > 0)
results["unique_semantic_power"] = 100 * sig_sem.sum() / n_m
print(f"unique_semantic on Y_sim2: {sig_sem.sum()}/{n_m} sig ({results['unique_semantic_power']:.1f}%)  "
      f"[expect HIGH -- real effect present]  ({time.time()-t0:.0f}s)", flush=True)

# ── DIRECTION 2: unique_controls (CALIBRATION check -- zero true effect) ────
print("\n" + "="*60 + "\nDIRECTION 2: unique_controls (calibration check, zero true effect)", flush=True)
tau_ctrl = compute_tau_embed(X_ctrl_sc)
min_lag_ctrl = max(int(np.ceil(tau_ctrl)), n_w // (N_OUTER * 2))
min_lag_ctrl = min(min_lag_ctrl, n_w // 2 - 1)
print(f"tau_ctrl={tau_ctrl:.0f}w  min_lag_ctrl={min_lag_ctrl}w", flush=True)

t0 = time.time()
rng2 = np.random.default_rng(SEED)
perm_uctrl = np.full((N_PERM_XCIRC, n_m), np.nan)
for pi in range(N_PERM_XCIRC):
    lag = int(rng2.integers(min_lag_ctrl, n_w - min_lag_ctrl))
    X_full_perm = np.hstack([X_sem_sc, np.roll(X_ctrl_sc, lag, axis=0)]).astype(np.float32)
    r2_full_perm = r2_from_ll(fit_one_model_ll(X_full_perm, Y_v))
    perm_uctrl[pi] = r2_full_perm - r2_sem_real
    if (pi + 1) % 10 == 0:
        print(f"  perm {pi+1}/{N_PERM_XCIRC}  {time.time()-t0:.0f}s", flush=True)
p_vals_ctrl = np.array([np.nanmean(perm_uctrl[:, m] >= unique_ctrl_real[m])
                         if not np.isnan(unique_ctrl_real[m]) else 1.0 for m in range(n_m)])
_, sig_ctrl = fdr_bh(p_vals_ctrl)
sig_ctrl = sig_ctrl & (unique_ctrl_real > 0)
results["unique_controls_calibration"] = 100 * sig_ctrl.sum() / n_m
print(f"unique_controls on Y_sim2: {sig_ctrl.sum()}/{n_m} sig ({results['unique_controls_calibration']:.1f}%)  "
      f"[expect ~5% -- zero true effect]  ({time.time()-t0:.0f}s)", flush=True)

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n\n" + "="*60)
print("=== CORRECTED SYMMETRIC CALIBRATION CHECK ===")
print("    Y_sim2 ~ Poisson(lambda_sem): real semantic effect, ZERO true controls effect")
for k, v in results.items():
    print(f"  {k:28s}  {v:.1f}% significant (FDR-corrected)")

print("\n--- raw (uncorrected) p-value distribution ---")
for name, pv in [("unique_semantic (power)", p_vals_sem),
                  ("unique_controls (calibration)", p_vals_ctrl)]:
    pv = pv[~np.isnan(pv)]
    pct_below_05 = 100 * np.mean(pv < 0.05)
    print(f"  {name:30s}  mean={pv.mean():.3f}  %raw_p<0.05={pct_below_05:.1f}%  "
          f"min={pv.min():.3f}  n={len(pv)}")
    hist, edges = np.histogram(pv, bins=10, range=(0, 1))
    print(f"      histogram (10 bins, 0->1): {hist.tolist()}")

out = pd.DataFrame([results])
out["p_vals_unique_semantic_power"] = [p_vals_sem]
out["p_vals_unique_controls_calibration"] = [p_vals_ctrl]
out_path = "/scratch/aniluchavez/ConvoDATAS/VPResults/null_calibration_check_symmetric_PTYEU_hippo_self.pkl"
with open(out_path, "wb") as f:
    pickle.dump(out, f)
print(f"\nSaved -> {out_path}")
