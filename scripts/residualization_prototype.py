"""
Prototype: residualization (Frisch-Waugh-Lovell-style) confound control vs.
model-comparison (xcirc) for testing whether semantic encoding survives
controlling for ALL confounds jointly (lexical + syntactic + acoustic).

Avoids ever fitting a joint [semantic + controls] model. Instead:
  1. Residualize the neural response against controls: fit a Poisson ridge
     model of Y on X_ctrl (held-out per outer fold), take Pearson residuals
     Y_resid = (Y - lambda_hat) / sqrt(lambda_hat).
  2. Residualize the semantic PCA features against controls: per outer fold,
     fit closed-form ridge of X_sem on X_ctrl (train only), subtract the
     fitted projection from BOTH train and test rows of that fold.
  3. Fit a plain (closed-form) ridge regression of Y_resid on X_sem_resid,
     5-fold block CV, held-out R² as the test statistic.

This is the two-sided residual method the user described: regress out
controls from BOTH sides, then see how much residual semantic content
explains residual response variance. Known caveats (vs. the joint model):
asymmetric (residualizing Y first vs. X_sem first need not agree), and not
exactly equivalent to the joint-model unique-variance decomposition the way
Frisch-Waugh-Lovell guarantees for plain OLS -- Poisson residualization on
one side breaks the exact equivalence.

Null: circularly shift the RAW semantic embeddings (same xcirc mechanism
used everywhere else), then recompute X_sem_resid using the SAME real
per-fold ridge projections (held fixed -- they don't depend on the
permutation), and refit only the cheap final-stage ridge regression. No
Poisson refitting needed in the permutation loop at all, since both
residualization stages are fixed/real -- much faster than xcirc's
joint-Poisson-refit-per-permutation.

One patient, one region/condition (PTYEU_task147, hippocampus, self) --
directly comparable to the existing xcirc all_controls result.

Usage:
  cd /scratch/aniluchavez/hippocampal-speaker-semantics
  python3 -u scripts/residualization_prototype.py
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
N_PERM       = 100
SEED         = 0

ALPHAS       = np.logspace(-2, 4, 20)   # Poisson controls-model alpha grid
RESID_ALPHA  = 10.0                      # fixed ridge alpha, semantic~controls projection (low-dim, p=14, untuned is fine)
FINAL_ALPHAS = np.logspace(1, 4, 10)     # ridge alpha grid, final residual~residual stage -- chosen via NESTED
                                          # inner CV per neuron (see fit_ridge_r2), not by leaking the outer test fold
N_OUTER      = 5
N_INNER      = 3
FDR_ALPHA    = 0.05
MIN_SPIKES   = 5
ETA_CLIP     = 20.0
LBFGS_ITER   = 20

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

VP_PKL = ("/scratch/aniluchavez/ConvoDATAS/VPResults/gpt2-xl_ctx200_worddur_xcirc/"
          f"pc100/{PATIENT_ID}_L{LAYER:02d}_VP.pkl")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={DEVICE}", flush=True)

# ── GPU UTILITIES (Poisson ridge, copied from variance_partitioning.py) ──────

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
    """Nested-CV Poisson ridge fit -> held-out log-likelihood per neuron."""
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
    """Same nested-CV Poisson ridge fit, but returns held-out predicted rate
    lambda_hat (n_w, n_m) instead of summed log-likelihood -- needed to build
    Pearson residuals of Y against controls."""
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

# ── CPU closed-form ridge utilities (continuous-on-continuous, fast) ─────────

def ridge_fit(X_tr, Y_tr, alpha):
    """Closed-form multivariate ridge. X_tr:(n,p) Y_tr:(n,q) -> B:(p,q)."""
    p = X_tr.shape[1]
    XtX = X_tr.T @ X_tr + alpha * np.eye(p)
    XtY = X_tr.T @ Y_tr
    return np.linalg.solve(XtX, XtY)


def residualize_semantic(X_ctrl_sc, X_sem_sc, n_w, alpha):
    """Per outer fold: fit ridge B (X_ctrl -> X_sem) on TRAIN only, residualize
    both train and test rows of that fold using that fold's B. Returns the
    full (n_w, P) residual array plus the per-fold B's (for reuse, unchanged,
    in the permutation null)."""
    resid = np.zeros_like(X_sem_sc)
    fold_B = []
    for tr_m, te_m in block_splits(n_w, N_OUTER):
        B = ridge_fit(X_ctrl_sc[tr_m], X_sem_sc[tr_m], alpha)
        fold_B.append((tr_m, te_m, B))
        resid[tr_m] = X_sem_sc[tr_m] - X_ctrl_sc[tr_m] @ B
        resid[te_m] = X_sem_sc[te_m] - X_ctrl_sc[te_m] @ B
    return resid, fold_B


def residualize_semantic_with_fixed_B(X_ctrl_sc, X_sem_sc, fold_B):
    """Same as above but reusing already-fit per-fold B's (held fixed) -- used
    inside the permutation loop on circularly-shifted X_sem."""
    resid = np.zeros_like(X_sem_sc)
    for tr_m, te_m, B in fold_B:
        resid[tr_m] = X_sem_sc[tr_m] - X_ctrl_sc[tr_m] @ B
        resid[te_m] = X_sem_sc[te_m] - X_ctrl_sc[te_m] @ B
    return resid


def fit_ridge_r2(X_resid, Y_resid, n_w, alphas):
    """Nested-CV ridge of Y_resid (n_w,n_m) on X_resid (n_w,p): inner CV picks
    a per-neuron alpha using ONLY inner train/val splits (never touches the
    outer test fold), outer fold then evaluates held-out R² at that alpha --
    same nested structure as fit_one_model_ll's Poisson alpha selection, to
    avoid leaking the outer test fold into the alpha choice (which inflated
    the result badly when alpha was tuned directly against outer-fold R²)."""
    n_m = Y_resid.shape[1]
    alphas = np.asarray(alphas, dtype=np.float64)
    ss_res = np.zeros(n_m); ss_tot = np.zeros(n_m)
    for tr_m, te_m in block_splits(n_w, N_OUTER):
        X_tr, Y_tr = X_resid[tr_m], Y_resid[tr_m]
        n_tr = X_tr.shape[0]
        inner_mse = np.zeros((len(alphas), n_m))
        for itr_m, iva_m in block_splits(n_tr, N_INNER):
            for ai, a in enumerate(alphas):
                B = ridge_fit(X_tr[itr_m], Y_tr[itr_m], a)
                pred_iv = X_tr[iva_m] @ B
                inner_mse[ai] += ((Y_tr[iva_m] - pred_iv) ** 2).sum(0)
        best_ai = np.argmin(inner_mse, axis=0)

        pred_te = np.zeros((int(te_m.sum()), n_m))
        for ai in np.unique(best_ai):
            ids = np.where(best_ai == ai)[0]
            B = ridge_fit(X_tr, Y_tr[:, ids], float(alphas[ai]))
            pred_te[:, ids] = X_resid[te_m] @ B
        train_mean = Y_tr.mean(0)
        ss_res += ((Y_resid[te_m] - pred_te) ** 2).sum(0)
        ss_tot += ((Y_resid[te_m] - train_mean) ** 2).sum(0)
    return 1 - ss_res / ss_tot.clip(1e-10)


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
X_ctrl_sc = StandardScaler().fit_transform(X_ctrl_raw.astype(np.float32)).astype(np.float32)

# ── Stage 1: residualize Y against controls (Poisson, Pearson residuals) ─────
t0 = time.time()
lam_ctrl = fit_poisson_held_out_lambda(X_ctrl_sc, Y_v)
Y_resid = (Y_v - lam_ctrl) / np.sqrt(lam_ctrl.clip(1e-6))
print(f"Y residualized against controls ({time.time()-t0:.1f}s)  "
      f"median lambda_ctrl={np.median(lam_ctrl):.4f}", flush=True)

# ── Stage 2: residualize X_sem against controls (closed-form ridge, per fold) ─
X_sem_resid_real, fold_B = residualize_semantic(X_ctrl_sc, X_sem_sc, n_w, RESID_ALPHA)
print(f"X_sem residualized against controls  "
      f"(mean |resid| / mean |raw| = "
      f"{np.abs(X_sem_resid_real).mean() / np.abs(X_sem_sc).mean():.3f})", flush=True)

# ── Stage 3: real residual~residual regression, held-out R² ──────────────────
t0 = time.time()
r2_resid_real = fit_ridge_r2(X_sem_resid_real, Y_resid, n_w, FINAL_ALPHAS)
print(f"residual~residual CV done ({time.time()-t0:.1f}s)  "
      f"median R²={np.nanmedian(r2_resid_real):.4f}", flush=True)

# ── Null: circular-shift raw X_sem, reuse fixed Y_resid + fold_B ────────────
tau = compute_tau_embed(X_sem_sc)
min_lag = max(int(np.ceil(tau)), n_w // (N_OUTER * 2))
min_lag = min(min_lag, n_w // 2 - 1)
print(f"tau={tau:.0f}w  min_lag={min_lag}w", flush=True)

rng = np.random.default_rng(SEED)
perm_r2 = np.full((N_PERM, n_m), np.nan)
t0 = time.time()
for pi in range(N_PERM):
    lag = int(rng.integers(min_lag, n_w - min_lag))
    X_sem_perm = np.roll(X_sem_sc, lag, axis=0)
    X_sem_resid_perm = residualize_semantic_with_fixed_B(X_ctrl_sc, X_sem_perm, fold_B)
    perm_r2[pi] = fit_ridge_r2(X_sem_resid_perm, Y_resid, n_w, FINAL_ALPHAS)
    if (pi + 1) % 25 == 0:
        print(f"  perm {pi+1}/{N_PERM}  {time.time()-t0:.1f}s", flush=True)
print(f"perms done ({time.time()-t0:.1f}s)", flush=True)

p_vals = np.array([
    np.nanmean(perm_r2[:, m] >= r2_resid_real[m]) if not np.isnan(r2_resid_real[m]) else 1.0
    for m in range(n_m)
])
p_fdr, sig = fdr_bh(p_vals)
sig = sig & (r2_resid_real > 0)
print(f"\n=== residualization method: {sig.sum()}/{n_m} sig ({100*sig.sum()/n_m:.1f}%) ===")

# ── Compare against the existing xcirc all_controls result ──────────────────
if os.path.exists(VP_PKL):
    vp = pickle.load(open(VP_PKL, "rb"))
    vp_sub = vp[(vp["region"] == REGION) & (vp["condition"] == COND)]
    if not vp_sub.empty:
        n_xcirc = len(vp_sub)
        sig_xcirc = vp_sub["significant"].sum()
        print(f"=== xcirc (joint model-comparison), SAME patient/region/condition ===")
        print(f"  xcirc:  {sig_xcirc}/{n_xcirc} sig ({100*sig_xcirc/n_xcirc:.1f}%)  "
              f"median unique_sem={vp_sub['unique_semantic'].median():.4f}")
else:
    print(f"\n(no existing xcirc VP result found at {VP_PKL} for comparison)")

out = pd.DataFrame({
    "neuron_idx": np.arange(n_m),
    "r2_resid": r2_resid_real,
    "p_perm_resid": p_vals, "p_fdr_resid": p_fdr, "significant_resid": sig,
})
out_path = "/scratch/aniluchavez/ConvoDATAS/VPResults/residualization_prototype_PTYEU_hippo_self.pkl"
with open(out_path, "wb") as f:
    pickle.dump(out, f)
print(f"\nSaved -> {out_path}")
