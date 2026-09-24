"""
Prototype: "is predictor block X significant on its own?" for semantic,
lexical, syntactic, acoustic — independent of how they compete with each
other (different question from unique_semantic | family, which asks whether
semantic survives a given control).

For each block: fit the block-alone model, build a null by circularly
shifting that block's OWN raw features (lag > tau, that block's own
autocorrelation) and refitting, compare real R² to the null. Same xcirc
logic used throughout this project, just pointed at each block in turn.

One patient (PTYEU_task147), hippocampus, self + other — prototype before
committing to a full 15-patient run.

Usage:
  cd /scratch/aniluchavez/hippocampal-speaker-semantics
  python3 -u scripts/block_significance_prototype.py
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
LAYER        = 36
N_COMPONENTS = 100
N_PERM       = 50
SEED         = 0

FEATURE_GROUPS = {
    "lexical":   ["log_word_freq", "word_length", "local_count", "local_rate", "surprisal"],
    "syntactic": ["dep_depth", "dep_children", "sent_position", "serial_position"],
    "acoustic":  ["f0_mean", "f0_std", "rms_mean", "spectral_flux", "speaking_rate"],
}

ALPHAS     = np.logspace(-2, 4, 20)
N_OUTER    = 5
N_INNER    = 3
FDR_ALPHA  = 0.05
MIN_SPIKES = 5
ETA_CLIP   = 20.0
LBFGS_ITER = 20

EMBED_DIR     = "/scratch/aniluchavez/ConvoDATAS/EmbedCache"
CONTROL_DIR   = "/scratch/aniluchavez/ConvoDATAS/ControlFeatures"
SURPRISAL_DIR = "/scratch/aniluchavez/ConvoDATAS/Surprisal"
SPIKE_ROOT    = "/scratch/aniluchavez/ConvoDATAS/SpikeWindows"
MODEL_TAG, CONTEXT_TAG, WINDOW_TYPE = "gpt2-xl", "_ctx200", "worddur"

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


def block_significance(X_sc, Y_v, ll_null, d_null, n_w, n_m, rng):
    """Real fit + circular-shift null for one predictor block alone."""
    ll_real = fit_one_model_ll(X_sc, Y_v)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2_real = np.where(d_null > 0.1, (ll_real - ll_null) / d_null, np.nan)

    tau = compute_tau_embed(X_sc)
    min_lag = max(int(np.ceil(tau)), n_w // (N_OUTER * 2))
    min_lag = min(min_lag, n_w // 2 - 1)

    perm_r2 = np.full((N_PERM, n_m), np.nan)
    for pi in range(N_PERM):
        lag = int(rng.integers(min_lag, n_w - min_lag))
        X_perm = np.roll(X_sc, lag, axis=0)
        ll_p = fit_one_model_ll(X_perm, Y_v)
        with np.errstate(divide="ignore", invalid="ignore"):
            perm_r2[pi] = np.where(d_null > 0.1, (ll_p - ll_null) / d_null, np.nan)

    p_vals = np.array([
        np.nanmean(perm_r2[:, m] >= r2_real[m]) if not np.isnan(r2_real[m]) else 1.0
        for m in range(n_m)
    ])
    _, sig = fdr_bh(p_vals)
    sig = sig & (r2_real > 0)
    return r2_real, p_vals, sig, tau

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
    mask_self  = assign == "Speaker1"
    mask_other = np.array([(a is not None and a != "Speaker1") for a in assign], dtype=bool)
    return assign, mask_self, mask_other, dir_membership


def load_condition_ordered(spike_dir, cond, region, spk_assignment, dir_membership):
    if cond == "self":
        return load_spike_matrix(spike_dir, "Speaker1", region)
    other_spks = sorted([c for c in dir_membership if c != "Speaker1"])
    mats = {s: load_spike_matrix(spike_dir, s, region) for s in other_spks}
    mats = {s: m for s, m in mats.items() if m is not None}
    if not mats:
        return None
    dir_pos = {s: 0 for s in mats}
    rows = []
    for i, spk in enumerate(spk_assignment):
        if spk is None or spk == "Speaker1":
            continue
        for s in mats:
            if dir_membership[s][i]:
                if spk == s:
                    rows.append(mats[s][dir_pos[s]])
                dir_pos[s] += 1
    return np.vstack(rows) if rows else None


def load_control_features(patient_ID):
    ctrl = pd.read_csv(os.path.join(CONTROL_DIR, f"{patient_ID}_control_features.csv"))
    surp_path = os.path.join(SURPRISAL_DIR, f"{patient_ID}_surprisal.csv")
    if os.path.exists(surp_path) and "surprisal" not in ctrl.columns:
        surp = pd.read_csv(surp_path)
        if len(surp) == len(ctrl):
            ctrl["surprisal"] = surp["surprisal"].values
    return ctrl


def clean_cols(df, cols):
    X = df[cols].values.astype(float)
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    return X

# ── MAIN ──────────────────────────────────────────────────────────────────────

spike_dir = find_spike_dir(PATIENT)
spk_assignment, mask_self, mask_other, dir_membership = load_speaker_assignment(spike_dir)
ctrl = load_control_features(PATIENT_ID)

npy_path = os.path.join(EMBED_DIR, f"{PATIENT_ID}_{MODEL_TAG}{CONTEXT_TAG}_word_emb_layers.npy")
raw_layer = np.load(npy_path, mmap_mode='r')[LAYER].astype(np.float32)
X_pca_all = gpu_pca(raw_layer, N_COMPONENTS)

results = []

for cond, mask in [("self", mask_self), ("other", mask_other)]:
    Y_mat = load_condition_ordered(spike_dir, cond, REGION, spk_assignment, dir_membership)
    valid = ~np.isnan(Y_mat).any(axis=1)
    Y_v_raw = Y_mat[valid].astype(np.float32)
    spike_ok = Y_v_raw.sum(0) >= MIN_SPIKES
    Y_v = Y_v_raw[:, spike_ok]
    n_w, n_m = Y_v.shape
    print(f"\n{'='*60}\n{REGION}/{cond}: {n_w}w x {n_m}n", flush=True)

    X_sem_sc = StandardScaler().fit_transform(X_pca_all[mask][valid]).astype(np.float32)
    blocks = {"semantic": X_sem_sc}
    for fam, feats in FEATURE_GROUPS.items():
        Xf = clean_cols(ctrl, feats)[mask][valid]
        blocks[fam] = StandardScaler().fit_transform(Xf).astype(np.float32)

    ll_null = np.zeros(n_m, np.float64)
    ll_sat  = np.zeros(n_m, np.float64)
    for tr_m, te_m in block_splits(n_w, N_OUTER):
        mean_tr = Y_v[tr_m].mean(0).clip(1e-10)
        ll_null += (Y_v[te_m] * np.log(mean_tr) - mean_tr).sum(0)
        Y_te_np = Y_v[te_m]
        ll_sat_fold = (Y_te_np * np.log(Y_te_np.clip(1e-10)) - Y_te_np).sum(0)
        ll_sat += np.where(Y_te_np.sum(0) > 0, ll_sat_fold, 0.0)
    d_null = ll_sat - ll_null

    rng = np.random.default_rng(SEED)
    for block_name, X_sc in blocks.items():
        t0 = time.time()
        r2_real, p_vals, sig, tau = block_significance(X_sc, Y_v, ll_null, d_null, n_w, n_m, rng)
        print(f"  {block_name:10s}  tau={tau:5.0f}w  median_R2={np.nanmedian(r2_real):.4f}  "
              f"sig={sig.sum()}/{n_m} ({100*sig.sum()/n_m:.1f}%)  ({time.time()-t0:.0f}s)", flush=True)
        results.append({
            "condition": cond, "block": block_name, "n_neurons": n_m,
            "pct_sig": 100*sig.sum()/n_m, "median_r2": np.nanmedian(r2_real),
        })

print("\n=== SUMMARY: % neurons significantly encoding each block alone ===")
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

out_path = "/scratch/aniluchavez/ConvoDATAS/VPResults/block_significance_prototype_PTYEU_hippo.pkl"
with open(out_path, "wb") as f:
    pickle.dump(df_results, f)
print(f"\nSaved -> {out_path}")
