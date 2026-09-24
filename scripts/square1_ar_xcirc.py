"""
Square-1, AR-control + circular-shift variant — the full dissociation of
semantic autocorrelation from neural autocorrelation, per the user's explicit
three-model design:

  1. Semantic model   : Y_i ~ X_i                       (real, word-aligned)
  2. Shifted-semantic : Y_i ~ X_{i+k}  (k > tau)         (preserves X's own
                         autocorrelation/topic structure, breaks true
                         word-to-spike alignment — implemented as circular
                         shift, same mechanism as xcirc elsewhere in this repo)
  3. Neural-history   : Y_i ~ Y_{i-1}, Y_{i-2}, Y_{i-3}  (controls for the
                         neuron's own slow rate drift)

Combined model: Y_i ~ Y_history + X_i  (or + X_{i+k} for the shifted version)

Key test:
  dLLH_real    = LLH(history + X_i)     - LLH(history)
  dLLH_shifted = LLH(history + X_{i+k}) - LLH(history)   (k drawn repeatedly)
  significant if dLLH_real sits outside the dLLH_shifted null distribution

This supersedes square1_ar_control.py, which used full random shuffle
(xshuffle) + random-fold CV — both now known to leak/inflate via
autocorrelation. Here: block CV (contiguous temporal folds) + circular shift
(preserves X's autocorrelation exactly, only breaks true alignment).

Usage: python3 -u scripts/square1_ar_xcirc.py
"""
import os, time, warnings
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"dev={DEVICE}", flush=True)

PATIENT_ID  = "PTYEU_task147"
PATIENT     = "ptYEU_task147"
LAYER       = 36
N_COMPONENTS = 100
REGION      = "hippocampus"
N_LAGS      = 3
N_OUTER     = 5
N_INNER     = 3
N_PERM      = 50
SEED        = 0
ALPHAS      = np.logspace(-1, 3, 10)
MIN_SPIKES  = 5
ETA_CLIP    = 20.0
LBFGS_ITER  = 20
FDR_ALPHA   = 0.05

EMBED_DIR  = "/scratch/aniluchavez/ConvoDATAS/EmbedCache"
SPIKE_ROOT = "/scratch/aniluchavez/ConvoDATAS/SpikeWindows"

# ── DATA LOADING (same as square1_ar_control.py) ──────────────────────────────

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
                assign[i] = col; break
    mask_self  = assign == "Speaker1"
    mask_other = np.array([(a is not None and a != "Speaker1") for a in assign], dtype=bool)
    return assign, mask_self, mask_other, dir_membership


def load_word_dur_ms(spike_dir):
    cands = [f for f in os.listdir(spike_dir) if f.endswith("_with_regress_dur.xlsx")]
    df = pd.read_excel(os.path.join(spike_dir, cands[0]))
    col = "word_dur" if "word_dur" in df.columns else "Duration"
    return df[col].values.astype(np.float64)


def load_spike_matrix(spike_dir, speaker, region):
    spk_dir = os.path.join(spike_dir, speaker)
    if not os.path.isdir(spk_dir):
        return None
    cands = [f for f in os.listdir(spk_dir)
             if f.lower().startswith(region.lower()) and f.endswith(".npy")]
    return np.load(os.path.join(spk_dir, cands[0])) if cands else None


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


@torch.no_grad()
def gpu_pca(X_np, n_components):
    X = torch.tensor(X_np, device=DEVICE, dtype=torch.float32)
    X = X - X.mean(0, keepdim=True)
    U, S, V = torch.linalg.svd(X, full_matrices=False)
    return (X @ V[:n_components].T).cpu().numpy()


def find_spike_dir():
    d = os.path.join(SPIKE_ROOT, f"output_{PATIENT}_english_only_worddur")
    return d if os.path.isdir(d) else None


def compute_tau_embed(X_raw, thresh=0.05):
    """Autocorrelation decay length of X's own PC1 — the minimum safe circular-
    shift lag, so the shifted version still breaks the true word alignment."""
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

# ── MODEL FIT: neuron is the batch dim (each neuron has its own X) ───────────

def gpu_fit_batched_by_neuron(X_t, Y_t, alpha, offset_t=None):
    n_m, n, p = X_t.shape
    n_eff = float(Y_t.sum().clamp(min=1).item())
    Xi = torch.cat([torch.ones(n_m, n, 1, device=DEVICE), X_t], dim=-1)
    beta = torch.zeros(n_m, 1, p + 1, device=DEVICE, requires_grad=True)
    opt = torch.optim.LBFGS([beta], max_iter=LBFGS_ITER, line_search_fn='strong_wolfe')
    a = alpha.to(DEVICE).view(n_m, 1, 1) if isinstance(alpha, torch.Tensor) else float(alpha)

    def closure():
        opt.zero_grad()
        eta = torch.bmm(Xi, beta.transpose(1, 2))
        if offset_t is not None:
            eta = eta + offset_t
        eta = torch.clamp(eta, -ETA_CLIP, ETA_CLIP)
        loss = ((torch.exp(eta) - Y_t * eta).sum()
                + (a * beta[:, :, 1:].pow(2)).sum()) / n_eff
        loss.backward()
        return loss

    opt.step(closure)
    b = beta.detach()
    return b[:, :, 1:], b[:, :, :1]


def block_splits(n, k):
    """Contiguous temporal folds — block CV, per today's CV-leakage finding."""
    b = n // k
    for i in range(k):
        s, e = i * b, (i * b + b if i < k - 1 else n)
        mask = np.zeros(n, bool); mask[s:e] = True
        yield ~mask, mask


def fit_model_ll_perneuron(X_shared_sc, ar_feat, Y_v, offset):
    """Same as square1_ar_control.py but with block (contiguous) CV folds."""
    n_w, n_m = Y_v.shape
    p_shared = 0 if X_shared_sc is None else X_shared_sc.shape[1]
    p = p_shared + N_LAGS
    ll = np.zeros(n_m, np.float64)

    def build_X(idx):
        ar_part = ar_feat[idx].transpose(1, 0, 2)
        if X_shared_sc is None:
            return ar_part.astype(np.float32)
        sel = X_shared_sc[idx]
        shared_part = np.broadcast_to(sel[None], (n_m, sel.shape[0], p_shared))
        return np.concatenate([shared_part.astype(np.float32), ar_part.astype(np.float32)], axis=-1)

    for tr_m, te_m in block_splits(n_w, N_OUTER):
        n_tr = int(tr_m.sum())
        Y_tr = Y_v[tr_m]; Y_te = Y_v[te_m]
        Y_tr_t = torch.tensor(Y_tr.T[:, :, None], device=DEVICE)
        Y_te_t = torch.tensor(Y_te.T[:, :, None], device=DEVICE)
        X_tr = build_X(tr_m)
        off_tr = offset[tr_m] if offset is not None else None
        off_te_t = (torch.tensor(offset[te_m], device=DEVICE).view(1, -1, 1).expand(n_m, -1, -1)
                    if offset is not None else None)

        inner_ll = np.zeros((len(ALPHAS), n_m), np.float64)
        for itr_m, iva_m in block_splits(n_tr, N_INNER):
            X_itr_t = torch.tensor(X_tr[:, itr_m, :], device=DEVICE)
            Y_itr_t = Y_tr_t[:, itr_m, :]
            off_itr_t = (torch.tensor(off_tr[itr_m], device=DEVICE).view(1, -1, 1).expand(n_m, -1, -1)
                         if off_tr is not None else None)
            for ai, alpha in enumerate(ALPHAS):
                bf, bi = gpu_fit_batched_by_neuron(X_itr_t, Y_itr_t, alpha, offset_t=off_itr_t)
                with torch.no_grad():
                    X_iva_t = torch.tensor(X_tr[:, iva_m, :], device=DEVICE)
                    Y_iva_t = Y_tr_t[:, iva_m, :]
                    eta_iva = torch.bmm(X_iva_t, bf.transpose(1, 2)) + bi.transpose(1, 2)
                    if off_tr is not None:
                        off_iva_t = torch.tensor(off_tr[iva_m], device=DEVICE).view(1, -1, 1).expand(n_m, -1, -1)
                        eta_iva = eta_iva + off_iva_t
                    eta_iva = torch.clamp(eta_iva, -ETA_CLIP, ETA_CLIP)
                    inner_ll[ai] += (Y_iva_t * eta_iva - torch.exp(eta_iva)).sum((1, 2)).cpu().numpy()
        best_ai = np.argmax(inner_ll, axis=0)
        alpha_vec = torch.tensor(ALPHAS[best_ai], dtype=torch.float32)

        X_tr_t = torch.tensor(X_tr, device=DEVICE)
        off_tr_t = (torch.tensor(off_tr, device=DEVICE).view(1, -1, 1).expand(n_m, -1, -1)
                    if off_tr is not None else None)
        bf, bi = gpu_fit_batched_by_neuron(X_tr_t, Y_tr_t, alpha_vec, offset_t=off_tr_t)

        X_te = build_X(te_m)
        X_te_t = torch.tensor(X_te, device=DEVICE)
        with torch.no_grad():
            eta_te = torch.bmm(X_te_t, bf.transpose(1, 2)) + bi.transpose(1, 2)
            if off_te_t is not None:
                eta_te = eta_te + off_te_t
            eta_te = torch.clamp(eta_te, -ETA_CLIP, ETA_CLIP)
            ll += (Y_te_t * eta_te - torch.exp(eta_te)).sum((1, 2)).cpu().numpy()

    return ll


def fdr_bh(pvals):
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.minimum(1.0, pvals[order] * n / np.arange(1, n + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n); out[order] = adj
    return out, out < FDR_ALPHA


def build_ar_features(Y_v):
    n_w, n_m = Y_v.shape
    feat = np.zeros((n_w, n_m, N_LAGS), dtype=np.float64)
    for lag in range(1, N_LAGS + 1):
        feat[lag:, :, lag - 1] = Y_v[:-lag, :]
    feat = feat[N_LAGS:]
    mu = feat.mean(0, keepdims=True); sd = feat.std(0, keepdims=True).clip(1e-6, None)
    feat = (feat - mu) / sd
    return feat.astype(np.float32)

# ── MAIN ──────────────────────────────────────────────────────────────────────

spike_dir = find_spike_dir()
assign, mask_self, mask_other, dir_membership = load_speaker_assignment(spike_dir)
all_word_dur = load_word_dur_ms(spike_dir)
raw_layer = np.load(os.path.join(EMBED_DIR, f"{PATIENT_ID}_gpt2-xl_ctx200_word_emb_layers.npy"),
                     mmap_mode='r')[LAYER].astype(np.float32)
X_pca_full = gpu_pca(raw_layer, N_COMPONENTS)

results_summary = []
rng = np.random.default_rng(SEED)

for cond, mask in [("self", mask_self), ("other", mask_other)]:
    Y_mat = load_condition_ordered(spike_dir, cond, REGION, assign, dir_membership)
    valid = ~np.isnan(Y_mat).any(axis=1)
    Y_v_full = Y_mat[valid].astype(np.float32)
    spike_ok = Y_v_full.sum(0) >= MIN_SPIKES
    Y_v_full = Y_v_full[:, spike_ok]
    n_w_full, n_m = Y_v_full.shape

    dur_v_full = all_word_dur[mask][valid].astype(np.float64).clip(1e-3, None)
    X_sem_full = StandardScaler().fit_transform(X_pca_full[mask][valid]).astype(np.float32)

    ar_feat = build_ar_features(Y_v_full)
    Y_v = Y_v_full[N_LAGS:]
    dur_v = dur_v_full[N_LAGS:]
    X_sem_sc = X_sem_full[N_LAGS:]
    offset_v = np.log(dur_v).astype(np.float32)
    n_w = Y_v.shape[0]

    tau = compute_tau_embed(X_sem_sc)
    min_lag = max(int(np.ceil(tau)), n_w // (N_OUTER * 2))
    min_lag = min(min_lag, n_w // 2 - 1)
    print(f"\n{REGION}/{cond}: {n_w}w x {n_m}n (after dropping {N_LAGS} lag rows)  "
          f"tau={tau:.0f}w  min_lag={min_lag}w", flush=True)

    t0 = time.time()
    ll_ar = fit_model_ll_perneuron(None, ar_feat, Y_v, offset_v)
    ll_full = fit_model_ll_perneuron(X_sem_sc, ar_feat, Y_v, offset_v)
    print(f"  real fits done {time.time()-t0:.1f}s", flush=True)

    ll_null = np.zeros(n_m, np.float64)
    ll_sat = np.zeros(n_m, np.float64)
    for tr_m, te_m in block_splits(n_w, N_OUTER):
        rate_tr = Y_v[tr_m].sum(0) / dur_v[tr_m].sum().clip(1e-10)
        mu_te = np.outer(dur_v[te_m], rate_tr).clip(1e-10)
        ll_null += (Y_v[te_m] * np.log(mu_te) - mu_te).sum(0)
        Y_te = Y_v[te_m]
        sat = (Y_te * np.log(Y_te.clip(1e-10)) - Y_te).sum(0)
        ll_sat += np.where(Y_te.sum(0) > 0, sat, 0.0)
    d_null = ll_sat - ll_null

    with np.errstate(divide="ignore", invalid="ignore"):
        r2_ar = np.where(d_null > 0.1, (ll_ar - ll_null) / d_null, np.nan)
        r2_full = np.where(d_null > 0.1, (ll_full - ll_null) / d_null, np.nan)
    unique_sem = r2_full - r2_ar

    t1 = time.time()
    perm_usem = np.full((N_PERM, n_m), np.nan)
    for pi in range(N_PERM):
        lag = int(rng.integers(min_lag, n_w - min_lag))
        X_sem_perm = np.roll(X_sem_sc, lag, axis=0)     # shifted embeddings X_{i+k}
        ll_full_p = fit_model_ll_perneuron(X_sem_perm, ar_feat, Y_v, offset_v)
        with np.errstate(divide="ignore", invalid="ignore"):
            r2f_p = np.where(d_null > 0.1, (ll_full_p - ll_null) / d_null, np.nan)
        perm_usem[pi] = r2f_p - r2_ar
        if (pi + 1) % 10 == 0:
            print(f"    perm {pi+1}/{N_PERM}  {time.time()-t1:.0f}s", flush=True)

    p_vals = np.array([
        np.nanmean(perm_usem[:, m] >= unique_sem[m]) if not np.isnan(unique_sem[m]) else 1.0
        for m in range(n_m)
    ])
    p_fdr, sig = fdr_bh(p_vals)
    sig = sig & (unique_sem > 0)

    print(f"  median R²: AR={np.nanmedian(r2_ar):.4f}  full={np.nanmedian(r2_full):.4f}", flush=True)
    print(f"  sig unique_sem|AR (xcirc-shift, block CV): {sig.sum()}/{n_m} ({100*sig.mean():.1f}%)  "
          f"raw p<0.05: {100*np.mean(p_vals<0.05):.1f}%", flush=True)

    results_summary.append({
        "condition": cond, "n_sig": int(sig.sum()), "n_m": n_m,
        "pct_sig": 100*sig.mean(), "pct_raw_p_below_05": 100*np.mean(p_vals < 0.05),
        "median_r2_AR": float(np.nanmedian(r2_ar)), "median_r2_full": float(np.nanmedian(r2_full)),
    })

print("\n" + "="*70)
print("SUMMARY  (semantic vs AR-history control, shifted-X circular null, block CV)")
print("="*70)
summary_df = pd.DataFrame(results_summary)
print(summary_df.to_string(index=False))
out_path = "/scratch/aniluchavez/ConvoDATAS/VPResults/square1_ar_xcirc_PTYEU_hippo.pkl"
summary_df.to_pickle(out_path)
print(f"\nSaved -> {out_path}")
