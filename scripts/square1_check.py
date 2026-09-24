"""
Square-1 check — minimal, transparent re-derivation of the semantic-vs-controls
result, stripped down to the fewest moving parts possible, to rebuild trust
after today's chain of bugs (duration offset, null-type miscalibration).

Kept:
  - Poisson ridge GLM with the (now-validated) word-duration log-offset
  - Nested CV (inner CV for per-neuron ridge alpha selection, outer CV for
    held-out R²) — NOT dropped, but run in two fold-construction variants:
      "shuffle" : both inner and outer splits are random (sklearn KFold,
                  shuffle=True) — ignores temporal order entirely
      "block"   : both inner and outer splits are contiguous temporal blocks
                  (no shuffling) — the convention used elsewhere in this repo
  - Three models only: semantic / controls / full (no per-family breakdown)

Dropped (deliberately, vs. the full pipeline):
  - circular-shift (xcirc) and block-shuffle (xblock) nulls — replaced with
    plain global X-shuffle (xshuffle): randomly permute the semantic PCA
    matrix's rows, refit, repeat. Simplest possible permutation test.
  - per-family (lexical/syntactic/acoustic) decomposition
  - per-neuron alpha grid search batching tricks — still nested CV, but a
    smaller alpha grid and explicit, readable loop (no implicit batching)

Usage: python3 -u scripts/square1_check.py
"""
import os, time, warnings
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"dev={DEVICE}", flush=True)

# ── CONFIG ────────────────────────────────────────────────────────────────────
PATIENT_ID  = "PTYEU_task147"
PATIENT     = "ptYEU_task147"
LAYER       = 36
N_COMPONENTS = 100
REGION      = "hippocampus"
CONDITIONS  = ["self", "other"]
N_OUTER     = 5
N_INNER     = 3
N_PERM      = 50
SEED        = 0
ALPHAS      = np.logspace(-1, 3, 10)   # small, readable grid
MIN_SPIKES  = 5
ETA_CLIP    = 20.0
LBFGS_ITER  = 20
FDR_ALPHA   = 0.05

FEATURE_GROUPS = {
    "lexical":   ["log_word_freq", "word_length", "local_count", "local_rate", "surprisal"],
    "syntactic": ["dep_depth", "dep_children", "sent_position", "serial_position"],
    "acoustic":  ["f0_mean", "f0_std", "rms_mean", "spectral_flux", "speaking_rate"],
}
CONTROL_FEATURES = sum(FEATURE_GROUPS.values(), [])

EMBED_DIR     = "/scratch/aniluchavez/ConvoDATAS/EmbedCache"
CONTROL_DIR   = "/scratch/aniluchavez/ConvoDATAS/ControlFeatures"
SURPRISAL_DIR = "/scratch/aniluchavez/ConvoDATAS/Surprisal"
SPIKE_ROOT    = "/scratch/aniluchavez/ConvoDATAS/SpikeWindows"

# ── DATA LOADING (copied from variance_partitioning.py) ──────────────────────

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


def load_control_features(patient_ID):
    ctrl = pd.read_csv(os.path.join(CONTROL_DIR, f"{patient_ID}_control_features.csv"))
    surp_path = os.path.join(SURPRISAL_DIR, f"{patient_ID}_surprisal.csv")
    if os.path.exists(surp_path) and "surprisal" not in ctrl.columns:
        surp = pd.read_csv(surp_path)
        if len(surp) == len(ctrl):
            ctrl["surprisal"] = surp["surprisal"].values
    cols = [c for c in CONTROL_FEATURES if c in ctrl.columns]
    X = ctrl[cols].values.astype(float)
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    return X, cols


@torch.no_grad()
def gpu_pca(X_np, n_components):
    X = torch.tensor(X_np, device=DEVICE, dtype=torch.float32)
    X = X - X.mean(0, keepdim=True)
    U, S, V = torch.linalg.svd(X, full_matrices=False)
    return (X @ V[:n_components].T).cpu().numpy()


def find_spike_dir():
    d = os.path.join(SPIKE_ROOT, f"output_{PATIENT}_english_only_worddur")
    return d if os.path.isdir(d) else None

# ── MODEL FIT ─────────────────────────────────────────────────────────────────

def gpu_fit(X_t, Y_t, alpha, offset_t=None):
    """k batch items x n samples x p features -> beta (k,m,p), bias (k,m,1)."""
    k, n, p = X_t.shape
    m = Y_t.shape[2]
    n_eff = float(Y_t.sum().clamp(min=1).item())
    Xi = torch.cat([torch.ones(k, n, 1, device=DEVICE), X_t], dim=-1)
    beta = torch.zeros(k, m, p + 1, device=DEVICE, requires_grad=True)
    opt = torch.optim.LBFGS([beta], max_iter=LBFGS_ITER, line_search_fn='strong_wolfe')
    a = alpha.to(DEVICE).view(k, 1, 1) if isinstance(alpha, torch.Tensor) else float(alpha)

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


def make_folds(n, k, mode, rng):
    """mode='block' -> contiguous temporal blocks; mode='shuffle' -> random KFold."""
    if mode == "block":
        b = n // k
        for i in range(k):
            s, e = i * b, (i * b + b if i < k - 1 else n)
            mask = np.zeros(n, bool); mask[s:e] = True
            yield ~mask, mask
    else:
        kf = KFold(n_splits=k, shuffle=True, random_state=rng.integers(0, 2**31))
        for tr_idx, te_idx in kf.split(np.arange(n)):
            tr_m = np.zeros(n, bool); tr_m[tr_idx] = True
            te_m = np.zeros(n, bool); te_m[te_idx] = True
            yield tr_m, te_m


def fit_model_ll(X_sc, Y_v, offset, cv_mode, rng):
    """Nested CV (inner: alpha selection, outer: held-out LL). Returns ll (n_m,)."""
    n_w, n_m = Y_v.shape
    p = X_sc.shape[1]
    ll = np.zeros(n_m, np.float64)

    for tr_m, te_m in make_folds(n_w, N_OUTER, cv_mode, rng):
        n_tr = int(tr_m.sum())
        Y_tr_t = torch.tensor(Y_v[tr_m], device=DEVICE)
        Y_te_t = torch.tensor(Y_v[te_m], device=DEVICE)
        X_tr = X_sc[tr_m]
        off_tr = offset[tr_m] if offset is not None else None
        off_te_t = (torch.tensor(offset[te_m], device=DEVICE).view(1, -1, 1)
                    if offset is not None else None)

        # inner CV: per-neuron alpha selection (small grid, explicit loop)
        inner_ll = np.zeros((len(ALPHAS), n_m), np.float64)
        for itr_m, iva_m in make_folds(n_tr, N_INNER, cv_mode, rng):
            X_itr = torch.tensor(X_tr[itr_m], device=DEVICE).unsqueeze(0)
            Y_itr = Y_tr_t[itr_m].unsqueeze(0)
            off_itr_t = (torch.tensor(off_tr[itr_m], device=DEVICE).view(1, -1, 1)
                         if off_tr is not None else None)
            for ai, alpha in enumerate(ALPHAS):
                bf, bi = gpu_fit(X_itr, Y_itr, alpha, offset_t=off_itr_t)
                with torch.no_grad():
                    X_iva = torch.tensor(X_tr[iva_m], device=DEVICE).unsqueeze(0)
                    Y_iva = Y_tr_t[iva_m].unsqueeze(0)
                    eta_iva = torch.bmm(X_iva, bf.transpose(1, 2)) + bi.transpose(1, 2)
                    if off_tr is not None:
                        off_iva_t = torch.tensor(off_tr[iva_m], device=DEVICE).view(1, -1, 1)
                        eta_iva = eta_iva + off_iva_t
                    eta_iva = torch.clamp(eta_iva, -ETA_CLIP, ETA_CLIP)
                    inner_ll[ai] += (Y_iva * eta_iva - torch.exp(eta_iva)).sum((0, 1)).cpu().numpy()
        best_ai = np.argmax(inner_ll, axis=0)

        # outer fit at each neuron's chosen alpha
        X_tr_t = torch.tensor(X_tr[None], device=DEVICE)
        off_tr_t = (torch.tensor(off_tr, device=DEVICE).view(1, -1, 1)
                    if off_tr is not None else None)
        beta_f = torch.zeros(1, n_m, p, device=DEVICE)
        bias_f = torch.zeros(1, n_m, 1, device=DEVICE)
        for ai in np.unique(best_ai):
            ids = np.where(best_ai == ai)[0]
            bf2, bi2 = gpu_fit(X_tr_t, Y_tr_t[None, :, ids], float(ALPHAS[ai]), offset_t=off_tr_t)
            beta_f[0, ids] = bf2[0]; bias_f[0, ids] = bi2[0]

        X_te_t = torch.tensor(X_sc[te_m][None], device=DEVICE)
        with torch.no_grad():
            eta_te = torch.bmm(X_te_t, beta_f.transpose(1, 2)) + bias_f.transpose(1, 2)
            if off_te_t is not None:
                eta_te = eta_te + off_te_t
            eta_te = torch.clamp(eta_te, -ETA_CLIP, ETA_CLIP)
            ll += (Y_te_t * eta_te[0] - torch.exp(eta_te[0])).sum(0).cpu().numpy()

    return ll


def fdr_bh(pvals):
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.minimum(1.0, pvals[order] * n / np.arange(1, n + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n); out[order] = adj
    return out, out < FDR_ALPHA

# ── MAIN ──────────────────────────────────────────────────────────────────────

spike_dir = find_spike_dir()
assign, mask_self, mask_other, dir_membership = load_speaker_assignment(spike_dir)
all_word_dur = load_word_dur_ms(spike_dir)
X_ctrl_raw, ctrl_cols = load_control_features(PATIENT_ID)
raw_layer = np.load(os.path.join(EMBED_DIR, f"{PATIENT_ID}_gpt2-xl_ctx200_word_emb_layers.npy"),
                     mmap_mode='r')[LAYER].astype(np.float32)
X_pca = gpu_pca(raw_layer, N_COMPONENTS)

results_summary = []

for cv_mode in ["shuffle", "block"]:
    print(f"\n{'='*70}\nCV MODE = {cv_mode}\n{'='*70}", flush=True)
    rng = np.random.default_rng(SEED)

    for cond, mask in [("self", mask_self), ("other", mask_other)]:
        Y_mat = load_condition_ordered(spike_dir, cond, REGION, assign, dir_membership)
        valid = ~np.isnan(Y_mat).any(axis=1)
        Y_v = Y_mat[valid].astype(np.float32)
        spike_ok = Y_v.sum(0) >= MIN_SPIKES
        Y_v = Y_v[:, spike_ok]
        n_w, n_m = Y_v.shape
        print(f"\n{REGION}/{cond}: {n_w}w x {n_m}n", flush=True)

        dur_v = all_word_dur[mask][valid].astype(np.float64).clip(1e-3, None)
        offset_v = np.log(dur_v).astype(np.float32)

        X_ctrl_sc = StandardScaler().fit_transform(X_ctrl_raw[mask][valid]).astype(np.float32)
        X_sem_sc = StandardScaler().fit_transform(X_pca[mask][valid]).astype(np.float32)
        X_full_sc = np.hstack([X_sem_sc, X_ctrl_sc]).astype(np.float32)

        t0 = time.time()
        ll_sem = fit_model_ll(X_sem_sc, Y_v, offset_v, cv_mode, rng)
        ll_ctrl = fit_model_ll(X_ctrl_sc, Y_v, offset_v, cv_mode, rng)
        ll_full = fit_model_ll(X_full_sc, Y_v, offset_v, cv_mode, rng)
        print(f"  real fits done {time.time()-t0:.1f}s", flush=True)

        # null model: rate-based offset-only MLE, via the same fold structure
        ll_null = np.zeros(n_m, np.float64)
        ll_sat = np.zeros(n_m, np.float64)
        for tr_m, te_m in make_folds(n_w, N_OUTER, cv_mode, rng):
            rate_tr = Y_v[tr_m].sum(0) / dur_v[tr_m].sum().clip(1e-10)
            mu_te = np.outer(dur_v[te_m], rate_tr).clip(1e-10)
            ll_null += (Y_v[te_m] * np.log(mu_te) - mu_te).sum(0)
            Y_te = Y_v[te_m]
            sat = (Y_te * np.log(Y_te.clip(1e-10)) - Y_te).sum(0)
            ll_sat += np.where(Y_te.sum(0) > 0, sat, 0.0)
        d_null = ll_sat - ll_null

        with np.errstate(divide="ignore", invalid="ignore"):
            r2_sem = np.where(d_null > 0.1, (ll_sem - ll_null) / d_null, np.nan)
            r2_ctrl = np.where(d_null > 0.1, (ll_ctrl - ll_null) / d_null, np.nan)
            r2_full = np.where(d_null > 0.1, (ll_full - ll_null) / d_null, np.nan)
        unique_sem = r2_full - r2_ctrl

        # significance: plain global X-shuffle (xshuffle) on the FULL model only
        t1 = time.time()
        perm_usem = np.full((N_PERM, n_m), np.nan)
        for pi in range(N_PERM):
            perm_idx = rng.permutation(n_w)
            X_sem_perm = X_sem_sc[perm_idx]
            X_full_perm = np.hstack([X_sem_perm, X_ctrl_sc]).astype(np.float32)
            ll_full_p = fit_model_ll(X_full_perm, Y_v, offset_v, cv_mode, rng)
            with np.errstate(divide="ignore", invalid="ignore"):
                r2f_p = np.where(d_null > 0.1, (ll_full_p - ll_null) / d_null, np.nan)
            perm_usem[pi] = r2f_p - r2_ctrl
            if (pi + 1) % 10 == 0:
                print(f"    perm {pi+1}/{N_PERM}  {time.time()-t1:.0f}s", flush=True)

        p_vals = np.array([
            np.nanmean(perm_usem[:, m] >= unique_sem[m]) if not np.isnan(unique_sem[m]) else 1.0
            for m in range(n_m)
        ])
        p_fdr, sig = fdr_bh(p_vals)
        sig = sig & (unique_sem > 0)

        print(f"  median R²: sem={np.nanmedian(r2_sem):.4f}  ctrl={np.nanmedian(r2_ctrl):.4f}  "
              f"full={np.nanmedian(r2_full):.4f}", flush=True)
        print(f"  sig unique_sem (xshuffle, {cv_mode} CV): {sig.sum()}/{n_m} ({100*sig.mean():.1f}%)  "
              f"raw p<0.05: {100*np.mean(p_vals<0.05):.1f}%", flush=True)

        results_summary.append({
            "cv_mode": cv_mode, "condition": cond,
            "n_sig": int(sig.sum()), "n_m": n_m, "pct_sig": 100*sig.mean(),
            "pct_raw_p_below_05": 100*np.mean(p_vals < 0.05),
            "median_r2_sem": float(np.nanmedian(r2_sem)),
        })

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
summary_df = pd.DataFrame(results_summary)
print(summary_df.to_string(index=False))
out_path = "/scratch/aniluchavez/ConvoDATAS/VPResults/square1_check_PTYEU_hippo.pkl"
summary_df.to_pickle(out_path)
print(f"\nSaved -> {out_path}")
