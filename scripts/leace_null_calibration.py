"""
Decisive check on the shuffle-CV-inflation question.

The LEACE pilot showed shuffle CV + xshuffle null gave 14.5%/64.9% sig
(self/other) on REAL data vs 0.0%/1.3% for block CV + xshuffle and 3.9%/9.2%
for the established nested-deviance baseline. On real data this is
ambiguous: maybe shuffle CV is just more sensitive to genuine slow semantic
drift that's naturally autocorrelated in conversation (the claim being
tested here), or maybe it's leakage. Real data can't disambiguate this.

This script builds a dataset with a KNOWN ground truth: fit the real
controls-only Poisson model -> held-out lambda_hat (zero functional
dependence on semantic embeddings, by construction), then simulate
Y_sim ~ Poisson(lambda_hat). Any "significant" semantic effect found in
Y_sim is mechanically guaranteed to be a false positive, full stop --
there is no real (or "natural autocorrelation") signal it could be
detecting, because Y_sim's generative process never saw X_sem.

Runs the LEACE-erased single-model fit under shuffle CV vs block CV (both
paired with xshuffle null) on Y_sim, for hippocampus self+other,
PTYEU_task147.

Usage
-----
  cd /scratch/aniluchavez/hippocampal-speaker-semantics
  python3 -u scripts/leace_null_calibration.py
"""

import os, time, pickle, warnings
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LAYER, N_COMPONENTS = 36, 100
ALPHAS = np.logspace(-2, 4, 20)
N_OUTER, N_INNER = 5, 3
FDR_ALPHA, MIN_SPIKES, ETA_CLIP, LBFGS_ITER = 0.05, 5, 20.0, 20
N_PERM = 50
SEED = 0
EPS = 1e-6

FEATURE_GROUPS = {
    "lexical":   ["log_word_freq", "word_length", "local_count", "local_rate", "surprisal"],
    "syntactic": ["dep_depth", "dep_children", "sent_position", "serial_position"],
    "acoustic":  ["f0_mean", "f0_std", "rms_mean", "spectral_flux", "speaking_rate"],
}
CONTROL_FEATURES = sum(FEATURE_GROUPS.values(), [])

EMBED_DIR   = "/scratch/aniluchavez/ConvoDATAS/EmbedCache"
CONTROL_DIR = "/scratch/aniluchavez/ConvoDATAS/ControlFeatures"
SURPRISAL_DIR = "/scratch/aniluchavez/ConvoDATAS/Surprisal"
SPIKE_ROOT  = "/scratch/aniluchavez/ConvoDATAS/SpikeWindows"

PATIENT_ID = "PTYEU_task147"
PATIENT    = "ptYEU_task147"
REGION = "hippocampus"
REGION_RANGES = {"hippocampus": [(1, 16), (25, 40)]}

# ── LEACE ─────────────────────────────────────────────────────────────────────

def leace_fit(X, Z):
    mu_x, mu_z = X.mean(0), Z.mean(0)
    Xc, Zc = X - mu_x, Z - mu_z
    n = X.shape[0]
    sigma_xx = (Xc.T @ Xc) / (n - 1)
    sigma_xz = (Xc.T @ Zc) / (n - 1)
    eigval, eigvec = np.linalg.eigh(sigma_xx)
    eps = 1e-6 * eigval.max()
    inv_sqrt = np.where(eigval > eps, eigval ** -0.5, 0.0)
    fwd_sqrt = np.where(eigval > eps, eigval ** 0.5, 0.0)
    W = eigvec @ np.diag(inv_sqrt) @ eigvec.T
    W_pinv = eigvec @ np.diag(fwd_sqrt) @ eigvec.T
    sigma_xz_tilde = W @ sigma_xz
    U, S, _ = np.linalg.svd(sigma_xz_tilde, full_matrices=False)
    tol = 1e-8 * S.max() if S.size else 0.0
    r = int((S > tol).sum())
    Ur = U[:, :r]
    P = Ur @ Ur.T
    eraser = W_pinv @ (np.eye(X.shape[1]) - P) @ W

    def erase(X_new):
        return (X_new - mu_x) @ eraser.T + mu_x
    return erase

# ── GPU GLM utilities ─────────────────────────────────────────────────────────

@torch.no_grad()
def gpu_pca(X_np, n_components):
    X = torch.tensor(X_np, dtype=torch.float32, device=DEVICE)
    X -= X.mean(0)
    k = min(n_components + 10, min(X.shape))
    Y = X @ torch.randn(X.shape[1], k, device=DEVICE)
    for _ in range(2):
        Y = X @ (X.T @ Y)
    Q, _ = torch.linalg.qr(Y)
    _, _, Vt = torch.linalg.svd(Q.T @ X, full_matrices=False)
    return (X @ Vt[:n_components].T).cpu().numpy().astype(np.float32)


def gpu_fit(X_t, Y_t, alpha):
    k, n, p = X_t.shape
    m = Y_t.shape[2]
    n_eff = float(Y_t.sum().clamp(min=1).item())
    Xi = torch.cat([torch.ones(k, n, 1, device=DEVICE), X_t], dim=-1)
    beta = torch.zeros(k, m, p + 1, device=DEVICE, requires_grad=True)
    opt = torch.optim.LBFGS([beta], max_iter=LBFGS_ITER, line_search_fn='strong_wolfe')
    a = alpha.to(DEVICE).view(k, 1, 1) if isinstance(alpha, torch.Tensor) else float(alpha)

    def closure():
        opt.zero_grad()
        eta = torch.clamp(torch.bmm(Xi, beta.transpose(1, 2)), -ETA_CLIP, ETA_CLIP)
        loss = ((torch.exp(eta) - Y_t * eta).sum()
                + (a * beta[:, :, 1:].pow(2)).sum()) / n_eff
        loss.backward()
        return loss

    opt.step(closure)
    b = beta.detach()
    return b[:, :, 1:], b[:, :, :1]


def block_splits(n, k, mode, rng_seed):
    if mode == "block":
        b = n // k
        for i in range(k):
            s, e = i * b, (i * b + b if i < k - 1 else n)
            mask = np.zeros(n, bool); mask[s:e] = True
            yield ~mask, mask
    else:
        kf = KFold(n_splits=k, shuffle=True, random_state=rng_seed)
        for tr_idx, te_idx in kf.split(np.arange(n)):
            tr_m = np.zeros(n, bool); tr_m[tr_idx] = True
            te_m = np.zeros(n, bool); te_m[te_idx] = True
            yield tr_m, te_m


def fdr_bh(pvals):
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.minimum(1.0, pvals[order] * n / np.arange(1, n + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n); out[order] = adj
    return out, out < FDR_ALPHA


def fit_one_model_ll(X_sc, Y_v, mode, cv_rng):
    n_w, n_m = Y_v.shape
    N_A = len(ALPHAS)
    alpha_t = torch.tensor(ALPHAS, dtype=torch.float32)
    p = X_sc.shape[1]
    ll = np.zeros(n_m, np.float64)
    for tr_m, te_m in block_splits(n_w, N_OUTER, mode, int(cv_rng.integers(0, 2**31))):
        n_tr = int(tr_m.sum())
        Y_tr_t = torch.tensor(Y_v[tr_m], device=DEVICE)
        Y_te_t = torch.tensor(Y_v[te_m], device=DEVICE)
        X_tr = X_sc[tr_m]
        inner_ll = np.zeros((N_A, n_m), np.float64)
        for itr_m, iva_m in block_splits(n_tr, N_INNER, mode, int(cv_rng.integers(0, 2**31))):
            Xii = torch.tensor(X_tr[itr_m], device=DEVICE).unsqueeze(0).expand(N_A, -1, -1).contiguous()
            Yii = Y_tr_t[itr_m].unsqueeze(0).expand(N_A, -1, -1).contiguous()
            bf, bi = gpu_fit(Xii, Yii, alpha_t)
            with torch.no_grad():
                Xiv = torch.tensor(X_tr[iva_m], device=DEVICE).unsqueeze(0).expand(N_A, -1, -1).contiguous()
                Yiv = Y_tr_t[iva_m].unsqueeze(0).expand(N_A, -1, -1).contiguous()
                eta_iv = torch.clamp(torch.bmm(Xiv, bf.transpose(1, 2)) + bi.transpose(1, 2),
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
            eta_te_t = torch.clamp(torch.bmm(X_te_t, beta_f.transpose(1, 2)) + bias_f.transpose(1, 2),
                                    -ETA_CLIP, ETA_CLIP)
            ll_fold = (Y_te_t * eta_te_t[0] - torch.exp(eta_te_t[0])).sum(0).cpu().numpy()
        ll += ll_fold
        del X_tr_t, X_te_t, beta_f, bias_f, eta_te_t, Y_tr_t, Y_te_t
        torch.cuda.empty_cache()
    return ll


def fit_poisson_held_out_lambda(X_sc, Y_v, mode, cv_rng):
    n_w, n_m = Y_v.shape
    N_A = len(ALPHAS)
    alpha_t = torch.tensor(ALPHAS, dtype=torch.float32)
    p = X_sc.shape[1]
    lam_hat = np.zeros((n_w, n_m), np.float64)
    for tr_m, te_m in block_splits(n_w, N_OUTER, mode, int(cv_rng.integers(0, 2**31))):
        n_tr = int(tr_m.sum())
        Y_tr_t = torch.tensor(Y_v[tr_m], device=DEVICE)
        X_tr = X_sc[tr_m]
        inner_ll = np.zeros((N_A, n_m), np.float64)
        for itr_m, iva_m in block_splits(n_tr, N_INNER, mode, int(cv_rng.integers(0, 2**31))):
            Xii = torch.tensor(X_tr[itr_m], device=DEVICE).unsqueeze(0).expand(N_A, -1, -1).contiguous()
            Yii = Y_tr_t[itr_m].unsqueeze(0).expand(N_A, -1, -1).contiguous()
            bf, bi = gpu_fit(Xii, Yii, alpha_t)
            with torch.no_grad():
                Xiv = torch.tensor(X_tr[iva_m], device=DEVICE).unsqueeze(0).expand(N_A, -1, -1).contiguous()
                Yiv = Y_tr_t[iva_m].unsqueeze(0).expand(N_A, -1, -1).contiguous()
                eta_iv = torch.clamp(torch.bmm(Xiv, bf.transpose(1, 2)) + bi.transpose(1, 2),
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
            eta_te_t = torch.clamp(torch.bmm(X_te_t, beta_f.transpose(1, 2)) + bias_f.transpose(1, 2),
                                    -ETA_CLIP, ETA_CLIP)
            lam_hat[te_m] = torch.exp(eta_te_t)[0].cpu().numpy()
        del X_tr_t, X_te_t, beta_f, bias_f, eta_te_t, Y_tr_t
        torch.cuda.empty_cache()
    return lam_hat

# ── data loading ──────────────────────────────────────────────────────────────

def find_spike_dir(patient):
    d = os.path.join(SPIKE_ROOT, f"output_{patient}_english_only_worddur")
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
    cols = [c for c in CONTROL_FEATURES if c in ctrl.columns]
    X = ctrl[cols].values.astype(float)
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    return X, cols

# ── one region/condition, one CV mode ────────────────────────────────────────

def run_one(cond, cv_mode, X_pca, X_ctrl_raw, all_word_dur, spike_dir,
            spk_assignment, mask_self, mask_other, dir_membership):
    mask = mask_self if cond == "self" else mask_other
    Y_mat = load_condition_ordered(spike_dir, cond, REGION, spk_assignment, dir_membership)
    valid = ~np.isnan(Y_mat).any(axis=1)
    Y_v_raw = Y_mat[valid].astype(np.float32)
    spike_ok = Y_v_raw.sum(0) >= MIN_SPIKES
    Y_v_real = Y_v_raw[:, spike_ok]
    n_w, n_m = Y_v_real.shape

    X_ctrl_sc = StandardScaler().fit_transform(X_ctrl_raw[mask][valid]).astype(np.float32)
    X_sem_sc = StandardScaler().fit_transform(X_pca[mask][valid]).astype(np.float32)

    # ── known-null simulated Y: zero true semantic effect by construction ──
    cv_rng = np.random.default_rng(SEED)
    lam_ctrl_real = fit_poisson_held_out_lambda(X_ctrl_sc, Y_v_real, "block", cv_rng).clip(EPS)
    rng_sim = np.random.default_rng(SEED)
    Y_v = rng_sim.poisson(lam_ctrl_real).astype(np.float32)
    print(f"  {cond}: {n_w}w x {n_m}n  Y_sim mean={Y_v.mean():.3f} vs real={Y_v_real.mean():.3f}",
          flush=True)

    # ── LEACE-erase, fit, xshuffle-null under the given CV mode ─────────────
    cv_rng2 = np.random.default_rng(SEED + 1)
    erase_fn = leace_fit(X_sem_sc.astype(np.float64), X_ctrl_sc.astype(np.float64))
    X_sem_erased = erase_fn(X_sem_sc.astype(np.float64)).astype(np.float32)

    ll_null = np.zeros(n_m, np.float64); ll_sat = np.zeros(n_m, np.float64)
    for tr_m, te_m in block_splits(n_w, N_OUTER, "block", 0):
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
    r2_real = r2_from_ll(fit_one_model_ll(X_sem_erased, Y_v, cv_mode, cv_rng2))

    rng = np.random.default_rng(SEED)
    perm_r2 = np.full((N_PERM, n_m), np.nan)
    for pi in range(N_PERM):
        X_sem_perm = X_sem_sc[rng.permutation(n_w)]
        erase_fn_p = leace_fit(X_sem_perm.astype(np.float64), X_ctrl_sc.astype(np.float64))
        X_perm_erased = erase_fn_p(X_sem_perm.astype(np.float64)).astype(np.float32)
        perm_r2[pi] = r2_from_ll(fit_one_model_ll(X_perm_erased, Y_v, cv_mode, cv_rng2))
        if (pi + 1) % 10 == 0:
            print(f"    [{cv_mode}] perm {pi+1}/{N_PERM}  {time.time()-t0:.0f}s", flush=True)

    p_vals = np.array([np.nanmean(perm_r2[:, m] >= r2_real[m]) if not np.isnan(r2_real[m]) else 1.0
                        for m in range(n_m)])
    _, sig = fdr_bh(p_vals)
    sig = sig & (r2_real > 0)
    pv_valid = p_vals[~np.isnan(p_vals)]
    print(f"  [{cv_mode}] on Y_sim (zero true effect): {sig.sum()}/{n_m} sig "
          f"({100*sig.sum()/n_m:.1f}%)  raw p mean={pv_valid.mean():.3f} (expect ~0.5)  "
          f"%raw_p<0.05={100*np.mean(pv_valid<0.05):.1f}% (expect ~5%)  ({time.time()-t0:.0f}s)",
          flush=True)
    return {"cond": cond, "cv_mode": cv_mode, "n_m": n_m, "n_sig": int(sig.sum()),
            "pct_sig": 100 * sig.sum() / n_m, "p_mean": float(pv_valid.mean()),
            "pct_p_below_05": float(100 * np.mean(pv_valid < 0.05))}

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    npy_path = os.path.join(EMBED_DIR, f"{PATIENT_ID}_gpt2-xl_ctx200_word_emb_layers.npy")
    spike_dir = find_spike_dir(PATIENT)
    spk_assignment, mask_self, mask_other, dir_membership = load_speaker_assignment(spike_dir)
    X_ctrl_raw, _ = load_control_features(PATIENT_ID)

    raw_layer = np.load(npy_path, mmap_mode='r')[LAYER].astype(np.float32)
    X_pca = gpu_pca(raw_layer, N_COMPONENTS)
    del raw_layer

    results = []
    for cond in ["self", "other"]:
        for cv_mode in ["shuffle", "block"]:
            print(f"\n=== {cond} / CV={cv_mode} ===", flush=True)
            r = run_one(cond, cv_mode, X_pca, X_ctrl_raw, None, spike_dir,
                        spk_assignment, mask_self, mask_other, dir_membership)
            results.append(r)
            torch.cuda.empty_cache()

    print("\n\n=== SUMMARY: LEACE-erased model on Y_sim (ZERO true semantic effect) ===")
    print("If a CV/null combo is calibrated: %sig should be near 0-5%, raw p mean ~0.5, %raw_p<0.05 ~5%.")
    for r in results:
        print(f"  {r['cond']:5s} CV={r['cv_mode']:8s}  sig {r['n_sig']:3d}/{r['n_m']} ({r['pct_sig']:.1f}%)  "
              f"raw_p_mean={r['p_mean']:.3f}  %raw_p<0.05={r['pct_p_below_05']:.1f}%")

    out_path = "/scratch/aniluchavez/ConvoDATAS/VPResults/leace_null_calibration_PTYEU_hippo.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(pd.DataFrame(results), f)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
