"""
LEACE pilot — single patient (PTYEU_task147), gpt2-xl L36 pc100, worddur+offset.

Erase lexical+syntactic+acoustic control features from the semantic PCA
embeddings via closed-form LEACE (Belrose et al. 2023), then fit the same
Poisson-ridge GLM machinery as variance_partitioning.py on the *erased*
embeddings alone (no nested controls model needed by construction). Compares
against the already-computed nested-deviance unique_semantic baseline in
VPResults/gpt2-xl_ctx200_worddur_xcirc/pc100/PTYEU_task147_L36_VP.pkl.

CV = contiguous block folds (CV_MODE='block', as in variance_partitioning.py),
null = xshuffle (global row permutation of the semantic embeddings, re-erased
against the real controls before each refit) — block CV paired with the
simpler global-shuffle null instead of circular-shift. Hippocampus only.

Usage
-----
  cd /scratch/aniluchavez/hippocampal-speaker-semantics
  python3 -u scripts/leace_pilot.py
"""

import os, time, pickle, warnings
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── fixed config (mirrors variance_partitioning.py defaults) ─────────────────
LAYER, N_COMPONENTS = 36, 100
ALPHAS = np.logspace(-2, 4, 20)
N_OUTER, N_INNER = 5, 3
FDR_ALPHA, MIN_SPIKES, ETA_CLIP, LBFGS_ITER = 0.05, 5, 20.0, 20
N_PERM = 50
SEED = 0

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
BASELINE_PKL = ("/scratch/aniluchavez/ConvoDATAS/VPResults/"
                "gpt2-xl_ctx200_worddur_xcirc/pc100/PTYEU_task147_L36_VP.pkl")

PATIENT_ID = "PTYEU_task147"
PATIENT    = "ptYEU_task147"
REGION_RANGES = {"hippocampus": [(1, 16), (25, 40)], "ACC": [(17, 24), (41, 48)]}

# ── LEACE (closed-form, Belrose et al. 2023) ─────────────────────────────────

def leace_fit(X, Z):
    """Closed-form least-squares concept eraser. X: (n,d) regressors to clean,
    Z: (n,k) concept matrix to remove (guarantees zero *linear* leakage of Z
    from the returned erase(X), with minimal change to X under the whitened
    norm). Returns (erase_fn, eraser_matrix)."""
    mu_x, mu_z = X.mean(0), Z.mean(0)
    Xc, Zc = X - mu_x, Z - mu_z
    n = X.shape[0]
    sigma_xx = (Xc.T @ Xc) / (n - 1)
    sigma_xz = (Xc.T @ Zc) / (n - 1)

    eigval, eigvec = np.linalg.eigh(sigma_xx)
    eps = 1e-6 * eigval.max()
    inv_sqrt  = np.where(eigval > eps, eigval ** -0.5, 0.0)
    fwd_sqrt  = np.where(eigval > eps, eigval ** 0.5, 0.0)
    W      = eigvec @ np.diag(inv_sqrt) @ eigvec.T   # sigma_xx^{-1/2}
    W_pinv = eigvec @ np.diag(fwd_sqrt) @ eigvec.T   # sigma_xx^{1/2}

    sigma_xz_tilde = W @ sigma_xz
    U, S, _ = np.linalg.svd(sigma_xz_tilde, full_matrices=False)
    tol = 1e-8 * S.max() if S.size else 0.0
    r = int((S > tol).sum())
    Ur = U[:, :r]
    P = Ur @ Ur.T  # projector onto range(sigma_xz_tilde)

    eraser = W_pinv @ (np.eye(X.shape[1]) - P) @ W

    def erase(X_new):
        return (X_new - mu_x) @ eraser.T + mu_x

    return erase, eraser


def max_abs_corr(X, Z):
    """Max |corr| between any column of X and any column of Z — used as the
    sanity-check metric for residual linear leakage after erasure."""
    Xc = (X - X.mean(0)) / (X.std(0) + 1e-12)
    Zc = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    n = X.shape[0]
    corr = (Xc.T @ Zc) / n
    return float(np.abs(corr).max())

# ── GPU GLM utilities (copied from variance_partitioning.py) ────────────────

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


def gpu_fit(X_t, Y_t, alpha, offset_t=None):
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


def block_splits(n, k):
    """Contiguous temporal folds (CV_MODE='block')."""
    b = n // k
    for i in range(k):
        s, e = i * b, (i * b + b if i < k - 1 else n)
        mask = np.zeros(n, bool); mask[s:e] = True
        yield ~mask, mask


def fdr_bh(pvals):
    n = len(pvals)
    if n == 0:
        return np.array([]), np.array([], dtype=bool)
    order = np.argsort(pvals)
    adj = np.minimum(1.0, pvals[order] * n / np.arange(1, n + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n); out[order] = adj
    return out, out < FDR_ALPHA


def fit_one_model_ll(X_sc, Y_v, offset=None):
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
        off_tr_np = offset[tr_m] if offset is not None else None
        off_te_t = (torch.tensor(offset[te_m], device=DEVICE).view(1, -1, 1)
                    if offset is not None else None)

        inner_ll = np.zeros((N_A, n_m), np.float64)
        for itr_m, iva_m in block_splits(n_tr, N_INNER):
            Xii = torch.tensor(X_tr[itr_m], device=DEVICE).unsqueeze(0).expand(N_A, -1, -1).contiguous()
            Yii = Y_tr_t[itr_m].unsqueeze(0).expand(N_A, -1, -1).contiguous()
            off_itr_t = (torch.tensor(off_tr_np[itr_m], device=DEVICE).view(1, -1, 1)
                         if off_tr_np is not None else None)
            bf, bi = gpu_fit(Xii, Yii, alpha_t, offset_t=off_itr_t)
            with torch.no_grad():
                Xiv = torch.tensor(X_tr[iva_m], device=DEVICE).unsqueeze(0).expand(N_A, -1, -1).contiguous()
                Yiv = Y_tr_t[iva_m].unsqueeze(0).expand(N_A, -1, -1).contiguous()
                eta_iv = torch.bmm(Xiv, bf.transpose(1, 2)) + bi.transpose(1, 2)
                if off_tr_np is not None:
                    off_iva_t = torch.tensor(off_tr_np[iva_m], device=DEVICE).view(1, -1, 1)
                    eta_iv = eta_iv + off_iva_t
                eta_iv = torch.clamp(eta_iv, -ETA_CLIP, ETA_CLIP)
                inner_ll += (Yiv * eta_iv - torch.exp(eta_iv)).sum(1).cpu().numpy()
        best_ai = np.argmax(inner_ll, axis=0)

        X_tr_t = torch.tensor(X_tr[None], device=DEVICE)
        off_tr_t = (torch.tensor(off_tr_np, device=DEVICE).view(1, -1, 1)
                    if off_tr_np is not None else None)
        beta_f = torch.zeros(1, n_m, p, device=DEVICE)
        bias_f = torch.zeros(1, n_m, 1, device=DEVICE)
        for ai in np.unique(best_ai):
            ids = np.where(best_ai == ai)[0]
            bf2, bi2 = gpu_fit(X_tr_t, Y_tr_t[None, :, ids], float(ALPHAS[ai]), offset_t=off_tr_t)
            beta_f[0, ids] = bf2[0]; bias_f[0, ids] = bi2[0]

        X_te_t = torch.tensor(X_sc[te_m][None], device=DEVICE)
        with torch.no_grad():
            eta_te_t = torch.bmm(X_te_t, beta_f.transpose(1, 2)) + bias_f.transpose(1, 2)
            if off_te_t is not None:
                eta_te_t = eta_te_t + off_te_t
            eta_te_t = torch.clamp(eta_te_t, -ETA_CLIP, ETA_CLIP)
            ll_fold = (Y_te_t * eta_te_t[0] - torch.exp(eta_te_t[0])).sum(0).cpu().numpy()
        ll += ll_fold
        del X_tr_t, X_te_t, beta_f, bias_f, eta_te_t, Y_tr_t, Y_te_t
        torch.cuda.empty_cache()
    return ll

# ── data loading (copied from variance_partitioning.py) ─────────────────────

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
    if not cands:
        raise FileNotFoundError(f"No _with_regress_dur.xlsx in {spike_dir}")
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


def load_word_dur_ms(spike_dir):
    cands = [f for f in os.listdir(spike_dir) if f.endswith("_with_regress_dur.xlsx")]
    if not cands:
        return None
    df = pd.read_excel(os.path.join(spike_dir, cands[0]))
    col = "word_dur" if "word_dur" in df.columns else "Duration"
    if col not in df.columns:
        return None
    return df[col].values.astype(np.float64)


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

# ── pipeline for one region/condition ────────────────────────────────────────

def run_region_condition(region, cond, X_pca, X_ctrl_raw, ctrl_cols, all_word_dur,
                          spike_dir, spk_assignment, mask_self, mask_other, dir_membership):
    mask = mask_self if cond == "self" else mask_other
    Y_mat = load_condition_ordered(spike_dir, cond, region, spk_assignment, dir_membership)
    if Y_mat is None or Y_mat.ndim < 2 or Y_mat.shape[0] != int(mask.sum()):
        print(f"  {region}/{cond}: no/mismatched data — skip"); return None
    valid = ~np.isnan(Y_mat).any(axis=1)
    Y_v_raw = Y_mat[valid].astype(np.float32)
    spike_ok = Y_v_raw.sum(0) >= MIN_SPIKES
    n_m = int(spike_ok.sum())
    if n_m == 0:
        print(f"  {region}/{cond}: 0 neurons pass filter"); return None
    Y_v = Y_v_raw[:, spike_ok]
    n_w = Y_v.shape[0]

    if n_w // N_OUTER < N_COMPONENTS + 2:
        print(f"  {region}/{cond}: too few samples — skip"); return None

    X_ctrl_masked = X_ctrl_raw[mask][valid]
    X_sem_sc  = StandardScaler().fit_transform(X_pca[mask][valid]).astype(np.float32)
    X_ctrl_sc = StandardScaler().fit_transform(X_ctrl_masked).astype(np.float32)

    dur_v = all_word_dur[mask][valid].astype(np.float64).clip(1e-3, None)
    offset_v = np.log(dur_v).astype(np.float32)

    print(f"  {region}/{cond}: {n_w}w x {n_m}n", flush=True)

    # ── LEACE: erase combined controls from semantic embeddings ─────────────
    erase_fn, _ = leace_fit(X_sem_sc.astype(np.float64), X_ctrl_sc.astype(np.float64))
    X_sem_erased = erase_fn(X_sem_sc.astype(np.float64)).astype(np.float32)

    corr_before = max_abs_corr(X_sem_sc, X_ctrl_sc)
    corr_after  = max_abs_corr(X_sem_erased, X_ctrl_sc)
    print(f"    max|corr(sem,ctrl)|  before={corr_before:.4f}  after={corr_after:.6f}", flush=True)

    # ── null/sat (same closed-form offset MLE as variance_partitioning.py) ──
    ll_null = np.zeros(n_m, np.float64)
    ll_sat  = np.zeros(n_m, np.float64)
    for tr_m, te_m in block_splits(n_w, N_OUTER):
        rate_tr = Y_v[tr_m].sum(0) / dur_v[tr_m].sum().clip(1e-10)
        mu_te = np.outer(dur_v[te_m], rate_tr).clip(1e-10)
        ll_null += (Y_v[te_m] * np.log(mu_te) - mu_te).sum(0)
        Y_te_np = Y_v[te_m]
        ll_sat_fold = (Y_te_np * np.log(Y_te_np.clip(1e-10)) - Y_te_np).sum(0)
        ll_sat += np.where(Y_te_np.sum(0) > 0, ll_sat_fold, 0.0)
    d_null = ll_sat - ll_null

    # ── fit Poisson GLM on erased embeddings alone ───────────────────────────
    t0 = time.time()
    ll_erased = fit_one_model_ll(X_sem_erased, Y_v, offset=offset_v)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2_erased = np.where(d_null > 0.1, (ll_erased - ll_null) / d_null, np.nan)
    print(f"    fit done {time.time()-t0:.1f}s  median r2_erased={np.nanmedian(r2_erased):.4f}", flush=True)

    # ── xshuffle null: globally permute embedding rows, re-erase against real Z, refit ─
    rng = np.random.default_rng(SEED)
    perm_r2 = np.full((N_PERM, n_m), np.nan)
    t_perm = time.time()
    for pi in range(N_PERM):
        X_sem_perm = X_sem_sc[rng.permutation(n_w)]
        erase_fn_p, _ = leace_fit(X_sem_perm.astype(np.float64), X_ctrl_sc.astype(np.float64))
        X_perm_erased = erase_fn_p(X_sem_perm.astype(np.float64)).astype(np.float32)
        ll_p = fit_one_model_ll(X_perm_erased, Y_v, offset=offset_v)
        with np.errstate(divide="ignore", invalid="ignore"):
            perm_r2[pi] = np.where(d_null > 0.1, (ll_p - ll_null) / d_null, np.nan)
        if (pi + 1) % 10 == 0:
            print(f"    perm {pi+1}/{N_PERM}  {time.time()-t_perm:.0f}s", flush=True)

    p_vals = np.array([
        np.nanmean(perm_r2[:, m] >= r2_erased[m]) if not np.isnan(r2_erased[m]) else 1.0
        for m in range(n_m)
    ])
    p_fdr, sig_fdr = fdr_bh(p_vals)
    significant = sig_fdr & (r2_erased > 0)
    n_sig = int(significant.sum())
    med_sig = float(np.nanmedian(r2_erased[significant])) if n_sig else float("nan")

    print(f"    LEACE-erased: sig {n_sig}/{n_m} ({100*n_sig/n_m:.1f}%)  "
          f"median_r2(sig)={med_sig:.4f}" if n_sig else
          f"    LEACE-erased: sig 0/{n_m} (0.0%)", flush=True)

    return {
        "region": region, "condition": cond, "n_neurons": n_m,
        "n_sig": n_sig, "pct_sig": 100 * n_sig / n_m, "median_r2_sig": med_sig,
        "corr_before": corr_before, "corr_after": corr_after,
    }

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    npy_path = os.path.join(EMBED_DIR, f"{PATIENT_ID}_gpt2-xl_ctx200_word_emb_layers.npy")
    spike_dir = find_spike_dir(PATIENT)

    spk_assignment, mask_self, mask_other, dir_membership = load_speaker_assignment(spike_dir)
    X_ctrl_raw, ctrl_cols = load_control_features(PATIENT_ID)
    all_word_dur = load_word_dur_ms(spike_dir)

    t0 = time.time()
    raw_layer = np.load(npy_path, mmap_mode='r')[LAYER].astype(np.float32)
    X_pca = gpu_pca(raw_layer, N_COMPONENTS)
    del raw_layer
    print(f"PCA done {X_pca.shape} ({time.time()-t0:.1f}s)", flush=True)

    results = []
    for region in ["hippocampus"]:
        for cond in ["self", "other"]:
            r = run_region_condition(region, cond, X_pca, X_ctrl_raw, ctrl_cols, all_word_dur,
                                      spike_dir, spk_assignment, mask_self, mask_other, dir_membership)
            if r:
                results.append(r)
        torch.cuda.empty_cache()

    print("\n=== LEACE pilot summary (PTYEU_task147) ===")
    for r in results:
        print(f"  {r['region']:12s} {r['condition']:5s}  {r['n_sig']:3d}/{r['n_neurons']}  "
              f"({r['pct_sig']:.1f}%)  median_r2(sig)={r['median_r2_sig']:.4f}  "
              f"corr(sem,ctrl) {r['corr_before']:.3f}->{r['corr_after']:.5f}")

    if os.path.exists(BASELINE_PKL):
        with open(BASELINE_PKL, "rb") as f:
            base = pickle.load(f)
        print("\n=== baseline (nested deviance, offset-corrected) ===")
        base = base[base["region"] == "hippocampus"]
        for (region, cond), sub in base.groupby(["region", "condition"]):
            n_sig, n_tot = sub["significant"].sum(), len(sub)
            med = sub.loc[sub["significant"], "unique_semantic"].median() if n_sig else float("nan")
            print(f"  {region:12s} {cond:5s}  {n_sig:3d}/{n_tot}  ({100*n_sig/n_tot:.1f}%)  "
                  f"median_usem={med:.4f}" if n_sig else f"  {region:12s} {cond:5s}    0/{n_tot}  (0%)")

    out_path = "/scratch/aniluchavez/ConvoDATAS/VPResults/leace_pilot_PTYEU_task147_xshuffle.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
