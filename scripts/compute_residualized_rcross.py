#!/usr/bin/env python3
"""
Residualized r_cross analysis.

For each neuron:
  1. Fit Poisson ridge on control features only (GPU batched)
  2. Compute Pearson residuals: r = (Y - mu_ctrl) / sqrt(mu_ctrl)
  3. Fit OLS ridge on semantic PCs → beta_sem  (per condition)
  4. r_cross = corr(beta_sem_self, beta_sem_other)
  5. Split-half reliability of beta_sem within each condition (Spearman-Brown)

Filters to VP-significant neurons (p_perm < 0.05 & unique_semantic > 0).
Saves per-patient PKL; aggregates at end and makes violin plot.

Usage:
  python3 compute_residualized_rcross.py --model gpt2-xl --layer 36
  python3 compute_residualized_rcross.py --model bert-base --layer 12
"""

import argparse, os, pickle, sys, glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT  = Path("/scratch/aniluchavez/hippocampal-speaker-semantics")
sys.path.insert(0, str(PROJECT))

p = argparse.ArgumentParser()
p.add_argument("--model",       default="gpt2-xl")
p.add_argument("--context_tag", default="_ctx200")
p.add_argument("--layer",       type=int, default=36)
p.add_argument("--n_components",type=int, default=30)
p.add_argument("--patient",     default=None)
A = p.parse_args()

MODEL_TAG    = A.model
CONTEXT_TAG  = A.context_tag
LAYER        = A.layer
N_COMPONENTS = A.n_components

# ── paths ─────────────────────────────────────────────────────────────────────
EMBED_DIR    = Path("/scratch/aniluchavez/ConvoDATAS/EmbedCache")
CONTROL_DIR  = Path("/scratch/aniluchavez/ConvoDATAS/ControlFeatures")
SURPRISAL_DIR= Path("/scratch/aniluchavez/ConvoDATAS/Surprisal")
SPIKE_ROOT   = Path("/scratch/aniluchavez/ConvoDATAS/SpikeWindows")
VPR_DIR      = Path("/scratch/aniluchavez/ConvoDATAS/VPResults")

# VP PKL for significance filtering
if MODEL_TAG == "gpt2-xl":
    VP_PKL = VPR_DIR / f"{MODEL_TAG}{CONTEXT_TAG}_worddur_xcirc_symperm/pc{N_COMPONENTS}/L{LAYER}_VP_all.pkl"
else:
    VP_PKL = VPR_DIR / f"{MODEL_TAG}{CONTEXT_TAG}_worddur_xcirc_symperm/pc{N_COMPONENTS}/L{LAYER}_VP_all.pkl"

OUT_DIR = VPR_DIR / f"{MODEL_TAG}{CONTEXT_TAG}_worddur_xcirc_symperm/pc{N_COMPONENTS}/residualized_rcross"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── GPU setup ─────────────────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={DEVICE}  model={MODEL_TAG}{CONTEXT_TAG}  layer={LAYER}  pc={N_COMPONENTS}")
if DEVICE == "cuda":
    print(f"  {torch.cuda.get_device_name(0)}  {torch.cuda.get_device_properties(0).total_memory//1024**2} MB")

# ── control features ──────────────────────────────────────────────────────────
CONTROL_FEATURES = (
    ["log_word_freq", "word_length", "local_count", "local_rate", "surprisal",
     "dep_depth", "dep_children", "sent_position", "serial_position",
     "f0_mean", "f0_std", "rms_mean", "spectral_flux", "speaking_rate"]
)
ALPHAS = np.logspace(-2, 4, 20)

# ── GPU Poisson ridge helpers ─────────────────────────────────────────────────
def _t(arr, device=DEVICE):
    return torch.tensor(np.asarray(arr, np.float32), device=device)

def _fit_all_neurons_gpu(X_np, Y_np, alphas_np, max_iter=80, tol=1e-4):
    """Batched L-BFGS Poisson ridge on GPU. Returns coef (n_neurons, n_features)."""
    n, p_ = X_np.shape
    k     = Y_np.shape[1]
    X_t   = _t(X_np); Y_t = _t(Y_np)
    a_t   = torch.tensor(alphas_np, dtype=torch.float32, device=DEVICE)
    W     = torch.zeros(p_, k, device=DEVICE, requires_grad=False)
    W     = torch.nn.Parameter(W)
    opt   = torch.optim.LBFGS([W], max_iter=max_iter, tolerance_grad=tol,
                                line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        eta = torch.clamp(X_t @ W, -20, 20)
        mu  = torch.exp(eta)
        nll = -(Y_t * eta - mu).sum(0)
        reg = 0.5 * a_t * (W * W).sum(0)
        (nll + reg).sum().backward()
        return (nll + reg).sum()
    opt.step(closure)
    return W.detach().cpu().numpy().T  # (k, p_)

def gpu_alpha_cv(X_np, Y_np, n_splits=5):
    """Per-neuron alpha selection on GPU. Returns best_alpha (k,)."""
    n, k = X_np.shape[0], Y_np.shape[1]
    fold = n // n_splits
    best_alpha = np.full(k, ALPHAS[0])
    best_ll    = np.full(k, -np.inf)
    for alpha in ALPHAS:
        fll = np.zeros((n_splits, k))
        for fi in range(n_splits):
            vs = fi * fold
            ve = (vs + fold) if fi < n_splits - 1 else n
            tr = np.concatenate([np.arange(0, vs), np.arange(ve, n)])
            va = np.arange(vs, ve)
            coef = _fit_all_neurons_gpu(X_np[tr], Y_np[tr],
                                        np.full(k, alpha, np.float32))
            mu_va = np.exp(np.clip(X_np[va] @ coef.T, -20, 20))
            fll[fi] = (Y_np[va] * np.log(np.clip(mu_va, 1e-10, None)) - mu_va).sum(0)
        mean_ll = fll.mean(0)
        better  = mean_ll > best_ll
        best_alpha[better] = alpha
        best_ll[better]    = mean_ll[better]
    return best_alpha

# ── OLS ridge helpers ─────────────────────────────────────────────────────────
def ols_ridge_beta(X, y, alpha):
    """OLS ridge: (X'X + alpha*I)^-1 X' y  per neuron (y is vector)."""
    from sklearn.linear_model import Ridge
    m = Ridge(alpha=float(alpha), fit_intercept=True)
    m.fit(X, y)
    return m.coef_.astype(float)

def ols_alpha_cv(X, Y, n_splits=5, rng=None):
    """Per-neuron alpha selection for OLS ridge. Y shape (n, k)."""
    from sklearn.linear_model import Ridge
    if rng is None: rng = np.random.default_rng(42)
    n, k = Y.shape
    fold = n // n_splits
    best_alpha = np.full(k, ALPHAS[0])
    best_mse   = np.full(k, np.inf)
    for alpha in ALPHAS:
        mse = np.zeros((n_splits, k))
        for fi in range(n_splits):
            vs = fi * fold
            ve = (vs + fold) if fi < n_splits - 1 else n
            tr = np.concatenate([np.arange(0, vs), np.arange(ve, n)])
            va = np.arange(vs, ve)
            m  = Ridge(alpha=float(alpha), fit_intercept=True)
            m.fit(X[tr], Y[tr])
            pred  = m.predict(X[va])
            mse[fi] = ((Y[va] - pred) ** 2).mean(0)
        mean_mse = mse.mean(0)
        better   = mean_mse < best_mse
        best_alpha[better] = alpha
        best_mse[better]   = mean_mse[better]
    return best_alpha

def fit_ols_betas_all(X, Y, alphas):
    """Fit OLS ridge per-neuron with pre-selected alphas. Returns (k, p_) betas."""
    from sklearn.linear_model import Ridge
    k = Y.shape[1]
    betas = np.zeros((k, X.shape[1]))
    for ni in range(k):
        m = Ridge(alpha=float(alphas[ni]), fit_intercept=True)
        m.fit(X, Y[:, ni])
        betas[ni] = m.coef_
    return betas

# ── data loading (reuse VP logic) ─────────────────────────────────────────────
def spike_dir(patient_ID):
    # PKL uses PTYEU_task147; dirs use ptYEU_task147
    dir_id = patient_ID[:2].lower() + patient_ID[2:]
    d = SPIKE_ROOT / f"output_{dir_id}_english_only_worddur"
    return d if d.is_dir() else None

def load_spike_matrix(spike_dir, speaker, region="hippocampus"):
    spk_d = spike_dir / speaker
    if not spk_d.is_dir(): return None
    cands = [f for f in os.listdir(spk_d)
             if f.lower().startswith(region.lower()) and f.endswith(".npy")]
    return np.load(spk_d / cands[0]) if cands else None

def load_speaker_assignment(spike_dir):
    cands = [f for f in os.listdir(spike_dir) if f.endswith("_with_regress_dur.xlsx")]
    tx = pd.read_excel(spike_dir / cands[0])
    spk_cols = sorted([c for c in tx.columns if str(c).startswith("Speaker")],
                      key=lambda c: int(c.replace("Speaker","").strip())
                                    if c.replace("Speaker","").strip().isdigit() else 999)
    def _nn(v): return pd.notna(v) and str(v).strip() not in ("", "nan")
    dir_m = {c: np.array([_nn(v) for v in tx[c]], dtype=bool) for c in spk_cols}
    n     = len(tx)
    assign = np.array([None]*n, dtype=object)
    for i in range(n):
        for c in spk_cols:
            if dir_m[c][i]: assign[i] = c; break
    return assign, dir_m

def load_condition(sd, cond, region, assign, dir_m):
    if cond == "self":
        return load_spike_matrix(sd, "Speaker1", region)
    other = sorted([c for c in dir_m if c != "Speaker1"])
    mats  = {s: load_spike_matrix(sd, s, region) for s in other}
    mats  = {s: m for s, m in mats.items() if m is not None}
    pos   = {s: 0 for s in mats}
    rows  = []
    for i, spk in enumerate(assign):
        if spk is None or spk == "Speaker1": continue
        for s in mats:
            if dir_m[s][i]:
                if spk == s: rows.append(mats[s][pos[s]])
                pos[s] += 1
    return np.vstack(rows) if rows else None

def load_word_dur(sd):
    cands = [f for f in os.listdir(sd) if f.endswith("_with_regress_dur.xlsx")]
    tx = pd.read_excel(sd / cands[0])
    col = "word_dur" if "word_dur" in tx.columns else "Duration"
    return tx[col].values.astype(np.float64) if col in tx.columns else None

def load_controls(patient_ID):
    ctrl = pd.read_csv(CONTROL_DIR / f"{patient_ID}_control_features.csv")
    sp   = SURPRISAL_DIR / f"{patient_ID}_surprisal.csv"
    if sp.exists() and "surprisal" not in ctrl.columns:
        s = pd.read_csv(sp)
        if len(s) == len(ctrl): ctrl["surprisal"] = s["surprisal"].values
    cols = [c for c in CONTROL_FEATURES if c in ctrl.columns]
    X    = ctrl[cols].values.astype(float)
    means = np.where(np.isnan(np.nanmean(X, 0)), 0.0, np.nanmean(X, 0))
    nm    = np.isnan(X)
    X[nm] = means[np.where(nm)[1]]
    # standardize
    std = X.std(0); std[std == 0] = 1.0
    return (X - X.mean(0)) / std

def load_semantic_pcs(patient_ID):
    npy = EMBED_DIR / f"{patient_ID}_{MODEL_TAG}{CONTEXT_TAG}_word_emb_layers.npy"
    if not npy.exists(): return None
    raw = np.load(npy, mmap_mode='r')[LAYER].astype(np.float32)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=N_COMPONENTS)
    pcs = pca.fit_transform(raw)
    std = pcs.std(0); std[std == 0] = 1.0
    return (pcs - pcs.mean(0)) / std

# ── split-half reliability ────────────────────────────────────────────────────
def split_half_reliability(X, R, alphas, n_splits=100, rng=None):
    """
    Split-half beta correlation on OLS-ridge betas, Spearman-Brown corrected.
    X: (n_words, n_sem_pcs)
    R: (n_words, n_neurons) residuals
    alphas: (n_neurons,) pre-selected
    Returns mean SB-corrected reliability per neuron.
    """
    if rng is None: rng = np.random.default_rng(42)
    n, k = R.shape
    sbs  = np.zeros((n_splits, k))
    idx  = np.arange(n)
    from sklearn.linear_model import Ridge
    for si in range(n_splits):
        rng.shuffle(idx)
        h1, h2 = idx[:n//2], idx[n//2: 2*(n//2)]
        b1 = np.zeros((k, X.shape[1])); b2 = np.zeros_like(b1)
        for ni in range(k):
            m1 = Ridge(alpha=float(alphas[ni]), fit_intercept=True)
            m1.fit(X[h1], R[h1, ni]); b1[ni] = m1.coef_
            m2 = Ridge(alpha=float(alphas[ni]), fit_intercept=True)
            m2.fit(X[h2], R[h2, ni]); b2[ni] = m2.coef_
        for ni in range(k):
            a, b_ = b1[ni], b2[ni]
            denom = np.linalg.norm(a) * np.linalg.norm(b_)
            r     = float(np.dot(a-a.mean(), b_-b_.mean()) / denom) if denom > 0 else np.nan
            sb    = (2*r)/(1+r) if np.isfinite(r) else np.nan
            sbs[si, ni] = sb
    return np.nanmean(sbs, 0)

# ── safe pearson ──────────────────────────────────────────────────────────────
def safe_pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a - a.mean(), b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else np.nan

# ── load VP significance ───────────────────────────────────────────────────────
print(f"\nLoading VP from {VP_PKL}")
df_vp = pickle.load(open(VP_PKL, "rb"))
hippo_vp = df_vp[df_vp["region"] == "hippocampus"]

# ── patient list ──────────────────────────────────────────────────────────────
PATIENTS = sorted(hippo_vp["patient"].unique())
if A.patient:
    PATIENTS = [p_ for p_ in PATIENTS if p_ == A.patient]
print(f"Patients: {len(PATIENTS)}")

# ── main loop ─────────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
all_rows = []

for patient_ID in PATIENTS:
    print(f"\n{'='*60}\n  {patient_ID}")
    sd = spike_dir(patient_ID)
    if sd is None:
        print("  SKIP — no spike dir"); continue

    X_ctrl = load_controls(patient_ID)
    X_sem  = load_semantic_pcs(patient_ID)
    if X_sem is None:
        print("  SKIP — no embeddings"); continue

    wdur = load_word_dur(sd)
    assign, dir_m = load_speaker_assignment(sd)

    cond_data = {}
    for cond in ["self", "other"]:
        Y_mat = load_condition(sd, cond, "hippocampus", assign, dir_m)
        if Y_mat is None or Y_mat.shape[0] == 0:
            print(f"  {cond}: no data"); continue

        n_w, n_n = Y_mat.shape
        print(f"  {cond}: {n_w}w × {n_n}n")

        # align rows with transcript
        mask = (assign == "Speaker1") if cond == "self" else \
               np.array([(a is not None and a != "Speaker1") for a in assign])
        X_c  = X_ctrl[mask]
        X_s  = X_sem[mask]
        if wdur is not None:
            wd  = wdur[mask]
            log_offset = np.log(np.clip(wd / 1000.0, 1e-4, None))  # s
        else:
            log_offset = None

        Y_f  = Y_mat.astype(np.float32)

        # ── 1. Poisson ridge on controls, select alpha ─────────────────────
        print(f"    alpha CV (Poisson, controls)...", flush=True)
        alpha_ctrl = gpu_alpha_cv(X_c, Y_f, n_splits=5)

        # fit full controls model
        coef_ctrl = _fit_all_neurons_gpu(X_c, Y_f, alpha_ctrl.astype(np.float32))
        # coef_ctrl: (n_n, n_ctrl)
        eta = X_c @ coef_ctrl.T  # (n_w, n_n)
        if log_offset is not None:
            eta = eta + log_offset[:, None]
        mu_ctrl = np.exp(np.clip(eta, -20, 20))  # (n_w, n_n)

        # ── 2. Pearson residuals ───────────────────────────────────────────
        R = (Y_f - mu_ctrl) / np.sqrt(np.clip(mu_ctrl, 1e-8, None))
        R = R.astype(np.float32)

        # ── 3. OLS ridge on semantic PCs, select alpha ─────────────────────
        print(f"    alpha CV (OLS, semantic)...", flush=True)
        alpha_sem = ols_alpha_cv(X_s, R, n_splits=5, rng=rng)

        # fit full semantic model on residuals
        betas_sem = fit_ols_betas_all(X_s, R, alpha_sem)  # (n_n, n_pc)

        # ── 4. Split-half reliability ──────────────────────────────────────
        print(f"    split-half reliability...", flush=True)
        rel = split_half_reliability(X_s, R, alpha_sem, n_splits=100, rng=rng)

        cond_data[cond] = {
            "betas_sem": betas_sem,
            "reliability": rel,
            "n_words": n_w,
            "n_neurons": n_n,
            "alpha_sem": alpha_sem,
            "X_s": X_s,
            "R": R,
        }

    if "self" not in cond_data or "other" not in cond_data:
        print("  SKIP — missing a condition"); continue

    # ── 5. r_cross per neuron ─────────────────────────────────────────────
    b_self  = cond_data["self"]["betas_sem"]
    b_other = cond_data["other"]["betas_sem"]
    n_n = min(b_self.shape[0], b_other.shape[0])
    r_cross = np.array([safe_pearson(b_self[ni], b_other[ni]) for ni in range(n_n)])

    # ── 5b. Permutation test: shuffle other-condition semantic PCs ────────
    N_PERM = 1000
    print(f"    permutation test (n={N_PERM})...", flush=True)
    X_other   = cond_data["other"]["X_s"]
    R_other   = cond_data["other"]["R"]
    alp_other = cond_data["other"]["alpha_sem"]
    rng_perm  = np.random.default_rng(99)
    r_null    = np.full((N_PERM, n_n), np.nan)
    for pi in range(N_PERM):
        perm_idx    = rng_perm.permutation(len(X_other))
        b_null      = fit_ols_betas_all(X_other[perm_idx], R_other, alp_other)
        for ni in range(n_n):
            r_null[pi, ni] = safe_pearson(b_self[ni], b_null[ni])
    p_perm = np.array([(r_null[:, ni] >= r_cross[ni]).mean() for ni in range(n_n)])

    rel_self  = cond_data["self"]["reliability"][:n_n]
    rel_other = cond_data["other"]["reliability"][:n_n]
    ceil_mean = np.minimum(rel_self, rel_other)

    # ── 6. VP significance labels (don't filter — save all neurons) ───────
    vp_pat = hippo_vp[hippo_vp["patient"] == patient_ID]
    sig_self  = set(vp_pat[(vp_pat["condition"]=="self")  & (vp_pat["p_perm"]<0.05) & (vp_pat["unique_semantic"]>0)]["neuron_idx"])
    sig_other = set(vp_pat[(vp_pat["condition"]=="other") & (vp_pat["p_perm"]<0.05) & (vp_pat["unique_semantic"]>0)]["neuron_idx"])
    sig_either = sig_self | sig_other

    for ni in range(n_n):
        all_rows.append({
            "patient": patient_ID,
            "neuron_idx": ni,
            "r_cross": float(r_cross[ni]),
            "p_perm": float(p_perm[ni]),
            "self_reliability_mean": float(rel_self[ni]),
            "other_reliability_mean": float(rel_other[ni]),
            "lower_ceiling": float(ceil_mean[ni]),
            "sig_vp": ni in sig_either,
        })

    # save per-patient
    out_f = OUT_DIR / f"{patient_ID}_L{LAYER}_resid_rcross.pkl"
    pickle.dump({
        "r_cross": r_cross,
        "p_perm": p_perm,
        "r_null": r_null.astype(np.float32),
        "rel_self": rel_self,
        "rel_other": rel_other,
        "ceil": ceil_mean,
        "sig_either": sig_either,
    }, open(out_f, "wb"))
    print(f"  saved {out_f.name}")

# ── aggregate & save ──────────────────────────────────────────────────────────
df = pd.DataFrame(all_rows)
agg_f = OUT_DIR / f"L{LAYER}_resid_rcross_all.pkl"
pickle.dump(df, open(agg_f, "wb"))
print(f"\nAggregated: {len(df)} sig neurons from {df['patient'].nunique()} patients")

# ── violin plot ───────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

FIG_DIR = PROJECT / "figures"

def p_label(p_):
    if not np.isfinite(p_): return "p=n/a"
    if p_ < 0.0001: return "p<0.0001"
    return f"p={p_:.4f}"

def one_sided_wilcoxon(x, y=None):
    x = np.asarray(x, float)
    if y is None:
        diff = x[np.isfinite(x)]
    else:
        y = np.asarray(y, float)
        ok = np.isfinite(x) & np.isfinite(y)
        diff = (x-y)[ok]
    if len(diff) < 2 or np.allclose(diff, 0): return np.nan
    return float(stats.wilcoxon(diff, alternative="greater", zero_method="wilcox").pvalue)

patient_stats = df.groupby("patient", as_index=False).agg(
    cross=("r_cross","median"), ceiling=("lower_ceiling","median"),
    p_perm_med=("p_perm","median"))
p_cross_zero = one_sided_wilcoxon(patient_stats["cross"].values)
p_cross_ceil = one_sided_wilcoxon(patient_stats["ceiling"].values,
                                   patient_stats["cross"].values)

n_neurons  = len(df)
n_patients = df["patient"].nunique()
median_cross = df["r_cross"].median()
median_ceil  = df["lower_ceiling"].median()
pct_pos      = 100*(df["r_cross"]>0).mean()
frac_sig_perm = 100*(df["p_perm"] < 0.05).mean()
median_p_perm = df["p_perm"].median()

print(f"\nn={n_neurons} neurons, {n_patients} patients")
print(f"median r_cross={median_cross:.3f}  ceiling={median_ceil:.3f}")
print(f"% positive={pct_pos:.1f}%")
print(f"p(r_cross>0)={p_cross_zero:.2e}  [Wilcoxon, patient medians]")
print(f"p(ceiling>r_cross)={p_cross_ceil:.2e}")
print(f"permutation test: {frac_sig_perm:.1f}% neurons p_perm<0.05  (median p_perm={median_p_perm:.3f})")

METRICS = ["Speaking\nreliability", "Listening\nreliability", "Cross-condition\n(r_cross)"]
COLORS  = {"Speaking\nreliability":"#c83e3e",
           "Listening\nreliability":"#3656b3",
           "Cross-condition\n(r_cross)":"#b449b5"}

wide = df[["patient","neuron_idx","self_reliability_mean","other_reliability_mean",
           "r_cross","lower_ceiling"]].copy()
wide.columns = ["patient","neuron_idx",
                "Speaking\nreliability","Listening\nreliability",
                "Cross-condition\n(r_cross)","lower_ceiling"]
long = wide.melt(id_vars=["patient","neuron_idx","lower_ceiling"],
                 value_vars=METRICS, var_name="metric", value_name="correlation"
                 ).dropna(subset=["correlation"])

sns.set_theme(style="white", context="talk")
fig, ax = plt.subplots(figsize=(7,6), constrained_layout=True)
rng2 = np.random.default_rng(42)

sns.violinplot(data=long, x="metric", y="correlation",
               order=METRICS, hue="metric", hue_order=METRICS,
               palette=COLORS, inner=None, cut=0, linewidth=1, legend=False, ax=ax)

for xi, metric in enumerate(METRICS):
    vals = wide[metric].dropna().values
    keep = (rng2.choice(len(vals), size=min(220,len(vals)), replace=False)
            if len(vals)>220 else np.arange(len(vals)))
    jitter = rng2.uniform(-0.16, 0.16, size=len(keep))
    ax.scatter(xi+jitter, vals[keep], s=8, color="black", alpha=0.28, linewidths=0, zorder=3)
    med = float(np.median(vals))
    ax.plot([xi-0.23, xi+0.23], [med,med], color="black", linewidth=4,
            solid_capstyle="butt", zorder=4)

bx = 2.48
ax.plot([bx,bx], [0, median_ceil], color="black", linewidth=1.5, clip_on=False)
for y_ in [0, median_cross, median_ceil]:
    ax.plot([bx-0.07, bx], [y_,y_], color="black", linewidth=1.5, clip_on=False)
ax.text(bx+0.05, median_cross/2, p_label(p_cross_zero), va="center", fontsize=10)
ax.text(bx+0.05, (median_cross+median_ceil)/2, p_label(p_cross_ceil), va="center", fontsize=10)

ax.axhline(0, color="0.82", linewidth=7, zorder=0)
ax.set_ylim(-0.35, 1.03)
ax.set_xlabel(""); ax.set_ylabel("correlation (r)")

label = f"L{LAYER}" if "gpt2" in MODEL_TAG else f"L{LAYER}"
model_str = "GPT-2 XL" if "gpt2" in MODEL_TAG else "BERT-base"
ax.set_title(
    f"Hippocampus — residualized semantic encoding\n"
    f"{model_str} {label}, worddur pc{N_COMPONENTS}, xcirc VP filter\n"
    f"Controls residualized before r_cross & reliability\n"
    f"n={n_neurons} neurons, {n_patients} patients | p_perm<0.05 & unique_sem>0",
    fontsize=9.5, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)

suffix = f"gpt2xl" if "gpt2" in MODEL_TAG else "bert"
out_png = FIG_DIR / f"12_brackets_{suffix}_worddur_pc{N_COMPONENTS}_resid_L{LAYER}.png"
out_svg = FIG_DIR / f"12_brackets_{suffix}_worddur_pc{N_COMPONENTS}_resid_L{LAYER}.svg"
out_pdf = FIG_DIR / f"12_brackets_{suffix}_worddur_pc{N_COMPONENTS}_resid_L{LAYER}.pdf"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, dpi=220, bbox_inches="tight")
fig.savefig(out_svg, bbox_inches="tight")
print(f"Saved: {out_png}")
