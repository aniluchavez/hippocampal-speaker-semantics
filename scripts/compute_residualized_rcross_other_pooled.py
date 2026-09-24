#!/usr/bin/env python3
"""
Residualized r_cross — pooled OTHER split in half.

Pools all non-Speaker1 words, splits randomly into two halves,
fits residualized OLS semantic betas on each, and correlates them.
Excludes PTYEZ_task60.

Usage:
  python3 compute_residualized_rcross_other_pooled.py --model bert-base --layer 12
  python3 compute_residualized_rcross_other_pooled.py --model gpt2-xl --layer 36
"""

import argparse, os, pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT  = Path("/scratch/aniluchavez/hippocampal-speaker-semantics")
sys.path.insert(0, str(PROJECT))

p = argparse.ArgumentParser()
p.add_argument("--model",        default="gpt2-xl")
p.add_argument("--context_tag",  default="_ctx200")
p.add_argument("--layer",        type=int, default=36)
p.add_argument("--n_components", type=int, default=30)
p.add_argument("--patient",      default=None)
A = p.parse_args()

MODEL_TAG    = A.model
CONTEXT_TAG  = A.context_tag
LAYER        = A.layer
N_COMPONENTS = A.n_components

EXCLUDE_PATIENTS = {"PTYEZ_task60"}

# ── paths ─────────────────────────────────────────────────────────────────────
EMBED_DIR     = Path("/scratch/aniluchavez/ConvoDATAS/EmbedCache")
CONTROL_DIR   = Path("/scratch/aniluchavez/ConvoDATAS/ControlFeatures")
SURPRISAL_DIR = Path("/scratch/aniluchavez/ConvoDATAS/Surprisal")
SPIKE_ROOT    = Path("/scratch/aniluchavez/ConvoDATAS/SpikeWindows")
VPR_DIR       = Path("/scratch/aniluchavez/ConvoDATAS/VPResults")

VP_PKL  = VPR_DIR / f"{MODEL_TAG}{CONTEXT_TAG}_worddur_xcirc_symperm/pc{N_COMPONENTS}/L{LAYER}_VP_all.pkl"
OUT_DIR = VPR_DIR / f"{MODEL_TAG}{CONTEXT_TAG}_worddur_xcirc_symperm/pc{N_COMPONENTS}/residualized_rcross_other_pooled"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── GPU setup ─────────────────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={DEVICE}  model={MODEL_TAG}{CONTEXT_TAG}  layer={LAYER}  pc={N_COMPONENTS}")
if DEVICE == "cuda":
    print(f"  {torch.cuda.get_device_name(0)}  {torch.cuda.get_device_properties(0).total_memory//1024**2} MB")

CONTROL_FEATURES = [
    "log_word_freq", "word_length", "local_count", "local_rate", "surprisal",
    "dep_depth", "dep_children", "sent_position", "serial_position",
    "f0_mean", "f0_std", "rms_mean", "spectral_flux", "speaking_rate"
]
ALPHAS = np.logspace(-2, 4, 20)

# ── GPU Poisson ridge ─────────────────────────────────────────────────────────
def _t(arr):
    return torch.tensor(np.asarray(arr, np.float32), device=DEVICE)

def _fit_all_neurons_gpu(X_np, Y_np, alphas_np, max_iter=80, tol=1e-4):
    n, p_ = X_np.shape; k = Y_np.shape[1]
    X_t = _t(X_np); Y_t = _t(Y_np)
    a_t = torch.tensor(alphas_np, dtype=torch.float32, device=DEVICE)
    W   = torch.nn.Parameter(torch.zeros(p_, k, device=DEVICE))
    opt = torch.optim.LBFGS([W], max_iter=max_iter, tolerance_grad=tol,
                              line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        eta = torch.clamp(X_t @ W, -20, 20); mu = torch.exp(eta)
        nll = -(Y_t * eta - mu).sum(0)
        reg = 0.5 * a_t * (W * W).sum(0)
        (nll + reg).sum().backward()
        return (nll + reg).sum()
    opt.step(closure)
    return W.detach().cpu().numpy().T

def gpu_alpha_cv(X_np, Y_np, n_splits=5):
    n, k = X_np.shape[0], Y_np.shape[1]; fold = n // n_splits
    best_alpha = np.full(k, ALPHAS[0]); best_ll = np.full(k, -np.inf)
    for alpha in ALPHAS:
        fll = np.zeros((n_splits, k))
        for fi in range(n_splits):
            vs = fi*fold; ve = (vs+fold) if fi < n_splits-1 else n
            tr = np.concatenate([np.arange(0,vs), np.arange(ve,n)]); va = np.arange(vs,ve)
            coef = _fit_all_neurons_gpu(X_np[tr], Y_np[tr], np.full(k, alpha, np.float32))
            mu_va = np.exp(np.clip(X_np[va] @ coef.T, -20, 20))
            fll[fi] = (Y_np[va] * np.log(np.clip(mu_va,1e-10,None)) - mu_va).sum(0)
        mean_ll = fll.mean(0); better = mean_ll > best_ll
        best_alpha[better] = alpha; best_ll[better] = mean_ll[better]
    return best_alpha

# ── OLS ridge ─────────────────────────────────────────────────────────────────
def ols_alpha_cv(X, Y, n_splits=5, rng=None):
    from sklearn.linear_model import Ridge
    if rng is None: rng = np.random.default_rng(42)
    n, k = Y.shape; fold = n // n_splits
    best_alpha = np.full(k, ALPHAS[0]); best_mse = np.full(k, np.inf)
    for alpha in ALPHAS:
        mse = np.zeros((n_splits, k))
        for fi in range(n_splits):
            vs = fi*fold; ve = (vs+fold) if fi < n_splits-1 else n
            tr = np.concatenate([np.arange(0,vs), np.arange(ve,n)]); va = np.arange(vs,ve)
            m = Ridge(alpha=float(alpha), fit_intercept=True)
            m.fit(X[tr], Y[tr]); mse[fi] = ((Y[va] - m.predict(X[va]))**2).mean(0)
        mean_mse = mse.mean(0); better = mean_mse < best_mse
        best_alpha[better] = alpha; best_mse[better] = mean_mse[better]
    return best_alpha

def fit_ols_betas_all(X, Y, alphas):
    from sklearn.linear_model import Ridge
    k = Y.shape[1]; betas = np.zeros((k, X.shape[1]))
    for ni in range(k):
        m = Ridge(alpha=float(alphas[ni]), fit_intercept=True)
        m.fit(X, Y[:, ni]); betas[ni] = m.coef_
    return betas

# ── data loading ──────────────────────────────────────────────────────────────
def spike_dir(patient_ID):
    dir_id = patient_ID[:2].lower() + patient_ID[2:]
    d = SPIKE_ROOT / f"output_{dir_id}_english_only_worddur"
    return d if d.is_dir() else None

def load_spike_matrix(sd, speaker, region="hippocampus"):
    spk_d = sd / speaker
    if not spk_d.is_dir(): return None
    cands = [f for f in os.listdir(spk_d)
             if f.lower().startswith(region.lower()) and f.endswith(".npy")]
    return np.load(spk_d / cands[0]) if cands else None

def load_speaker_assignment(sd):
    cands = [f for f in os.listdir(sd) if f.endswith("_with_regress_dur.xlsx")]
    tx = pd.read_excel(sd / cands[0])
    spk_cols = sorted([c for c in tx.columns if str(c).startswith("Speaker")],
                      key=lambda c: int(c.replace("Speaker","").strip())
                                    if c.replace("Speaker","").strip().isdigit() else 999)
    def _nn(v): return pd.notna(v) and str(v).strip() not in ("","nan")
    dir_m = {c: np.array([_nn(v) for v in tx[c]], dtype=bool) for c in spk_cols}
    n = len(tx); assign = np.array([None]*n, dtype=object)
    for i in range(n):
        for c in spk_cols:
            if dir_m[c][i]: assign[i] = c; break
    return assign, dir_m

def load_other_condition(sd, region, assign, dir_m):
    other = sorted([c for c in dir_m if c != "Speaker1"])
    mats  = {s: load_spike_matrix(sd, s, region) for s in other}
    mats  = {s: m for s, m in mats.items() if m is not None}
    pos   = {s: 0 for s in mats}; rows = []
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
    sp = SURPRISAL_DIR / f"{patient_ID}_surprisal.csv"
    if sp.exists() and "surprisal" not in ctrl.columns:
        s = pd.read_csv(sp)
        if len(s) == len(ctrl): ctrl["surprisal"] = s["surprisal"].values
    cols = [c for c in CONTROL_FEATURES if c in ctrl.columns]
    X = ctrl[cols].values.astype(float)
    means = np.where(np.isnan(np.nanmean(X,0)), 0.0, np.nanmean(X,0))
    nm = np.isnan(X); X[nm] = means[np.where(nm)[1]]
    std = X.std(0); std[std==0] = 1.0
    return (X - X.mean(0)) / std

def load_semantic_pcs(patient_ID):
    npy = EMBED_DIR / f"{patient_ID}_{MODEL_TAG}{CONTEXT_TAG}_word_emb_layers.npy"
    if not npy.exists(): return None
    raw = np.load(npy, mmap_mode='r')[LAYER].astype(np.float32)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=N_COMPONENTS); pcs = pca.fit_transform(raw)
    std = pcs.std(0); std[std==0] = 1.0
    return (pcs - pcs.mean(0)) / std

def safe_pearson(a, b):
    a, b = np.asarray(a,float), np.asarray(b,float)
    a, b = a-a.mean(), b-b.mean()
    d = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/d) if d>0 else np.nan

# ── load VP ───────────────────────────────────────────────────────────────────
print(f"\nLoading VP from {VP_PKL}")
df_vp = pickle.load(open(VP_PKL,"rb"))
hippo_vp = df_vp[df_vp["region"]=="hippocampus"]

PATIENTS = sorted(hippo_vp["patient"].unique())
PATIENTS = [p_ for p_ in PATIENTS if p_ not in EXCLUDE_PATIENTS]
if A.patient:
    PATIENTS = [p_ for p_ in PATIENTS if p_ == A.patient]
print(f"Patients: {len(PATIENTS)}  (excluded: {EXCLUDE_PATIENTS})")

# ── main loop ─────────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
all_rows = []

for patient_ID in PATIENTS:
    print(f"\n{'='*60}\n  {patient_ID}")
    sd = spike_dir(patient_ID)
    if sd is None: print("  SKIP — no spike dir"); continue

    X_ctrl = load_controls(patient_ID)
    X_sem  = load_semantic_pcs(patient_ID)
    if X_sem is None: print("  SKIP — no embeddings"); continue

    wdur = load_word_dur(sd)
    assign, dir_m = load_speaker_assignment(sd)

    # ── load pooled other ─────────────────────────────────────────────────────
    Y_other = load_other_condition(sd, "hippocampus", assign, dir_m)
    if Y_other is None or Y_other.shape[0] == 0:
        print("  SKIP — no other data"); continue

    n_w, n_n = Y_other.shape
    print(f"  other (pooled): {n_w}w × {n_n}n")

    mask_other = np.array([(a is not None and a != "Speaker1") for a in assign])
    X_c = X_ctrl[mask_other]; X_s = X_sem[mask_other]
    log_offset = (np.log(np.clip(wdur[mask_other]/1000.0,1e-4,None))
                  if wdur is not None else None)
    Y_f = Y_other.astype(np.float32)

    # ── 1. Poisson ridge on controls ──────────────────────────────────────────
    print(f"    alpha CV (Poisson, controls)...", flush=True)
    alpha_ctrl = gpu_alpha_cv(X_c, Y_f, n_splits=5)
    coef_ctrl  = _fit_all_neurons_gpu(X_c, Y_f, alpha_ctrl.astype(np.float32))
    eta = X_c @ coef_ctrl.T
    if log_offset is not None: eta = eta + log_offset[:, None]
    mu_ctrl = np.exp(np.clip(eta, -20, 20))

    # ── 2. Pearson residuals ──────────────────────────────────────────────────
    R = ((Y_f - mu_ctrl) / np.sqrt(np.clip(mu_ctrl,1e-8,None))).astype(np.float32)

    # ── 3. Random split into two halves ──────────────────────────────────────
    idx = rng.permutation(n_w)
    h1, h2 = idx[:n_w//2], idx[n_w//2: 2*(n_w//2)]
    print(f"    split: h1={len(h1)}, h2={len(h2)}", flush=True)

    # ── 4. OLS ridge on each half ─────────────────────────────────────────────
    print(f"    alpha CV (OLS, h1)...", flush=True)
    alpha_h1 = ols_alpha_cv(X_s[h1], R[h1], n_splits=5, rng=rng)
    b_h1     = fit_ols_betas_all(X_s[h1], R[h1], alpha_h1)

    print(f"    alpha CV (OLS, h2)...", flush=True)
    alpha_h2 = ols_alpha_cv(X_s[h2], R[h2], n_splits=5, rng=rng)
    b_h2     = fit_ols_betas_all(X_s[h2], R[h2], alpha_h2)

    # ── 5. r_cross ────────────────────────────────────────────────────────────
    r_cross = np.array([safe_pearson(b_h1[ni], b_h2[ni]) for ni in range(n_n)])

    # ── 6. Permutation test ───────────────────────────────────────────────────
    N_PERM = 1000
    print(f"    permutation test (n={N_PERM})...", flush=True)
    rng_perm = np.random.default_rng(99)
    r_null   = np.full((N_PERM, n_n), np.nan)
    for pi in range(N_PERM):
        perm_idx = rng_perm.permutation(len(h2))
        b_null   = fit_ols_betas_all(X_s[h2][perm_idx], R[h2], alpha_h2)
        for ni in range(n_n):
            r_null[pi, ni] = safe_pearson(b_h1[ni], b_null[ni])
    p_perm = np.array([(r_null[:, ni] >= r_cross[ni]).mean() for ni in range(n_n)])

    # ── 7. VP labels ──────────────────────────────────────────────────────────
    vp_pat = hippo_vp[hippo_vp["patient"]==patient_ID]
    sig_self  = set(vp_pat[(vp_pat["condition"]=="self")  & (vp_pat["p_perm"]<0.05) & (vp_pat["unique_semantic"]>0)]["neuron_idx"])
    sig_other = set(vp_pat[(vp_pat["condition"]=="other") & (vp_pat["p_perm"]<0.05) & (vp_pat["unique_semantic"]>0)]["neuron_idx"])
    sig_either = sig_self | sig_other

    for ni in range(n_n):
        all_rows.append({
            "patient": patient_ID, "neuron_idx": ni,
            "r_cross": float(r_cross[ni]), "p_perm": float(p_perm[ni]),
            "sig_vp": ni in sig_either,
        })

    out_f = OUT_DIR / f"{patient_ID}_L{LAYER}_resid_rcross_op.pkl"
    pickle.dump({"r_cross": r_cross, "p_perm": p_perm, "sig_either": sig_either},
                open(out_f,"wb"))
    print(f"  saved {out_f.name}")

# ── aggregate ─────────────────────────────────────────────────────────────────
df = pd.DataFrame(all_rows)
agg_f = OUT_DIR / f"L{LAYER}_resid_rcross_op_all.pkl"
pickle.dump(df, open(agg_f,"wb"))

df_sig = df[df["sig_vp"]]
print(f"\nAggregated: {len(df)} total, {len(df_sig)} VP-sig, {df['patient'].nunique()} patients")

if len(df_sig) > 0:
    rc = df_sig["r_cross"].values; pp = df_sig["p_perm"].values
    pat_rc = df_sig.groupby("patient")["r_cross"].median().values
    W, p_w = stats.wilcoxon(pat_rc, alternative="greater", zero_method="wilcox")
    print(f"VP-sig: median r_cross={np.median(rc):.3f}, {100*(rc>0).mean():.1f}% positive")
    print(f"Wilcoxon r_cross>0: W={W:.0f}, p={p_w:.2e}")
    print(f"perm: {100*(pp<0.05).mean():.1f}% p_perm<0.05, median p_perm={np.median(pp):.3f}")
