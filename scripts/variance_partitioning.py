"""
Variance Partitioning — GPU Poisson ridge, training-mean null, deviance R².

For each neuron fits semantic / controls / full, plus per-family controls
(lexical, syntactic, acoustic — see FEATURE_GROUPS) and semantic+family
combos, to dissociate which nuisance family any apparent shared variance
sits with:
  semantic           : GPT-2 XL layer embeddings (N_COMPONENTS PCA)
  controls           : lexical + syntactic + acoustic (all combined)
  full               : semantic + controls
  lexical/syntactic/acoustic        : each family alone
  sem+lexical/sem+syntactic/sem+acoustic : semantic + one family

Unique semantic            = R²_full − R²_controls
Unique semantic | lexical  = R²_(sem+lex) − R²_lex   (specificity check)
Unique semantic | syntactic= R²_(sem+syn) − R²_syn
Unique semantic | acoustic = R²_(sem+aco) − R²_aco
Unique controls = R²_full − R²_semantic
Shared          = R²_sem + R²_ctrl − R²_full
All BH-FDR corrected per region × condition × comparison.

Families are additive only — no cross-family or semantic×family interaction
terms (not identifiable at this sample size, and not interpretable within a
variance-partition framework).

Significance:
  yblock : block-shuffle Y, no refit (only tests unique_semantic vs combined
           controls — fixed eta reused from real CV).
  xcirc  : circular-shift semantic features by lag>tau, refit only the model
           that touches the shifted features (full, or sem+family) — the
           untouched baseline (controls/family-only) keeps its real fit.
           Supports the per-family breakdown; yblock does not.

Usage
-----
  cd /scratch/aniluchavez/hippocampal-speaker-semantics
  python3 -u scripts/variance_partitioning.py --layer 36 --perm_type xcirc
"""

import os, sys, argparse, time, pickle, warnings
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ── CLI ───────────────────────────────────────────────────────────────────────

def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--layer",        type=int, default=36)
    p.add_argument("--n_components", type=int, default=100)
    p.add_argument("--n_perm",       type=int, default=100)
    p.add_argument("--block_size",   type=int, default=20)
    p.add_argument("--perm_type",    type=str, default="yblock",
                   choices=["yblock", "xcirc", "xshuffle"],
                   help="yblock=block-shuffle Y, no refit (existing); "
                        "xcirc=circular shift semantic features by lag>tau, "
                        "refit full model only; "
                        "xshuffle=globally permute semantic feature rows, "
                        "refit full model only")
    p.add_argument("--cv_mode",      type=str, default="block",
                   choices=["block", "shuffle"],
                   help="block=contiguous temporal folds (existing); "
                        "shuffle=random KFold, ignores temporal order")
    p.add_argument("--min_spikes",   type=int, default=None,
                   help="Minimum total spikes per neuron to include (default: 5)")
    p.add_argument("--sym_perm",     action="store_true",
                   help="Also run symmetric permutation: xcirc-shift X_ctrl (holding X_sem "
                        "fixed) to get null distribution for unique_controls. Produces "
                        "p_perm_ctrl/significant_ctrl columns. xcirc/xshuffle only.")
    p.add_argument("--n_outer",      type=int, default=5,
                   help="Number of outer CV folds (default 5; use 3 for larger test blocks)")
    p.add_argument("--fixed_lag_ms", type=float, default=0,
                   help="If >0, override tau-adaptive lag with a fixed symmetric ±lag: "
                        "draw lag uniformly from [-fixed_lag_words, -1] U [1, fixed_lag_words] "
                        "where fixed_lag_words = round(fixed_lag_ms / avg_word_dur_ms). "
                        "Implements the paper-style fixed-range null (e.g. 500 for ±500ms). "
                        "Only used with --perm_type xcirc.")
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--model",        type=str, default="gpt2-xl")
    p.add_argument("--context_tag",  type=str, default="_ctx200")
    p.add_argument("--window_type",  type=str, default="worddur")
    p.add_argument("--patient",      type=str, default=None,
                   help="Run only this patient_ID; runs all if omitted")
    return p.parse_args()

A              = _args()
LAYER          = A.layer
N_COMPONENTS   = A.n_components
N_PERM         = A.n_perm
BLOCK_SIZE     = A.block_size
PERM_TYPE      = A.perm_type
CV_MODE        = A.cv_mode
SYM_PERM       = A.sym_perm
FIXED_LAG_MS   = A.fixed_lag_ms
N_OUTER_ARG    = A.n_outer
MIN_SPIKES     = 5 if A.min_spikes is None else A.min_spikes
SEED           = A.seed
MODEL_TAG      = A.model
CONTEXT_TAG    = A.context_tag
WINDOW_TYPE    = A.window_type
PATIENT_FILTER = A.patient
CV_RNG         = np.random.default_rng(SEED + 12345)

# ── FIXED HYPERPARAMETERS ─────────────────────────────────────────────────────

ALPHAS     = np.logspace(-2, 4, 20)
N_OUTER    = N_OUTER_ARG
N_INNER    = 3
FDR_ALPHA  = 0.05
ETA_CLIP   = 20.0
LBFGS_ITER = 20

FEATURE_GROUPS = {
    "lexical":   ["log_word_freq", "word_length", "local_count", "local_rate", "surprisal"],
    "syntactic": ["dep_depth", "dep_children", "sent_position", "serial_position"],
    "acoustic":  ["f0_mean", "f0_std", "rms_mean", "spectral_flux", "speaking_rate"],
}
CONTROL_FEATURES = sum(FEATURE_GROUPS.values(), [])

# ── PATHS ─────────────────────────────────────────────────────────────────────

EMBED_DIR     = "/scratch/aniluchavez/ConvoDATAS/EmbedCache"
CONTROL_DIR   = "/scratch/aniluchavez/ConvoDATAS/ControlFeatures"
SURPRISAL_DIR = "/scratch/aniluchavez/ConvoDATAS/Surprisal"
SPIKE_ROOT    = "/scratch/aniluchavez/ConvoDATAS/SpikeWindows"
_perm_suffix  = "" if PERM_TYPE == "yblock" else f"_{PERM_TYPE}"
_cv_suffix    = "" if CV_MODE == "block" else f"_cv{CV_MODE}"
_sym_suffix   = "_symperm" if SYM_PERM else ""
_fixlag_suffix = f"_fixlag{int(FIXED_LAG_MS)}ms" if FIXED_LAG_MS > 0 else ""
_nfold_suffix  = f"_cv{N_OUTER}fold" if N_OUTER != 5 else ""
OUT_DIR       = os.path.join("/scratch/aniluchavez/ConvoDATAS/VPResults",
                              f"{MODEL_TAG}{CONTEXT_TAG}_{WINDOW_TYPE}{_perm_suffix}{_cv_suffix}{_sym_suffix}{_fixlag_suffix}{_nfold_suffix}",
                              f"pc{N_COMPONENTS}")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"layer={LAYER}  PCs={N_COMPONENTS}  perms={N_PERM}  perm_type={PERM_TYPE}  cv_mode={CV_MODE}  "
      f"model={MODEL_TAG}{CONTEXT_TAG}  window={WINDOW_TYPE}  dev={DEVICE}",
      flush=True)
if DEVICE.type == "cuda":
    _p = torch.cuda.get_device_properties(0)
    print(f"  {_p.name}  {_p.total_memory//1024**2} MB", flush=True)

# ── PATIENTS ──────────────────────────────────────────────────────────────────

PATIENTS = [
    {"patient_ID": "PTYEU_task147", "patient": "ptYEU_task147",
     "region_ranges": {"hippocampus": [(1,16),(25,40)], "ACC": [(17,24),(41,48)]}},
    {"patient_ID": "PTYFF_task17",  "patient": "ptYFF_task17",
     "region_ranges": {"hippocampus": [(9,16),(25,40)], "ACC": [(17,24),(41,48)]}},
    {"patient_ID": "PTYFG_task18",  "patient": "ptYFG_task18",
     "region_ranges": {"hippocampus": [(9,16)],         "ACC": [(25,56)]}},
    {"patient_ID": "PTYFI_task81",  "patient": "ptYFI_task81",
     "region_ranges": {"hippocampus": [(1,8),(25,40)],  "ACC": [(9,16)]}},
    {"patient_ID": "PTYFA_task25",  "patient": "ptYFA_task25",
     "region_ranges": {"hippocampus": [(1,16),(25,40)], "ACC": [(17,24)]}},
    {"patient_ID": "PTYFK_task40",  "patient": "ptYFK_task40",
     "region_ranges": {"hippocampus": [(1,16),(25,40)], "ACC": [(49,56)]}},
    {"patient_ID": "PTYEY_task86",  "patient": "ptYEY_task86",
     "region_ranges": {"hippocampus": [(1,16)]}},
    {"patient_ID": "PTYEV_task37",  "patient": "ptYEV_task37",
     "region_ranges": {"hippocampus": [(1,16),(25,40)], "ACC": [(17,24),(41,48)]}},
    {"patient_ID": "PTYEZ_task60",  "patient": "ptYEZ_task60",
     "region_ranges": {"hippocampus": [(1,16)],         "ACC": [(17,24)]}},
    {"patient_ID": "PTYFC_task28",  "patient": "ptYFC_task28",
     "region_ranges": {"hippocampus": [(1,8),(33,48)],  "ACC": [(17,32),(49,64)]}},
    {"patient_ID": "PTYFM_task104", "patient": "ptYFM_task104",
     "region_ranges": {"hippocampus": [(33,48)]}},
    {"patient_ID": "PTYFP_task88",  "patient": "ptYFP_task88",
     "region_ranges": {"hippocampus": [(17,24),(25,32),(49,56),(57,64)]}},
    {"patient_ID": "PTYFR_task91",  "patient": "ptYFR_task91",
     "region_ranges": {"hippocampus": [(1,16),(41,56)]}},
    {"patient_ID": "PTYFS_task95",  "patient": "ptYFS_task95",
     "region_ranges": {"hippocampus": [(1,24)]}},
    {"patient_ID": "PTYFU_task224", "patient": "ptYFU_task224",
     "region_ranges": {"hippocampus": [(17,32),(41,56)]}},
]

# ── GPU / MATH UTILITIES ──────────────────────────────────────────────────────

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


def gpu_fit(X_t, Y_t, alpha, offset_t=None):
    """offset_t : (1, n, 1) log-exposure (log word duration), or None — added to
    eta before the link, coefficient fixed at 1 (standard Poisson rate offset)."""
    k, n, p = X_t.shape
    m  = Y_t.shape[2]
    n_eff = float(Y_t.sum().clamp(min=1).item())
    Xi = torch.cat([torch.ones(k, n, 1, device=DEVICE), X_t], dim=-1)
    beta = torch.zeros(k, m, p + 1, device=DEVICE, requires_grad=True)
    opt  = torch.optim.LBFGS([beta], max_iter=LBFGS_ITER, line_search_fn='strong_wolfe')
    if isinstance(alpha, torch.Tensor):
        a = alpha.to(DEVICE).view(k, 1, 1)
    else:
        a = float(alpha)
    def closure():
        opt.zero_grad()
        eta  = torch.bmm(Xi, beta.transpose(1, 2))
        if offset_t is not None:
            eta = eta + offset_t
        eta  = torch.clamp(eta, -ETA_CLIP, ETA_CLIP)
        loss = ((torch.exp(eta) - Y_t * eta).sum()
                + (a * beta[:, :, 1:].pow(2)).sum()) / n_eff
        loss.backward()
        return loss
    opt.step(closure)
    b = beta.detach()
    return b[:, :, 1:], b[:, :, :1]


def block_splits(n, k):
    """Outer/inner CV fold generator. CV_MODE='block' (default): contiguous
    temporal folds. CV_MODE='shuffle': random KFold, ignores temporal order
    entirely (kept for explicit methodology comparison, not the default)."""
    if CV_MODE == "block":
        b = n // k
        for i in range(k):
            s, e = i * b, (i * b + b if i < k - 1 else n)
            mask = np.zeros(n, bool); mask[s:e] = True
            yield ~mask, mask
    else:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=k, shuffle=True, random_state=int(CV_RNG.integers(0, 2**31)))
        for tr_idx, te_idx in kf.split(np.arange(n)):
            tr_m = np.zeros(n, bool); tr_m[tr_idx] = True
            te_m = np.zeros(n, bool); te_m[te_idx] = True
            yield tr_m, te_m


def block_shuffle(Y, block_size, rng):
    idx = np.arange(Y.shape[0])
    for s in range(0, len(idx), block_size):
        idx[s : s + block_size] = rng.permutation(idx[s : s + block_size])
    return Y[idx]


def fdr_bh(pvals):
    n = len(pvals)
    if n == 0:
        return np.array([]), np.array([], dtype=bool)
    order = np.argsort(pvals)
    adj   = np.minimum(1.0, pvals[order] * n / np.arange(1, n + 1))
    adj   = np.minimum.accumulate(adj[::-1])[::-1]
    out   = np.empty(n); out[order] = adj
    return out, out < FDR_ALPHA

# ── DATA LOADING ──────────────────────────────────────────────────────────────

def find_spike_dir(patient):
    d = os.path.join(SPIKE_ROOT, f"output_{patient}_english_only_{WINDOW_TYPE}")
    return d if os.path.isdir(d) else None


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
    dir_membership = {
        col: np.array([_nn(v) for v in tx[col]], dtype=bool) for col in spk_cols
    }
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


def load_word_dur_ms(spike_dir):
    """Word acoustic duration in ms, transcript order — Poisson log-exposure
    offset for the worddur window (each word's spike count is collected over
    its own variable-length window, so longer words naturally get more spikes
    independent of any feature; the offset removes that purely mechanical effect)."""
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
    col_means  = np.where(np.isnan(col_means), 0.0, col_means)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    return X, cols

# ── CORE MODEL FIT ────────────────────────────────────────────────────────────

def run_cv_models(X_list, Y_v, offset=None):
    """
    Nested CV for an arbitrary list of design matrices in one outer-fold pass.
    Y is transferred to GPU once per fold (not once per model).
    offset : (n_w,) log word duration, or None — same offset reused for every
              model in X_list (it's a property of the row/word, not the model).
    Returns list of (ll, fold_store), one per entry in X_list.
    """
    n_w = Y_v.shape[0]
    n_m = Y_v.shape[1]
    N_A = len(ALPHAS)
    alpha_t = torch.tensor(ALPHAS, dtype=torch.float32)

    ll_arr    = [np.zeros(n_m, np.float64) for _ in X_list]
    ll_arr_tr = [np.zeros(n_m, np.float64) for _ in X_list]
    fs_arr    = [[] for _ in X_list]

    for tr_m, te_m in block_splits(n_w, N_OUTER):
        n_tr = int(tr_m.sum())
        Y_tr_t = torch.tensor(Y_v[tr_m], device=DEVICE)
        Y_te_t = torch.tensor(Y_v[te_m], device=DEVICE)
        off_tr_np = offset[tr_m] if offset is not None else None
        off_te_t  = (torch.tensor(offset[te_m], device=DEVICE).view(1, -1, 1)
                     if offset is not None else None)

        for mi, X_sc in enumerate(X_list):
            p    = X_sc.shape[1]
            X_tr = X_sc[tr_m]

            # Inner CV — batch over all alphas
            inner_ll = np.zeros((N_A, n_m), np.float64)
            for itr_m, iva_m in block_splits(n_tr, N_INNER):
                Xii = torch.tensor(X_tr[itr_m], device=DEVICE).unsqueeze(0).expand(N_A,-1,-1).contiguous()
                Yii = Y_tr_t[itr_m].unsqueeze(0).expand(N_A,-1,-1).contiguous()
                off_itr_t = (torch.tensor(off_tr_np[itr_m], device=DEVICE).view(1, -1, 1)
                             if off_tr_np is not None else None)
                bf, bi = gpu_fit(Xii, Yii, alpha_t, offset_t=off_itr_t)
                with torch.no_grad():
                    Xiv = torch.tensor(X_tr[iva_m], device=DEVICE).unsqueeze(0).expand(N_A,-1,-1).contiguous()
                    Yiv = Y_tr_t[iva_m].unsqueeze(0).expand(N_A,-1,-1).contiguous()
                    eta_iv = torch.bmm(Xiv, bf.transpose(1,2)) + bi.transpose(1,2)
                    if off_tr_np is not None:
                        off_iva_t = torch.tensor(off_tr_np[iva_m], device=DEVICE).view(1, -1, 1)
                        eta_iv = eta_iv + off_iva_t
                    eta_iv = torch.clamp(eta_iv, -ETA_CLIP, ETA_CLIP)
                    inner_ll += (Yiv * eta_iv - torch.exp(eta_iv)).sum(1).cpu().numpy()
            best_ai = np.argmax(inner_ll, axis=0)

            # Outer fit
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
                eta_te_t = torch.bmm(X_te_t, beta_f.transpose(1,2)) + bias_f.transpose(1,2)
                if off_te_t is not None:
                    eta_te_t = eta_te_t + off_te_t
                eta_te_t = torch.clamp(eta_te_t, -ETA_CLIP, ETA_CLIP)
                eta_te  = eta_te_t[0].cpu().numpy()
                ll_fold = (Y_te_t * eta_te_t[0] - torch.exp(eta_te_t[0])).sum(0).cpu().numpy()

                # Train LL: evaluate fitted model on training data
                eta_tr_t = torch.bmm(X_tr_t, beta_f.transpose(1,2)) + bias_f.transpose(1,2)
                if off_tr_t is not None:
                    eta_tr_t = eta_tr_t + off_tr_t
                eta_tr_t = torch.clamp(eta_tr_t, -ETA_CLIP, ETA_CLIP)
                ll_fold_tr = (Y_tr_t * eta_tr_t[0] - torch.exp(eta_tr_t[0])).sum(0).cpu().numpy()

            ll_arr[mi]    += ll_fold
            ll_arr_tr[mi] += ll_fold_tr
            fs_arr[mi].append({"eta_te": eta_te, "sum_exp": np.exp(eta_te).sum(0)})
            del X_tr_t, X_te_t, beta_f, bias_f, eta_te_t, eta_tr_t

        del Y_tr_t, Y_te_t
        torch.cuda.empty_cache()

    return [(ll_arr[i], ll_arr_tr[i], fs_arr[i]) for i in range(len(X_list))]


def fit_one_model_ll(X_sc, Y_v, offset=None):
    """Same nested-CV + per-neuron alpha selection as one model slot in
    run_cv_three, but for a single design matrix — used to refit the 'full'
    model under xcirc permutation (semantic & controls models are untouched
    by shuffling X_sem, so only this needs to be redone per permutation).
    offset : (n_w,) log word duration, or None — unaffected by the X_sem shift,
              since it's a property of the word/row, not of the semantic features."""
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
        off_tr_np = offset[tr_m] if offset is not None else None
        off_te_t  = (torch.tensor(offset[te_m], device=DEVICE).view(1, -1, 1)
                     if offset is not None else None)

        inner_ll = np.zeros((N_A, n_m), np.float64)
        for itr_m, iva_m in block_splits(n_tr, N_INNER):
            Xii = torch.tensor(X_tr[itr_m], device=DEVICE).unsqueeze(0).expand(N_A,-1,-1).contiguous()
            Yii = Y_tr_t[itr_m].unsqueeze(0).expand(N_A,-1,-1).contiguous()
            off_itr_t = (torch.tensor(off_tr_np[itr_m], device=DEVICE).view(1, -1, 1)
                         if off_tr_np is not None else None)
            bf, bi = gpu_fit(Xii, Yii, alpha_t, offset_t=off_itr_t)
            with torch.no_grad():
                Xiv = torch.tensor(X_tr[iva_m], device=DEVICE).unsqueeze(0).expand(N_A,-1,-1).contiguous()
                Yiv = Y_tr_t[iva_m].unsqueeze(0).expand(N_A,-1,-1).contiguous()
                eta_iv = torch.bmm(Xiv, bf.transpose(1,2)) + bi.transpose(1,2)
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
            eta_te_t = torch.bmm(X_te_t, beta_f.transpose(1,2)) + bias_f.transpose(1,2)
            if off_te_t is not None:
                eta_te_t = eta_te_t + off_te_t
            eta_te_t = torch.clamp(eta_te_t, -ETA_CLIP, ETA_CLIP)
            ll_fold = (Y_te_t * eta_te_t[0] - torch.exp(eta_te_t[0])).sum(0).cpu().numpy()
        ll += ll_fold
        del X_tr_t, X_te_t, beta_f, bias_f, eta_te_t, Y_tr_t, Y_te_t
        torch.cuda.empty_cache()

    return ll

# ── MAIN ──────────────────────────────────────────────────────────────────────

all_rows = []

for cfg in PATIENTS:
    patient_ID = cfg["patient_ID"]
    patient    = cfg["patient"]

    if PATIENT_FILTER and patient_ID != PATIENT_FILTER:
        continue

    out_path = os.path.join(OUT_DIR, f"{patient_ID}_L{LAYER:02d}_VP.pkl")
    if os.path.exists(out_path):
        print(f"[SKIP] {patient_ID}", flush=True)
        with open(out_path, "rb") as f:
            all_rows.append(pickle.load(f))
        continue

    print(f"\n{'='*60}\n  {patient_ID}", flush=True)

    npy_path  = os.path.join(EMBED_DIR, f"{patient_ID}_{MODEL_TAG}{CONTEXT_TAG}_word_emb_layers.npy")
    ctrl_path = os.path.join(CONTROL_DIR, f"{patient_ID}_control_features.csv")
    spike_dir = find_spike_dir(patient)

    missing = []
    if not os.path.exists(npy_path):  missing.append("embeddings")
    if not os.path.exists(ctrl_path): missing.append("control features")
    if spike_dir is None:             missing.append("spike dir")
    if missing:
        print(f"  SKIP — missing: {', '.join(missing)}"); continue

    try:
        spk_assignment, mask_self, mask_other, dir_membership = \
            load_speaker_assignment(spike_dir)
        X_ctrl_raw, ctrl_cols = load_control_features(patient_ID)
    except Exception as e:
        print(f"  SKIP — {e}"); continue

    all_word_dur = load_word_dur_ms(spike_dir) if WINDOW_TYPE == "worddur" else None
    if WINDOW_TYPE == "worddur" and all_word_dur is None:
        print("  SKIP — word_dur not found for worddur mode"); continue

    # PCA on all words for this patient
    t0 = time.time()
    raw_layer = np.load(npy_path, mmap_mode='r')[LAYER].astype(np.float32)
    X_pca = gpu_pca(raw_layer, N_COMPONENTS)
    del raw_layer
    print(f"  PCA done  {X_pca.shape}  ({time.time()-t0:.1f}s)", flush=True)

    n_words = X_pca.shape[0]
    if X_ctrl_raw.shape[0] != n_words or len(spk_assignment) != n_words:
        print(f"  SKIP — row mismatch"); continue
    if all_word_dur is not None and len(all_word_dur) != n_words:
        print(f"  SKIP — word_dur row mismatch"); continue

    patient_rows = []

    for region in cfg["region_ranges"]:
        cond_data = {}
        raw_n_neurons = None
        for cond, mask in [("self", mask_self), ("other", mask_other)]:
            Y_mat = load_condition_ordered(spike_dir, cond, region, spk_assignment, dir_membership)
            if Y_mat is None or Y_mat.ndim < 2:
                cond_data[cond] = None; continue
            if Y_mat.shape[0] != int(mask.sum()):
                print(f"  {region}/{cond}: row mismatch — skip")
                cond_data[cond] = None; continue
            if raw_n_neurons is None:
                raw_n_neurons = Y_mat.shape[1]
            elif Y_mat.shape[1] != raw_n_neurons:
                cond_data[cond] = None; continue
            valid = ~np.isnan(Y_mat).any(axis=1)
            cond_data[cond] = (Y_mat[valid].astype(np.float32), mask, valid)

        available = {c: d for c, d in cond_data.items() if d is not None}
        if not available: continue

        # Shared neuron filter: pass MIN_SPIKES in ALL conditions
        spike_ok = None
        for _, (Y_v_raw, _, _) in available.items():
            ok = Y_v_raw.sum(0) >= MIN_SPIKES
            spike_ok = ok if spike_ok is None else (spike_ok & ok)
        n_m = int(spike_ok.sum())
        if n_m == 0:
            print(f"  {region}: 0 neurons pass filter"); continue

        for cond, (Y_v_raw, mask, valid) in available.items():
            Y_v = Y_v_raw[:, spike_ok]
            n_w = Y_v.shape[0]
            print(f"  {region}/{cond}: {n_w}w × {n_m}n", flush=True)

            if n_w // N_OUTER < N_COMPONENTS + 2:
                print("    skip — too few samples"); continue

            tau = compute_tau_embed(X_pca[mask][valid])
            print(f"    tau={tau:.0f}w  blk={n_w//N_OUTER}w", flush=True)

            # Condition-masked and valid-filtered feature matrices
            X_ctrl_masked = X_ctrl_raw[mask][valid]
            X_sem_sc  = StandardScaler().fit_transform(X_pca[mask][valid]).astype(np.float32)
            X_ctrl_sc = StandardScaler().fit_transform(X_ctrl_masked).astype(np.float32)

            if all_word_dur is not None:
                dur_v    = all_word_dur[mask][valid].astype(np.float64).clip(1e-3, None)
                offset_v = np.log(dur_v).astype(np.float32)
            else:
                dur_v, offset_v = None, None

            group_idx = {
                g: [ctrl_cols.index(c) for c in feats if c in ctrl_cols]
                for g, feats in FEATURE_GROUPS.items()
            }
            X_group_sc = {}
            for g, idx in group_idx.items():
                if idx:
                    X_group_sc[g] = StandardScaler().fit_transform(
                        X_ctrl_masked[:, idx]).astype(np.float32)
                else:
                    X_group_sc[g] = np.zeros((n_w, 0), np.float32)

            X_full_sc = np.hstack([X_sem_sc, X_ctrl_sc]).astype(np.float32)
            X_sem_group_sc = {
                g: np.hstack([X_sem_sc, X_group_sc[g]]).astype(np.float32)
                for g in FEATURE_GROUPS
            }

            # ── Shared null/sat (depends only on Y, same for all 3 models) ──
            ll_null    = np.zeros(n_m, np.float64)
            ll_sat     = np.zeros(n_m, np.float64)
            ll_null_tr = np.zeros(n_m, np.float64)
            ll_sat_tr  = np.zeros(n_m, np.float64)
            fold_Y  = []
            for tr_m, te_m in block_splits(n_w, N_OUTER):
                if dur_v is not None:
                    # intercept-only Poisson MLE with fixed log(duration) offset:
                    # rate = sum(Y_train) / sum(duration_train); mu_test_i = rate * duration_test_i
                    rate_tr = Y_v[tr_m].sum(0) / dur_v[tr_m].sum().clip(1e-10)
                    mu_te   = np.outer(dur_v[te_m], rate_tr).clip(1e-10)
                    ll_null_fold = (Y_v[te_m] * np.log(mu_te) - mu_te).sum(0)
                    mu_tr = np.outer(dur_v[tr_m], rate_tr).clip(1e-10)
                    ll_null_tr_fold = (Y_v[tr_m] * np.log(mu_tr) - mu_tr).sum(0)
                else:
                    mean_tr = Y_v[tr_m].mean(0).clip(1e-10)
                    ll_null_fold = (Y_v[te_m] * np.log(mean_tr) - mean_tr).sum(0)
                    ll_null_tr_fold = (Y_v[tr_m] * np.log(mean_tr) - mean_tr).sum(0)
                ll_null    += ll_null_fold
                ll_null_tr += ll_null_tr_fold
                Y_te_np = Y_v[te_m]
                Y_tr_np = Y_v[tr_m]
                ll_sat_fold = (Y_te_np * np.log(Y_te_np.clip(1e-10)) - Y_te_np).sum(0)
                ll_sat_fold = np.where(Y_te_np.sum(0) > 0, ll_sat_fold, 0.0)
                ll_sat_tr_fold = (Y_tr_np * np.log(Y_tr_np.clip(1e-10)) - Y_tr_np).sum(0)
                ll_sat_tr_fold = np.where(Y_tr_np.sum(0) > 0, ll_sat_tr_fold, 0.0)
                ll_sat    += ll_sat_fold
                ll_sat_tr += ll_sat_tr_fold
                fold_Y.append({"Y_te": Y_te_np.copy(), "ll_null_fold": ll_null_fold})
            d_null    = ll_sat    - ll_null     # (n_m,) always ≥ 0
            d_null_tr = ll_sat_tr - ll_null_tr  # train version

            # ── Fit semantic / controls / full + per-family models ───────────
            # Order: semantic, controls(all), full, lexical, syntactic, acoustic,
            #        sem+lexical, sem+syntactic, sem+acoustic
            group_names = list(FEATURE_GROUPS.keys())
            # Leave-one-control-family-out: sem + all ctrl except group g
            X_noctrl = {
                g: np.hstack(
                    [X_sem_sc] + [X_group_sc[h] for h in group_names if h != g]
                ).astype(np.float32)
                for g in group_names
            }
            X_list = ([X_sem_sc, X_ctrl_sc, X_full_sc]
                      + [X_group_sc[g] for g in group_names]
                      + [X_sem_group_sc[g] for g in group_names]
                      + [X_noctrl[g] for g in group_names])

            t_cv = time.time()
            results = run_cv_models(X_list, Y_v, offset=offset_v)
            (ll_sem, ll_sem_tr, folds_sem), (ll_ctrl, ll_ctrl_tr, folds_ctrl), (ll_full, ll_full_tr, folds_full) = results[:3]
            ll_group      = {g: results[3+i][0] for i, g in enumerate(group_names)}
            ll_semgroup   = {g: results[3+len(group_names)+i][0] for i, g in enumerate(group_names)}
            n_base = 3 + 2 * len(group_names)
            ll_noctrl = {g: results[n_base+i][0] for i, g in enumerate(group_names)}
            print(f"    CV done {time.time()-t_cv:.1f}s", flush=True)

            with np.errstate(divide="ignore", invalid="ignore"):
                r2_sem  = np.where(d_null > 0.1, (ll_sem  - ll_null) / d_null, np.nan)
                r2_ctrl = np.where(d_null > 0.1, (ll_ctrl - ll_null) / d_null, np.nan)
                r2_full = np.where(d_null > 0.1, (ll_full - ll_null) / d_null, np.nan)
                r2_group    = {g: np.where(d_null > 0.1, (ll_group[g]    - ll_null) / d_null, np.nan)
                               for g in group_names}
                r2_semgroup = {g: np.where(d_null > 0.1, (ll_semgroup[g] - ll_null) / d_null, np.nan)
                               for g in group_names}
                # Train R²
                r2_sem_tr  = np.where(d_null_tr > 0.1, (ll_sem_tr  - ll_null_tr) / d_null_tr, np.nan)
                r2_ctrl_tr = np.where(d_null_tr > 0.1, (ll_ctrl_tr - ll_null_tr) / d_null_tr, np.nan)
                r2_full_tr = np.where(d_null_tr > 0.1, (ll_full_tr - ll_null_tr) / d_null_tr, np.nan)
                # Leave-one-control-family-out R² (test folds)
                r2_noctrl = {g: np.where(d_null > 0.1, (ll_noctrl[g] - ll_null) / d_null, np.nan)
                             for g in group_names}

            unique_sem  = r2_full - r2_ctrl
            unique_ctrl = r2_full - r2_sem
            shared      = r2_sem + r2_ctrl - r2_full
            # unique semantic controlling for ONLY one family at a time —
            # tests whether the semantic effect is specific to (not absorbed
            # by) any single nuisance family.
            unique_sem_group = {g: r2_semgroup[g] - r2_group[g] for g in group_names}
            # unique_<family> = R²_full − R²_{full minus that control family}
            unique_ctrl_group = {g: r2_full - r2_noctrl[g] for g in group_names}

            # ── Score permutation for unique_semantic ────────────────────────
            # yblock: block-shuffle Y, no refit (d_null/r2_ctrl invariant).
            #         Only tests unique_sem (vs combined controls) — the
            #         per-family breakdown is xcirc-only (see below).
            # xcirc : circular-shift X_sem by lag>tau, refit the full model
            #         AND each sem+family model (controls/family-only models
            #         don't touch the shifted features, so their R² is fixed
            #         at the real fit).
            rng = np.random.default_rng(SEED)
            perm_usem = np.full((N_PERM, n_m), np.nan)
            perm_usem_group = {g: np.full((N_PERM, n_m), np.nan) for g in group_names}
            t_perm = time.time()

            if PERM_TYPE == "yblock":
                cse_full = sum(fd["sum_exp"] for fd in folds_full)
                cse_ctrl = sum(fd["sum_exp"] for fd in folds_ctrl)
                for pi in range(N_PERM):
                    ct_full = np.zeros(n_m, np.float64)
                    ct_ctrl = np.zeros(n_m, np.float64)
                    for fd_f, fd_c, fd_y in zip(folds_full, folds_ctrl, fold_Y):
                        Y_shuf = block_shuffle(fd_y["Y_te"], BLOCK_SIZE, rng)
                        ct_full += (Y_shuf * fd_f["eta_te"]).sum(0)
                        ct_ctrl += (Y_shuf * fd_c["eta_te"]).sum(0)
                    ll_full_p = ct_full - cse_full
                    ll_ctrl_p = ct_ctrl - cse_ctrl
                    with np.errstate(divide="ignore", invalid="ignore"):
                        r2f_p = np.where(d_null > 0.1, (ll_full_p - ll_null) / d_null, np.nan)
                        r2c_p = np.where(d_null > 0.1, (ll_ctrl_p - ll_null) / d_null, np.nan)
                    perm_usem[pi] = r2f_p - r2c_p
                print("    note: family-level null is xcirc-only — skipping", flush=True)

            else:  # xcirc or xshuffle
                if FIXED_LAG_MS > 0 and PERM_TYPE == "xcirc":
                    avg_word_dur_ms = 1000.0 * np.mean(dur_v) if dur_v is not None else 250.0
                    max_lag_w = max(1, round(FIXED_LAG_MS / avg_word_dur_ms))
                    _lag_pool = np.concatenate([np.arange(-max_lag_w, 0), np.arange(1, max_lag_w + 1)])
                    print(f"    fixed_lag_ms={FIXED_LAG_MS:.0f} → ±{max_lag_w}w (avg_word={avg_word_dur_ms:.0f}ms)", flush=True)
                else:
                    min_lag = max(int(np.ceil(tau)), n_w // (N_OUTER * 2))
                    min_lag = min(min_lag, n_w // 2 - 1)
                    _lag_pool = None
                for pi in range(N_PERM):
                    if PERM_TYPE == "xshuffle":
                        X_sem_perm = X_sem_sc[rng.permutation(n_w)]
                    else:
                        if _lag_pool is not None:
                            lag = int(rng.choice(_lag_pool))
                        else:
                            lag = int(rng.integers(min_lag, n_w - min_lag))
                        X_sem_perm  = np.roll(X_sem_sc, lag, axis=0)
                    X_full_perm = np.hstack([X_sem_perm, X_ctrl_sc]).astype(np.float32)

                    ll_full_p = fit_one_model_ll(X_full_perm, Y_v, offset=offset_v)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        r2f_p = np.where(d_null > 0.1, (ll_full_p - ll_null) / d_null, np.nan)
                    perm_usem[pi] = r2f_p - r2_ctrl

                    for g in group_names:
                        X_semg_perm = np.hstack([X_sem_perm, X_group_sc[g]]).astype(np.float32)
                        ll_g_p = fit_one_model_ll(X_semg_perm, Y_v, offset=offset_v)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            r2g_p = np.where(d_null > 0.1, (ll_g_p - ll_null) / d_null, np.nan)
                        perm_usem_group[g][pi] = r2g_p - r2_group[g]

                    if (pi + 1) % 50 == 0:
                        print(f"    perm {pi+1}/{N_PERM}  {time.time()-t_perm:.0f}s", flush=True)
            print(f"    perms done {time.time()-t_perm:.1f}s", flush=True)

            # ── Symmetric permutation for unique_controls (optional) ──────────
            # Xcirc-shift X_ctrl (hold X_sem fixed) → null for unique_ctrl.
            # unique_ctrl_perm = r2_full_perm - r2_sem  (r2_sem unchanged)
            perm_uctrl = np.full((N_PERM, n_m), np.nan)
            if SYM_PERM and PERM_TYPE in ("xcirc", "xshuffle"):
                tau_ctrl = compute_tau_embed(X_ctrl_sc)
                if FIXED_LAG_MS > 0 and PERM_TYPE == "xcirc":
                    _lag_pool_c = _lag_pool  # reuse same fixed pool computed above
                else:
                    min_lag_c = max(int(np.ceil(tau_ctrl)), n_w // (N_OUTER * 2))
                    min_lag_c = min(min_lag_c, n_w // 2 - 1)
                    _lag_pool_c = None
                t_sym = time.time()
                for pi in range(N_PERM):
                    if PERM_TYPE == "xshuffle":
                        X_ctrl_perm = X_ctrl_sc[rng.permutation(n_w)]
                    else:
                        lag = int(rng.choice(_lag_pool_c)) if _lag_pool_c is not None else int(rng.integers(min_lag_c, n_w - min_lag_c))
                        X_ctrl_perm = np.roll(X_ctrl_sc, lag, axis=0)
                    X_full_cp = np.hstack([X_sem_sc, X_ctrl_perm]).astype(np.float32)
                    ll_full_cp = fit_one_model_ll(X_full_cp, Y_v, offset=offset_v)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        r2f_cp = np.where(d_null > 0.1, (ll_full_cp - ll_null) / d_null, np.nan)
                    perm_uctrl[pi] = r2f_cp - r2_sem
                    if (pi + 1) % 50 == 0:
                        print(f"    sym perm {pi+1}/{N_PERM}  {time.time()-t_sym:.0f}s", flush=True)
                print(f"    sym perms done {time.time()-t_sym:.1f}s", flush=True)

            # Per-neuron p-values (one-sided: real unique_sem > perm)
            def _pvals_and_sig(real, perm):
                pv = np.array([
                    np.nanmean(perm[:, m] >= real[m]) if not np.isnan(real[m]) else 1.0
                    for m in range(n_m)
                ])
                pf, sf = fdr_bh(pv)
                sig = sf & (real > 0)
                return pv, pf, sig

            p_vals, p_fdr, significant = _pvals_and_sig(unique_sem, perm_usem)

            if SYM_PERM and PERM_TYPE in ("xcirc", "xshuffle"):
                p_vals_ctrl, p_fdr_ctrl, significant_ctrl = _pvals_and_sig(unique_ctrl, perm_uctrl)
            else:
                p_vals_ctrl   = np.full(n_m, np.nan)
                p_fdr_ctrl    = np.full(n_m, np.nan)
                significant_ctrl = np.zeros(n_m, bool)

            group_stats = {}
            for g in group_names:
                if PERM_TYPE in ("xcirc", "xshuffle"):
                    gp_vals, gp_fdr, gsig = _pvals_and_sig(unique_sem_group[g], perm_usem_group[g])
                else:
                    n_ = len(unique_sem_group[g])
                    gp_vals = np.full(n_, np.nan); gp_fdr = np.full(n_, np.nan)
                    gsig = np.zeros(n_, dtype=bool)
                group_stats[g] = (gp_vals, gp_fdr, gsig)

            n_sig = significant.sum()
            med_usem = np.nanmedian(unique_sem[significant]) if n_sig > 0 else np.nan
            print(f"    sig unique_sem: {n_sig}/{n_m}  "
                  f"({100*n_sig/n_m:.1f}%)  "
                  f"median={med_usem:.4f}" if n_sig > 0 else
                  f"    sig unique_sem: 0/{n_m}", flush=True)
            print(f"    median R²: sem={np.nanmedian(r2_sem):.4f}  "
                  f"ctrl={np.nanmedian(r2_ctrl):.4f}  "
                  f"full={np.nanmedian(r2_full):.4f}", flush=True)
            for g in group_names:
                gsig = group_stats[g][2]
                print(f"    sig unique_sem|{g}: {gsig.sum()}/{n_m}  "
                      f"({100*gsig.sum()/n_m:.1f}%)  r2_{g}={np.nanmedian(r2_group[g]):.4f}",
                      flush=True)

            for m in range(n_m):
                row = {
                    "patient":        patient_ID,
                    "region":         region,
                    "condition":      cond,
                    "neuron_idx":     int(m),
                    "layer":          LAYER,
                    "r2_semantic":    float(r2_sem[m])   if not np.isnan(r2_sem[m])  else np.nan,
                    "r2_controls":    float(r2_ctrl[m])  if not np.isnan(r2_ctrl[m]) else np.nan,
                    "r2_full":        float(r2_full[m])  if not np.isnan(r2_full[m]) else np.nan,
                    "unique_semantic":float(unique_sem[m])  if not np.isnan(unique_sem[m])  else np.nan,
                    "unique_controls":float(unique_ctrl[m]) if not np.isnan(unique_ctrl[m]) else np.nan,
                    "shared":         float(shared[m])   if not np.isnan(shared[m])  else np.nan,
                    "p_perm":         float(p_vals[m]),
                    "p_fdr":          float(p_fdr[m]),
                    "significant":    bool(significant[m]),
                    "p_perm_ctrl":    float(p_vals_ctrl[m]),
                    "p_fdr_ctrl":     float(p_fdr_ctrl[m]),
                    "significant_ctrl": bool(significant_ctrl[m]),
                    "n_spikes":       int(Y_v[:, m].sum()),
                    "r2_semantic_train":  float(r2_sem_tr[m])  if not np.isnan(r2_sem_tr[m])  else np.nan,
                    "r2_controls_train":  float(r2_ctrl_tr[m]) if not np.isnan(r2_ctrl_tr[m]) else np.nan,
                    "r2_full_train":      float(r2_full_tr[m]) if not np.isnan(r2_full_tr[m]) else np.nan,
                }
                for g in group_names:
                    gp_vals, gp_fdr, gsig = group_stats[g]
                    row[f"r2_{g}"]              = float(r2_group[g][m])    if not np.isnan(r2_group[g][m])    else np.nan
                    row[f"unique_semantic_{g}"] = float(unique_sem_group[g][m]) if not np.isnan(unique_sem_group[g][m]) else np.nan
                    row[f"unique_{g}"]          = float(unique_ctrl_group[g][m]) if not np.isnan(unique_ctrl_group[g][m]) else np.nan
                    row[f"p_perm_{g}"]          = float(gp_vals[m])
                    row[f"p_fdr_{g}"]           = float(gp_fdr[m])
                    row[f"significant_{g}"]     = bool(gsig[m])
                patient_rows.append(row)

    if patient_rows:
        df_p = pd.DataFrame(patient_rows)
        with open(out_path, "wb") as f:
            pickle.dump(df_p, f)
        print(f"  Saved → {out_path}", flush=True)
        all_rows.append(df_p)

    torch.cuda.empty_cache()

# ── AGGREGATE + FDR ───────────────────────────────────────────────────────────

if not all_rows:
    print("No results."); sys.exit(0)

data     = pd.concat(all_rows, ignore_index=True)
agg_path = os.path.join(OUT_DIR, f"L{LAYER:02d}_VP_all.pkl")
with open(agg_path, "wb") as f:
    pickle.dump(data, f)
print(f"\nAggregated → {agg_path}")

print("\n=== SUMMARY ===")
for region in data["region"].unique():
    for cond in ["self", "other"]:
        sub = data[(data["region"] == region) & (data["condition"] == cond)]
        if sub.empty: continue
        n_sig = sub["significant"].sum()
        n_tot = len(sub)
        med   = sub.loc[sub["significant"], "unique_semantic"].median() if n_sig else np.nan
        print(f"  {region:12s} {cond:5s}  {n_sig:3d}/{n_tot}  "
              f"({100*n_sig/n_tot:.1f}%)  median_usem={med:.4f}" if n_sig
              else f"  {region:12s} {cond:5s}    0/{n_tot}  (0%)")

print("\nDone.")
