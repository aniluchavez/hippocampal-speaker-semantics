"""
Poisson ridge semantic encoding GLM — collaborator-matched pipeline.

Pipeline matches collaborator notebook (word_level_duration_cv_all_n.ipynb):
  - Per-fold PCA: standardise → PCA → re-standardise, all fit on train only
  - McFadden's pseudo-R²: 1 - LL_model / LL_null  (complete Poisson LL with log-factorial)
  - X-permutation significance: shuffle embedding matrix globally, refit model from scratch
    per permutation per fold; p-value = fraction of permutations where mean(LL_perm) ≥ mean(LL_real)
  - 200-iter L-BFGS for real fits, 100-iter for permutation fits
  - 5-fold temporal inner CV for per-neuron alpha selection, 40 alphas on log-scale
  - Rate-based Poisson null (varwin/worddur):
    λ_k = ΣY_train / Σδ_train; μ_null = λ_k × δ_i
  - Mean-count null (fixed window): μ_null = mean(Y_train)

Usage
-----
  cd /scratch/aniluchavez/hippocampal-speaker-semantics
  python3 -u scripts/semantic_glm.py --model gpt2-xl --layer 24 \\
      --context_tag _ctx200 --window_type varwin --n_perm 500
"""

import os, sys, time, pickle, warnings, argparse
import numpy as np
import pandas as pd
import torch
from itertools import combinations
from scipy.special import gammaln

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ── CLI ───────────────────────────────────────────────────────────────────────

def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--layer",        type=int,   default=18)
    p.add_argument("--n_components", type=int,   default=100,
                   help="PCA components (per fold, fit on train only)")
    p.add_argument("--n_perm",       type=int,   default=500)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--model",        type=str,   default="llama-3.1-8b")
    p.add_argument("--context_tag",  type=str,   default="")
    p.add_argument("--window_type",  type=str,   default="fixed",
                   choices=["fixed", "varwin", "worddur"])
    p.add_argument("--spike_tag", type=str, default=None,
                   help="Override SpikeWindows output tag, e.g. "
                        "varwin_m100tooffset_p20tooffset. If omitted, use the "
                        "default tag for --window_type.")
    p.add_argument("--all_conditions", action="store_true",
                   help="Combine self+other words per region (matches collaborator's no-split approach)")
    p.add_argument("--outer_cv", type=str, default="block",
                   choices=["block", "shuffle", "purged_shuffle", "earlylate"],
                   help="Outer CV split strategy: 'block' = contiguous temporal folds (default); "
                        "'shuffle' = random KFold (matches collaborator); "
                        "'purged_shuffle' = random KFold with training words removed "
                        "when they fall within --embargo_ms of any test word; "
                        "'earlylate' = two temporal half-splits: early→late and late→early")
    p.add_argument("--n_outer", type=int, default=5,
                   help="Number of outer CV folds for block/shuffle/purged_shuffle.")
    p.add_argument("--embargo_ms", type=float, default=300.0,
                   help="Temporal gap around test-word onsets for purged_shuffle CV.")
    p.add_argument("--patient", type=str, default=None,
                   help="Run only this patient_ID (e.g. PTYEY_task86); runs all if omitted")
    p.add_argument("--exclude_patient", type=str, default=None,
                   help="Skip this patient_ID (e.g. for a known data/embedding "
                        "alignment mismatch); runs all if omitted")
    p.add_argument("--perm_type", type=str, default="xcirc",
                   choices=["xshuffle", "xfoldshuffle", "xcirc", "xblock", "yshuffle", "yblock", "ycirc"],
                   help="Permutation method: "
                        "xshuffle=global X shuffle+refit; "
                        "xfoldshuffle=shuffle train-fold X rows+refit, score real test X/Y; "
                        "xcirc=circular shift X by lag>tau+refit; "
                        "xblock=shuffle X within blocks+refit; "
                        "yshuffle=shuffle train-fold Y rows+refit, score real test Y; "
                        "yblock=shuffle Y within blocks (no refit); "
                        "ycirc=circular shift Y (no refit)")
    p.add_argument("--perm_block_size", type=int, default=20,
                   help="Block size for yblock permutation (words)")
    p.add_argument("--region", type=str, default=None,
                   help="Run only this region (e.g. hippocampus); runs all if omitted")
    p.add_argument("--force_all", action="store_true",
                   help="Include all patients regardless of word count; "
                        "reduces n_components to fit if needed")
    p.add_argument("--reliability", action="store_true",
                   help="Also compute proper split-half beta reliability + cross-condition "
                        "noise ceiling per neuron (neural_encoding/reliability.py), matching "
                        "BERTRidgeBetaCorr.ipynb's pipeline. Self/other only (skipped when "
                        "--all_conditions is set). Adds real CPU time per region (PoissonRegressor "
                        "refits per neuron x null x half-split) — off by default.")
    p.add_argument("--rel_n_null", type=int, default=100,
                   help="Null permutation iterations for r_cross significance (--reliability)")
    p.add_argument("--rel_n_half_splits", type=int, default=100,
                   help="Random half-split repeats for within-condition reliability (--reliability)")
    p.add_argument("--rel_n_jobs", type=int, default=8,
                   help="Parallel jobs (threads) for reliability per-neuron fits (--reliability)")
    p.add_argument("--rel_balance_trials", action="store_true",
                   help="With --reliability, randomly downsample the larger condition "
                        "(self or other) so X/Y have matched trial counts before "
                        "computing r_cross and split-half noise ceilings. This affects "
                        "only the reliability analysis, not the main CV fit.")
    p.add_argument("--rel_balance_seed", type=int, default=20260709,
                   help="Seed for --rel_balance_trials downsampling.")
    p.add_argument("--cosine_bin_split", action="store_true",
                   help="With --reliability, additionally split each condition's words into "
                        "quantile bins by cosine distance to the OTHER condition's raw-embedding "
                        "centroid (computed in the raw pre-PCA embedding space), and rerun the "
                        "r_cross/noise-ceiling reliability analysis separately on each bin. Tests "
                        "whether the self/other beta-vector alignment depends on how similar a "
                        "word's context is to the opposite condition's overall content. Saved "
                        "under 'reliability_cosine_bin' (n_bins=2) or "
                        "'reliability_cosine_bin_nb{n_bins}' otherwise.")
    p.add_argument("--cosine_n_bins", type=int, default=2,
                   help="Number of quantile bins for --cosine_bin_split (2=near/far median "
                        "split, 3=near/mid/far terciles, etc.)")
    p.add_argument("--cosine_min_words_per_bin", type=int, default=60,
                   help="Minimum words required in EVERY bin, for BOTH self and other, to run "
                        "--cosine_bin_split for a given patient/region; otherwise skipped "
                        "entirely for that patient/region (avoids near-degenerate beta fits on "
                        "too few words). Default matches 3x the n_components=20 used for this "
                        "analysis; adjust if using a different PC count.")
    p.add_argument("--notebook_exact", action="store_true",
                   help="Match notebooks/word_level_duration_cv_all_n.ipynb as closely as possible: "
                        "alpha grid 1e-3..1e3 with 30 values, shuffled inner KFold, "
                        "inner-CV warm start for the outer fit, and no min-spike filter.")
    p.add_argument("--r2_only", action="store_true",
                   help="Run only the real CV fit and skip permutation nulls. "
                        "This writes to a separate _r2only output folder; p-values are set to 1.")
    p.add_argument("--min_spikes", type=int, default=None,
                   help="Override the default min-spike-count neuron filter (20, or 0 under "
                        "--notebook_exact). Set to match variance_partitioning.py's threshold (5) "
                        "for cross-script neuron-count consistency.")
    return p.parse_args()

A            = _args()
LAYER        = A.layer
N_COMPONENTS = A.n_components
N_PERM       = A.n_perm
SEED         = A.seed
MODEL_TAG      = A.model
CONTEXT_TAG    = A.context_tag
WINDOW_TYPE    = A.window_type
SPIKE_TAG_OVERRIDE = A.spike_tag
USE_OFFSET     = WINDOW_TYPE in ("varwin", "worddur")
WORDDUR_MODE   = WINDOW_TYPE == "worddur"
ALL_CONDITIONS = A.all_conditions
OUTER_CV        = A.outer_cv   # "block", "shuffle", or "earlylate"
N_OUTER         = A.n_outer
EMBARGO_MS      = A.embargo_ms
PERM_TYPE       = A.perm_type  # "xshuffle", "yblock", "ycirc"
PERM_BLOCK_SIZE = A.perm_block_size
PATIENT_FILTER  = A.patient
EXCLUDE_PATIENT = A.exclude_patient
REGION_FILTER   = A.region
FORCE_ALL       = A.force_all
RELIABILITY        = A.reliability
REL_N_NULL          = A.rel_n_null
REL_N_HALF_SPLITS   = A.rel_n_half_splits
REL_N_JOBS          = A.rel_n_jobs
REL_BALANCE_TRIALS  = A.rel_balance_trials
REL_BALANCE_SEED    = A.rel_balance_seed
COSINE_BIN_SPLIT    = A.cosine_bin_split
COSINE_N_BINS        = A.cosine_n_bins
COSINE_MIN_WORDS_PER_BIN = A.cosine_min_words_per_bin
COSINE_BIN_KEY = ("reliability_cosine_bin" if COSINE_N_BINS == 2
                  else f"reliability_cosine_bin_nb{COSINE_N_BINS}")
if COSINE_N_BINS == 2:
    COSINE_BIN_LABELS = ["near", "far"]
elif COSINE_N_BINS == 3:
    COSINE_BIN_LABELS = ["near", "mid", "far"]
else:
    COSINE_BIN_LABELS = [f"bin{i}" for i in range(COSINE_N_BINS)]
NOTEBOOK_EXACT      = A.notebook_exact
R2_ONLY             = A.r2_only

_reliability_mod = None
if RELIABILITY:
    import importlib.util as _ilu
    _rel_spec = _ilu.spec_from_file_location(
        "nn_reliability",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "neural_encoding", "reliability.py"))
    _reliability_mod = _ilu.module_from_spec(_rel_spec)
    sys.modules["nn_reliability"] = _reliability_mod   # required before exec_module: dataclass
    _rel_spec.loader.exec_module(_reliability_mod)     # decorators resolve cls.__module__ via sys.modules

# ── HYPERPARAMETERS ───────────────────────────────────────────────────────────

ALPHAS          = np.logspace(-3, 3, 30) if NOTEBOOK_EXACT else np.logspace(-3, 6, 40)
COLLAB_LOSS     = WORDDUR_MODE             # True = sum loss (no /n_eff), matches collaborator
N_INNER         = 5                        # was 3; matches collaborators
FDR_ALPHA       = 0.05
MIN_SPIKES      = (0 if NOTEBOOK_EXACT else 20) if A.min_spikes is None else A.min_spikes
ETA_CLIP        = 20.0
LBFGS_ITER      = 200                      # matches collaborators
LBFGS_TOL       = 1e-6                     # matches collaborators (tolerance_grad/change)
LBFGS_HISTORY   = 10                       # matches collaborators history_size

# ── PATHS ─────────────────────────────────────────────────────────────────────

EMBED_DIR  = "/scratch/aniluchavez/ConvoDATAS/EmbedCache"
SPIKE_ROOT = "/scratch/aniluchavez/ConvoDATAS/SpikeWindows"
_win_suffix  = ("_worddur_exposure" if WORDDUR_MODE
                else f"_{WINDOW_TYPE}" if USE_OFFSET
                else "")
if SPIKE_TAG_OVERRIDE:
    _win_suffix = f"_{SPIKE_TAG_OVERRIDE}"
if NOTEBOOK_EXACT:
    _reg_suffix = "_notebookexact"
else:
    _reg_suffix  = "_a1e6_tinner_min20_nullinit" if WORDDUR_MODE else "_a1e6_min20"
_cond_suffix = "_allcond" if ALL_CONDITIONS else ""
_cv_suffix   = {
    "shuffle": "_shuf",
    "purged_shuffle": f"_purgedshuf_emb{EMBARGO_MS:g}ms",
    "earlylate": "_earlylate",
}.get(OUTER_CV, "")
if OUTER_CV in ("block", "shuffle", "purged_shuffle") and N_OUTER != 5:
    _cv_suffix += f"_k{N_OUTER}"
_perm_suffix = f"_{PERM_TYPE}" + ("_r2only" if R2_ONLY else "")
_rel_suffix = "_relbal" if (RELIABILITY and REL_BALANCE_TRIALS) else ""
_out_model   = (f"{MODEL_TAG}{CONTEXT_TAG}{_win_suffix}{_reg_suffix}"
                f"{_cond_suffix}{_cv_suffix}{_perm_suffix}{_rel_suffix}")
OUT_DIR      = os.path.join("/scratch/aniluchavez/ConvoDATAS/SemanticGLM",
                            _out_model, f"pc{N_COMPONENTS}")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"layer={LAYER}  PCs={N_COMPONENTS}  perms={N_PERM}  "
      f"model={MODEL_TAG}{CONTEXT_TAG}  window={WINDOW_TYPE}  outer_cv={OUTER_CV}  "
      f"embargo_ms={EMBARGO_MS if OUTER_CV == 'purged_shuffle' else 'n/a'}  "
      f"dev={DEVICE}", flush=True)
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

# ── MATH / GPU UTILITIES ──────────────────────────────────────────────────────

@torch.no_grad()
def gpu_pca_fit(X_np, n_components):
    """
    Randomised SVD PCA on GPU, fit on X_np.
    Returns (X_pca, train_mean, Vt) — Vt shape (n_components, D).
    """
    X  = torch.tensor(X_np, dtype=torch.float32, device=DEVICE)
    Xm = X.mean(0)
    Xc = X - Xm
    k  = min(n_components + 10, min(Xc.shape))
    Y  = Xc @ torch.randn(Xc.shape[1], k, device=DEVICE)
    for _ in range(2):
        Y = Xc @ (Xc.T @ Y)
    Q, _ = torch.linalg.qr(Y)
    _, _, Vt = torch.linalg.svd(Q.T @ Xc, full_matrices=False)
    Vt = Vt[:n_components]
    X_pca = (Xc @ Vt.T).cpu().numpy().astype(np.float32)
    return X_pca, Xm.cpu().numpy(), Vt.cpu().numpy()


@torch.no_grad()
def gpu_pca_transform(X_np, train_mean_np, Vt_np):
    """Project X_np using pre-fitted PCA (centre with train mean, then project)."""
    X  = torch.tensor(X_np - train_mean_np, dtype=torch.float32, device=DEVICE)
    Vt = torch.tensor(Vt_np, dtype=torch.float32, device=DEVICE)
    return (X @ Vt.T).cpu().numpy().astype(np.float32)


def prep_fold_features(X_tr_raw, X_te_raw, n_components):
    """
    Standardise → PCA → re-standardise.  All three steps fit on X_tr_raw only.
    Matches collaborator _prep_X_with_pca.
    worddur mode: uses exact sklearn PCA (collaborator uses sklearn).
    Other modes: uses randomised GPU SVD.
    """
    mu_r = X_tr_raw.mean(0)
    sd_r = X_tr_raw.std(0, ddof=0); sd_r[sd_r == 0] = 1.0
    Xtr_s = (X_tr_raw - mu_r) / sd_r
    Xte_s = (X_te_raw - mu_r) / sd_r

    Xtr_pca, pca_mean, Vt = gpu_pca_fit(Xtr_s, n_components)
    Xte_pca = gpu_pca_transform(Xte_s, pca_mean, Vt)

    mu_p = Xtr_pca.mean(0)
    sd_p = Xtr_pca.std(0, ddof=0); sd_p[sd_p == 0] = 1.0
    return ((Xtr_pca - mu_p) / sd_p).astype(np.float32), \
           ((Xte_pca - mu_p) / sd_p).astype(np.float32)




def poisson_ll_full(Y_np, mu_np):
    """
    Complete Poisson log-likelihood including log-factorial term, summed over
    words (axis=0), kept per neuron.  Matches collaborator poisson_ll_per_neuron.
    """
    mu = np.clip(mu_np, 1e-10, None)
    return (Y_np * np.log(mu) - mu - gammaln(Y_np + 1)).sum(0)   # (n_m,)


poisson_ll_full_neuron = poisson_ll_full   # alias used in inner_cv_sequential


def gpu_fit_per_neuron(X_t, Y_t, alpha_vec, offset_t=None, max_iter=None):
    """
    Per-neuron L-BFGS Poisson ridge.  Exactly matches the collaborator pipeline:
    one independent LBFGS instance per neuron so each solves a (p+1)-parameter
    problem with its own well-conditioned Hessian approximation.

    X_t      : (1, n, p)   features
    Y_t      : (1, n, m)   spike counts
    alpha_vec: (m,) tensor  per-neuron regularisation strengths
    offset_t : (1, n, 1)   log-exposure, or None
    """
    assert X_t.shape[0] == 1
    n, p = X_t.shape[1], X_t.shape[2]
    m = Y_t.shape[2]
    iters = max_iter if max_iter is not None else LBFGS_ITER
    n_eff = max(float(Y_t.sum().item()), 1.0)

    Xi   = torch.cat([torch.ones(n, 1, device=DEVICE), X_t[0]], dim=-1)  # (n, p+1)
    Y_all = Y_t[0]                                                         # (n, m)
    off   = offset_t[0, :, 0] if offset_t is not None else None           # (n,) or None

    betas = torch.zeros(m, p + 1, device=DEVICE)

    for j in range(m):
        alpha_j = float(alpha_vec[j])
        y_j     = Y_all[:, j]
        beta_j  = torch.zeros(p + 1, device=DEVICE, requires_grad=True)
        opt_j   = torch.optim.LBFGS([beta_j], max_iter=iters,
                                     line_search_fn='strong_wolfe',
                                     tolerance_grad=LBFGS_TOL, tolerance_change=LBFGS_TOL,
                                     history_size=LBFGS_HISTORY)

        def _make_closure(bj, yj, aj, offj):
            def closure():
                opt_j.zero_grad()
                eta = Xi @ bj
                if offj is not None:
                    eta = eta + offj
                eta = torch.clamp(eta, -ETA_CLIP, ETA_CLIP)
                raw = (torch.exp(eta) - yj * eta).sum() + aj * bj[1:].pow(2).sum()
                loss = raw if COLLAB_LOSS else raw / n_eff
                loss.backward()
                return loss
            return closure

        opt_j.step(_make_closure(beta_j, y_j, alpha_j, off))
        betas[j] = beta_j.detach()

    betas = betas.unsqueeze(0)                        # (1, m, p+1)
    return betas[:, :, 1:], betas[:, :, :1]          # (1,m,p), (1,m,1)


def gpu_fit(X_t, Y_t, alpha, offset_t=None, max_iter=None):
    """
    L-BFGS Poisson ridge, batched over k contexts × m neurons.
    Used for inner CV only (alpha is scalar per batch item — well conditioned).

    X_t      : (k, n, p)   features  (intercept added internally, NOT regularised)
    Y_t      : (k, n, m)   spike counts
    alpha    : scalar  — same for all batches and neurons
               (k,) tensor — one per batch item (inner CV)
    offset_t : (k, n, 1) log-exposure offset, or None
    max_iter : overrides LBFGS_ITER if provided

    Returns beta_feat (k, m, p), bias (k, m, 1).
    """
    k, n, p = X_t.shape
    m = Y_t.shape[2]
    n_eff = float(Y_t.sum().clamp(min=1).item())
    iters = max_iter if max_iter is not None else LBFGS_ITER

    Xi = torch.cat([torch.ones(k, n, 1, device=DEVICE), X_t], dim=-1)
    beta = torch.zeros(k, m, p + 1, device=DEVICE, requires_grad=True)
    opt  = torch.optim.LBFGS([beta], max_iter=iters, line_search_fn='strong_wolfe',
                              tolerance_grad=LBFGS_TOL, tolerance_change=LBFGS_TOL,
                              history_size=LBFGS_HISTORY)

    if isinstance(alpha, torch.Tensor):
        if alpha.ndim == 1 and alpha.shape[0] == k:
            a = alpha.to(DEVICE).view(k, 1, 1)    # one per batch item
        elif alpha.ndim == 1 and k == 1 and alpha.shape[0] == m:
            a = alpha.to(DEVICE).view(1, m, 1)    # one per neuron
        else:
            a = float(alpha.item()) if alpha.numel() == 1 else float(alpha)
    else:
        a = float(alpha)

    def closure():
        opt.zero_grad()
        eta = torch.bmm(Xi, beta.transpose(1, 2))
        if offset_t is not None:
            eta = eta + offset_t
        eta  = torch.clamp(eta, -ETA_CLIP, ETA_CLIP)
        raw  = (torch.exp(eta) - Y_t * eta).sum() + (a * beta[:, :, 1:].pow(2)).sum()
        loss = raw if COLLAB_LOSS else raw / n_eff
        loss.backward()
        return loss

    opt.step(closure)
    b = beta.detach()
    return b[:, :, 1:], b[:, :, :1]   # (k,m,p), (k,m,1)


def fit_collab(Xt, Yt, alpha, off_t=None, init_W=None, init_b=None, max_iter=None):
    """
    Single LBFGS fit for all neurons together — collaborator style.
    Xt : (n, p)   Yt : (n, K)   off_t : (n,) or None
    Returns model, W (p,K), b (K,).
    No /n_eff in loss.
    """
    n, p = Xt.shape; K = Yt.shape[1]
    iters = max_iter or LBFGS_ITER
    W = torch.zeros(p, K, device=DEVICE)
    if off_t is not None:
        exposure = torch.exp(off_t).sum().clamp_min(1e-10)
        rate = Yt.sum(dim=0).div(exposure).clamp_min(1e-10)
        b = torch.log(rate)
    else:
        b = torch.log(Yt.mean(dim=0).clamp_min(1e-10))
    if init_W is not None:
        W = init_W.clone().to(DEVICE)
    if init_b is not None:
        b = init_b.clone().to(DEVICE)
    W.requires_grad_(True); b.requires_grad_(True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters,
                             tolerance_grad=LBFGS_TOL, tolerance_change=LBFGS_TOL,
                             history_size=LBFGS_HISTORY,
                             line_search_fn="strong_wolfe")
    a_t = torch.as_tensor(alpha, dtype=torch.float32, device=DEVICE)
    if a_t.ndim == 0:
        a_t = a_t.expand(K)
    def closure():
        opt.zero_grad(set_to_none=True)
        eta = Xt @ W + b
        if off_t is not None:
            eta = eta + off_t[:, None]
        eta = torch.clamp(eta, -ETA_CLIP, ETA_CLIP)
        mu  = torch.exp(eta)
        nll = torch.sum(mu - Yt * torch.log(mu.clamp_min(1e-10)))
        reg = 0.5 * torch.sum(a_t * (W**2).sum(dim=0))
        loss = nll + reg
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        W_out = W.detach().clone()
        b_out = b.detach().clone()
    return W_out, b_out


def inner_cv_sequential(X_raw_tr, Y_tr, off_tr, seed, n_components=None):
    """
    Sequential alpha descent (high→low) with warm starts per fold. In normal
    script mode this uses temporal inner folds; in --notebook_exact mode it uses
    shuffled KFold, matching word_level_duration_cv_all_n.ipynb.
    """
    if NOTEBOOK_EXACT:
        from sklearn.model_selection import KFold as _KFold
        splits = list(_KFold(n_splits=N_INNER, shuffle=True,
                             random_state=seed).split(X_raw_tr))
    else:
        splits = []
        for tr_mask, va_mask in block_splits(len(X_raw_tr), N_INNER):
            splits.append((np.flatnonzero(tr_mask), np.flatnonzero(va_mask)))

    fold_data = []
    for tr_f, va_f in splits:
        Xtr_p, Xva_p = prep_fold_features(X_raw_tr[tr_f], X_raw_tr[va_f],
                                           n_components or N_COMPONENTS)
        fold_data.append({
            "Xtr": torch.tensor(Xtr_p, dtype=torch.float32, device=DEVICE),
            "Ytr": torch.tensor(Y_tr[tr_f], dtype=torch.float32, device=DEVICE),
            "Xva": torch.tensor(Xva_p, dtype=torch.float32, device=DEVICE),
            "Yva": Y_tr[va_f].astype(np.float64),
            "otr": (torch.tensor(off_tr[tr_f], dtype=torch.float32, device=DEVICE)
                    if off_tr is not None else None),
            "ova": (torch.tensor(off_tr[va_f], dtype=torch.float32, device=DEVICE)
                    if off_tr is not None else None),
        })

    alphas_desc = np.sort(ALPHAS)[::-1]   # high → low
    n_folds = len(fold_data)
    n_m = Y_tr.shape[1]
    scores = np.zeros((len(alphas_desc), n_folds, n_m))
    fold_cache = {i: (None, None) for i in range(n_folds)}

    for a_idx, a in enumerate(alphas_desc):
        for fi, fd in enumerate(fold_data):
            W0, b0 = fold_cache[fi]
            W, b = fit_collab(fd["Xtr"], fd["Ytr"], float(a), fd["otr"], W0, b0)
            fold_cache[fi] = (W, b)
            with torch.no_grad():
                eta = fd["Xva"] @ W + b
                if fd["ova"] is not None:
                    eta = eta + fd["ova"][:, None]
                mu = torch.exp(torch.clamp(eta, -ETA_CLIP, ETA_CLIP)).cpu().numpy()
            scores[a_idx, fi] = poisson_ll_full_neuron(fd["Yva"], mu)

    mean_ll = scores.mean(axis=1)           # (n_alphas, n_m)
    best_a_idx = np.argmax(mean_ll, axis=0) # (n_m,)
    best_alpha = alphas_desc[best_a_idx]    # (n_m,) per-neuron best alpha
    init_W, init_b = fold_cache[0]
    return best_alpha, init_W, init_b


def block_splits(n, k):
    """k contiguous time-block folds."""
    b = n // k
    for i in range(k):
        s, e = i * b, (i * b + b if i < k - 1 else n)
        mask = np.zeros(n, bool); mask[s:e] = True
        yield ~mask, mask


def purged_shuffle_splits(onset, k, embargo_ms, seed):
    """Random K-fold test sets with temporally adjacent train rows purged.

    Test rows remain distributed across the recording, preserving broad topic
    coverage. A candidate training row is removed whenever its onset is within
    ``embargo_ms`` of any onset in that fold's test set.
    """
    from sklearn.model_selection import KFold as _KFold

    onset = np.asarray(onset, dtype=float)
    kf = _KFold(n_splits=k, shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(np.arange(len(onset))):
        test_onsets = np.sort(onset[test_idx])
        candidate_onsets = onset[train_idx]
        positions = np.searchsorted(test_onsets, candidate_onsets)
        left = np.clip(positions - 1, 0, len(test_onsets) - 1)
        right = np.clip(positions, 0, len(test_onsets) - 1)
        nearest = np.minimum(
            np.abs(candidate_onsets - test_onsets[left]),
            np.abs(candidate_onsets - test_onsets[right]),
        )
        kept_train_idx = train_idx[nearest > embargo_ms]
        train_mask = np.zeros(len(onset), dtype=bool)
        test_mask = np.zeros(len(onset), dtype=bool)
        train_mask[kept_train_idx] = True
        test_mask[test_idx] = True
        yield train_mask, test_mask


def earlylate_splits(n):
    """Two temporal half-splits: train early/test late, then train late/test early."""
    mid = n // 2
    early = np.zeros(n, bool); early[:mid] = True
    late  = np.zeros(n, bool); late[mid:] = True
    yield early, late
    yield late, early


def compute_tau_embed(X_raw, thresh=0.05):
    """
    Estimate embedding autocorrelation timescale (tau) in word-index units.

    X_raw  : (n_words, D) — raw embeddings sorted in temporal order.
    thresh : ACF value considered negligible (default 0.05).

    Projects onto PC1 (dominant direction of semantic drift in the conversation)
    and returns the first lag where |ACF| < thresh.  This is the minimum
    temporal separation needed so that test-set embeddings are decorrelated
    from their nearest training neighbours.
    """
    n_words = X_raw.shape[0]
    max_lag = min(n_words // 4, 500)
    X = X_raw.astype(np.float64)
    X -= X.mean(0)
    v = X[0].copy(); v /= np.linalg.norm(v) + 1e-10
    for _ in range(20):
        v = X.T @ (X @ v); v /= np.linalg.norm(v) + 1e-10
    pc1 = X @ v
    pc1 = (pc1 - pc1.mean()) / (pc1.std() + 1e-10)
    acf = np.array([np.dot(pc1[:n_words - lag], pc1[lag:]) / (n_words - lag)
                    for lag in range(1, max_lag + 1)])
    below = np.where(np.abs(acf) < thresh)[0]
    return float(below[0] + 1) if len(below) > 0 else float(max_lag)


BLOCK_TAU_SCALE = 3.0   # block_size = ceil(BLOCK_TAU_SCALE × tau); conservative


def fdr_bh(pvals):
    """Benjamini-Hochberg FDR.  Returns (adjusted_pvals, reject_bool)."""
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
    if SPIKE_TAG_OVERRIDE:
        tag = SPIKE_TAG_OVERRIDE
    elif WORDDUR_MODE:
        tag = "worddur"
    elif USE_OFFSET:
        tag = "varwin_m150p500_p200p500"
    else:
        tag = "tshift-150_tlen500_oshift+200_olen500"
    d = os.path.join(SPIKE_ROOT, f"output_{patient}_english_only_{tag}")
    return d if os.path.isdir(d) else None


def load_regress_dur(spike_dir, patient_ID):
    cands = [f for f in os.listdir(spike_dir) if f.endswith("_with_regress_dur.xlsx")]
    if not cands:
        return None
    df = pd.read_excel(os.path.join(spike_dir, cands[0]))
    if "regress_dur" not in df.columns:
        return None
    return df["regress_dur"].values.astype(np.float64)


def load_word_dur_ms(spike_dir):
    """Load word acoustic duration in ms (= offset - onset) for worddur mode."""
    cands = [f for f in os.listdir(spike_dir) if f.endswith("_with_regress_dur.xlsx")]
    if not cands:
        return None
    df = pd.read_excel(os.path.join(spike_dir, cands[0]))
    col = "word_dur" if "word_dur" in df.columns else "Duration"
    if col not in df.columns:
        return None
    return df[col].values.astype(np.float64)


def load_spike_matrix(spike_dir, speaker, region):
    spk_dir = os.path.join(spike_dir, speaker)
    if not os.path.isdir(spk_dir):
        return None
    # worddur files are named "{region}_worddur_spike_counts.npy";
    # varwin/fixed files are named "{region}_spike_counts.npy"
    suffix = "_worddur_spike_counts.npy" if WORDDUR_MODE else "_spike_counts.npy"
    cands = [f for f in os.listdir(spk_dir)
             if f.lower().startswith(region.lower()) and f.endswith(suffix)]
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
                assign[i] = col; break
    mask_self  = assign == "Speaker1"
    mask_other = np.array([(a is not None and a != "Speaker1") for a in assign], dtype=bool)
    onset_ms   = tx["onset"].values.astype(np.float64) if "onset" in tx.columns else np.arange(n, dtype=np.float64)
    return assign, mask_self, mask_other, dir_membership, onset_ms


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

# ── MAIN ──────────────────────────────────────────────────────────────────────

all_rows = []

for cfg in PATIENTS:
    patient_ID = cfg["patient_ID"]
    patient    = cfg["patient"]
    if PATIENT_FILTER and patient_ID != PATIENT_FILTER:
        continue

    out_path = os.path.join(OUT_DIR, f"{patient_ID}_L{LAYER:02d}_sem.pkl")
    _prior_obj = None
    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            obj = pickle.load(f)
        _prior_obj = obj if isinstance(obj, dict) else None
        # A pkl saved without --reliability won't have the "reliability" key; if it's
        # now requested, recompute fully rather than silently skipping it forever.
        if RELIABILITY and not (isinstance(obj, dict) and "reliability" in obj):
            print(f"  {patient_ID}: cached result lacks reliability data — recomputing", flush=True)
        elif COSINE_BIN_SPLIT and not (isinstance(obj, dict) and COSINE_BIN_KEY in obj):
            print(f"  {patient_ID}: cached result lacks {COSINE_BIN_KEY} data — recomputing", flush=True)
        elif not (isinstance(obj, dict) and "perm_nulls" in obj):
            print(f"  {patient_ID}: cached result lacks permutation-null arrays — recomputing", flush=True)
        else:
            print(f"[SKIP] {patient_ID}", flush=True)
            all_rows.append(obj["df"] if isinstance(obj, dict) else obj)
            continue

    print(f"\n{'='*60}\n  {patient_ID}", flush=True)

    npy_path  = os.path.join(EMBED_DIR,
                              f"{patient_ID}_{MODEL_TAG}{CONTEXT_TAG}_word_emb_layers.npy")
    spike_dir = find_spike_dir(patient)
    if not os.path.exists(npy_path) or spike_dir is None:
        print("  SKIP — missing embeddings or spike dir"); continue

    try:
        spk_assignment, mask_self, mask_other, dir_membership, onset_ms = \
            load_speaker_assignment(spike_dir)
    except FileNotFoundError as e:
        print(f"  SKIP — {e}"); continue

    if WORDDUR_MODE:
        all_regress_dur = load_word_dur_ms(spike_dir)
        if all_regress_dur is None:
            print("  SKIP — word_dur not found for worddur mode"); continue
    elif USE_OFFSET:
        all_regress_dur = load_regress_dur(spike_dir, patient_ID)
        if all_regress_dur is None:
            print("  SKIP — regress_dur not found for varwin mode"); continue
    else:
        all_regress_dur = None

    # Load raw embeddings for this layer — PCA happens per-fold inside the CV loop
    t0 = time.time()
    X_layers = np.load(npy_path, mmap_mode='r')
    if X_layers.ndim == 3:
        X_layer_raw = X_layers[LAYER].astype(np.float32)  # (n_words, D)
    elif X_layers.ndim == 2:
        # Some precomputed max-context/static embeddings are already collapsed
        # to one matrix per patient rather than layer x word x dim.
        X_layer_raw = X_layers.astype(np.float32)  # (n_words, D)
    else:
        raise ValueError(f"Unexpected embedding shape for {npy_path}: {X_layers.shape}")
    print(f"  Raw embeddings loaded  {X_layer_raw.shape}  ({time.time()-t0:.1f}s)", flush=True)

    # One joint PCA per patient on the FULL word set — matches BERTRidgeBetaCorr.ipynb's
    # load_and_reduce_bert_embeddings (PCA fit once on all words, self/other only split
    # out afterwards). Only needed for --reliability; cheap (CPU, 768-dim sklearn PCA).
    X_layer_pca_rel = None
    if RELIABILITY:
        from sklearn.decomposition import PCA as _PCA
        X_layer_pca_rel = _PCA(n_components=N_COMPONENTS).fit_transform(
            np.asarray(X_layer_raw, dtype=np.float64))

    patient_rows = []
    patient_betas = {}
    patient_perm_nulls = {}
    patient_reliability = {}
    patient_reliability_meta = {}
    patient_reliability_cosine = {}

    for region in cfg["region_ranges"]:
        if REGION_FILTER and region.lower() != REGION_FILTER.lower():
            continue
        cond_data = {}
        raw_n_neurons = None
        region_cond_xy = {}   # cond -> (X_rel_std, Y_v) captured below, for --reliability
        for cond, mask in [("self", mask_self), ("other", mask_other)]:
            Y_mat = load_condition_ordered(
                spike_dir, cond, region, spk_assignment, dir_membership)
            if Y_mat is None or Y_mat.ndim < 2:
                cond_data[cond] = None; continue
            if Y_mat.shape[0] != int(mask.sum()):
                print(f"  {region}/{cond}: row mismatch — skip")
                cond_data[cond] = None; continue
            if raw_n_neurons is None:
                raw_n_neurons = Y_mat.shape[1]
            elif Y_mat.shape[1] != raw_n_neurons:
                print(f"  {region}/{cond}: neuron-count mismatch — skip")
                cond_data[cond] = None; continue
            valid = ~np.isnan(Y_mat).any(axis=1)
            cond_data[cond] = (Y_mat[valid].astype(np.float32), mask, valid)

        available = {c: d for c, d in cond_data.items() if d is not None}
        if not available:
            continue

        # Cosine-distance-to-other-centroid quantile bins (--reliability --cosine_bin_split).
        # Computed in the raw pre-PCA embedding space so the split doesn't depend on the
        # PCA truncation used for the encoding model itself. For each self word, distance
        # is to the OTHER condition's centroid (mean raw embedding over other's words in
        # this region), and vice versa — a direct per-word measure of "how much does this
        # word's context resemble the opposite condition's overall content." Skips the
        # whole patient/region if any bin would fall below --cosine_min_words_per_bin for
        # either condition, rather than fitting betas on too few words.
        cosine_bin_masks = {}
        if RELIABILITY and COSINE_BIN_SPLIT and "self" in available and "other" in available:
            _, mask_s, valid_s = available["self"]
            _, mask_o, valid_o = available["other"]
            Xs_raw = np.asarray(X_layer_raw[mask_s][valid_s], dtype=np.float64)
            Xo_raw = np.asarray(X_layer_raw[mask_o][valid_o], dtype=np.float64)
            c_self  = Xs_raw.mean(axis=0)
            c_other = Xo_raw.mean(axis=0)

            def _cos_dist_to(X, c):
                num = X @ c
                den = np.linalg.norm(X, axis=1) * np.linalg.norm(c)
                den = np.where(den == 0, np.nan, den)
                return 1.0 - num / den

            def _quantile_bin_masks(dist, n_bins):
                edges = np.nanquantile(dist, np.linspace(0, 1, n_bins + 1))
                edges[-1] += 1e-9  # include the max value in the last bin
                masks = []
                for i in range(n_bins):
                    lo, hi = edges[i], edges[i + 1]
                    m = (dist >= lo) & (dist < hi) if i < n_bins - 1 else (dist >= lo) & (dist <= hi)
                    masks.append(m)
                return masks

            dist_self_to_other = _cos_dist_to(Xs_raw, c_other)
            dist_other_to_self = _cos_dist_to(Xo_raw, c_self)
            masks_self  = _quantile_bin_masks(dist_self_to_other, COSINE_N_BINS)
            masks_other = _quantile_bin_masks(dist_other_to_self, COSINE_N_BINS)
            counts_self  = [int(m.sum()) for m in masks_self]
            counts_other = [int(m.sum()) for m in masks_other]
            min_count = min(counts_self + counts_other)
            print(f"  {region}: cosine-bin split ({COSINE_N_BINS} bins) — "
                  f"self counts={counts_self}, other counts={counts_other}", flush=True)
            if min_count < COSINE_MIN_WORDS_PER_BIN:
                print(f"  {region}: SKIP cosine-bin reliability — smallest bin ({min_count} words) "
                      f"< --cosine_min_words_per_bin ({COSINE_MIN_WORDS_PER_BIN})", flush=True)
            else:
                cosine_bin_masks["self"]  = masks_self
                cosine_bin_masks["other"] = masks_other

        spike_ok = None
        for _, (Y_v_raw, _, _) in available.items():
            ok = Y_v_raw.sum(0) >= MIN_SPIKES
            spike_ok = ok if spike_ok is None else (spike_ok & ok)
        n_m = int(spike_ok.sum())
        if n_m == 0:
            print(f"  {region}: 0 neurons pass spike filter — skip"); continue

        # Build iteration list: combine self+other if --all_conditions, else keep split
        if ALL_CONDITIONS and len(available) > 0:
            pieces_Y, pieces_X, pieces_dur, pieces_onset = [], [], [], []
            for c, (Y_v_raw, mask, valid) in available.items():
                Y_piece = Y_v_raw[:, spike_ok]
                if USE_OFFSET and all_regress_dur is not None:
                    d = all_regress_dur[mask][valid].copy()
                    d = np.where(np.isnan(d) | (d <= 0), 500.0, d)
                    pieces_dur.append(d)
                pieces_Y.append(Y_piece)
                pieces_X.append(X_layer_raw[mask][valid].astype(np.float32))
                pieces_onset.append(onset_ms[mask][valid])
            # Sort all pieces by global onset time so block-CV folds are truly temporal
            all_Y   = np.concatenate(pieces_Y,     axis=0)
            all_X   = np.concatenate(pieces_X,     axis=0)
            all_dur = np.concatenate(pieces_dur,   axis=0) if pieces_dur else None
            all_on  = np.concatenate(pieces_onset, axis=0)
            t_order = np.argsort(all_on, kind="stable")
            all_Y   = all_Y[t_order]
            all_X   = all_X[t_order]
            all_dur = all_dur[t_order] if all_dur is not None else None
            cond_iter = [("all", all_Y, all_X, all_dur, all_on[t_order])]
        else:
            cond_iter = []
            for cond, (Y_v_raw, _mask, _valid) in available.items():
                Y_piece = Y_v_raw[:, spike_ok]
                onset_piece = onset_ms[_mask][_valid]
                cond_iter.append((cond, Y_piece, None, None, onset_piece))

        for cond_item in cond_iter:
            if ALL_CONDITIONS:
                cond, Y_v, X_raw_c_pre, dur_pre, onset_c = cond_item
            else:
                cond, Y_v_raw, _, _, onset_c = cond_item
                _, mask, valid = available[cond]
                Y_v = Y_v_raw
                X_raw_c_pre = None
                dur_pre = None

            n_w = Y_v.shape[0]
            print(f"  {region}/{cond}: {n_w}w × {n_m}n", flush=True)

            n_comp_eff = min(N_COMPONENTS, n_w // N_OUTER - 2)
            if n_comp_eff < 2:
                print("    skip — too few samples even for minimal PCA"); continue
            if n_comp_eff < N_COMPONENTS:
                if FORCE_ALL:
                    print(f"    warn — too few samples; reducing n_components {N_COMPONENTS}→{n_comp_eff}")
                else:
                    print("    skip — too few samples (use --force_all to include)"); continue

            # Raw (pre-PCA) features for this condition — PCA done per fold
            if ALL_CONDITIONS:
                X_raw_c = X_raw_c_pre
            else:
                X_raw_c = X_layer_raw[mask][valid].astype(np.float32)  # (n_w, D)

            # Capture self/other features+counts for --reliability (uses the same joint
            # per-patient PCA as BERTRidgeBetaCorr.ipynb, standardized per condition).
            if RELIABILITY and not ALL_CONDITIONS and X_layer_pca_rel is not None:
                from sklearn.preprocessing import StandardScaler as _SS
                X_rel_std = _SS().fit_transform(X_layer_pca_rel[mask][valid])
                region_cond_xy[cond] = (X_rel_std.astype(np.float64), Y_v.astype(np.float64))

            # Poisson offset
            # Exposure offset for variable-duration windows. Y remains spike
            # counts; exp(Xβ + b) is therefore a firing rate in spikes/second.
            # Both duration sources are stored in ms, so convert them to seconds.
            if ALL_CONDITIONS:
                if dur_pre is not None:
                    offset_np = np.log(dur_pre / 1000.0).astype(np.float32)
                else:
                    offset_np = None
            elif USE_OFFSET and all_regress_dur is not None:
                dur_c = all_regress_dur[mask][valid].copy()
                dur_c = np.where(np.isnan(dur_c) | (dur_c <= 0), 500.0, dur_c)
                offset_np = np.log(dur_c / 1000.0).astype(np.float32)
            else:
                offset_np = None

            # ── REAL NESTED CV ────────────────────────────────────────────────
            ll_model_folds = []   # per-fold model LL arrays (n_m,)
            ll_null_folds  = []   # per-fold null  LL arrays (n_m,)
            ll_train_model_folds = []
            ll_train_null_folds  = []
            fold_betas     = []
            best_alpha_folds = []  # per-fold (n_m,) best-alpha indices

            # Store fold metadata for permutation loop
            fold_meta = []   # list of {tr_m, te_m, best_alpha_vec, off_tr, off_te}

            if DEVICE.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t_cv = time.time()

            if OUTER_CV == "shuffle":
                from sklearn.model_selection import KFold as _KFold
                _kf = _KFold(n_splits=N_OUTER, shuffle=True, random_state=SEED)
                _outer_iter = []
                for _tri, _tei in _kf.split(np.arange(n_w)):
                    _tr_m = np.zeros(n_w, bool); _tr_m[_tri] = True
                    _te_m = np.zeros(n_w, bool); _te_m[_tei] = True
                    _outer_iter.append((_tr_m, _te_m))
            elif OUTER_CV == "purged_shuffle":
                _outer_iter = list(
                    purged_shuffle_splits(
                        onset_c, N_OUTER, EMBARGO_MS, seed=SEED
                    )
                )
                fold_sizes = ", ".join(
                    f"{int(tr.sum())}/{int(te.sum())}"
                    for tr, te in _outer_iter
                )
                print(
                    f"    purged-shuffle outer CV: embargo={EMBARGO_MS:g}ms "
                    f"train/test words={fold_sizes}",
                    flush=True,
                )
            elif OUTER_CV == "earlylate":
                _outer_iter = list(earlylate_splits(n_w))
                print(f"    earlylate outer CV: "
                      f"early={int(_outer_iter[0][0].sum())}w  "
                      f"late={int(_outer_iter[0][1].sum())}w  "
                      f"n_folds={len(_outer_iter)}", flush=True)
            else:
                # Compute tau from embedding PC1 — this is the timescale of semantic
                # autocorrelation in the conversation, which is what block CV protects against
                tau = compute_tau_embed(X_raw_c)
                min_block = int(np.ceil(BLOCK_TAU_SCALE * tau))
                n_folds   = max(2, min(N_OUTER, n_w))   # always force N_OUTER; min_block only for diagnostics
                print(f"    tau={tau:.1f}w  min_block={min_block}w  n_folds={n_folds}  blk={n_w//n_folds}w", flush=True)
                _outer_iter = list(block_splits(n_w, n_folds))

            for fold_i, (tr_m, te_m) in enumerate(_outer_iter):
                n_tr = int(tr_m.sum())
                if n_tr <= n_comp_eff + N_INNER:
                    print(
                        f"    skip fold {fold_i} — only {n_tr} training words "
                        f"remain after CV purge",
                        flush=True,
                    )
                    continue

                # Per-fold PCA: fit on outer-train only
                X_tr, X_te = prep_fold_features(X_raw_c[tr_m], X_raw_c[te_m], n_comp_eff)

                X_tr_np = X_tr   # (n_tr, P)
                Y_tr_np = Y_v[tr_m]
                Y_te_np = Y_v[te_m]

                # Offset slices
                off_tr_np = offset_np[tr_m] if offset_np is not None else None
                off_te_np = offset_np[te_m] if offset_np is not None else None

                # ── Null model LL (complete, with gammaln) ──────────────────
                if USE_OFFSET and off_tr_np is not None:
                    delta_tr = np.exp(off_tr_np)
                    delta_te = np.exp(off_te_np)
                    lambda_k = (Y_tr_np.sum(0) / delta_tr.sum()).clip(1e-10)
                    mu_null_te = lambda_k[None, :] * delta_te[:, None]
                    mu_null_tr = lambda_k[None, :] * delta_tr[:, None]
                else:
                    mean_tr = Y_tr_np.mean(0).clip(1e-10)
                    mu_null_te = np.broadcast_to(mean_tr[None, :], Y_te_np.shape).copy()
                    mu_null_tr = np.broadcast_to(mean_tr[None, :], Y_tr_np.shape).copy()
                ll_null_folds.append(poisson_ll_full(Y_te_np, mu_null_te))
                ll_train_null_folds.append(poisson_ll_full(Y_tr_np, mu_null_tr))

                # ── Inner CV: per-neuron alpha selection ────────────────────
                if WORDDUR_MODE:
                    # Sequential warm-start inner CV — matches collaborator exactly
                    best_alpha_np, init_W, init_b = inner_cv_sequential(
                        X_raw_c[tr_m], Y_tr_np, off_tr_np,
                        seed=SEED + 1000 * fold_i, n_components=n_comp_eff)
                    best_alpha_vec = torch.tensor(best_alpha_np, dtype=torch.float32)
                    best_alpha_folds.append(np.searchsorted(np.sort(ALPHAS), best_alpha_np))
                else:
                    N_A     = len(ALPHAS)
                    alpha_t = torch.tensor(ALPHAS, dtype=torch.float32)
                    inner_ll = np.zeros((N_A, n_m), np.float64)

                    X_raw_tr = X_raw_c[tr_m]
                    off_tr_t = (torch.tensor(off_tr_np[None, :, None],
                                             dtype=torch.float32, device=DEVICE)
                                if off_tr_np is not None else None)

                    if OUTER_CV == "purged_shuffle":
                        inner_onset = onset_c[tr_m]
                        _inner_iter = purged_shuffle_splits(
                            inner_onset,
                            N_INNER,
                            EMBARGO_MS,
                            seed=SEED + 1000 * fold_i,
                        )
                    else:
                        _inner_iter = block_splits(n_tr, N_INNER)

                    for itr_m, iva_m in _inner_iter:
                        if int(itr_m.sum()) <= n_comp_eff + 1:
                            continue
                        X_itr, X_iva = prep_fold_features(
                            X_raw_tr[itr_m], X_raw_tr[iva_m], n_comp_eff)
                        Y_itr = Y_tr_np[itr_m]
                        Y_iva = Y_tr_np[iva_m]
                        Xii = torch.tensor(X_itr, device=DEVICE
                                           ).unsqueeze(0).expand(N_A, -1, -1).contiguous()
                        Yii = torch.tensor(Y_itr, device=DEVICE
                                           ).unsqueeze(0).expand(N_A, -1, -1).contiguous()
                        off_itr = None
                        if off_tr_np is not None:
                            off_itr = (torch.tensor(off_tr_np[itr_m][None, :, None],
                                                    dtype=torch.float32, device=DEVICE)
                                       .expand(N_A, -1, -1).contiguous())
                        bf, bi = gpu_fit(Xii, Yii, alpha_t, offset_t=off_itr)
                        with torch.no_grad():
                            Xiv = torch.tensor(X_iva, device=DEVICE
                                               ).unsqueeze(0).expand(N_A, -1, -1).contiguous()
                            Yiv = torch.tensor(Y_iva, device=DEVICE
                                               ).unsqueeze(0).expand(N_A, -1, -1).contiguous()
                            off_iva = None
                            if off_tr_np is not None:
                                off_iva = (torch.tensor(off_tr_np[iva_m][None, :, None],
                                                        dtype=torch.float32, device=DEVICE)
                                           .expand(N_A, -1, -1).contiguous())
                            eta_iv = torch.bmm(Xiv, bf.transpose(1, 2)) + bi.transpose(1, 2)
                            if off_iva is not None:
                                eta_iv = eta_iv + off_iva
                            eta_iv = torch.clamp(eta_iv, -ETA_CLIP, ETA_CLIP)
                            inner_ll += (Yiv * eta_iv - torch.exp(eta_iv)
                                         ).sum(1).cpu().numpy()
                    best_ai = np.argmax(inner_ll, axis=0)
                    best_alpha_vec = torch.tensor(ALPHAS[best_ai], dtype=torch.float32)
                    best_alpha_folds.append(best_ai)
                    init_W = init_b = None

                # ── Outer fit ────────────────────────────────────────────────
                if WORDDUR_MODE:
                    # In --notebook_exact mode, use the inner-CV warm start, matching
                    # word_level_duration_cv_all_n.ipynb. Otherwise fit from the
                    # training-fold null model, which is more stable for sparse units.
                    Xtr_t  = torch.tensor(X_tr_np, dtype=torch.float32, device=DEVICE)
                    Ytr_t  = torch.tensor(Y_tr_np, dtype=torch.float32, device=DEVICE)
                    otr_t  = (torch.tensor(off_tr_np, dtype=torch.float32, device=DEVICE)
                              if off_tr_np is not None else None)
                    init_W_out = init_W if NOTEBOOK_EXACT else None
                    init_b_out = init_b if NOTEBOOK_EXACT else None
                    W_out, b_out = fit_collab(Xtr_t, Ytr_t, best_alpha_np,
                                             otr_t, init_W_out, init_b_out, max_iter=300)
                    with torch.no_grad():
                        Xte_t = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)
                        ote_t = (torch.tensor(off_te_np, dtype=torch.float32, device=DEVICE)
                                 if off_te_np is not None else None)
                        eta_te = Xte_t @ W_out + b_out
                        if ote_t is not None:
                            eta_te = eta_te + ote_t[:, None]
                        eta_te = torch.clamp(eta_te, -ETA_CLIP, ETA_CLIP)
                        mu_te = torch.exp(eta_te).cpu().numpy()
                        eta_tr = Xtr_t @ W_out + b_out
                        if otr_t is not None:
                            eta_tr = eta_tr + otr_t[:, None]
                        mu_tr = torch.exp(torch.clamp(eta_tr, -ETA_CLIP, ETA_CLIP)).cpu().numpy()
                    ll_model_folds.append(poisson_ll_full(Y_te_np, mu_te))
                    ll_train_model_folds.append(poisson_ll_full(Y_tr_np, mu_tr))
                    fold_betas.append(W_out.T.cpu().numpy())   # (n_m, P)
                    eta_te_np = eta_te.cpu().numpy()
                    del Xtr_t, Ytr_t, Xte_t, W_out, b_out, eta_te, eta_tr
                else:
                    X_tr_t = torch.tensor(X_tr_np[None], device=DEVICE)
                    Y_tr_t = torch.tensor(Y_tr_np[None], device=DEVICE)
                    off_tr_t_outer = (torch.tensor(off_tr_np[None, :, None],
                                                   dtype=torch.float32, device=DEVICE)
                                      if off_tr_np is not None else None)
                    bf, bi = gpu_fit_per_neuron(X_tr_t, Y_tr_t, best_alpha_vec,
                                                offset_t=off_tr_t_outer)
                    X_te_t = torch.tensor(X_te[None], device=DEVICE)
                    off_te_t = (torch.tensor(off_te_np[None, :, None],
                                             dtype=torch.float32, device=DEVICE)
                                if off_te_np is not None else None)
                    with torch.no_grad():
                        eta_te = torch.bmm(X_te_t, bf.transpose(1, 2)) + bi.transpose(1, 2)
                        if off_te_t is not None:
                            eta_te = eta_te + off_te_t
                        eta_te = torch.clamp(eta_te, -ETA_CLIP, ETA_CLIP)
                        mu_te = torch.exp(eta_te[0]).cpu().numpy()
                        eta_tr_b = torch.bmm(X_tr_t, bf.transpose(1, 2)) + bi.transpose(1, 2)
                        if off_tr_t_outer is not None:
                            eta_tr_b = eta_tr_b + off_tr_t_outer
                        mu_tr = torch.exp(torch.clamp(eta_tr_b, -ETA_CLIP, ETA_CLIP)[0]).cpu().numpy()
                    ll_model_folds.append(poisson_ll_full(Y_te_np, mu_te))
                    ll_train_model_folds.append(poisson_ll_full(Y_tr_np, mu_tr))
                    fold_betas.append(bf[0].cpu().numpy())   # (n_m, P)
                    eta_te_np = eta_te[0].cpu().numpy()
                    del X_tr_t, Y_tr_t, X_te_t, bf, bi, eta_te, eta_tr_b

                # Store fold predictions for Y-circular-shift permutation
                fold_meta.append({
                    "tr_m":          tr_m.copy(),
                    "te_m":          te_m.copy(),
                    "best_alpha_vec": best_alpha_vec.clone(),
                    "best_alpha_np": (best_alpha_np.copy() if WORDDUR_MODE
                                      else ALPHAS[best_alpha_folds[-1]]),
                    "off_tr_np":     off_tr_np.copy() if off_tr_np is not None else None,
                    "off_te_np":     off_te_np.copy() if off_te_np is not None else None,
                    "Y_te_np":       Y_te_np.copy(),
                    "eta_te_np":     eta_te_np.copy(),
                })

                torch.cuda.empty_cache()

            # ── Aggregate LL across folds ─────────────────────────────────────
            ll_model_real = sum(ll_model_folds)   # (n_m,)
            ll_null_real  = sum(ll_null_folds)    # (n_m,)
            ll_train_model_real = sum(ll_train_model_folds)
            ll_train_null_real  = sum(ll_train_null_folds)

            # McFadden's pseudo-R²: 1 - LL_model / LL_null
            with np.errstate(divide="ignore", invalid="ignore"):
                r2_real = np.where(ll_null_real < -0.1,
                                   1.0 - ll_model_real / ll_null_real,
                                   np.nan)
                r2_train = np.where(ll_train_null_real < -0.1,
                                    1.0 - ll_train_model_real / ll_train_null_real,
                                    np.nan)

            # Cross-fold beta reliability
            beta_mat = np.stack(fold_betas)   # (n_outer_folds, n_m, P)
            fold_pairs = list(combinations(range(beta_mat.shape[0]), 2))
            corr_sum = np.zeros(n_m)
            for fi, fj in fold_pairs:
                for m in range(n_m):
                    b1, b2 = beta_mat[fi, m], beta_mat[fj, m]
                    if b1.std() > 0 and b2.std() > 0:
                        corr_sum[m] += np.corrcoef(b1, b2)[0, 1]
            fold_beta_corr = (corr_sum / len(fold_pairs)
                              if fold_pairs else np.full(n_m, np.nan))
            patient_betas[(region, cond)] = beta_mat.mean(0)

            t_cv_done = time.time() - t_cv
            gpu_peak_mb = (torch.cuda.max_memory_allocated() / 1024**2
                           if DEVICE.type == "cuda" else 0.0)
            print(f"    CV done {t_cv_done:.1f}s  "
                  f"train_R²={np.nanmedian(r2_train):.4f}  "
                  f"test_R²={np.nanmedian(r2_real):.4f}  "
                  f"GPU_peak={gpu_peak_mb:.0f}MB", flush=True)

            # ── PERMUTATION SIGNIFICANCE TEST ─────────────────────────────────
            # Options controlled by --perm_type:
            #   xshuffle : global shuffle of X embeddings + model refit per perm
            #   xfoldshuffle : shuffle train-fold X rows + model refit; score original test X/Y
            #   xcirc    : circular shift X by random lag > tau + model refit
            #   xblock   : shuffle X within PERM_BLOCK_SIZE-word blocks + model refit
            #   yshuffle : shuffle train-fold Y rows + model refit; score original test Y
            #   yblock   : shuffle Y within PERM_BLOCK_SIZE-word windows (no refit)
            #   ycirc    : circular shift Y by random lag > tau (no refit)
            rng = np.random.default_rng(SEED)
            n_perm_eff = 0 if R2_ONLY else N_PERM
            ll_model_perms = np.full((n_perm_eff, n_m), np.nan)
            print(f"    perm_type={PERM_TYPE}{' (r2_only: skipping perms)' if R2_ONLY else ''}", flush=True)

            t_perm = time.time()

            if R2_ONLY:
                pass

            elif PERM_TYPE in ("xshuffle", "xcirc", "xblock"):
                if PERM_TYPE == "xcirc":
                    tau_words = max(1, int(np.ceil(compute_tau_embed(X_raw_c))))
                    min_lag   = max(tau_words, n_w // (N_OUTER * 2))
                for pi in range(N_PERM):
                    # ── generate permuted X ──
                    if PERM_TYPE == "xshuffle":
                        X_perm = X_raw_c[rng.permutation(n_w)]
                    elif PERM_TYPE == "xcirc":
                        lag = int(rng.integers(min_lag, n_w - min_lag))
                        X_perm = np.roll(X_raw_c, lag, axis=0)
                    else:  # xblock
                        idx = np.arange(n_w)
                        for s in range(0, n_w, PERM_BLOCK_SIZE):
                            blk = idx[s:s+PERM_BLOCK_SIZE]
                            idx[s:s+PERM_BLOCK_SIZE] = rng.permutation(blk)
                        X_perm = X_raw_c[idx]
                    # ── refit + score each fold ──
                    ll_perm_accum = np.zeros(n_m, np.float64)
                    for fd in fold_meta:
                        X_p_tr, X_p_te = prep_fold_features(
                            X_perm[fd["tr_m"]], X_perm[fd["te_m"]], n_comp_eff)
                        if WORDDUR_MODE:
                            Xs_tr_t = torch.tensor(X_p_tr, dtype=torch.float32, device=DEVICE)
                            Y_tr_f_t = torch.tensor(Y_v[fd["tr_m"]], dtype=torch.float32, device=DEVICE)
                            o_tr_t = (torch.tensor(fd["off_tr_np"], dtype=torch.float32, device=DEVICE)
                                      if fd["off_tr_np"] is not None else None)
                            W_p, b_p = fit_collab(Xs_tr_t, Y_tr_f_t, fd["best_alpha_np"], o_tr_t,
                                                  max_iter=LBFGS_ITER)
                            with torch.no_grad():
                                Xs_te_t = torch.tensor(X_p_te, dtype=torch.float32, device=DEVICE)
                                o_te_t = (torch.tensor(fd["off_te_np"], dtype=torch.float32, device=DEVICE)
                                          if fd["off_te_np"] is not None else None)
                                eta_p = Xs_te_t @ W_p + b_p
                                if o_te_t is not None: eta_p = eta_p + o_te_t[:, None]
                                mu_p = torch.exp(torch.clamp(eta_p, -ETA_CLIP, ETA_CLIP)).cpu().numpy()
                            del Xs_tr_t, Y_tr_f_t, Xs_te_t, W_p, b_p, eta_p
                        else:
                            Xstr_t = torch.tensor(X_p_tr[None], device=DEVICE)
                            Ytr_t  = torch.tensor(Y_v[fd["tr_m"]][None], device=DEVICE)
                            otr_t  = (torch.tensor(fd["off_tr_np"][None,:,None], dtype=torch.float32, device=DEVICE)
                                      if fd["off_tr_np"] is not None else None)
                            Xste_t = torch.tensor(X_p_te[None], device=DEVICE)
                            ote_t  = (torch.tensor(fd["off_te_np"][None,:,None], dtype=torch.float32, device=DEVICE)
                                      if fd["off_te_np"] is not None else None)
                            bf_p, bi_p = gpu_fit(Xstr_t, Ytr_t, fd["best_alpha_vec"],
                                                 offset_t=otr_t, max_iter=LBFGS_ITER)
                            with torch.no_grad():
                                eta_p = torch.bmm(Xste_t, bf_p.transpose(1,2)) + bi_p.transpose(1,2)
                                if ote_t is not None: eta_p = eta_p + ote_t
                                mu_p = torch.exp(torch.clamp(eta_p,-ETA_CLIP,ETA_CLIP)[0]).cpu().numpy()
                            del Xstr_t, Ytr_t, Xste_t, bf_p, bi_p, eta_p
                        ll_perm_accum += poisson_ll_full(fd["Y_te_np"], mu_p)
                        torch.cuda.empty_cache()
                    ll_model_perms[pi] = ll_perm_accum
                    if (pi + 1) % 50 == 0:
                        print(f"    perm {pi+1}/{N_PERM}  {time.time()-t_perm:.0f}s", flush=True)

            elif PERM_TYPE == "xfoldshuffle":
                # Shuffle embedding rows inside each outer-train fold, refit the
                # model on scrambled semantic predictors, then score on the original
                # held-out X/Y. This avoids train/test feature leakage from a global
                # shuffle while destroying the train-fold semantic↔neural alignment.
                for pi in range(N_PERM):
                    ll_perm_accum = np.zeros(n_m, np.float64)
                    for fd in fold_meta:
                        X_tr_real = X_raw_c[fd["tr_m"]]
                        X_p_tr_raw = X_tr_real[rng.permutation(X_tr_real.shape[0])]
                        X_p_tr, X_p_te = prep_fold_features(
                            X_p_tr_raw, X_raw_c[fd["te_m"]], n_comp_eff)

                        if WORDDUR_MODE:
                            Xs_tr_t = torch.tensor(X_p_tr, dtype=torch.float32, device=DEVICE)
                            Y_tr_f_t = torch.tensor(Y_v[fd["tr_m"]], dtype=torch.float32, device=DEVICE)
                            o_tr_t = (torch.tensor(fd["off_tr_np"], dtype=torch.float32, device=DEVICE)
                                      if fd["off_tr_np"] is not None else None)
                            W_p, b_p = fit_collab(Xs_tr_t, Y_tr_f_t, fd["best_alpha_np"], o_tr_t,
                                                  max_iter=LBFGS_ITER)
                            with torch.no_grad():
                                Xs_te_t = torch.tensor(X_p_te, dtype=torch.float32, device=DEVICE)
                                o_te_t = (torch.tensor(fd["off_te_np"], dtype=torch.float32, device=DEVICE)
                                          if fd["off_te_np"] is not None else None)
                                eta_p = Xs_te_t @ W_p + b_p
                                if o_te_t is not None: eta_p = eta_p + o_te_t[:, None]
                                mu_p = torch.exp(torch.clamp(eta_p, -ETA_CLIP, ETA_CLIP)).cpu().numpy()
                            del Xs_tr_t, Y_tr_f_t, Xs_te_t, W_p, b_p, eta_p
                        else:
                            Xstr_t = torch.tensor(X_p_tr[None], device=DEVICE)
                            Ytr_t  = torch.tensor(Y_v[fd["tr_m"]][None], device=DEVICE)
                            otr_t  = (torch.tensor(fd["off_tr_np"][None,:,None], dtype=torch.float32, device=DEVICE)
                                      if fd["off_tr_np"] is not None else None)
                            Xste_t = torch.tensor(X_p_te[None], device=DEVICE)
                            ote_t  = (torch.tensor(fd["off_te_np"][None,:,None], dtype=torch.float32, device=DEVICE)
                                      if fd["off_te_np"] is not None else None)
                            bf_p, bi_p = gpu_fit(Xstr_t, Ytr_t, fd["best_alpha_vec"],
                                                 offset_t=otr_t, max_iter=LBFGS_ITER)
                            with torch.no_grad():
                                eta_p = torch.bmm(Xste_t, bf_p.transpose(1,2)) + bi_p.transpose(1,2)
                                if ote_t is not None: eta_p = eta_p + ote_t
                                mu_p = torch.exp(torch.clamp(eta_p,-ETA_CLIP,ETA_CLIP)[0]).cpu().numpy()
                            del Xstr_t, Ytr_t, Xste_t, bf_p, bi_p, eta_p

                        ll_perm_accum += poisson_ll_full(fd["Y_te_np"], mu_p)
                        torch.cuda.empty_cache()
                    ll_model_perms[pi] = ll_perm_accum
                    if (pi + 1) % 50 == 0:
                        print(f"    perm {pi+1}/{N_PERM}  {time.time()-t_perm:.0f}s", flush=True)

            elif PERM_TYPE == "yshuffle":
                # Shuffle spike-count rows inside each outer-train fold, refit the
                # model on scrambled neural responses, then score on the original
                # held-out Y. This keeps the test target identical to the real model
                # while destroying the train-fold semantic↔neural alignment.
                for pi in range(N_PERM):
                    ll_perm_accum = np.zeros(n_m, np.float64)
                    for fd in fold_meta:
                        X_p_tr, X_p_te = prep_fold_features(
                            X_raw_c[fd["tr_m"]], X_raw_c[fd["te_m"]], n_comp_eff)
                        Y_tr_real = Y_v[fd["tr_m"]]
                        Y_p_tr = Y_tr_real[rng.permutation(Y_tr_real.shape[0])]

                        if WORDDUR_MODE:
                            Xs_tr_t = torch.tensor(X_p_tr, dtype=torch.float32, device=DEVICE)
                            Y_tr_f_t = torch.tensor(Y_p_tr, dtype=torch.float32, device=DEVICE)
                            o_tr_t = (torch.tensor(fd["off_tr_np"], dtype=torch.float32, device=DEVICE)
                                      if fd["off_tr_np"] is not None else None)
                            W_p, b_p = fit_collab(Xs_tr_t, Y_tr_f_t, fd["best_alpha_np"], o_tr_t,
                                                  max_iter=LBFGS_ITER)
                            with torch.no_grad():
                                Xs_te_t = torch.tensor(X_p_te, dtype=torch.float32, device=DEVICE)
                                o_te_t = (torch.tensor(fd["off_te_np"], dtype=torch.float32, device=DEVICE)
                                          if fd["off_te_np"] is not None else None)
                                eta_p = Xs_te_t @ W_p + b_p
                                if o_te_t is not None: eta_p = eta_p + o_te_t[:, None]
                                mu_p = torch.exp(torch.clamp(eta_p, -ETA_CLIP, ETA_CLIP)).cpu().numpy()
                            del Xs_tr_t, Y_tr_f_t, Xs_te_t, W_p, b_p, eta_p
                        else:
                            Xstr_t = torch.tensor(X_p_tr[None], device=DEVICE)
                            Ytr_t  = torch.tensor(Y_p_tr[None], device=DEVICE)
                            otr_t  = (torch.tensor(fd["off_tr_np"][None,:,None], dtype=torch.float32, device=DEVICE)
                                      if fd["off_tr_np"] is not None else None)
                            Xste_t = torch.tensor(X_p_te[None], device=DEVICE)
                            ote_t  = (torch.tensor(fd["off_te_np"][None,:,None], dtype=torch.float32, device=DEVICE)
                                      if fd["off_te_np"] is not None else None)
                            bf_p, bi_p = gpu_fit(Xstr_t, Ytr_t, fd["best_alpha_vec"],
                                                 offset_t=otr_t, max_iter=LBFGS_ITER)
                            with torch.no_grad():
                                eta_p = torch.bmm(Xste_t, bf_p.transpose(1,2)) + bi_p.transpose(1,2)
                                if ote_t is not None: eta_p = eta_p + ote_t
                                mu_p = torch.exp(torch.clamp(eta_p,-ETA_CLIP,ETA_CLIP)[0]).cpu().numpy()
                            del Xstr_t, Ytr_t, Xste_t, bf_p, bi_p, eta_p

                        ll_perm_accum += poisson_ll_full(fd["Y_te_np"], mu_p)
                        torch.cuda.empty_cache()
                    ll_model_perms[pi] = ll_perm_accum
                    if (pi + 1) % 50 == 0:
                        print(f"    perm {pi+1}/{N_PERM}  {time.time()-t_perm:.0f}s", flush=True)

            elif PERM_TYPE == "yblock":
                # Shuffle Y within fixed-size blocks — preserves local autocorrelation
                for pi in range(N_PERM):
                    idx = np.arange(n_w)
                    for s in range(0, n_w, PERM_BLOCK_SIZE):
                        blk = idx[s:s+PERM_BLOCK_SIZE]
                        idx[s:s+PERM_BLOCK_SIZE] = rng.permutation(blk)
                    Y_shuf = Y_v[idx]
                    ll_perm_accum = np.zeros(n_m, np.float64)
                    for fd in fold_meta:
                        Y_te_s = Y_shuf[fd["te_m"]]
                        eta_te = fd["eta_te_np"]
                        ll_perm_accum += (Y_te_s * eta_te - np.exp(eta_te) - gammaln(Y_te_s + 1)).sum(0)
                    ll_model_perms[pi] = ll_perm_accum
                    if (pi + 1) % 100 == 0:
                        print(f"    perm {pi+1}/{N_PERM}  {time.time()-t_perm:.0f}s", flush=True)

            elif PERM_TYPE == "ycirc":  # Y circular shift
                tau_words = max(1, int(np.ceil(compute_tau_embed(X_raw_c))))
                min_lag   = max(tau_words, n_w // (N_OUTER * 2))
                for pi in range(N_PERM):
                    lag = int(rng.integers(min_lag, n_w - min_lag))
                    Y_shifted = np.roll(Y_v, lag, axis=0)
                    ll_perm_accum = np.zeros(n_m, np.float64)
                    for fd in fold_meta:
                        Y_te_s = Y_shifted[fd["te_m"]]
                        eta_te = fd["eta_te_np"]
                        ll_perm_accum += (Y_te_s * eta_te - np.exp(eta_te) - gammaln(Y_te_s + 1)).sum(0)
                    ll_model_perms[pi] = ll_perm_accum
                    if (pi + 1) % 100 == 0:
                        print(f"    perm {pi+1}/{N_PERM}  {time.time()-t_perm:.0f}s", flush=True)

            t_perm_done = time.time() - t_perm
            print(f"    perms done {t_perm_done:.1f}s", flush=True)

            # p-value: fraction of permutations where perm LL >= real LL (+1 correction)
            p_vals = (np.ones(n_m, dtype=np.float64) if R2_ONLY
                      else (np.sum(ll_model_perms >= ll_model_real[None, :], axis=0) + 1) / (N_PERM + 1))
            p_fdr, fdr_significant = fdr_bh(p_vals)
            raw_significant = p_vals < FDR_ALPHA
            significant = raw_significant

            ll_perm_mean = (np.full(n_m, np.nan, dtype=np.float64) if R2_ONLY
                            else np.nanmean(ll_model_perms, axis=0))  # (n_m,) mean shuffled-null LL
            patient_perm_nulls[(region, cond)] = {
                "ll_perms": ll_model_perms.astype(np.float32, copy=True),
                "ll_real": ll_model_real.astype(np.float32, copy=True),
                "ll_null": ll_null_real.astype(np.float32, copy=True),
                "r2_real": r2_real.astype(np.float32, copy=True),
                "p_perm": p_vals.astype(np.float32, copy=True),
                "perm_type": PERM_TYPE,
                "outer_cv": OUTER_CV,
                "embargo_ms": (
                    EMBARGO_MS if OUTER_CV == "purged_shuffle" else np.nan
                ),
            }

            n_sig = significant.sum()
            n_sig_fdr = fdr_significant.sum()
            med_r2 = float(np.nanmedian(r2_real[significant])) if n_sig else float("nan")
            print(f"    raw significant: {n_sig}/{n_m} ({100*n_sig/n_m:.1f}%)  "
                  f"median_R²(raw_sig)={med_r2:.4f}  FDR_sig={n_sig_fdr}/{n_m}" if n_sig else
                  f"    raw significant: 0/{n_m}  FDR_sig={n_sig_fdr}/{n_m}", flush=True)

            for m in range(n_m):
                patient_rows.append({
                    "patient":        patient_ID,
                    "region":         region,
                    "condition":      cond,
                    "neuron_idx":     int(m),
                    "layer":          LAYER,
                    "r2":             float(r2_real[m])  if not np.isnan(r2_real[m])  else np.nan,
                    "r2_train":       float(r2_train[m]) if not np.isnan(r2_train[m]) else np.nan,
                    "ll_real":        float(ll_model_real[m]),
                    "ll_null":        float(ll_null_real[m]),
                    "ll_perm_mean":   float(ll_perm_mean[m]),
                    "p_perm":         float(p_vals[m]),
                    "p_fdr":          float(p_fdr[m]),
                    "significant":    bool(significant[m]),
                    "raw_significant": bool(raw_significant[m]),
                    "fdr_significant": bool(fdr_significant[m]),
                    "n_spikes":       int(Y_v[:, m].sum()),
                    "best_alpha":     float(np.nanmedian(
                                          [ALPHAS[ba[m]] for ba in best_alpha_folds])),
                    "fold_beta_corr": float(fold_beta_corr[m]),
                    "outer_cv":       OUTER_CV,
                    "embargo_ms":     (
                        EMBARGO_MS if OUTER_CV == "purged_shuffle" else np.nan
                    ),
                    "cv_time_s":      float(t_cv_done),
                    "gpu_peak_mb":    float(gpu_peak_mb),
                })

        # ── Split-half reliability + r_cross + noise ceiling (--reliability) ──────
        # Matches BERTRidgeBetaCorr.ipynb's reliability_mod.run_beta_reliability_all_neurons:
        # per neuron, fits beta on full self/other (r_cross), a null distribution for
        # r_cross, and within-condition split-half reliability (Spearman-Brown corrected)
        # combined into a cross-condition noise ceiling sqrt(SB(r_self)*SB(r_other)).
        if RELIABILITY and not ALL_CONDITIONS and "self" in region_cond_xy and "other" in region_cond_xy:
            X_self_rel,  Y_self_rel  = region_cond_xy["self"]
            X_other_rel, Y_other_rel = region_cond_xy["other"]
            rel_meta = {
                "balanced_trials": bool(REL_BALANCE_TRIALS),
                "n_self_raw": int(X_self_rel.shape[0]),
                "n_other_raw": int(X_other_rel.shape[0]),
                "n_self_used": int(X_self_rel.shape[0]),
                "n_other_used": int(X_other_rel.shape[0]),
                "balance_seed": int(REL_BALANCE_SEED),
            }
            if REL_BALANCE_TRIALS:
                n_s = int(X_self_rel.shape[0])
                n_o = int(X_other_rel.shape[0])
                n_t = min(n_s, n_o)
                seed_offset = sum(ord(ch) for ch in f"{patient_ID}:{region}")
                rng_bal = np.random.default_rng(REL_BALANCE_SEED + seed_offset)
                idx_s = np.arange(n_s)
                idx_o = np.arange(n_o)
                if n_s > n_t:
                    idx_s = np.sort(rng_bal.choice(idx_s, size=n_t, replace=False))
                if n_o > n_t:
                    idx_o = np.sort(rng_bal.choice(idx_o, size=n_t, replace=False))
                X_self_rel, Y_self_rel = X_self_rel[idx_s], Y_self_rel[idx_s]
                X_other_rel, Y_other_rel = X_other_rel[idx_o], Y_other_rel[idx_o]
                rel_meta.update({
                    "n_self_used": int(X_self_rel.shape[0]),
                    "n_other_used": int(X_other_rel.shape[0]),
                })
                print(f"  {region}: balanced reliability trials "
                      f"self {n_s}->{X_self_rel.shape[0]}, "
                      f"other {n_o}->{X_other_rel.shape[0]}", flush=True)
            print(f"  {region}: running reliability (split-half + r_cross + ceiling)...", flush=True)
            t_rel = time.time()
            rel_cfg = _reliability_mod.ReliabilityConfig(
                n_null=REL_N_NULL,
                n_half_splits=REL_N_HALF_SPLITS,
                alphas=tuple(float(a) for a in ALPHAS),
                n_jobs=REL_N_JOBS,
            )
            rel_results = _reliability_mod.run_beta_reliability_all_neurons(
                X_self=X_self_rel, X_other=X_other_rel,
                Y_self=Y_self_rel, Y_other=Y_other_rel,
                cfg=rel_cfg, verbose=True,
            )
            patient_reliability[region] = rel_results
            if REL_BALANCE_TRIALS:
                patient_reliability_meta[region] = rel_meta
            print(f"  {region}: reliability done ({time.time()-t_rel:.1f}s)", flush=True)

            if COSINE_BIN_SPLIT and "self" in cosine_bin_masks and "other" in cosine_bin_masks:
                masks_s = cosine_bin_masks["self"]
                masks_o = cosine_bin_masks["other"]
                bin_results = {}
                for bin_name, bm_s, bm_o in zip(COSINE_BIN_LABELS, masks_s, masks_o):
                    print(f"  {region}: running reliability [{bin_name} bin, "
                          f"n_self={bm_s.sum()}, n_other={bm_o.sum()}]...", flush=True)
                    t_bin = time.time()
                    bin_res = _reliability_mod.run_beta_reliability_all_neurons(
                        X_self=X_self_rel[bm_s], X_other=X_other_rel[bm_o],
                        Y_self=Y_self_rel[bm_s], Y_other=Y_other_rel[bm_o],
                        cfg=rel_cfg, verbose=True,
                    )
                    bin_results[bin_name] = bin_res
                    print(f"  {region}: [{bin_name}] done ({time.time()-t_bin:.1f}s)", flush=True)
                patient_reliability_cosine[region] = bin_results

    if patient_rows:
        df_p = pd.DataFrame(patient_rows)
        save_obj = {"df": df_p, "betas": patient_betas, "perm_nulls": patient_perm_nulls}
        # Carry forward any cosine-bin results computed by a *different* --cosine_n_bins
        # setting in a prior run (e.g. the earlier near/far median-split), so re-running
        # with a new n_bins value doesn't silently discard it.
        if _prior_obj is not None:
            for k, v in _prior_obj.items():
                if k.startswith("reliability_cosine_bin") and k != COSINE_BIN_KEY:
                    save_obj[k] = v
        if RELIABILITY:
            save_obj["reliability"] = patient_reliability
            if patient_reliability_meta:
                save_obj["reliability_meta"] = patient_reliability_meta
        if COSINE_BIN_SPLIT:
            save_obj[COSINE_BIN_KEY] = patient_reliability_cosine
        with open(out_path, "wb") as f:
            pickle.dump(save_obj, f)
        print(f"  Saved → {out_path}", flush=True)
        all_rows.append(df_p)

    torch.cuda.empty_cache()

# ── AGGREGATE ─────────────────────────────────────────────────────────────────

if not all_rows:
    print("No results."); sys.exit(0)

data     = pd.concat(all_rows, ignore_index=True)
agg_path = os.path.join(OUT_DIR, f"L{LAYER:02d}_all.pkl")
with open(agg_path, "wb") as f:
    pickle.dump(data, f)
print(f"\nAggregated → {agg_path}")

print("\n=== SUMMARY ===")
for region in data["region"].unique():
    for cond in data["condition"].unique():
        sub = data[(data["region"] == region) & (data["condition"] == cond)]
        if sub.empty: continue
        n_sig = sub["significant"].sum()
        n_tot = len(sub)
        med   = sub.loc[sub["significant"], "r2"].median() if n_sig else float("nan")
        print(f"  {region:12s} {cond:5s}  {n_sig:3d}/{n_tot}  "
              f"({100*n_sig/n_tot:.1f}%)  median_R²(sig)={med:.4f}" if n_sig
              else f"  {region:12s} {cond:5s}  {n_sig:3d}/{n_tot}  (0%)")

print("\nDone.")
