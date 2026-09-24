"""
Beta reliability + noise ceiling analysis for Poisson-ridge encoding models.

This module implements:
1) Per-neuron cross-condition beta similarity (self vs other): r_cross = corr(beta_self, beta_other)
2) A null distribution for r_cross via permutation (configurable)
3) Within-condition split-half beta reliability for SELF and OTHER, with Spearman–Brown correction
4) A combined cross-condition noise ceiling per neuron: ceil_i = sqrt(SB(r_self_i) * SB(r_other_i))
5) Group-level inference:
   - pooled null test of mean r_cross
   - paired bootstrap test that mean ceiling exceeds mean observed r_cross

Design choices / fixes vs common pitfalls:
- Train/test leakage is irrelevant here because this is *parameter reliability*, not generalization;
  split-halves are random by default but you can choose contiguous halves if you want to respect time structure.
- Permutation null is done in a controlled way: for each null iteration we break X↔y mapping within each condition
  inside the same dataset, fit betas, and compute r_cross.
- Alpha can be chosen once per neuron per condition (recommended) or re-selected per split (slower/noisier).

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple, Any, cast

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

# ── GPU helpers (used when torch+CUDA is available) ───────────────────────────
try:
    import torch as _torch
    _GPU_AVAILABLE = _torch.cuda.is_available()
except ImportError:
    _torch = None          # type: ignore[assignment]
    _GPU_AVAILABLE = False

_GPU_DEVICE = "cuda" if _GPU_AVAILABLE else "cpu"


def _t(arr: np.ndarray, device: str = _GPU_DEVICE):
    return _torch.tensor(np.asarray(arr, dtype="float32"), device=device)


def _fit_all_neurons_gpu(
    X_np: np.ndarray,
    Y_np: np.ndarray,
    alpha_per_neuron: np.ndarray,
    *,
    max_iter: int = 100,
    tol: float = 1e-5,
    device: str = _GPU_DEVICE,
) -> np.ndarray:
    """
    Batched Poisson-ridge Newton-Raphson for ALL neurons simultaneously on GPU.
    X_np            : (n, p)  float32
    Y_np            : (n, k)  float32
    alpha_per_neuron: (k,)    float32
    Returns coef    : (k, p)  float32  (no intercept)
    """
    X_t = _t(X_np, device); Y_t = _t(Y_np, device)
    n, p = X_t.shape; k = Y_t.shape[1]; dev = X_t.device
    Xb = _torch.cat([_torch.ones(n, 1, device=dev), X_t], dim=1)
    W  = _torch.zeros(p + 1, k, device=dev)
    rm = _torch.ones(p + 1, device=dev); rm[0] = 0.0
    at = _t(alpha_per_neuron, device)
    for _ in range(max_iter):
        eta = _torch.clamp(Xb @ W, -30.0, 30.0)
        mu  = _torch.exp(eta)
        G   = Xb.T @ (mu - Y_t) + (rm[:, None] * at[None, :]) * W
        if G.abs().max().item() < tol:
            break
        XW  = Xb.unsqueeze(2) * _torch.sqrt(mu).unsqueeze(1)
        H   = _torch.einsum("npi,nqi->ipq", XW, XW)
        H  += at[:, None, None] * _torch.diag(rm)[None]
        W  -= _torch.linalg.solve(H, G.T).T
    return W[1:].T.cpu().numpy().astype(np.float32)   # (k, p)


def _alpha_cv_gpu_batched(
    X_np: np.ndarray,
    Y_np: np.ndarray,
    alphas: Sequence[float],
    *,
    n_splits: int = 5,
    device: str = _GPU_DEVICE,
) -> np.ndarray:
    """
    K-fold alpha selection for ALL neurons simultaneously on GPU.
    Returns best_alpha : (k,) float32
    """
    n, k  = X_np.shape[0], Y_np.shape[1]
    fold  = n // n_splits
    best  = np.full(k, float(alphas[0]), dtype=np.float32)
    best_ll = np.full(k, -np.inf)
    for alpha in alphas:
        at      = _torch.full((k,), float(alpha), dtype=_torch.float32, device=device)
        fll     = np.zeros((n_splits, k))
        for fi in range(n_splits):
            vs = fi * fold
            ve = (vs + fold) if fi < n_splits - 1 else n
            va = np.arange(vs, ve)
            tr = np.concatenate([np.arange(0, vs), np.arange(ve, n)])
            X_tr = _t(X_np[tr], device); Y_tr = _t(Y_np[tr], device)
            X_va = _t(X_np[va], device); Y_va = _t(Y_np[va], device)
            coef = _torch.tensor(
                _fit_all_neurons_gpu(X_np[tr], Y_np[tr],
                                     np.full(k, float(alpha), np.float32),
                                     max_iter=80, tol=1e-4, device=device),
                device=device,
            ).T                                              # (p, k)
            # intercept: fit a separate intercept-only model for stability
            mu_va = _torch.exp(_torch.clamp(X_va @ coef, -30, 30))
            fll[fi] = (
                Y_va * _torch.log(_torch.clamp(mu_va, 1e-10, None)) - mu_va
            ).sum(0).cpu().numpy()
        mean_ll = fll.mean(0)
        better  = mean_ll > best_ll
        best[better]    = alpha
        best_ll[better] = mean_ll[better]
    return best
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt  # keep this at top level


try:
    import matplotlib.pyplot as plt
    _HAVE_PLT = True
except Exception:
    plt = None
    _HAVE_PLT = False


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()

    if a.size < 2 or b.size < 2:
        return float("nan")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return float("nan")

    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")

    return float(np.dot(a, b) / denom)


def _spearman_brown(r: float) -> float:
    """Spearman–Brown prophecy formula for split-half reliability."""
    if not np.isfinite(r):
        return np.nan
    denom = 1.0 + r
    if denom == 0:
        return np.nan
    return float((2.0 * r) / denom)


def poisson_ll_numpy(y_true: np.ndarray, mu_pred: np.ndarray) -> float:
    """Poisson log-likelihood sum (includes constants)."""
    from scipy.special import gammaln
    y_true = np.asarray(y_true, dtype=float)
    mu_pred = np.clip(np.asarray(mu_pred, dtype=float), 1e-10, None)
    return float(np.sum(y_true * np.log(mu_pred) - mu_pred - gammaln(y_true + 1)))


# -------------------------
# Model fitting helpers
# -------------------------

FitBetaFn = Callable[[np.ndarray, np.ndarray, float], np.ndarray]


def fit_poisson_ridge_beta_sklearn(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    *,
    fit_intercept: bool = True,
    max_iter: int = 1000,
) -> np.ndarray:
    """
    Default beta fitter using sklearn PoissonRegressor (L2 penalty = alpha).
    Returns coefficients in whatever feature space X is given in (so standardize X outside if desired).
    """
    from sklearn.linear_model import PoissonRegressor

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    model = PoissonRegressor(alpha=float(alpha), fit_intercept=fit_intercept, max_iter=max_iter)
    model.fit(X, y)
    return model.coef_.astype(float, copy=True)


def select_alpha_cv(
    X: np.ndarray,
    y: np.ndarray,
    alphas: Sequence[float],
    *,
    split: Literal["random_kfold", "contiguous_kfold"] = "random_kfold",
    n_splits: int = 5,
    random_state: int = 0,
    standardize: bool = True,
) -> float:
    """
    Choose alpha by inner CV on the provided (X,y) only.
    Scoring: mean Poisson log-likelihood on validation splits.

    Note: uses PoissonRegressor internally for LL scoring (keeps intercept correct).
    """
    from sklearn.linear_model import PoissonRegressor

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = X.shape[0]
    if n < max(10, 2 * n_splits):
        return float(alphas[0])

    if split == "contiguous_kfold":
        idx = np.arange(n)
        folds = np.array_split(idx, n_splits)
        splits_list = []
        for k in range(n_splits):
            va = folds[k]
            tr = np.concatenate([folds[j] for j in range(n_splits) if j != k])
            splits_list.append((tr, va))
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits_list = list(kf.split(X, y))

    best_alpha = float(alphas[0])
    best_score = -np.inf

    for a in alphas:
        ll_list: List[float] = []
        for tr, va in splits_list:
            Xtr, Xva = X[tr], X[va]
            ytr, yva = y[tr], y[va]

            if standardize:
                sc = StandardScaler(with_mean=True, with_std=True)
                Xtr_s = sc.fit_transform(Xtr)
                Xva_s = sc.transform(Xva)
            else:
                Xtr_s, Xva_s = Xtr, Xva

            m = PoissonRegressor(alpha=float(a), fit_intercept=True, max_iter=1000)
            m.fit(Xtr_s, ytr)
            mu_va = m.predict(Xva_s)
            ll_list.append(poisson_ll_numpy(yva, mu_va))

        score = float(np.mean(ll_list)) if ll_list else -np.inf
        if score > best_score:
            best_score = score
            best_alpha = float(a)

    return best_alpha


# -------------------------
# Split logic for reliability
# -------------------------

@dataclass(frozen=True)
class ReliabilityConfig:
    n_null: int = 200
    n_half_splits: int = 200
    alphas: Tuple[float, ...] = tuple(np.logspace(-3, 3, 30))
    alpha_cv_split: Literal["random_kfold", "contiguous_kfold"] = "random_kfold"
    alpha_cv_folds: int = 5
    half_split_mode: Literal["random", "contiguous"] = "random"
    clamp_negative_to_zero: bool = True
    reuse_alpha: bool = True  # choose alpha once per neuron/condition using full data
    random_state: int = 0
    n_jobs: int = 8
    standardize_X: bool = True

    # Null type for r_cross
    null_mode: Literal["row_permute_X", "shuffle_y"] = "row_permute_X"
    # Which side(s) to randomize when building the cross-condition null.
    # Recommended default: "other" (keep self fixed, randomize other).
    null_side: Literal["self", "other", "both"] = "other"


def _half_split_indices(n: int, rng: np.random.Generator, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    if mode == "contiguous":
        cut = n // 2
        return idx[:cut], idx[cut:]
    perm = rng.permutation(idx)
    cut = n // 2
    return perm[:cut], perm[cut:]


def _standardize_train_only(X: np.ndarray) -> np.ndarray:
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


# -------------------------
# Per-neuron computations
# -------------------------

def compute_r_cross_and_null_for_neuron(
    X_self: np.ndarray,
    y_self: np.ndarray,
    X_other: np.ndarray,
    y_other: np.ndarray,
    cfg: ReliabilityConfig,
    *,
    fit_beta: FitBetaFn = fit_poisson_ridge_beta_sklearn,
    alpha_self: Optional[float] = None,
    alpha_other: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """
    For one neuron:
      - fit betas on full self and other
      - r_cross between betas
      - null distribution for r_cross by breaking mapping within selected side(s)

    Null construction:
      - null_side="other" (default): keep beta_self fixed; randomize OTHER data each iteration to get beta_other^null
      - null_side="self": keep beta_other fixed; randomize SELF data each iteration
      - null_side="both": randomize both sides each iteration

    Randomization method:
      - null_mode="row_permute_X": permute rows of X relative to y (preserves X covariance, preserves y marginal)
      - null_mode="shuffle_y": permute y relative to X
    """
    if rng is None:
        rng = np.random.default_rng(cfg.random_state)

    y_self = np.asarray(y_self, dtype=float)
    y_other = np.asarray(y_other, dtype=float)
    if y_self.size == 0 or y_other.size == 0:
        return {"r_cross": np.nan, "null_distribution": np.array([]), "alpha_self": np.nan, "alpha_other": np.nan}

    # Choose alpha once per condition (recommended) unless provided
    if alpha_self is None or alpha_other is None:
        if cfg.reuse_alpha:
            alpha_self = select_alpha_cv(
                X_self, y_self, cfg.alphas,
                split=cfg.alpha_cv_split, n_splits=cfg.alpha_cv_folds,
                random_state=cfg.random_state + 11,
                standardize=cfg.standardize_X
            )
            alpha_other = select_alpha_cv(
                X_other, y_other, cfg.alphas,
                split=cfg.alpha_cv_split, n_splits=cfg.alpha_cv_folds,
                random_state=cfg.random_state + 17,
                standardize=cfg.standardize_X
            )
        else:
            alpha_self = float(cfg.alphas[0])
            alpha_other = float(cfg.alphas[0])

    # Standardize within each condition separately
    Xs_s = _standardize_train_only(X_self) if cfg.standardize_X else np.asarray(X_self, dtype=float)
    Xs_o = _standardize_train_only(X_other) if cfg.standardize_X else np.asarray(X_other, dtype=float)

    beta_s = fit_beta(Xs_s, y_self, float(alpha_self))
    beta_o = fit_beta(Xs_o, y_other, float(alpha_other))
    r_cross = _safe_pearson(beta_s, beta_o)

    null = np.full(cfg.n_null, np.nan)
    for b in range(cfg.n_null):
        shuffle_self = (cfg.null_side in ("self", "both"))
        shuffle_other = (cfg.null_side in ("other", "both"))

        if shuffle_self:
            if cfg.null_mode == "shuffle_y":
                ys = rng.permutation(y_self)
                Xs_s_null = Xs_s
            else:
                Xs_s_null = Xs_s[rng.permutation(Xs_s.shape[0])]
                ys = y_self
            beta_s_null = fit_beta(Xs_s_null, ys, float(alpha_self))
        else:
            beta_s_null = beta_s

        if shuffle_other:
            if cfg.null_mode == "shuffle_y":
                yo = rng.permutation(y_other)
                Xs_o_null = Xs_o
            else:
                Xs_o_null = Xs_o[rng.permutation(Xs_o.shape[0])]
                yo = y_other
            beta_o_null = fit_beta(Xs_o_null, yo, float(alpha_other))
        else:
            beta_o_null = beta_o

        null[b] = _safe_pearson(beta_s_null, beta_o_null)

    return {
        "r_cross": float(r_cross) if np.isfinite(r_cross) else np.nan,
        "null_distribution": null.astype(float),
        "alpha_self": float(alpha_self),
        "alpha_other": float(alpha_other),
    }


def compute_split_half_beta_corr_samples(
    X: np.ndarray,
    y: np.ndarray,
    cfg: ReliabilityConfig,
    *,
    fit_beta: FitBetaFn,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Split-half correlation samples of betas (uncorrected) for one condition.
    """
    n = X.shape[0]
    samples = np.full(cfg.n_half_splits, np.nan)
    if n < 4:
        return samples

    for b in range(cfg.n_half_splits):
        i1, i2 = _half_split_indices(n, rng, cfg.half_split_mode)

        X1, y1 = X[i1], y[i1]
        X2, y2 = X[i2], y[i2]

        X1s = _standardize_train_only(X1) if cfg.standardize_X else np.asarray(X1, dtype=float)
        X2s = _standardize_train_only(X2) if cfg.standardize_X else np.asarray(X2, dtype=float)

        b1 = fit_beta(X1s, y1, float(alpha))
        b2 = fit_beta(X2s, y2, float(alpha))
        samples[b] = _safe_pearson(b1, b2)

    return samples


def compute_noise_ceiling_for_neuron(
    X_self: np.ndarray,
    y_self: np.ndarray,
    X_other: np.ndarray,
    y_other: np.ndarray,
    cfg: ReliabilityConfig,
    *,
    fit_beta: FitBetaFn = fit_poisson_ridge_beta_sklearn,
    alpha_self: float,
    alpha_other: float,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """
    For one neuron:
      - split-half reliability in SELF and OTHER
      - Spearman–Brown correction
      - combine into cross-condition ceiling: sqrt(SB(r_self)*SB(r_other))

    NOTE: pass a per-neuron rng so different neurons use different split draws.
    """
    if rng is None:
        rng = np.random.default_rng(cfg.random_state + 12345)

    rs = compute_split_half_beta_corr_samples(X_self, y_self, cfg, fit_beta=fit_beta, alpha=alpha_self, rng=rng)
    ro = compute_split_half_beta_corr_samples(X_other, y_other, cfg, fit_beta=fit_beta, alpha=alpha_other, rng=rng)

    ceil_samples = np.full(cfg.n_half_splits, np.nan)
    rs_sb_samples = np.full(cfg.n_half_splits, np.nan)
    ro_sb_samples = np.full(cfg.n_half_splits, np.nan)
    for i in range(cfg.n_half_splits):
        r_s = rs[i]
        r_o = ro[i]

        if cfg.clamp_negative_to_zero:
            if np.isfinite(r_s):
                r_s = max(r_s, 0.0)
            if np.isfinite(r_o):
                r_o = max(r_o, 0.0)

        rs_sb = _spearman_brown(r_s)
        ro_sb = _spearman_brown(r_o)
        rs_sb_samples[i] = rs_sb
        ro_sb_samples[i] = ro_sb

        if np.isfinite(rs_sb) and np.isfinite(ro_sb) and rs_sb >= 0 and ro_sb >= 0:
            ceil_samples[i] = float(np.sqrt(rs_sb * ro_sb))

    valid = ceil_samples[np.isfinite(ceil_samples)]
    valid_self = rs_sb_samples[np.isfinite(rs_sb_samples)]
    valid_other = ro_sb_samples[np.isfinite(ro_sb_samples)]
    return {
        "self_reliability_samples": rs_sb_samples,
        "other_reliability_samples": ro_sb_samples,
        "self_reliability_mean": float(np.mean(valid_self)) if valid_self.size else np.nan,
        "self_reliability_median": float(np.median(valid_self)) if valid_self.size else np.nan,
        "other_reliability_mean": float(np.mean(valid_other)) if valid_other.size else np.nan,
        "other_reliability_median": float(np.median(valid_other)) if valid_other.size else np.nan,
        "ceil_samples": ceil_samples,
        "ceil_mean": float(np.mean(valid)) if valid.size else np.nan,
        "ceil_median": float(np.median(valid)) if valid.size else np.nan,
        "ceil_ci_low": float(np.percentile(valid, 2.5)) if valid.size else np.nan,
        "ceil_ci_high": float(np.percentile(valid, 97.5)) if valid.size else np.nan,
    }


def run_beta_reliability_all_neurons(
    X_self: np.ndarray,
    X_other: np.ndarray,
    Y_self: np.ndarray,
    Y_other: np.ndarray,
    cfg: ReliabilityConfig,
    *,
    fit_beta: FitBetaFn = fit_poisson_ridge_beta_sklearn,
    neuron_indices: Optional[Sequence[int]] = None,
    verbose: bool = True,
) -> List[Dict]:
    """
    Computes per-neuron:
      - r_cross + null distribution
      - cross-condition noise ceiling from split-halves within condition

    Returns a list of dicts (one per neuron).
    """
    if neuron_indices is None:
        neuron_indices = list(range(Y_self.shape[1]))

    # Precompute alphas per neuron per condition (recommended)
    cached_alphas: Dict[int, Tuple[float, float]] = {}
    if cfg.reuse_alpha:
        active = [i for i in neuron_indices
                  if np.all(np.isfinite(Y_self[:, i])) and np.all(np.isfinite(Y_other[:, i]))
                  and np.std(Y_self[:, i]) > 0 and np.std(Y_other[:, i]) > 0]
        for i in set(neuron_indices) - set(active):
            cached_alphas[i] = (np.nan, np.nan)

        if active:
            if _GPU_AVAILABLE:
                # Batch all active neurons simultaneously on GPU
                Ys_mat = Y_self[:, active].astype(np.float32)
                Yo_mat = Y_other[:, active].astype(np.float32)
                Xs = (X_self - X_self.mean(0)) / np.where(X_self.std(0) == 0, 1, X_self.std(0))
                Xo = (X_other - X_other.mean(0)) / np.where(X_other.std(0) == 0, 1, X_other.std(0))
                alphas_s = _alpha_cv_gpu_batched(
                    Xs.astype(np.float32), Ys_mat, cfg.alphas,
                    n_splits=cfg.alpha_cv_folds,
                )
                alphas_o = _alpha_cv_gpu_batched(
                    Xo.astype(np.float32), Yo_mat, cfg.alphas,
                    n_splits=cfg.alpha_cv_folds,
                )
                for j, i in enumerate(active):
                    cached_alphas[i] = (float(alphas_s[j]), float(alphas_o[j]))
            else:
                # CPU fallback: parallelise across neurons
                def _cv_one(i):
                    return i, (
                        select_alpha_cv(X_self, Y_self[:, i], cfg.alphas,
                                        split=cfg.alpha_cv_split, n_splits=cfg.alpha_cv_folds,
                                        random_state=cfg.random_state + 101 + i,
                                        standardize=cfg.standardize_X),
                        select_alpha_cv(X_other, Y_other[:, i], cfg.alphas,
                                        split=cfg.alpha_cv_split, n_splits=cfg.alpha_cv_folds,
                                        random_state=cfg.random_state + 303 + i,
                                        standardize=cfg.standardize_X),
                    )
                for i, pair in Parallel(n_jobs=cfg.n_jobs, prefer="threads")(
                    delayed(_cv_one)(i) for i in active
                ):
                    cached_alphas[i] = pair

    def _worker(i: int) -> Dict[str, Any]:
        ys = Y_self[:, i]
        yo = Y_other[:, i]
        out: Dict[str, Any] = {"neuron": int(i)}

        if (not np.all(np.isfinite(ys))) or (not np.all(np.isfinite(yo))) or np.std(ys) == 0 or np.std(yo) == 0:
            out.update({"r_cross": np.nan, "null_distribution": np.array([]),
                        "alpha_self": np.nan, "alpha_other": np.nan,
                        "self_reliability_mean": np.nan, "self_reliability_median": np.nan,
                        "other_reliability_mean": np.nan, "other_reliability_median": np.nan,
                        "self_reliability_samples": np.array([]), "other_reliability_samples": np.array([]),
                        "ceil_mean": np.nan, "ceil_median": np.nan, "ceil_ci_low": np.nan, "ceil_ci_high": np.nan})  # pyright: ignore[reportCallIssue]
            return out

        a_s, a_o = (None, None)
        if cfg.reuse_alpha:
            a_s, a_o = cached_alphas.get(i, (np.nan, np.nan))
            if not (np.isfinite(a_s) and np.isfinite(a_o)):
                out.update({"r_cross": np.nan, "null_distribution": np.array([]),
                            "alpha_self": np.nan, "alpha_other": np.nan,
                            "self_reliability_mean": np.nan, "self_reliability_median": np.nan,
                            "other_reliability_mean": np.nan, "other_reliability_median": np.nan,
                            "self_reliability_samples": np.array([]), "other_reliability_samples": np.array([]),
                            "ceil_mean": np.nan, "ceil_median": np.nan, "ceil_ci_low": np.nan, "ceil_ci_high": np.nan})
                return out

        rng_i = np.random.default_rng(cfg.random_state + 10_000 + i)
        res_r = compute_r_cross_and_null_for_neuron(
            X_self, ys, X_other, yo, cfg,
            fit_beta=fit_beta,
            alpha_self=a_s, alpha_other=a_o,
            rng=rng_i
        )
        out.update(res_r)

        rng_c = np.random.default_rng(cfg.random_state + 20_000 + i)
        ceil = compute_noise_ceiling_for_neuron(
            X_self, ys, X_other, yo, cfg,
            fit_beta=fit_beta,
            alpha_self=float(out["alpha_self"]),
            alpha_other=float(out["alpha_other"]),
            rng=rng_c
        )
        out.update({k: v for k, v in ceil.items() if k != "ceil_samples"})
        out["ceil_samples"] = ceil["ceil_samples"]
        return out

    results = cast(List[Dict[str, Any]],
                Parallel(n_jobs=cfg.n_jobs, prefer="threads")(
                    delayed(_worker)(i) for i in neuron_indices
                ))

    if verbose:
        ok = sum(np.isfinite(r.get("r_cross", np.nan)) for r in results)
        print(f"[beta_reliability] computed {ok}/{len(results)} neurons with finite r_cross.")
    return results


# -------------------------
# Aggregation / group-level tests
# -------------------------

def aggregate_null_distribution_from_neurons(reliability_results: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      R_null: distribution over null iterations of mean r_cross across neurons
      r_obs: per-neuron observed r_cross vector

    We truncate to the minimum n_null across neurons (conservative).
    """
    nulls_list = []
    r_obs = []
    for r in reliability_results:
        if r is None:
            continue
        r_obs.append(r.get("r_cross", np.nan))
        nd = r.get("null_distribution", None)
        if nd is None:
            continue
        nd = np.asarray(nd, dtype=float)
        if nd.size:
            nulls_list.append(nd)

    r_obs = np.asarray(r_obs, dtype=float)
    if len(nulls_list) == 0:
        return np.array([]), r_obs

    min_nulls = min(n.size for n in nulls_list)
    mat = np.vstack([n[:min_nulls] for n in nulls_list])
    R_null = np.nanmean(mat, axis=0)
    return np.asarray(R_null, dtype=float), r_obs


def paired_bootstrap_difference(
    r_obs_vec: np.ndarray,
    ceil_means: np.ndarray,
    B: int = 2000,
    random_state: int = 0,
) -> Tuple[np.ndarray, Tuple[float, float], float]:
    """
    Paired bootstrap across neurons.
    Returns distribution of (Ceil - R), CI, and p-value for (Ceil - R) <= 0.
    """
    rng = np.random.default_rng(random_state)
    r_obs_vec = np.asarray(r_obs_vec, dtype=float)
    ceil_means = np.asarray(ceil_means, dtype=float)

    valid = np.isfinite(r_obs_vec) & np.isfinite(ceil_means)
    r_obs_vec = r_obs_vec[valid]
    ceil_means = ceil_means[valid]
    n = r_obs_vec.size
    if n == 0:
        return np.array([]), (np.nan, np.nan), np.nan

    D = np.zeros(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        D[b] = float(np.nanmean(ceil_means[idx]) - np.nanmean(r_obs_vec[idx]))

    lo, hi = np.percentile(D, [2.5, 97.5])
    p_below = float((np.sum(D <= 0) + 1) / (B + 1))
    return D, (float(lo), float(hi)), p_below

from matplotlib.axes import Axes
def analyze_reliability_results(
    reliability_results: List[Dict],
    *,
    B_boot: int = 2000,
    make_plots: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    Group-level summary:
      - pooled null test: mean(r_cross) > null
      - paired bootstrap: mean(ceiling) > mean(r_cross)
    """
    R_null, r_obs_vec = aggregate_null_distribution_from_neurons(reliability_results)
    r_obs_vec = np.asarray(r_obs_vec, dtype=float)

    valid_obs = np.isfinite(r_obs_vec)
    R_obs = float(np.nanmean(r_obs_vec[valid_obs])) if np.any(valid_obs) else np.nan

    if R_null.size:
        p_null = float((np.sum(R_null >= R_obs) + 1) / (R_null.size + 1))
        null_mean = float(np.nanmean(R_null))
        null_std = float(np.nanstd(R_null))
    else:
        p_null = np.nan
        null_mean = np.nan
        null_std = np.nan

    ceil_means = np.array([r.get("ceil_mean", np.nan) for r in reliability_results], dtype=float)
    D_boot, D_ci, p_below = paired_bootstrap_difference(r_obs_vec, ceil_means, B=B_boot, random_state=1)

    summary = {
        "n_neurons_total": int(len(reliability_results)),
        "n_neurons_valid_r": int(np.sum(valid_obs)),
        "R_obs_mean": float(R_obs),
        "R_null_mean": float(null_mean),
        "R_null_std": float(null_std),
        "p_null_one_sided": float(p_null),
        "Ceil_mean": float(np.nanmean(ceil_means)),
        "Ceil_std": float(np.nanstd(ceil_means)),
        "D_ci_95": (float(D_ci[0]), float(D_ci[1])),
        "p_mean_ceiling_leq_obs": float(p_below),
        "R_null": R_null,
        "r_obs_vec": r_obs_vec,
        "ceil_means_vec": ceil_means,
        "D_boot": D_boot,
    }

    if verbose:
        print("[beta_reliability summary]")
        print(f"  n_neurons_valid_r: {summary['n_neurons_valid_r']}")
        print(f"  R_obs_mean:        {summary['R_obs_mean']:.4f}")
        print(f"  null mean±sd:      {summary['R_null_mean']:.4f} ± {summary['R_null_std']:.4f}")
        print(f"  p_null (1-sided):  {summary['p_null_one_sided']}")
        print(f"  Ceil_mean:         {summary['Ceil_mean']:.4f}")
        print(f"  D_ci_95:           {summary['D_ci_95']}")
        print(f"  p(Ceil<=Obs):      {summary['p_mean_ceiling_leq_obs']}")

    if make_plots and _HAVE_PLT and R_null.size:
        assert plt is not None

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs = np.asarray(axs)


        axs[0].hist(R_null, bins=40, alpha=0.8)
        axs[0].axvline(R_obs, color="r", lw=2, label=f"R_obs={R_obs:.3f}")
        axs[0].axvline(null_mean, color="k", lw=1, ls="--", label=f"null mean={null_mean:.3f}")
        axs[0].legend()
        axs[0].set_title("Pooled null of mean r_cross")

        axs[1].scatter(summary["ceil_means_vec"], summary["r_obs_vec"], alpha=0.5)
        axs[1].plot([0, 1], [0, 1], "k--", lw=1)
        axs[1].set_xlabel("Per-neuron ceiling (mean)")
        axs[1].set_ylabel("Observed r_cross")
        axs[1].set_title("Ceiling vs observed (per neuron)")

        plt.tight_layout() # type: ignore
        plt.show() # type: ignore

    return summary
