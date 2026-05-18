
"""regression_refactor.py

Refactor of your Poisson ridge regression utilities with:
- Flexible design-matrix construction (semantic PCs, duration covariate, interactions, or duration as *offset*)
- Flexible train/test splitting (random, contiguous blocks, time-series splits)
- Null models (Y-shuffle and semantic-X shuffle) consistent with your earlier approach
- Alpha selection with inner CV that can also be contiguous (no accidental shuffle leakage)
- Cleaner organization and fewer repeated functions

Notes
-----
1) Offset support:
   sklearn.linear_model.PoissonRegressor does NOT support an offset/exposure term.
   If you choose duration/window length as an offset, use backend='statsmodels'.
2) Scaling / leakage:
   Standardization is fit on TRAIN only by default (recommended).
3) PCA:
   By default, PCA is fit on the full embedding matrix (stimulus-only). You can set pca_fit='train'
   for stricter generalization.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

import ast
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Literal, Optional, Tuple
from typing import Any, Literal
import numpy as np
from scipy.stats import pearsonr, spearmanr

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.linear_model import PoissonRegressor

from torch.optim import LBFGS, Adam


# -------------------------
# Config dataclasses
# -------------------------

SplitMethod = Literal[
    "random",
    "contiguous_holdout",
    "contiguous_kfold",
    "time_series_split",
]

DurationMode = Literal[
    "none",          # do not include duration
    "covariate",     # include duration as a predictor column
    "log_covariate", # include log(duration) as predictor
    "offset",        # use log(duration) as offset/exposure (statsmodels backend required)
]

Backend = Literal["sklearn", "statsmodels", "torch"]


@dataclass(frozen=True)
class FeatureConfig:
    n_pcs: int = 30
    duration_mode: DurationMode = "none"
    include_interactions: bool = False  # interactions between semantic PCs and duration covariate (not offset)
    interaction_uses: Literal["duration", "log_duration"] = "duration"
    standardize: bool = True
    standardize_on_train: bool = True
    pca_fit: Literal["all", "train"] = "all"  # 'all' fits PCA on all embeddings; 'train' refits per split
    semantic_columns_first: bool = True  # ensure semantic dims are at columns [0:n_pcs)


@dataclass(frozen=True)
class SplitConfig:
    method: SplitMethod = "contiguous_holdout"
    test_size: float = 0.2              # for holdout methods
    n_splits: int = 5                   # for CV methods
    shuffle: bool = False               # only used in random methods
    random_state: int = 0               # used when shuffle=True or random holdout


@dataclass(frozen=True)
class AlphaConfig:
    alphas: np.ndarray = np.logspace(-3, 3, 30)
    inner_cv: SplitConfig = SplitConfig(method="contiguous_kfold", n_splits=5, shuffle=False, random_state=0)
    scoring: Literal["ll"] = "ll"


@dataclass(frozen=True)
class NullConfig:
    n_shuffles: int = 200
    x_shuffle_mode: Literal["semantic_columns", "all_columns", "row_permute"] = "semantic_columns"
    seed: int = 0


# -------------------------
# Utilities
# -------------------------

def poisson_loglik(y: np.ndarray, mu: np.ndarray) -> float:
    """Poisson log-likelihood up to additive constant."""
    mu = np.clip(mu, 1e-12, None)
    return float(np.sum(y * np.log(mu) - mu - gammaln(y + 1)))


def safe_corr(a: np.ndarray, b: np.ndarray, kind: Literal["pearson", "spearman"] = "pearson") -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()

    if a.size < 2 or b.size < 2:
        return float("nan")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return float("nan")
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")

    try:
        res: Any = pearsonr(a, b) if kind == "pearson" else spearmanr(a, b)

        # Newer SciPy: result object with .statistic
        if hasattr(res, "statistic"):
            return float(res.statistic)

        # Older SciPy: tuple-like (statistic, pvalue)
        return float(res[0])

    except Exception:
        return float("nan")


def contiguous_holdout_indices(n: int, test_size: float, fold_id: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic contiguous holdout with a rolling start point."""
    test_n = int(np.round(n * test_size))
    test_n = max(1, min(test_n, n - 1))
    start = (fold_id * test_n) % n
    idx = np.arange(n)
    test_idx = np.roll(idx, -start)[:test_n]
    train_idx = np.setdiff1d(idx, test_idx, assume_unique=False)
    return train_idx, test_idx


def build_split_iterator(n: int, cfg: SplitConfig) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yields (train_idx, test_idx) splits."""
    if cfg.method == "random":
        rng = np.random.default_rng(cfg.random_state)
        idx = np.arange(n)
        rng.shuffle(idx)
        test_n = int(np.round(n * cfg.test_size))
        test_n = max(1, min(test_n, n - 1))
        test_idx = idx[:test_n]
        train_idx = idx[test_n:]
        yield np.sort(train_idx), np.sort(test_idx)

    elif cfg.method == "contiguous_holdout":
        # Yield one split; callers can loop fold_id externally if desired
        train_idx, test_idx = contiguous_holdout_indices(n, cfg.test_size, fold_id=0)
        yield train_idx, test_idx

    elif cfg.method == "contiguous_kfold":
        # contiguous blocks without shuffling
        fold_sizes = np.full(cfg.n_splits, n // cfg.n_splits, dtype=int)
        fold_sizes[: (n % cfg.n_splits)] += 1
        starts = np.cumsum(np.concatenate([[0], fold_sizes[:-1]]))
        for k in range(cfg.n_splits):
            test_start = starts[k]
            test_end = test_start + fold_sizes[k]
            test_idx = np.arange(test_start, test_end)
            train_idx = np.concatenate([np.arange(0, test_start), np.arange(test_end, n)])
            yield train_idx, test_idx

    elif cfg.method == "time_series_split":
        tss = TimeSeriesSplit(n_splits=cfg.n_splits)
        for train_idx, test_idx in tss.split(np.arange(n)):
            yield train_idx, test_idx

    else:
        raise ValueError(f"Unknown split method: {cfg.method}")


# -------------------------
# Embeddings + PCA
# -------------------------

def load_embeddings_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Embedding" not in df.columns:
        raise ValueError("Expected an 'Embedding' column with list-like strings.")
    df = df.copy()
    df["Parsed_Embedding"] = df["Embedding"].apply(ast.literal_eval)
    if "onset" in df.columns:
        df = df.sort_values("onset").reset_index(drop=True)
    return df.reset_index(drop=True)


def embeddings_matrix(df: pd.DataFrame) -> np.ndarray:
    mat = np.asarray(df["Parsed_Embedding"].tolist(), dtype=float)
    if mat.ndim != 2:
        raise ValueError("Embeddings could not be converted to a 2D matrix.")
    return mat


def pca_reduce(
    E: np.ndarray,
    n_pcs: int,
    fit_E: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, PCA]:
    """Return PCs for E; fit PCA on fit_E if provided else on E."""
    if fit_E is None:
        fit_E = E
    pca = PCA(n_components=n_pcs)
    pca.fit(fit_E)
    pcs = pca.transform(E)
    return pcs, pca


# -------------------------
# Duration + design matrix
# -------------------------

def _load_duration_vector(duration_path: str, df_meta: pd.DataFrame, duration_col: str = "regress_dur") -> np.ndarray:
    if duration_path.endswith(".xlsx"):
        ddf = pd.read_excel(duration_path)
    else:
        ddf = pd.read_csv(duration_path)
    ddf = ddf.reset_index(drop=True)
    if duration_col not in ddf.columns:
        raise ValueError(f"Duration file missing column '{duration_col}'. Found: {list(ddf.columns)}")
    if len(ddf) < len(df_meta):
        raise ValueError(f"Duration file has fewer rows ({len(ddf)}) than metadata ({len(df_meta)}).")
    dur = np.asarray(ddf.loc[df_meta.index, duration_col], dtype=float)
    dur = np.clip(dur, 1e-6, None)
    return dur


def build_design_matrix(
    pcs: np.ndarray,
    dur: Optional[np.ndarray],
    fcfg: FeatureConfig,
) -> Tuple[np.ndarray, Optional[np.ndarray], int, List[str]]:
    """Build X and optional offset. Returns (X, offset, n_semantic_dims, colnames)."""
    if pcs.ndim != 2:
        raise ValueError("pcs must be [n_samples x n_pcs]")
    n = pcs.shape[0]
    colnames: List[str] = [f"PC{i+1}" for i in range(pcs.shape[1])]
    X_parts = [pcs]
    offset = None

    # duration handling
    if fcfg.duration_mode != "none":
        if dur is None:
            raise ValueError("duration_mode != 'none' but dur is None.")
        dur = np.asarray(dur, dtype=float).reshape(n)
        if fcfg.duration_mode == "covariate":
            dcol = dur.reshape(-1, 1)
            X_parts.append(dcol)
            colnames.append("dur")
        elif fcfg.duration_mode == "log_covariate":
            dcol = np.log(dur).reshape(-1, 1)
            X_parts.append(dcol)
            colnames.append("log_dur")
        elif fcfg.duration_mode == "offset":
            # offset enters additively in log-mean; do not add it to X
            offset = np.log(dur).reshape(-1)
        else:
            raise ValueError(f"Unknown duration_mode: {fcfg.duration_mode}")

        # interactions only make sense if duration is a covariate, not offset
        if fcfg.include_interactions:
            if fcfg.duration_mode == "offset":
                raise ValueError("include_interactions=True is incompatible with duration_mode='offset'.")
            if fcfg.interaction_uses == "duration":
                v = dur.reshape(-1, 1)
                name = "dur"
            else:
                v = np.log(dur).reshape(-1, 1)
                name = "log_dur"
            inter = pcs * v
            X_parts.append(inter)
            colnames.extend([f"PC{i+1}x{name}" for i in range(pcs.shape[1])])

    X = np.hstack(X_parts).astype(float)
    n_semantic = pcs.shape[1]  # semantic PCs are always first block
    return X, offset, n_semantic, colnames


def standardize_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    do_standardize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[StandardScaler]]:
    if not do_standardize:
        return X_train, X_test, None
    sc = StandardScaler()
    X_train_s = sc.fit_transform(X_train)
    X_test_s = sc.transform(X_test)
    return X_train_s, X_test_s, sc


# -------------------------
# Model fitting backends
# -------------------------


# -------------------------
# Torch backend (GPU-friendly) Poisson ridge with optional offset
# -------------------------
# This mirrors the faster implementation you pasted:
# - full-batch LBFGS (or Adam warmup + LBFGS)
# - warm-start across alpha grid (descending alphas)
# - supports offset directly (eta += offset)
#
# Notes:
# - This backend is a drop-in alternative for fit/predict, but alpha tuning still uses your split configs.
# - Standardization should be done outside (train-only) exactly as in the sklearn/statsmodels path.



if TYPE_CHECKING:
    import torch
    import torch.nn as nn

try:
    import torch  # type: ignore[import-not-found]
    import torch.nn as nn  # type: ignore[import-not-found]
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError("Torch backend requested but PyTorch is not available in this environment.")


def _to_device(x: Any, device: str):
    _require_torch()
    return torch.as_tensor(x, dtype=torch.float32, device=device)


if _TORCH_AVAILABLE:
    class PoissonRidgeTorch(nn.Module):
        def __init__(self, d: int, alpha: float = 1.0):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(d))
            self.b = nn.Parameter(torch.zeros(()))
            self.alpha = float(alpha)

        def set_params(self, w0=None, b0=None):
            with torch.no_grad():
                if w0 is not None:
                    self.w.copy_(w0)
                if b0 is not None:
                    self.b.copy_(b0)

        def forward(self, X, offset=None):
            eta = X @ self.w + self.b
            if offset is not None:
                eta = eta + offset
            return torch.exp(eta)

        def loss(self, X, y, offset=None):
            mu = self.forward(X, offset)
            nll = torch.sum(mu - y * torch.log(mu.clamp_min(1e-10)))
            reg = 0.5 * self.alpha * torch.sum(self.w * self.w)
            return nll + reg




def fit_poisson_ridge_lbfgs(
    X_t, y_t, alpha: float, *,
    offset_t=None,
    init_w=None, init_b=None,
    max_iter: int = 200, tol: float = 1e-6,
    use_full_batch: bool = True,
    batch_size: int = 65536,
    adam_steps: int | None = None,
):
    if not _TORCH_AVAILABLE:
        raise ImportError("Torch backend requested but PyTorch is not available.")
    n, d = X_t.shape
    model = PoissonRidgeTorch(d, alpha=alpha).to(X_t.device)
    if init_w is not None or init_b is not None:
        model.set_params(init_w, init_b)

    if use_full_batch:
        optimizer = LBFGS(
            model.parameters(), lr=1.0, max_iter=max_iter,
            tolerance_grad=tol, tolerance_change=tol,
            history_size=10, line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(X_t, y_t, offset_t)
            loss.backward()
            return loss

        optimizer.step(closure)
    else:
        opt = Adam(model.parameters(), lr=1e-2)
        if adam_steps is None:
            adam_steps = min(2000, max(400, 4 * (n // batch_size + 1)))
        for _ in range(adam_steps):
            idx = torch.randint(0, n, (min(batch_size, n),), device=X_t.device)
            loss = model.loss(X_t[idx], y_t[idx], None if offset_t is None else offset_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        optimizer = LBFGS(
            model.parameters(), lr=1.0, max_iter=max_iter // 2,
            tolerance_grad=tol, tolerance_change=tol,
            history_size=10, line_search_fn="strong_wolfe",
        )

        def closure2():
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(X_t, y_t, offset_t)
            loss.backward()
            return loss

        optimizer.step(closure2)

    with torch.no_grad():
        w = model.w.detach().clone()
        b = model.b.detach().clone()
    return model, w, b


def approx_edf_ridge_poisson_torch(Xs_t, mu_t, alpha: float, n_probe: int = 64) -> float:
    #"\"\"Hutchinson trace estimator for edf = tr(B (B+alpha I)^-1), B = X^T W X, W=diag(mu).\"\"\"
    if not _TORCH_AVAILABLE:
        raise ImportError("Torch backend requested but PyTorch is not available.")
    d = Xs_t.shape[1]
    XT_W = (Xs_t.T * mu_t)          # d x n
    B = XT_W @ Xs_t                 # d x d
    A = B + alpha * torch.eye(d, device=Xs_t.device)
    z = torch.randn(d, n_probe, device=Xs_t.device)
    v = torch.linalg.solve(A, z)
    Bv = B @ v
    tr_est = torch.sum(z * Bv, dim=0).mean()
    return float(tr_est.item())


def fit_predict_poisson(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
    backend: Backend = "sklearn",
    offset_train: Optional[np.ndarray] = None,
    offset_test: Optional[np.ndarray] = None,
    max_iter: int = 2000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (mu_train, mu_test, coef)."""
    if backend == "sklearn":
        if offset_train is not None or offset_test is not None:
            raise ValueError("sklearn backend does not support offsets. Use backend='statsmodels' for offset.")
        model = PoissonRegressor(alpha=alpha, max_iter=max_iter)
        model.fit(X_train, y_train)
        mu_tr = np.clip(model.predict(X_train), 1e-12, None)
        mu_te = np.clip(model.predict(X_test), 1e-12, None)
        return mu_tr, mu_te, model.coef_.copy()

    if backend == "torch":
        if not _TORCH_AVAILABLE:
            raise ImportError("Torch backend requested but PyTorch is not available.")
        device = "cuda" if (torch.cuda.is_available()) else "cpu"
        Xtr_t = _to_device(X_train, device)
        ytr_t = _to_device(y_train, device)
        Xte_t = _to_device(X_test, device)
        off_tr_t = _to_device(offset_train, device) if offset_train is not None else None
        off_te_t = _to_device(offset_test, device) if offset_test is not None else None

        model_t, w_t, b_t = fit_poisson_ridge_lbfgs(
            Xtr_t, ytr_t, alpha=float(alpha),
            offset_t=off_tr_t,
            max_iter=max_iter, tol=1e-6,
            use_full_batch=True,
        )
        with torch.no_grad():
            mu_tr = model_t(Xtr_t, off_tr_t).clamp_min(1e-12).cpu().numpy()
            mu_te = model_t(Xte_t, off_te_t).clamp_min(1e-12).cpu().numpy()
        coef = w_t.detach().cpu().numpy().copy()
        return mu_tr, mu_te, coef

    # statsmodels backend
    import statsmodels.api as sm
    X_train_sm = sm.add_constant(X_train, has_constant="add")
    X_test_sm = sm.add_constant(X_test, has_constant="add")

    glm = sm.GLM(
        y_train,
        X_train_sm,
        family=sm.families.Poisson(),
        offset=offset_train,
    )
    # L2 (ridge) via elastic_net with L1_wt=0
    res = glm.fit_regularized(alpha=float(alpha), L1_wt=0.0, maxiter=max_iter)
    mu_tr = np.clip(res.predict(X_train_sm, offset=offset_train), 1e-12, None)
    mu_te = np.clip(res.predict(X_test_sm, offset=offset_test), 1e-12, None)
    # drop intercept from coef for consistency with sklearn
    coef = np.asarray(res.params[1:], dtype=float)
    return mu_tr, mu_te, coef


# -------------------------
# Alpha selection (inner CV)
# -------------------------

from typing import Dict, List, Optional, Tuple, Any, cast

import numpy as np


def select_alpha_inner_cv(
    X: np.ndarray,
    y: np.ndarray,
    acfg: AlphaConfig,
    backend: Backend = "sklearn",
    offset: Optional[np.ndarray] = None,
    torch_device: str = "cuda",
) -> Tuple[float, Dict[float, float]]:
    """
    Inner-CV alpha selection on TRAIN only.

    - sklearn/statsmodels: evaluate each alpha per fold
    - torch: evaluates alphas in descending order with per-fold warm-start
    """
    scores: Dict[float, float] = {}
    n = int(X.shape[0])

    # ------------------ non-torch path ------------------
    if backend != "torch":
        for a_raw in acfg.alphas:
            a = float(a_raw)
            ll_list: List[float] = []
            for tr, va in build_split_iterator(n, acfg.inner_cv):
                Xtr, Xva = X[tr], X[va]
                ytr, yva = y[tr], y[va]

                # standardize on train (always for alpha selection)
                Xtr_s, Xva_s, _ = standardize_train_test(Xtr, Xva, do_standardize=True)

                off_tr = offset[tr] if offset is not None else None
                off_va = offset[va] if offset is not None else None

                _, mu_va, _ = fit_predict_poisson(
                    Xtr_s, ytr, Xva_s, alpha=a,
                    backend=backend, offset_train=off_tr, offset_test=off_va
                )
                ll_list.append(float(poisson_loglik(yva, mu_va)))

            scores[a] = float(np.mean(ll_list)) if ll_list else float("-inf")

        if not scores:
            raise RuntimeError("Alpha selection failed: no scores computed.")

        # Pylance-safe max (avoids Optional return from dict.get)
        best_alpha = max(scores.items(), key=lambda kv: kv[1])[0]
        return float(best_alpha), scores

    # ------------------ torch path ------------------
    if not _TORCH_AVAILABLE:
        raise ImportError("Torch backend requested but PyTorch is not available.")

    # Narrow torch for the type checker (if you used optional imports)
    import torch  # type: ignore[import-not-found]

    device = torch_device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    fold_splits = list(build_split_iterator(n, acfg.inner_cv))

    # We'll store tensors / numpy arrays in a dict -> type as Any to avoid Pylance noise
    fold_data: List[Dict[str, Any]] = []

    for tr, va in fold_splits:
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]
        Xtr_s, Xva_s, _ = standardize_train_test(Xtr, Xva, do_standardize=True)

        off_tr = offset[tr] if offset is not None else None
        off_va = offset[va] if offset is not None else None

        fd: Dict[str, Any] = {
            "Xtr_t": _to_device(Xtr_s, device),
            "ytr_t": _to_device(ytr, device),
            "Xva_t": _to_device(Xva_s, device),
            "yva_np": np.asarray(yva, dtype=float),
            "off_tr_t": _to_device(off_tr, device) if off_tr is not None else None,
            "off_va_t": _to_device(off_va, device) if off_va is not None else None,
        }
        fold_data.append(fd)

    alphas_desc = np.array(sorted(map(float, acfg.alphas), reverse=True), dtype=float)

    # Warm-start cache: explicitly allow Any so it can hold torch tensors later
    cache: Dict[int, Tuple[Any, Any]] = {i: (None, None) for i in range(len(fold_data))}
    alpha_to_ll: Dict[float, List[float]] = {float(a): [] for a in alphas_desc}

    for a in alphas_desc:
        a_f = float(a)
        for fidx, fd in enumerate(fold_data):
            w0, b0 = cache[fidx]

            model_t, w_t, b_t = fit_poisson_ridge_lbfgs(
                fd["Xtr_t"], fd["ytr_t"], alpha=a_f,
                offset_t=fd["off_tr_t"],
                init_w=w0, init_b=b0,
                max_iter=200, tol=1e-6,
                use_full_batch=True,
            )
            cache[fidx] = (w_t, b_t)

            with torch.no_grad():
                mu_va = model_t(fd["Xva_t"], fd["off_va_t"]).clamp_min(1e-12).cpu().numpy()

            alpha_to_ll[a_f].append(float(poisson_loglik(fd["yva_np"], mu_va)))

    for a_f, lst in alpha_to_ll.items():
        scores[a_f] = float(np.mean(lst)) if lst else float("-inf")

    if not scores:
        raise RuntimeError("Alpha selection failed: no scores computed (torch path).")

    best_alpha = max(scores.items(), key=lambda kv: kv[1])[0]
    return float(best_alpha), scores



# -------------------------
# Main evaluation: one neuron
# -------------------------

def run_neuron_encoding(
    X: np.ndarray,
    Y: np.ndarray,
    neuron_idx: int,
    fcfg: FeatureConfig,
    split_cfg: SplitConfig,
    alpha_cfg: AlphaConfig,
    null_cfg: NullConfig,
    backend: Backend = "sklearn",
    offset: Optional[np.ndarray] = None,
    n_repeats: int = 10,
    semantic_dim_count: Optional[int] = None,
) -> Dict:
    """Runs repeated contiguous holdouts (or repeats of random splits) and returns metrics + null tests."""
    rng = np.random.default_rng(null_cfg.seed + neuron_idx)
    y = np.asarray(Y[:, neuron_idx], dtype=float)

    if (not np.all(np.isfinite(y))) or np.std(y) == 0 or np.all(y == 0):
        return {"neuron": neuron_idx, "failed": True}

    n = X.shape[0]
    if semantic_dim_count is None:
        semantic_dim_count = fcfg.n_pcs

    rows = []
    coefs = []

    for rep in range(n_repeats):
        # build split
        if split_cfg.method == "contiguous_holdout":
            tr, te = contiguous_holdout_indices(n, split_cfg.test_size, fold_id=rep)
        elif split_cfg.method == "random":
            # re-sample each repeat
            tmp = SplitConfig(method="random", test_size=split_cfg.test_size, shuffle=True, random_state=int(rng.integers(0, 2**31-1)))
            tr, te = next(build_split_iterator(n, tmp))
        else:
            # for kfold methods, rep ignored: just loop all folds once
            # if user asks repeats w/ kfold they can call this externally
            tr, te = next(build_split_iterator(n, split_cfg))

        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        off_tr = offset[tr] if offset is not None else None
        off_te = offset[te] if offset is not None else None

        # alpha selection (inner CV) on training only
        best_a, _ = select_alpha_inner_cv(Xtr, ytr, alpha_cfg, backend=backend, offset=off_tr)

        # standardize on train
        Xtr_s, Xte_s, _ = standardize_train_test(Xtr, Xte, do_standardize=fcfg.standardize)

        # fit real model
        mu_tr, mu_te, coef = fit_predict_poisson(
            Xtr_s, ytr, Xte_s, alpha=best_a, backend=backend,
            offset_train=off_tr, offset_test=off_te
        )
        ll_real = poisson_loglik(yte, mu_te)
        coefs.append(coef)

        pear = safe_corr(yte, mu_te, "pearson")
        spear = safe_corr(yte, mu_te, "spearman")

        # null 1: Y-shuffle on train
        ytr_sh = rng.permutation(ytr)
        mu_tr_n, mu_te_n, _ = fit_predict_poisson(
            Xtr_s, ytr_sh, Xte_s, alpha=best_a, backend=backend,
            offset_train=off_tr, offset_test=off_te
        )
        ll_yshuf = poisson_loglik(yte, mu_te_n)
        ll_diff_y = ll_real - ll_yshuf

        # null 2: X-shuffle (configurable)
        # Goal: destroy the stimulus→spike mapping while keeping the train/test split fixed.
        # Modes:
        #   - semantic_columns: shuffle semantic PC columns within TRAIN only (isolates semantic contribution)
        #   - all_columns: shuffle ALL columns within TRAIN only (destroys any feature mapping)
        #   - row_permute: permute rows of X within TRAIN only (preserves joint feature covariance; destroys alignment)
        Xtr_xs = Xtr_s.copy()

        if null_cfg.x_shuffle_mode == "row_permute":
            Xtr_xs = Xtr_xs[rng.permutation(Xtr_xs.shape[0])]
        elif null_cfg.x_shuffle_mode == "semantic_columns":
            for c in range(semantic_dim_count):
                rng.shuffle(Xtr_xs[:, c])
        elif null_cfg.x_shuffle_mode == "all_columns":
            for c in range(Xtr_xs.shape[1]):
                rng.shuffle(Xtr_xs[:, c])
        else:
            raise ValueError(f"Unknown x_shuffle_mode: {null_cfg.x_shuffle_mode}")

        mu_tr_x, mu_te_x, _ = fit_predict_poisson(
            Xtr_xs, ytr, Xte_s, alpha=best_a, backend=backend,
            offset_train=off_tr, offset_test=off_te
        )
        ll_xshuf = poisson_loglik(yte, mu_te_x)
        ll_diff_x = ll_real - ll_xshuf

        rows.append({
            "neuron": neuron_idx,
            "rep": rep,
            "best_alpha": best_a,
            "ll_real": ll_real,
            "ll_yshuf": ll_yshuf,
            "ll_diff_yshuf": ll_diff_y,
            "ll_xshuf": ll_xshuf,
            "ll_diff_xshuf": ll_diff_x,
            "pearson_r": pear,
            "spearman_r": spear,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
        })

    df = pd.DataFrame(rows)
    out = {
        "neuron": neuron_idx,
        "failed": False,
        "summary": df.groupby("neuron").mean(numeric_only=True).iloc[0].to_dict(),
        "by_repeat": df,
        "coef_mean": np.mean(np.vstack(coefs), axis=0).tolist() if len(coefs) else None,
        "coef_std": np.std(np.vstack(coefs), axis=0).tolist() if len(coefs) else None,
    }
    return out


# -------------------------
# End-to-end builder for self/other
# -------------------------

def build_self_other_design(
    embedding_csv_path: str,
    duration_path: str,
    fcfg: FeatureConfig,
    target_speaker: str = "SPK1",
    duration_col: str = "regress_dur",
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], pd.DataFrame]:
    """Returns X_self, X_other, offset_self, offset_other, df_meta."""
    df = load_embeddings_csv(embedding_csv_path)
    E = embeddings_matrix(df)
    dur = _load_duration_vector(duration_path, df, duration_col=duration_col)

    # masks
    mask_self = (df["Speaker"].astype(str) == target_speaker)
    mask_other = ~mask_self

    E_self = E[mask_self.values]
    E_other = E[mask_other.values]
    dur_self = dur[mask_self.values]
    dur_other = dur[mask_other.values]

    # PCA
    if fcfg.pca_fit == "all":
        pcs_all, pca = pca_reduce(E, fcfg.n_pcs, fit_E=E)
        pcs_self = pcs_all[mask_self.values]
        pcs_other = pcs_all[mask_other.values]
    else:
        # Fit PCA separately later per split (harder).
        # Here we provide a reasonable default: fit on each condition's full data.
        pcs_self, _ = pca_reduce(E_self, fcfg.n_pcs, fit_E=E_self)
        pcs_other, _ = pca_reduce(E_other, fcfg.n_pcs, fit_E=E_other)

    Xs, off_s, nsem, _ = build_design_matrix(pcs_self, dur_self, fcfg)
    Xo, off_o, _, _ = build_design_matrix(pcs_other, dur_other, fcfg)

    return Xs, Xo, off_s, off_o, df

# ======================================================================
# Y-building utilities (refactored)
# ======================================================================


from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class YBuildConfig:
    speaker_col: str = "Speaker"
    word_col: str = "Word"
    target_speaker: str = "SPK1"
    # If your metadata speaker labels are "SPK1" but your spike containers are keyed "Speaker1",
    # set speaker_key_style="SpeakerN". Otherwise set to "SPKN" to use labels verbatim.
    speaker_key_style: str = "SpeakerN"  # "SpeakerN" or "SPKN"
    # For raw spikes mode: which columns in speaker_events correspond to [start, end] in samples
    # Your earlier code used columns 1:3 for onset/offset. (eventTime, pre_onset, post_offset, duration)
    event_start_col: int = 1
    event_end_col: int = 2
    # Safety
    require_region_in_cells: bool = True


def speaker_label_to_key(label: str, style: str) -> str:
    """Map metadata speaker label -> key in spike containers."""
    label = str(label)
    if style == "SPKN":
        return label
    if style == "SpeakerN":
        # SPK1 -> Speaker1, SPK08 -> Speaker8
        if label.upper().startswith("SPK"):
            n = label.upper().replace("SPK", "").lstrip("0") or "0"
            return f"Speaker{n}"
        # already SpeakerX
        if label.lower().startswith("speaker"):
            return label
        # fallback
        return label
    raise ValueError(f"Unknown speaker_key_style: {style}")


def _empty_Y(n_neurons: int) -> np.ndarray:
    return np.zeros((0, n_neurons), dtype=float)


def _stack_or_empty(rows: List[np.ndarray], n_neurons: int) -> np.ndarray:
    if len(rows) == 0:
        return _empty_Y(n_neurons)
    return np.vstack(rows).astype(float)


def validate_region_Y(region_Y: Dict) -> None:
    """Raises on obvious mismatches."""
    Y_s = region_Y["self"]
    Y_o = region_Y["other"]
    if Y_s.ndim != 2 or Y_o.ndim != 2:
        raise ValueError("Y matrices must be 2D.")
    if Y_s.shape[1] != Y_o.shape[1]:
        raise ValueError(f"Neuron dimension mismatch self={Y_s.shape[1]} other={Y_o.shape[1]}")
    if Y_s.shape[0] != len(region_Y["indices_self"]):
        raise ValueError("self row count != indices_self length")
    if Y_o.shape[0] != len(region_Y["indices_other"]):
        raise ValueError("other row count != indices_other length")


# ---------------------------------------------------------------------
# Mode A: pre-binned spike_data by speaker/region (your build_Y_matrices_from_raw_spikes)
# ---------------------------------------------------------------------
from typing import Hashable
def build_region_Y_from_binned_by_speaker(
    metadata: pd.DataFrame,
    spike_data: Dict[str, Dict[str, np.ndarray]],
    regions: Sequence[str],
    cfg: YBuildConfig = YBuildConfig(),
) -> Dict[str, Dict]:
    """
    spike_data layout:
        spike_data[speaker_key][region] -> np.ndarray [n_events_for_that_speaker x n_neurons]
    """
    region_Ys: Dict[str, Dict] = {}

    # speakers available in spike_data
    speakers_in_data = list(spike_data.keys())
    counters = {spk: 0 for spk in speakers_in_data}

    for region in regions:
        Y_self: List[np.ndarray] = []
        Y_other: List[np.ndarray] = []
        idx_self: List[Hashable] = []
        idx_other: List[Hashable] = []
        w_self: List[str] = []
        w_other: List[str] = []
        s_self: List[str] = []
        s_other: List[str] = []

        # infer neuron count from first available speaker/region
        n_neurons = None
        for spk_key in speakers_in_data:
            if region in spike_data[spk_key]:
                n_neurons = spike_data[spk_key][region].shape[1]
                break
        if n_neurons is None:
            region_Ys[region] = {
                "self": np.zeros((0, 0)),
                "other": np.zeros((0, 0)),
                "indices_self": [],
                "indices_other": [],
                "word_ref_self": [],
                "word_ref_other": [],
                "speaker_ref_self": [],
                "speaker_ref_other": [],
                "speaker_counters_final": counters.copy(),
            }
            continue

        # local counters so each region consumes rows independently (matches your prior behavior)
        speaker_counters = {spk: 0 for spk in speakers_in_data}

        for i, row in metadata.iterrows():
            spk_label = row[cfg.speaker_col]
            spk_key = speaker_label_to_key(spk_label, cfg.speaker_key_style)
            if spk_key not in spike_data:
                continue
            if region not in spike_data[spk_key]:
                continue

            cur = speaker_counters[spk_key]
            mat = spike_data[spk_key][region]
            if cur >= mat.shape[0]:
                continue

            vec = mat[cur, :].astype(float)
            word = str(row.get(cfg.word_col, ""))

            if str(spk_label) == cfg.target_speaker:
                Y_self.append(vec)
                idx_self.append(i)
                w_self.append(word)
                s_self.append(str(spk_label))
            else:
                Y_other.append(vec)
                idx_other.append(i)
                w_other.append(word)
                s_other.append(str(spk_label))

            speaker_counters[spk_key] += 1

        region_Y = {
            "self": _stack_or_empty(Y_self, n_neurons),
            "other": _stack_or_empty(Y_other, n_neurons),
            "indices_self": idx_self,
            "indices_other": idx_other,
            "word_ref_self": w_self,
            "word_ref_other": w_other,
            "speaker_ref_self": s_self,
            "speaker_ref_other": s_other,
            "speaker_counters_final": speaker_counters,
        }
        validate_region_Y(region_Y)
        region_Ys[region] = region_Y

    return region_Ys


# ---------------------------------------------------------------------
# Mode B: raw spikes + speaker_events windows (your build_Y_matrices_from_raw_spikes_direct)
# ---------------------------------------------------------------------

def build_region_Y_from_raw_spikes_and_events(
    metadata: pd.DataFrame,
    spikes: np.ndarray,
    region_cells: Dict[str, np.ndarray],
    speaker_events: Dict[str, np.ndarray],
    regions: Sequence[str],
    cfg: YBuildConfig = YBuildConfig(),
    agg: str = "sum",  # "sum" or "mean"
) -> Dict[str, Dict]:
    """
    spikes: np.ndarray [time_samples x n_total_neurons]
    region_cells[region] -> indices into second axis of spikes
    speaker_events[speaker_key] -> np.ndarray [n_events x k], with start/end cols set in cfg

    Each metadata row consumes the next event for that speaker and extracts spikes[start:end, region_neurons],
    then aggregates across time.
    """
    if spikes.ndim != 2:
        raise ValueError("spikes must be [time x neurons].")

    speakers_in_events = list(speaker_events.keys())
    region_Ys: Dict[str, Dict] = {}

    for region in regions:
        if cfg.require_region_in_cells and region not in region_cells:
            continue
        region_idx = np.asarray(region_cells.get(region, []), dtype=int)
        if region_idx.size == 0:
            # no neurons in this region
            region_Ys[region] = {
                "self": np.zeros((0, 0)),
                "other": np.zeros((0, 0)),
                "indices_self": [],
                "indices_other": [],
                "word_ref_self": [],
                "word_ref_other": [],
                "speaker_ref_self": [],
                "speaker_ref_other": [],
                "speaker_counters_final": {spk: 0 for spk in speakers_in_events},
            }
            continue
        n_neurons = int(region_idx.size)

        Y_self: List[np.ndarray] = []
        Y_other: List[np.ndarray] = []
        idx_self: List[Hashable] = []
        idx_other: List[Hashable] = []
        w_self: List[str] = []
        w_other: List[str] = []
        s_self: List[str] = []
        s_other: List[str] = []

        speaker_counters = {spk: 0 for spk in speakers_in_events}

        for i, row in metadata.iterrows():
            spk_label = row[cfg.speaker_col]
            spk_key = speaker_label_to_key(spk_label, cfg.speaker_key_style)

            if spk_key not in speaker_events:
                continue

            cur = speaker_counters[spk_key]
            ev = speaker_events[spk_key]
            if cur >= ev.shape[0]:
                continue

            start = int(ev[cur, cfg.event_start_col])
            end = int(ev[cur, cfg.event_end_col])

            # bounds safety
            start = max(0, min(start, spikes.shape[0]))
            end = max(0, min(end, spikes.shape[0]))
            if end <= start:
                continue

            chunk = spikes[start:end, :][:, region_idx]
            if agg == "sum":
                vec = np.sum(chunk, axis=0)
            elif agg == "mean":
                vec = np.mean(chunk, axis=0)
            else:
                raise ValueError("agg must be 'sum' or 'mean'")

            word = str(row.get(cfg.word_col, ""))

            if str(spk_label) == cfg.target_speaker:
                Y_self.append(vec)
                idx_self.append(i)
                w_self.append(word)
                s_self.append(str(spk_label))
            else:
                Y_other.append(vec)
                idx_other.append(i)
                w_other.append(word)
                s_other.append(str(spk_label))

            speaker_counters[spk_key] += 1

        region_Y = {
            "self": _stack_or_empty(Y_self, n_neurons),
            "other": _stack_or_empty(Y_other, n_neurons),
            "indices_self": idx_self,
            "indices_other": idx_other,
            "word_ref_self": w_self,
            "word_ref_other": w_other,
            "speaker_ref_self": s_self,
            "speaker_ref_other": s_other,
            "speaker_counters_final": speaker_counters,
        }
        validate_region_Y(region_Y)
        region_Ys[region] = region_Y

    return region_Ys
