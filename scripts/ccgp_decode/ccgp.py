import warnings
import numpy as np
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, make_scorer

from .models import ModelSpec, make_pipeline
from .balance import balance_self_other
from .labels import remap_labels_to_contiguous

_SCORER = make_scorer(balanced_accuracy_score)


def _safe_n_splits(y, k_desired: int) -> int:
    """
    Ensure n_splits <= min count per class (otherwise StratifiedKFold/RepeatedStratifiedKFold breaks).
    Returns at least 2.
    """
    y = np.asarray(y)
    _, counts = np.unique(y, return_counts=True)
    k_eff = int(min(int(k_desired), int(counts.min())))
    return max(2, k_eff)


def _make_cv(y, cv_folds: int, seed: int, use_repeated: bool, n_repeats: int):
    k = _safe_n_splits(y, cv_folds)
    if use_repeated:
        return RepeatedStratifiedKFold(n_splits=k, n_repeats=int(n_repeats), random_state=seed)
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)


def _compute_metrics(y_true, y_pred):
    return dict(
        bal_acc=float(balanced_accuracy_score(y_true, y_pred)),
        acc=float(accuracy_score(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred, average="macro")),
    )


def _cv_score_fixed_params(Y, y, spec: ModelSpec, params: dict, cv):
    """
    Within-condition CV (balanced accuracy) using fixed hyperparameters (no selection inside folds).
    Returns (mean, std).
    """
    pipe = make_pipeline(spec)
    if params:
        pipe.set_params(**params)

    scores = []
    for tr, va in cv.split(Y, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(Y[tr], y[tr])
        pred = pipe.predict(Y[va])
        scores.append(balanced_accuracy_score(y[va], pred))

    scores = np.asarray(scores, dtype=float)
    if len(scores) > 1:
        return float(scores.mean()), float(scores.std(ddof=1))
    return float(scores.mean()), 0.0


def _cv_ccgp_trainfolds_to_other(Y_train, y_train, Y_test, y_test, spec: ModelSpec, params: dict, cv):
    """
    Cross-condition generalization with a CV loop *only on the training condition*.

    For each fold:
      - fit on training fold of Y_train/y_train
      - evaluate on ALL of the opposite condition Y_test/y_test

    Returns (mean_bal_acc, std_bal_acc) across folds.
    """
    pipe = make_pipeline(spec)
    if params:
        pipe.set_params(**params)

    scores = []
    for tr, _va in cv.split(Y_train, y_train):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(Y_train[tr], y_train[tr])
        pred = pipe.predict(Y_test)
        scores.append(balanced_accuracy_score(y_test, pred))

    scores = np.asarray(scores, dtype=float)
    if len(scores) > 1:
        return float(scores.mean()), float(scores.std(ddof=1))
    return float(scores.mean()), 0.0


def ccgp_one_patient(
    Y_self, y_self, Y_other, y_other,
    spec: ModelSpec,
    min_per_class=10,
    balance="gate_only",
    match_self_other=None,
    class_balance=None,
    soft_target="median",
    soft_min_keep=None,
    cv_folds=5,
    seed=0,
    n_jobs=1,
    n_perm=0,
    perm_mode="within",
    return_null=False,

    # ---------------- NEW OPTIONS ----------------
    use_repeated_cv=False,
    cv_repeats=10,

    # How to compute within-condition ceilings:
    #  - "gs_best": use GridSearchCV.best_score_ (optimistic because selection happens within CV)
    #  - "fixed_params": compute CV score with the selected best params fixed (less optimistic)
    within_cv_mode="fixed_params",

    # Whether to compute "CV-CCGP" (CCGP with folds on the training condition)
    compute_cv_ccgp=False,
):
    """
    Returns a dict including:
      - Cross-condition (CCGP) metrics: ccgp_bal, ccgp_acc, ccgp_f1 (+ direction-specific)
      - Within-condition ceilings (balanced acc): ceiling_self_matched, ceiling_other_matched, mean/min
      - Optional CV-CCGP (balanced acc): ccgp_bal_cvmean, ccgp_bal_cvstd (+ per-direction)

    Permutation test:
      - tests ccgp_bal (balanced accuracy) against a label-shuffle null (default perm_mode="within")
      - keeps hyperparameters fixed to those selected on real labels
    """
    # ---------------- balancing / gating ----------------
    Ys, ys, Yo, yo, diag = balance_self_other(
        Y_self, y_self, Y_other, y_other,
        rng=seed,
        min_per_class=min_per_class,
        balance=balance,
        match_self_other=match_self_other,
        class_balance=class_balance,
        soft_target=soft_target,
        soft_min_keep=soft_min_keep,
    )

    kept_classes = np.asarray(diag["kept_classes"], dtype=int)
    ys, _, _ = remap_labels_to_contiguous(ys, classes=kept_classes)
    yo, _, _ = remap_labels_to_contiguous(yo, classes=kept_classes)

    # CV objects (adaptive n_splits + optional repeats)
    cv_s = _make_cv(ys, cv_folds=cv_folds, seed=seed, use_repeated=use_repeated_cv, n_repeats=cv_repeats)
    cv_o = _make_cv(yo, cv_folds=cv_folds, seed=seed + 1, use_repeated=use_repeated_cv, n_repeats=cv_repeats)

    # ================= SELF → OTHER (fit on self) =================
    pipe_s = make_pipeline(spec)
    best_s = {}
    ceiling_s = np.nan

    if spec.param_grid:
        gs_s = GridSearchCV(
            pipe_s, spec.param_grid,
            cv=cv_s, scoring=_SCORER,
            n_jobs=n_jobs, refit=True
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gs_s.fit(Ys, ys)
        est_s = gs_s.best_estimator_
        best_s = gs_s.best_params_

        if within_cv_mode == "gs_best":
            ceiling_s = float(gs_s.best_score_)
        elif within_cv_mode == "fixed_params":
            ceiling_s, _ = _cv_score_fixed_params(Ys, ys, spec, best_s, cv_s)
        else:
            raise ValueError("within_cv_mode must be 'gs_best' or 'fixed_params'")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est_s = pipe_s.fit(Ys, ys)
        ceiling_s, _ = _cv_score_fixed_params(Ys, ys, spec, {}, cv_s)

    pred_s2o = est_s.predict(Yo)
    metrics_s2o = _compute_metrics(yo, pred_s2o)

    # ================= OTHER → SELF (fit on other) =================
    pipe_o = make_pipeline(spec)
    best_o = {}
    ceiling_o = np.nan

    if spec.param_grid:
        gs_o = GridSearchCV(
            pipe_o, spec.param_grid,
            cv=cv_o, scoring=_SCORER,
            n_jobs=n_jobs, refit=True
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gs_o.fit(Yo, yo)
        est_o = gs_o.best_estimator_
        best_o = gs_o.best_params_

        if within_cv_mode == "gs_best":
            ceiling_o = float(gs_o.best_score_)
        elif within_cv_mode == "fixed_params":
            ceiling_o, _ = _cv_score_fixed_params(Yo, yo, spec, best_o, cv_o)
        else:
            raise ValueError("within_cv_mode must be 'gs_best' or 'fixed_params'")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est_o = pipe_o.fit(Yo, yo)
        ceiling_o, _ = _cv_score_fixed_params(Yo, yo, spec, {}, cv_o)

    pred_o2s = est_o.predict(Ys)
    metrics_o2s = _compute_metrics(ys, pred_o2s)

    # ================= CCGP aggregate =================
    ccgp_bal = 0.5 * (metrics_s2o["bal_acc"] + metrics_o2s["bal_acc"])
    ccgp_acc = 0.5 * (metrics_s2o["acc"] + metrics_o2s["acc"])
    ccgp_f1  = 0.5 * (metrics_s2o["f1"] + metrics_o2s["f1"])

    # ================= within-condition aggregates =================
    ceiling_min = float(min(ceiling_s, ceiling_o))
    ceiling_mean = float(0.5 * (ceiling_s + ceiling_o))

    # ================= optional CV-CCGP =================
    cv_ccgp = {}
    if compute_cv_ccgp:
        s2o_mean, s2o_std = _cv_ccgp_trainfolds_to_other(Ys, ys, Yo, yo, spec, best_s, cv_s)
        o2s_mean, o2s_std = _cv_ccgp_trainfolds_to_other(Yo, yo, Ys, ys, spec, best_o, cv_o)

        cv_ccgp = dict(
            ccgp_self_to_other_bal_cvmean=float(s2o_mean),
            ccgp_self_to_other_bal_cvstd=float(s2o_std),
            ccgp_other_to_self_bal_cvmean=float(o2s_mean),
            ccgp_other_to_self_bal_cvstd=float(o2s_std),
            ccgp_bal_cvmean=float(0.5 * (s2o_mean + o2s_mean)),
            ccgp_bal_cvstd=float(0.5 * (s2o_std + o2s_std)),
        )

    # ---- output dict ----
    out = dict(diag)
    out.update(dict(
        model=spec.name,

        # Cross-condition (CCGP)
        ccgp_bal=float(ccgp_bal),
        acc_self_to_other_bal=float(metrics_s2o["bal_acc"]),
        acc_other_to_self_bal=float(metrics_o2s["bal_acc"]),

        ccgp_acc=float(ccgp_acc),
        acc_self_to_other=float(metrics_s2o["acc"]),
        acc_other_to_self=float(metrics_o2s["acc"]),

        ccgp_f1=float(ccgp_f1),
        f1_self_to_other=float(metrics_s2o["f1"]),
        f1_other_to_self=float(metrics_o2s["f1"]),

        # Within-condition ceilings (balanced acc)
        ceiling_self_matched=float(ceiling_s),
        ceiling_other_matched=float(ceiling_o),
        ceiling_mean_matched=float(ceiling_mean),
        ceiling_min_matched=float(ceiling_min),

        within_cv_mode=str(within_cv_mode),
        use_repeated_cv=bool(use_repeated_cv),
        cv_repeats=int(cv_repeats) if use_repeated_cv else 1,
        compute_cv_ccgp=bool(compute_cv_ccgp),
    ))
    if compute_cv_ccgp:
        out.update(cv_ccgp)

    # ---------------- permutation (tests CCGP balanced accuracy) ----------------
    out["p_perm"] = np.nan
    out["null_mean"] = np.nan
    out["null_sd"] = np.nan

    if n_perm > 0:
        rng = np.random.default_rng(seed)
        null_vals = np.zeros(n_perm)

        for i in range(n_perm):
            if perm_mode in ("within", "within_stratified"):
                ys_p = ys[rng.permutation(len(ys))]
                yo_p = yo[rng.permutation(len(yo))]

                pipe_sp = make_pipeline(spec)
                if best_s:
                    pipe_sp.set_params(**best_s)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipe_sp.fit(Ys, ys_p)
                m1 = balanced_accuracy_score(yo, pipe_sp.predict(Yo))

                pipe_op = make_pipeline(spec)
                if best_o:
                    pipe_op.set_params(**best_o)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipe_op.fit(Yo, yo_p)
                m2 = balanced_accuracy_score(ys, pipe_op.predict(Ys))

                null_vals[i] = 0.5 * (m1 + m2)

            elif perm_mode == "cross_label_map":
                K = len(np.unique(ys))
                pi = rng.permutation(K)
                inv_pi = np.argsort(pi)

                pred1 = pi[pred_s2o.astype(int)]
                pred2 = inv_pi[pred_o2s.astype(int)]

                m1 = balanced_accuracy_score(yo, pred1)
                m2 = balanced_accuracy_score(ys, pred2)
                null_vals[i] = 0.5 * (m1 + m2)

            else:
                raise ValueError(f"Unknown perm_mode: {perm_mode}")

        out["p_perm"] = float((np.sum(null_vals >= ccgp_bal) + 1) / (n_perm + 1))
        out["null_mean"] = float(null_vals.mean())
        out["null_sd"] = float(null_vals.std(ddof=1))

        if return_null:
            out["null_vals"] = null_vals

    return out
