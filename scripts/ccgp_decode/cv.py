import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score, make_scorer

from .models import make_pipeline, ModelSpec

_SCORER = make_scorer(balanced_accuracy_score)


def within_condition_cv(Y, y, spec: ModelSpec, cv_folds=5, seed=0, n_jobs=1):
    Y = np.asarray(Y)
    y = np.asarray(y).astype(int)
    if np.unique(y).size < 2:
        return np.nan, np.nan

    pipe = make_pipeline(spec)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(pipe, Y, y, scoring=_SCORER, cv=cv, n_jobs=n_jobs)
    return float(scores.mean()), float(scores.std(ddof=1))


def run_within_condition_diagnostics(Y_self, y_self, Y_other, y_other, spec: ModelSpec, cv_folds=5, seed=0, n_jobs=1):
    mS, sS = within_condition_cv(Y_self, y_self, spec, cv_folds=cv_folds, seed=seed, n_jobs=n_jobs)
    mO, sO = within_condition_cv(Y_other, y_other, spec, cv_folds=cv_folds, seed=seed + 1, n_jobs=n_jobs)

    all_classes = np.unique(np.concatenate([np.asarray(y_self).astype(int), np.asarray(y_other).astype(int)]))
    chance = 1.0 / len(all_classes)

    print("\n=== Within-condition decoding ===")
    print(f"model={spec.name} | chance≈{chance:.3f} | n_classes_total={len(all_classes)}")
    print(f"SELF  CV bal-acc: {mS:.3f} ± {sS:.3f}  (n={len(y_self)})")
    print(f"OTHER CV bal-acc: {mO:.3f} ± {sO:.3f}  (n={len(y_other)})")

    return {"self_mean": mS, "self_sd": sS, "other_mean": mO, "other_sd": sO, "chance": chance}