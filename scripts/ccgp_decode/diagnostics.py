# ccgp_decode/diagnostics.py
from __future__ import annotations
import numpy as np
import pandas as pd

import sklearn
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score, make_scorer

from .balance import balance_self_other # type: ignore
from .models import make_pipeline, ModelSpec # type: ignore

_SCORER = make_scorer(balanced_accuracy_score)

def speaker_sanity(spk_all, speaker_of_interest="Speaker1"):
    spk_all = np.asarray(spk_all, dtype=object)
    uniq, cnt = np.unique(spk_all, return_counts=True)
    soi = speaker_of_interest.lower()
    mask_self = np.array([str(s).lower() == soi for s in spk_all], dtype=bool)

    print("\n=== Speaker sanity ===")
    print("unique spk_all:", list(zip(uniq.tolist(), cnt.tolist())))
    print("speaker_of_interest:", speaker_of_interest)
    print("self fraction:", float(mask_self.mean()))

def _class_count_table(y, name=""):
    y = np.asarray(y).astype(int)
    u, c = np.unique(y, return_counts=True)
    return pd.DataFrame({"class": u, f"n_{name}": c}).sort_values("class").reset_index(drop=True)

def print_class_counts(y_self, y_other, min_per_class=10, balance="gate_only", seed=0):
    print("\n=== Class counts BEFORE gating ===")
    dfS = _class_count_table(y_self, "self")
    dfO = _class_count_table(y_other, "other")
    df = dfS.merge(dfO, on="class", how="outer").fillna(0)
    df["class"] = df["class"].astype(int)
    df["n_self"] = df["n_self"].astype(int)
    df["n_other"] = df["n_other"].astype(int)
    df["min_both"] = df[["n_self", "n_other"]].min(axis=1)
    print(df.to_string(index=False))

    # dummy matrices to reuse gating logic
    YS_dummy = np.zeros((len(y_self), 1))
    YO_dummy = np.zeros((len(y_other), 1))
    _, _, _, _, diag = balance_self_other(
        YS_dummy, y_self, YO_dummy, y_other,
        rng=seed,
        min_per_class=min_per_class,
        balance=balance,
    )

    print("\n=== AFTER gating/balancing ===")
    print("kept_classes:", diag["kept_classes"])
    print("dropped_classes:", diag["dropped_classes"])
    print("n_classes:", diag["n_classes"], "chance:", diag["chance"])
    print("min_after:", diag["min_after"], "n_self:", diag["n_self"], "n_other:", diag["n_other"])
    print("counts_after:", diag.get("counts_after"))
    return diag

def within_condition_cv(Y, y, spec: ModelSpec, cv_folds=5, seed=0, n_jobs=1):
    Y = np.asarray(Y)
    y = np.asarray(y).astype(int)
    if np.unique(y).size < 2:
        return np.nan, np.nan

    pipe = make_pipeline(spec)
    cv = sklearn.model_selection.StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scores = sklearn.model_selection.cross_val_score(pipe, Y, y, scoring=_SCORER, cv=cv, n_jobs=n_jobs)
    return float(scores.mean()), float(scores.std(ddof=1))

def run_within_condition_diagnostics(Y_self, y_self, Y_other, y_other, spec: ModelSpec,
                                     cv_folds=5, seed=0, n_jobs=1, kept_classes=None):
    mS, sS = within_condition_cv(Y_self, y_self, spec, cv_folds=cv_folds, seed=seed, n_jobs=n_jobs)
    mO, sO = within_condition_cv(Y_other, y_other, spec, cv_folds=cv_folds, seed=seed+1, n_jobs=n_jobs)

    if kept_classes is not None and len(kept_classes) > 0:
        chance = 1.0 / len(kept_classes)
        n_classes_total = len(kept_classes)
    else:
        all_classes = np.unique(np.concatenate([np.asarray(y_self).astype(int), np.asarray(y_other).astype(int)]))
        chance = 1.0 / len(all_classes)
        n_classes_total = len(all_classes)

    print("\n=== Within-condition decoding ===")
    print(f"model={spec.name} | chance≈{chance:.3f} | n_classes_total={n_classes_total}")
    print(f"SELF  CV bal-acc: {mS:.3f} ± {sS:.3f}  (n={len(y_self)})")
    print(f"OTHER CV bal-acc: {mO:.3f} ± {sO:.3f}  (n={len(y_other)})")

    return {"self_mean": mS, "self_sd": sS, "other_mean": mO, "other_sd": sO, "chance": chance}

