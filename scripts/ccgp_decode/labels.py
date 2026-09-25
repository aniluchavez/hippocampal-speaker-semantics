# ccgp_decode/labels.py
from __future__ import annotations
import numpy as np
from typing import Tuple

def recode_function_vs_content(y, function_class_id: int):
    """
    Binary labels:
      1 = function word class
      0 = content (all other classes)
    """
    y = np.asarray(y).astype(int)
    return (y == int(function_class_id)).astype(int)

def filter_drop_class(Y, y, drop_class_id: int):
    """
    Drops one class entirely, returns filtered (Y, y).
    """
    y = np.asarray(y).astype(int)
    keep = (y != int(drop_class_id))
    return Y[keep], y[keep]

# ccgp_decode/labels.py
import numpy as np

def recode_function_vs_content(y, function_class_id: int):
    """
    Binary labels:
      1 = function word class
      0 = content (all other classes)
    """
    y = np.asarray(y).astype(int)
    return (y == int(function_class_id)).astype(int)

def filter_drop_class(Y, y, drop_class_id: int):
    """
    Drops one class entirely (e.g., function words), returns filtered (Y, y).
    """
    y = np.asarray(y).astype(int)
    keep = (y != int(drop_class_id))
    return Y[keep], y[keep]

def remap_labels_to_contiguous(y, classes=None):
    """
    Remap labels to contiguous 0..K-1.
    If `classes` is provided, uses that ordering (and checks membership).
    Returns (y_new, mapping_old_to_new, classes_used).
    """
    y = np.asarray(y).astype(int)

    if classes is None:
        classes = np.unique(y)
    else:
        classes = np.asarray(classes).astype(int)
        if not np.all(np.isin(y, classes)):
            missing = np.setdiff1d(np.unique(y), classes)
            raise RuntimeError(f"Labels contain values not in provided classes: {missing.tolist()}")

    mapping = {int(c): int(i) for i, c in enumerate(classes)}
    y_new = np.array([mapping[int(v)] for v in y], dtype=int)
    return y_new, mapping, classes

def balance_self_other(
    Y_self, y_self, Y_other, y_other,
    rng=0,
    min_per_class=10,
    balance="match",  # "none"|"gate_only"|"match"|"downsample_total"
):
    """
    Returns Ys, ys, Yo, yo, diag

    balance:
      - "none": keep all trials; just drop classes not shared or < min_per_class in either condition
      - "gate_only": same as "none" (alias; clearer intent)
      - "match": per-class exact matching to m=min(count_self, count_other)
      - "downsample_total": keep gated classes; then downsample the *larger condition* to match total N (stratified)
    """
    rng = np.random.default_rng(rng)

    y_self = np.asarray(y_self).astype(int)
    y_other = np.asarray(y_other).astype(int)

    classes = np.intersect1d(np.unique(y_self), np.unique(y_other))
    counts_self = {int(c): int(np.sum(y_self == c)) for c in classes}
    counts_other = {int(c): int(np.sum(y_other == c)) for c in classes}

    kept = [c for c in classes if (counts_self[int(c)] >= min_per_class and counts_other[int(c)] >= min_per_class)]
    kept = np.asarray(kept, dtype=int)
    dropped = np.setdiff1d(classes.astype(int), kept)

    if kept.size == 0:
        raise RuntimeError(
            f"No classes survive min_per_class={min_per_class}. "
            f"Intersection had {len(classes)} classes."
        )

    keep_s = np.isin(y_self, kept)
    keep_o = np.isin(y_other, kept)

    Ys0, ys0 = Y_self[keep_s], y_self[keep_s]
    Yo0, yo0 = Y_other[keep_o], y_other[keep_o]

    if balance in ("none", "gate_only"):
        Ys, ys, Yo, yo = Ys0, ys0, Yo0, yo0
        counts_after = {int(c): (int(np.sum(ys == c)), int(np.sum(yo == c))) for c in kept}
        min_after = int(min(min(v) for v in counts_after.values()))
        diag = dict(
            kept_classes=[int(c) for c in kept.tolist()],
            dropped_classes=[int(c) for c in dropped.tolist()],
            counts_self=counts_self,
            counts_other=counts_other,
            counts_after=counts_after,
            n_classes=int(len(kept)),
            min_after=int(min_after),
            n_self=int(len(ys)),
            n_other=int(len(yo)),
            chance=float(1.0 / len(kept)),
            balance=str(balance),
        )
        return Ys, ys, Yo, yo, diag

    if balance == "match":
        idx_s, idx_o = [], []
        counts_after = {}
        for c in kept:
            is_ = np.where(ys0 == c)[0]
            io_ = np.where(yo0 == c)[0]
            m = min(len(is_), len(io_))
            idx_s.append(rng.choice(is_, size=m, replace=False))
            idx_o.append(rng.choice(io_, size=m, replace=False))
            counts_after[int(c)] = int(m)
        idx_s = np.concatenate(idx_s)
        idx_o = np.concatenate(idx_o)
        Ys, ys = Ys0[idx_s], ys0[idx_s]
        Yo, yo = Yo0[idx_o], yo0[idx_o]
        min_after = int(min(counts_after.values()))
        diag = dict(
            kept_classes=[int(c) for c in kept.tolist()],
            dropped_classes=[int(c) for c in dropped.tolist()],
            counts_self=counts_self,
            counts_other=counts_other,
            counts_after={int(k): int(v) for k, v in counts_after.items()},
            n_classes=int(len(kept)),
            min_after=int(min_after),
            n_self=int(len(ys)),
            n_other=int(len(yo)),
            chance=float(1.0 / len(kept)),
            balance=str(balance),
        )
        return Ys, ys, Yo, yo, diag

    if balance == "downsample_total":
        nS, nO = len(ys0), len(yo0)
        if nS == nO:
            Ys, ys, Yo, yo = Ys0, ys0, Yo0, yo0
        elif nS > nO:
            idx = []
            for c in kept:
                ic = np.where(ys0 == c)[0]
                target_c = int(np.sum(yo0 == c))
                target_c = min(target_c, len(ic))
                idx.append(rng.choice(ic, size=target_c, replace=False))
            idx = np.concatenate(idx)
            Ys, ys = Ys0[idx], ys0[idx]
            Yo, yo = Yo0, yo0
        else:
            idx = []
            for c in kept:
                ic = np.where(yo0 == c)[0]
                target_c = int(np.sum(ys0 == c))
                target_c = min(target_c, len(ic))
                idx.append(rng.choice(ic, size=target_c, replace=False))
            idx = np.concatenate(idx)
            Yo, yo = Yo0[idx], yo0[idx]
            Ys, ys = Ys0, ys0

        counts_after = {int(c): (int(np.sum(ys == c)), int(np.sum(yo == c))) for c in kept}
        min_after = int(min(min(v) for v in counts_after.values()))
        diag = dict(
            kept_classes=[int(c) for c in kept.tolist()],
            dropped_classes=[int(c) for c in dropped.tolist()],
            counts_self=counts_self,
            counts_other=counts_other,
            counts_after=counts_after,
            n_classes=int(len(kept)),
            min_after=int(min_after),
            n_self=int(len(ys)),
            n_other=int(len(yo)),
            chance=float(1.0 / len(kept)),
            balance=str(balance),
        )
        return Ys, ys, Yo, yo, diag

    raise ValueError(f"Unknown balance='{balance}'")


# ------------------------------------------------------------
# 11 → 4 super-category mapping
# ------------------------------------------------------------
# Original labels:
# 1.Personal pronouns
# 2.Name/place
# 3.Actions
# 4.Body/clothing
# 5.Descriptors
# 6.Numerical
# 7.Objects
# 8.Abstract concepts
# 9.Social
# 10.Health/medicine
# 11.Function words

MAP_11_TO_SUPER4 = {

    # 0 = Function
    11: 0,

    # 1 = People & Social
    1: 1,   # personal pronouns
    2: 1,   # name/place
    9: 1,   # social

    # 2 = Actions & Modifiers
    3: 2,   # actions
    5: 2,   # descriptors
    6: 2,   # numerical

    # 3 = Concrete & Conceptual Content
    4: 3,   # body/clothing
    7: 3,   # objects
    10: 3,  # health/medicine
    8: 3,   # abstract concepts
}
def recode_to_super4(y):
    """
    Map 11-way labels -> 4-way super labels (0..3 or 1..4, your choice)
    """
    y = np.asarray(y).astype(int)

    # if you already have a dict mapping, use it here
    # e.g. MAP_11_TO_SUPER4 = {1:..., 2:..., ..., 11:...}

    y4 = np.array([MAP_11_TO_SUPER4[int(v)] for v in y], dtype=int)
    return y4
