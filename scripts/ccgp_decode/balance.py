# ccgp_decode/balance.py
from __future__ import annotations

import numpy as np


def _resolve_balance_legacy(balance: str | None):
    """
    Backwards-compatible mapping from the legacy `balance` string to the new, orthogonal controls.

    Returns:
        gate_only: bool
        match_self_other: bool
        class_balance: str  # "none"|"hard"|"soft"
    """
    if balance is None:
        return False, True, "none"

    bal = str(balance).lower().strip()
    if bal in ("none", "gate_only"):
        return True, False, "none"
    if bal in ("match",):
        return True, True, "none"
    if bal in ("match_all", "hard", "hard_all", "force_match_all"):
        return True, True, "hard"
    if bal in ("soft", "soft_all"):
        return True, True, "soft"
    if bal in ("downsample_total",):
        # Old behavior: try to match totals (and class-wise counts on the larger side).
        # This is closest to match_self_other=True with class_balance="none",
        # but we keep the original branch below for exact compatibility.
        return None, None, None  # sentinel: handled specially
    raise ValueError(f"Unknown balance='{balance}'")


def balance_self_other(
    Y_self, y_self, Y_other, y_other,
    rng=0,
    min_per_class=10,
    balance="match",  # legacy: "none"|"gate_only"|"match"|"downsample_total"
    *,
    match_self_other: bool | None = None,
    class_balance: str | None = None,  # "none"|"hard"|"soft"
    soft_target: float | int | str = "median",  # "median"|float quantile (0-1)|int count
    soft_min_keep: int | None = None,  # drop classes that end up < this after soft capping
):
    """
    Balance two conditions (self vs other) and optionally balance across categories (classes).

    There are TWO orthogonal "balances":

    1) Match self vs other *within each class* (match_self_other=True)
       - ensures each class has the same n in self and other

    2) Balance *across classes* (class_balance="hard" or "soft")
       - "hard": forces every class to have the same n (equalized to the minimum across classes)
       - "soft": caps only large classes to a target n (median / quantile / explicit), without forcing all
                classes down to the absolute minimum

    Backwards compatibility:
      - balance="gate_only" => gating only, no downsampling
      - balance="match"     => match_self_other=True, class_balance="none"
      - balance="match_all" => match_self_other=True, class_balance="hard"
      - balance="soft"      => match_self_other=True, class_balance="soft"
      - balance="downsample_total" preserved exactly as before

    Returns:
        Ys, ys, Yo, yo, diag
    """
    rng = np.random.default_rng(rng)

    y_self = np.asarray(y_self).astype(int)
    y_other = np.asarray(y_other).astype(int)

    # ---------------- gating: keep only classes present in BOTH and meeting min_per_class in BOTH ----------------
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

    # ---------------- legacy compatibility: downsample_total branch ----------------
    if str(balance).lower().strip() == "downsample_total":
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
            match_self_other=None,
            class_balance=None,
        )
        return Ys, ys, Yo, yo, diag

    # ---------------- determine new controls (either from explicit args or legacy `balance`) ----------------
    gate_only, mso_legacy, cb_legacy = _resolve_balance_legacy(balance)

    if match_self_other is None:
        match_self_other = bool(mso_legacy) if mso_legacy is not None else False
    if class_balance is None:
        class_balance = str(cb_legacy) if cb_legacy is not None else "none"

    class_balance = str(class_balance).lower().strip()
    if class_balance not in ("none", "hard", "soft"):
        raise ValueError("class_balance must be one of {'none','hard','soft'}")

    # If user explicitly asked for no matching and no class balance, it's gating only.
    if gate_only is True and (not match_self_other) and class_balance == "none":
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
            match_self_other=bool(match_self_other),
            class_balance=str(class_balance),
        )
        return Ys, ys, Yo, yo, diag

    # ---------------- Stage 1: (optional) match self vs other WITHIN each class ----------------
    # After this stage: within each class, n_self == n_other if match_self_other=True.
    idx_s_by_c, idx_o_by_c = {}, {}
    for c in kept:
        is_ = np.where(ys0 == c)[0]
        io_ = np.where(yo0 == c)[0]
        if match_self_other:
            m = min(len(is_), len(io_))
            idx_s_by_c[int(c)] = rng.choice(is_, size=m, replace=False)
            idx_o_by_c[int(c)] = rng.choice(io_, size=m, replace=False)
        else:
            idx_s_by_c[int(c)] = is_
            idx_o_by_c[int(c)] = io_

    # Count after stage 1 (paired counts if match_self_other)
    n1 = {int(c): (len(idx_s_by_c[int(c)]), len(idx_o_by_c[int(c)])) for c in kept}

    # ---------------- Stage 2: (optional) balance ACROSS classes ----------------
    if class_balance == "none":
        pass

    elif class_balance == "hard":
        # equalize to the MIN across classes, separately per condition.
        if match_self_other:
            # counts are paired, so take min over self counts
            m = min(len(idx_s_by_c[int(c)]) for c in kept)
            for c in kept:
                c = int(c)
                idx_s_by_c[c] = rng.choice(idx_s_by_c[c], size=m, replace=False)
                idx_o_by_c[c] = rng.choice(idx_o_by_c[c], size=m, replace=False)
        else:
            # if not matching within class, equalize each condition independently
            mS = min(len(idx_s_by_c[int(c)]) for c in kept)
            mO = min(len(idx_o_by_c[int(c)]) for c in kept)
            for c in kept:
                c = int(c)
                idx_s_by_c[c] = rng.choice(idx_s_by_c[c], size=mS, replace=False)
                idx_o_by_c[c] = rng.choice(idx_o_by_c[c], size=mO, replace=False)

    elif class_balance == "soft":
        # Determine target cap:
        # - "median" (default) or "qXX" strings, or numeric quantile in [0,1], or explicit int count.
        sizes = np.array([len(idx_s_by_c[int(c)]) for c in kept], dtype=int) if match_self_other else \
                np.array([min(len(idx_s_by_c[int(c)]), len(idx_o_by_c[int(c)])) for c in kept], dtype=int)

        if isinstance(soft_target, str):
            st = soft_target.lower().strip()
            if st == "median":
                target = int(np.median(sizes))
            elif st.startswith("q"):
                q = float(st[1:])
                if q > 1:
                    q = q / 100.0
                target = int(np.quantile(sizes, q))
            else:
                raise ValueError("soft_target string must be 'median' or like 'q75'")
        elif isinstance(soft_target, (float, np.floating)):
            q = float(soft_target)
            if not (0.0 < q <= 1.0):
                raise ValueError("soft_target as float must be in (0,1] (quantile)")
            target = int(np.quantile(sizes, q))
        elif isinstance(soft_target, (int, np.integer)):
            target = int(soft_target)
        else:
            raise ValueError("soft_target must be 'median', 'q75', a float quantile, or an int")

        target = max(1, target)

        # Cap only large classes; keep small ones as-is.
        for c in kept:
            c = int(c)
            if match_self_other:
                n = len(idx_s_by_c[c])
                if n > target:
                    idx_s_by_c[c] = rng.choice(idx_s_by_c[c], size=target, replace=False)
                    idx_o_by_c[c] = rng.choice(idx_o_by_c[c], size=target, replace=False)
            else:
                # If not matching within class, cap each condition to target separately
                if len(idx_s_by_c[c]) > target:
                    idx_s_by_c[c] = rng.choice(idx_s_by_c[c], size=target, replace=False)
                if len(idx_o_by_c[c]) > target:
                    idx_o_by_c[c] = rng.choice(idx_o_by_c[c], size=target, replace=False)

        # Optionally drop classes that became too small (useful if target is small and some classes tiny)
        if soft_min_keep is not None:
            soft_min_keep = int(soft_min_keep)
            kept2 = []
            for c in kept:
                c = int(c)
                if match_self_other:
                    if len(idx_s_by_c[c]) >= soft_min_keep:
                        kept2.append(c)
                else:
                    if min(len(idx_s_by_c[c]), len(idx_o_by_c[c])) >= soft_min_keep:
                        kept2.append(c)
            kept2 = np.asarray(kept2, dtype=int)
            if kept2.size == 0:
                raise RuntimeError(
                    f"After soft balance, no classes survive soft_min_keep={soft_min_keep}."
                )
            dropped2 = np.setdiff1d(kept.astype(int), kept2)
            kept = kept2
            dropped = np.unique(np.concatenate([dropped.astype(int), dropped2.astype(int)]))
            # also prune dicts
            idx_s_by_c = {int(c): idx_s_by_c[int(c)] for c in kept}
            idx_o_by_c = {int(c): idx_o_by_c[int(c)] for c in kept}

    # ---------------- materialize indices ----------------
    idx_s = np.concatenate([idx_s_by_c[int(c)] for c in kept])
    idx_o = np.concatenate([idx_o_by_c[int(c)] for c in kept])

    Ys, ys = Ys0[idx_s], ys0[idx_s]
    Yo, yo = Yo0[idx_o], yo0[idx_o]

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
        match_self_other=bool(match_self_other),
        class_balance=str(class_balance),
        soft_target=soft_target if class_balance == "soft" else None,
        soft_min_keep=int(soft_min_keep) if (class_balance == "soft" and soft_min_keep is not None) else None,
        stage1_counts=n1,
    )
    return Ys, ys, Yo, yo, diag
