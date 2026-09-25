import pandas as pd
from joblib import Parallel, delayed

from .io import load_multi_patient_ccgp_inputs_word_order as load_many
from .ccgp import ccgp_one_patient
from .labels import recode_function_vs_content, filter_drop_class
from .labels import recode_to_super4

def run_ccgp_models_multi_patient_parallel(
    data_dict,
    patient_tags,
    model_specs,
    min_per_class=10,
    cv_folds=5,
    seed=0,
    n_workers=4,
    n_jobs_inner=1,
    balance="gate_only",
    match_self_other=True,
    return_null: bool = False,
    class_balance="none",
    soft_target="median",
    soft_min_keep=None,
    n_perm=0,
    perm_mode="within",
    use_repeated_cv=False,
    cv_repeats=10,
    within_cv_mode="fixed_params",
    compute_cv_ccgp=False,
    verbose=True,
    label_tasks=("all_11way",),
    function_id=11,
):
    import numpy as np
    import pandas as pd
    import traceback
    from joblib import Parallel, delayed

    # columns we want to ALWAYS exist in the output df (filled with NaN on failures)
    METRIC_KEYS = [
        "ccgp_bal", "ccgp_acc", "ccgp_f1",
        "ccgp_bal_cvmean", "ccgp_bal_cvstd",
        "chance",
        "ceiling_self_matched", "ceiling_other_matched",
        "ceiling_mean_matched", "ceiling_min_matched",
        "p_perm", "null_mean", "null_sd",
    ]

    def _fail_row(pid, spec_name=None, task=None, err="unknown", tb=None):
        row = {
            "patient": pid,
            "model": spec_name,
            "label_task": task,
            "function_id": int(function_id),
            "ok": False,
            "error": str(err),
        }
        if tb is not None:
            row["traceback"] = tb
        for k in METRIC_KEYS:
            row[k] = np.nan
        return row

    def _normalize_success(res, pid, spec_name, task):
        if not isinstance(res, dict):
            raise TypeError(f"ccgp_one_patient returned {type(res)}; expected dict")

        res["patient"] = pid
        res["model"] = spec_name
        res["label_task"] = task
        res["function_id"] = int(function_id)
        res["ok"] = True

        for k in METRIC_KEYS:
            res.setdefault(k, np.nan)

        return res

    def _run_one(pid):
        # consistent early-return schema
        if pid not in data_dict or (not data_dict[pid].get("ok", True)) or ("error" in data_dict[pid]):
            err = data_dict.get(pid, {}).get("error", "missing")
            return [_fail_row(pid, spec_name=None, task=None, err=err, tb=None)]

        Ys = data_dict[pid]["Y_self"];  ys = data_dict[pid]["y_self"]
        Yo = data_dict[pid]["Y_other"]; yo = data_dict[pid]["y_other"]

        rows = []
        for spec in model_specs:
            job_seed = int(seed + (abs(hash((pid, spec.name))) % 1_000_000))

            for task in label_tasks:
                try:
                    if task == "all_11way":
                        res = ccgp_one_patient(
                            Ys, ys, Yo, yo,
                            spec=spec,
                            return_null=return_null,
                            min_per_class=min_per_class,
                            balance=balance,
                            match_self_other=match_self_other,
                            class_balance=class_balance,
                            soft_target=soft_target,
                            soft_min_keep=soft_min_keep,
                            cv_folds=cv_folds,
                            seed=job_seed,
                            n_jobs=n_jobs_inner,
                            n_perm=n_perm,
                            perm_mode=perm_mode,
                            use_repeated_cv=use_repeated_cv,
                            cv_repeats=cv_repeats,
                            within_cv_mode=within_cv_mode,
                            compute_cv_ccgp=compute_cv_ccgp,
                        )

                    elif task == "func_vs_sem":
                        ys_bin = recode_function_vs_content(ys, function_id)
                        yo_bin = recode_function_vs_content(yo, function_id)
                        res = ccgp_one_patient(
                            Ys, ys_bin, Yo, yo_bin,
                            spec=spec,
                            return_null=return_null,
                            min_per_class=min(5, int(min_per_class)),
                            balance=balance,
                            match_self_other=match_self_other,
                            class_balance=class_balance,
                            soft_target=soft_target,
                            soft_min_keep=soft_min_keep,
                            cv_folds=cv_folds,
                            seed=job_seed + 101,
                            n_jobs=n_jobs_inner,
                            n_perm=n_perm,
                            perm_mode=perm_mode,
                            use_repeated_cv=use_repeated_cv,
                            cv_repeats=cv_repeats,
                            within_cv_mode=within_cv_mode,
                            compute_cv_ccgp=compute_cv_ccgp,
                        )

                    elif task == "sem_only":
                        Ys2, ys2 = filter_drop_class(Ys, ys, function_id)
                        Yo2, yo2 = filter_drop_class(Yo, yo, function_id)
                        res = ccgp_one_patient(
                            Ys2, ys2, Yo2, yo2,
                            spec=spec,
                            return_null=return_null,
                            min_per_class=min_per_class,
                            balance=balance,
                            match_self_other=match_self_other,
                            class_balance=class_balance,
                            soft_target=soft_target,
                            soft_min_keep=soft_min_keep,
                            cv_folds=cv_folds,
                            seed=job_seed + 202,
                            n_jobs=n_jobs_inner,
                            n_perm=n_perm,
                            perm_mode=perm_mode,
                            use_repeated_cv=use_repeated_cv,
                            cv_repeats=cv_repeats,
                            within_cv_mode=within_cv_mode,
                            compute_cv_ccgp=compute_cv_ccgp,
                        )

                    elif task == "super4":
                        ys4 = recode_to_super4(ys)
                        yo4 = recode_to_super4(yo)
                        res = ccgp_one_patient(
                            Ys, ys4, Yo, yo4,
                            spec=spec,
                            min_per_class=min_per_class,
                            balance=balance,
                            match_self_other=match_self_other,
                            class_balance=class_balance,
                            soft_target=soft_target,
                            soft_min_keep=soft_min_keep,
                            return_null=return_null,
                            cv_folds=cv_folds,
                            seed=job_seed + 303,
                            n_jobs=n_jobs_inner,
                            n_perm=n_perm,
                            perm_mode=perm_mode,
                            use_repeated_cv=use_repeated_cv,
                            cv_repeats=cv_repeats,
                            within_cv_mode=within_cv_mode,
                            compute_cv_ccgp=compute_cv_ccgp,
                        )

                    else:
                        raise ValueError(f"Unknown label task: {task}")

                    res = _normalize_success(res, pid, spec.name, task)
                    rows.append(res)

                    if verbose:
                        # robust formatting (won't crash on nan / weird types)
                        ccgp_val = res.get("ccgp_bal_cvmean", res.get("ccgp_bal", np.nan))
                        try:
                            ccgp_str = f"{float(ccgp_val):.3f}"
                        except Exception:
                            ccgp_str = str(ccgp_val)

                        print(
                            f"✅ {pid} | {spec.name} | {task} | ccgp={ccgp_str} | "
                            f"classes={res.get('n_classes')} | min_after={res.get('min_after')}"
                        )

                except Exception as e:
                    tb = traceback.format_exc()
                    if verbose:
                        print(f"❌ {pid} | {spec.name} | {task} failed: {e}")
                    rows.append(_fail_row(pid, spec_name=spec.name, task=task, err=e, tb=tb))

        return rows

    if verbose:
        print(f"Running {len(patient_tags)} patients × {len(model_specs)} models")

    all_rows_nested = Parallel(n_jobs=n_workers, backend="loky")(
        delayed(_run_one)(pid) for pid in patient_tags
    )
    rows = [r for sub in all_rows_nested for r in sub]
    df = pd.DataFrame(rows)

    # ensure metric columns always exist even if df is empty or all failures
    for k in METRIC_KEYS:
        if k not in df.columns:
            df[k] = np.nan

    if "patient" in df.columns and "model" in df.columns:
        df = df.sort_values(["patient", "model"]).reset_index(drop=True)

    return df

import numpy as np

def apply_label_task(Ys, ys, Yo, yo, task: str, function_id: int, min_per_class: int):
    """
    Returns (Ys2, ys2, Yo2, yo2, min_per_class_task) consistent with label_tasks.
    """
    task = str(task)

    if task == "all_11way":
        return Ys, np.asarray(ys).astype(int), Yo, np.asarray(yo).astype(int), int(min_per_class)

    elif task == "func_vs_sem":
        ys2 = recode_function_vs_content(ys, function_id)
        yo2 = recode_function_vs_content(yo, function_id)
        return Ys, np.asarray(ys2).astype(int), Yo, np.asarray(yo2).astype(int), int(min(5, int(min_per_class)))

    elif task == "sem_only":
        Ys2, ys2 = filter_drop_class(Ys, ys, function_id)
        Yo2, yo2 = filter_drop_class(Yo, yo, function_id)
        return Ys2, np.asarray(ys2).astype(int), Yo2, np.asarray(yo2).astype(int), int(min_per_class)

    elif task in ("super4", "super_4"):
        ys2 = recode_to_super4(ys)
        yo2 = recode_to_super4(yo)
        return Ys, np.asarray(ys2).astype(int), Yo, np.asarray(yo2).astype(int), int(min_per_class)

    else:
        raise ValueError(f"Unknown label task: {task}")