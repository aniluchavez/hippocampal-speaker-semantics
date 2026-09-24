#!/usr/bin/env python3
"""
cluster_glm_legacy_replication.py

Literal reconstruction of PoissonRidgeContigModular.ipynb's original
clusterwise self-vs-other pipeline -- same balancing logic
(report_cluster_balance -> minimal_balancing -> run_clusterwise_cosine_distance
with balance_function_words=True, cap_large_clusters=False,
min_trials_per_condition=10, max_size_ratio=2, cap_reference="median",
balance_all_clusters=True, soft_balance=True) and the same n_components=10 --
but on the current data (LLaMA-3.1-8b embeddings, current 15-patient list,
via cluster_glm_reliability.py's data loading) and the validated GLM fitter
(select_alpha_cv + fit_poisson_ridge_beta_sklearn from reliability.py)
instead of the old ad hoc PoissonRegressor+GridSearchCV.

Uses the bug-fixed report_cluster_balance/minimal_balancing/
run_clusterwise_cosine_distance in neural_encoding/cluster_analysis.py (the
sign-blind-argmax and row-scrambling fixes made earlier this session) rather
than literally reintroducing those bugs -- everything else matches the
original recipe.

Usage:
    cd /scratch/aniluchavez/hippocampal-speaker-semantics
    python3 -u scripts/cluster_glm_legacy_replication.py
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import cluster_glm_reliability as glm  # noqa: E402

RESULTS_ROOT = "/scratch/aniluchavez/ConvoDATAS/SemanticGLM/clusterwise_legacy_replication"
N_COMPONENTS = 10  # matches the original notebook's n_components, not the
                    # 100 (or adaptive-20) used in the rest of this session's work

_cluster_mod = glm._cluster_mod
report_cluster_balance = _cluster_mod.report_cluster_balance
minimal_balancing = _cluster_mod.minimal_balancing
run_clusterwise_cosine_distance = _cluster_mod.run_clusterwise_cosine_distance


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--patient", type=str, default=None)
    p.add_argument("--region", type=str, default=None)
    p.add_argument("--n_jobs", type=int, default=8)
    p.add_argument("--window-tag", type=str, default=None)
    p.add_argument("--exclude-clusters", type=int, nargs="*", default=[])
    p.add_argument("--hard-balance", action="store_true")
    p.add_argument("--fixed-balance-target", type=int, default=None,
                   help="Match every retained cluster to exactly this many trials per condition.")
    p.add_argument("--upsample-to-median", action="store_true",
                   help=("For soft balancing, resample every retained cluster/condition "
                         "to the patient-specific median count: downsample above-median "
                         "without replacement and upsample below-median with replacement."))
    p.add_argument("--balance-target-mode", type=str, default="median",
                   choices=["median", "mean", "pooled_condition_median", "pooled_condition_mean"],
                   help=("Soft-balance target. 'median'/'mean' use per-category matched "
                         "min(self, other). 'pooled_condition_median'/'mean' use all "
                         "category-condition cell counts before self/other matching."))
    p.add_argument("--no-balance", action="store_true",
                   help="Use all natural self/other trials in every cluster.")
    p.add_argument("--results-root", type=str, default=RESULTS_ROOT)
    p.add_argument("--model-tag", type=str, default=None)
    p.add_argument("--context-tag", type=str, default=None)
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--min-matched-trials", type=int, default=None,
                   help="Keep only clusters having at least this many trials in both conditions.")
    p.add_argument("--min-retained-categories", type=int, default=2)
    p.add_argument("--content-macro-categories", action="store_true",
                   help="Pool content words into 3 semantic domains and exclude function words.")
    return p.parse_args()


def main():
    args = _args()
    if args.window_tag:
        glm.WINDOW_TAG = args.window_tag
    if args.model_tag:
        glm.MODEL_TAG = args.model_tag
    if args.context_tag is not None:
        glm.CONTEXT_TAG = args.context_tag
    layer = args.layer if args.layer is not None else glm.LAYER
    run_poisson_ridge = glm.make_beta_fit_adapter(glm.ALPHAS)

    patients = [p for p in glm.PATIENTS if args.patient is None or p["patient_ID"] == args.patient]

    for cfg in patients:
        patient_ID = cfg["patient_ID"]
        regions = [r for r in cfg["region_ranges"] if args.region is None or r == args.region]

        for region in regions:
            print(f"\n{'='*60}\n  {patient_ID} / {region}", flush=True)

            data = glm.build_patient_region_data(cfg, region, layer, N_COMPONENTS)
            if data is None:
                continue

            if args.content_macro_categories:
                macro_map = {
                    1: 1, 2: 1, 6: 1,          # concrete/referential
                    3: 2, 4: 2, 5: 2, 10: 2,  # socio-affective/cognitive
                    7: 3, 8: 3, 9: 3,         # perceptual/action
                }
                for cond in ("self", "other"):
                    meta = data[f"metadata_{cond}"]
                    keep = meta["ClusterID"].isin(macro_map).to_numpy()
                    data[f"X_{cond}"] = data[f"X_{cond}"][keep]
                    data[f"Y_{cond}"] = data[f"Y_{cond}"][keep]
                    data[f"metadata_{cond}"] = pd.DataFrame({
                        "ClusterID": meta.loc[keep, "ClusterID"].map(macro_map).to_numpy()
                    })
                print("  Remapped to 3 content supercategories; excluded Function Words",
                      flush=True)

            report_cluster_balance(
                data["metadata_self"], data["metadata_other"], patient_ID, region)

            # minimal_balancing drops rows and reset_index(drop=True)s the
            # result, so its output index no longer maps to row positions in
            # data["X_self"]/data["Y_self"] whenever it actually downsamples a
            # cluster (confirmed via the "Downsampling ..." log lines -- this
            # hit nearly every patient/region this batch). Track original row
            # positions through the call so X/Y can be re-sliced to match.
            if args.no_balance:
                X_self_bal, Y_self_bal = data["X_self"], data["Y_self"]
                X_other_bal, Y_other_bal = data["X_other"], data["Y_other"]
                metadata_self_bal = data["metadata_self"][["ClusterID"]].reset_index(drop=True)
                metadata_other_bal = data["metadata_other"][["ClusterID"]].reset_index(drop=True)
                print("  Using all natural trials (no balancing)", flush=True)
            else:
                meta_self_ix = data["metadata_self"].copy()
                meta_self_ix["_orig_ix"] = np.arange(len(meta_self_ix))
                meta_other_ix = data["metadata_other"].copy()
                meta_other_ix["_orig_ix"] = np.arange(len(meta_other_ix))

                metadata_self_bal, metadata_other_bal = minimal_balancing(
                    meta_self_ix, meta_other_ix, cluster_column="ClusterID")

                orig_ix_self = metadata_self_bal["_orig_ix"].to_numpy()
                orig_ix_other = metadata_other_bal["_orig_ix"].to_numpy()
                X_self_bal = data["X_self"][orig_ix_self]
                Y_self_bal = data["Y_self"][orig_ix_self]
                X_other_bal = data["X_other"][orig_ix_other]
                Y_other_bal = data["Y_other"][orig_ix_other]
                metadata_self_bal = metadata_self_bal[["ClusterID"]].reset_index(drop=True)
                metadata_other_bal = metadata_other_bal[["ClusterID"]].reset_index(drop=True)

            if args.exclude_clusters:
                keep_s = ~metadata_self_bal["ClusterID"].isin(args.exclude_clusters).to_numpy()
                keep_o = ~metadata_other_bal["ClusterID"].isin(args.exclude_clusters).to_numpy()
                X_self_bal, Y_self_bal = X_self_bal[keep_s], Y_self_bal[keep_s]
                metadata_self_bal = metadata_self_bal.loc[keep_s].reset_index(drop=True)
                X_other_bal, Y_other_bal = X_other_bal[keep_o], Y_other_bal[keep_o]
                metadata_other_bal = metadata_other_bal.loc[keep_o].reset_index(drop=True)
                print(f"  Excluded clusters: {args.exclude_clusters}", flush=True)

            if args.min_matched_trials is not None:
                counts_s = metadata_self_bal["ClusterID"].value_counts()
                counts_o = metadata_other_bal["ClusterID"].value_counts()
                eligible = sorted(
                    c for c in set(counts_s.index) | set(counts_o.index)
                    if counts_s.get(c, 0) >= args.min_matched_trials
                    and counts_o.get(c, 0) >= args.min_matched_trials
                )
                print(f"  Eligible clusters at >= {args.min_matched_trials}/condition: "
                      f"{eligible}", flush=True)
                if len(eligible) < args.min_retained_categories:
                    print(f"  SKIP: only {len(eligible)} eligible categories", flush=True)
                    continue
                keep_s = metadata_self_bal["ClusterID"].isin(eligible).to_numpy()
                keep_o = metadata_other_bal["ClusterID"].isin(eligible).to_numpy()
                X_self_bal, Y_self_bal = X_self_bal[keep_s], Y_self_bal[keep_s]
                metadata_self_bal = metadata_self_bal.loc[keep_s].reset_index(drop=True)
                X_other_bal, Y_other_bal = X_other_bal[keep_o], Y_other_bal[keep_o]
                metadata_other_bal = metadata_other_bal.loc[keep_o].reset_index(drop=True)

            run_clusterwise_cosine_distance(
                X_self=X_self_bal, X_other=X_other_bal,
                Y_self=Y_self_bal, Y_other=Y_other_bal,
                metadata_self=metadata_self_bal, metadata_other=metadata_other_bal,
                cluster_column="ClusterID",
                region_name=region, patient_id=patient_ID,
                n_components=N_COMPONENTS,
                results_root=args.results_root,
                run_poisson_ridge=run_poisson_ridge,
                balance_function_words=not args.no_balance,
                compute_half_splits=True,
                n_jobs=args.n_jobs,
                cap_large_clusters=False,
                min_trials_per_condition=10,
                print_trial_counts=True,
                max_size_ratio=2,
                cap_reference="median",
                balance_all_clusters=not args.no_balance,
                soft_balance=(
                    args.fixed_balance_target is not None
                    or (not args.hard_balance and not args.no_balance)
                ),
                soft_balance_target=(
                    args.fixed_balance_target
                    if args.fixed_balance_target is not None else args.balance_target_mode
                ),
                resample_to_balance_target=args.upsample_to_median,
            )

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
