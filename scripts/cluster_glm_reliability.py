#!/usr/bin/env python3
"""
cluster_glm_reliability.py

Per-semantic-cluster self-vs-other beta cosine distance, using:
  - the same LLaMA-pipeline data loading as scripts/semantic_glm.py
    (find_spike_dir / load_speaker_assignment / load_condition_ordered,
     copied verbatim below since semantic_glm.py runs its full analysis
     loop at import time and can't be imported directly), and
  - the validated sklearn Poisson-ridge fitter from
    neural_encoding/reliability.py (fit_poisson_ridge_beta_sklearn +
    select_alpha_cv) -- the same fitter run_beta_reliability_all_neurons
    uses for the validated self/other r_cross comparison -- instead of the
    ad hoc PoissonRegressor+GridSearchCV fitter the old BERT-pipeline
    notebook used for clusterwise betas.

Cluster labels come from the canonical
Transcripts/{pid}_filtered_used_rows_withNP_withClusterIDNew.xlsx
(FinalClusterID column), which is also the source the embedding-cache
builder (p6/run_all_patients_multi_model.py) reads before writing
{pid}_llama-3.1-8b_word_emb_layers.npy. That builder applies
dropna(onset).sort_values(onset).reset_index before writing the cache, so
cluster IDs go through the identical transform here to land on the correct
row -- guarded by a hard row-count check against the cache shape.

Usage:
    cd /scratch/aniluchavez/hippocampal-speaker-semantics
    python3 -u scripts/cluster_glm_reliability.py [--patient PID] [--region NAME]
"""

import os
import sys
import argparse
import warnings
import importlib.util

# Must be set before numpy/sklearn import. With n_jobs threads each also
# spawning a multi-threaded BLAS call underneath, a 112-core box lets every
# thread try to grab dozens of cores for matrix ops far too small to benefit
# -- pure contention. The actual parallelism here comes from the outer
# joblib threads (one PoissonRegressor fit per thread), not from BLAS.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.decomposition import PCA
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT    = "/scratch/aniluchavez/hippocampal-speaker-semantics"
EMBED_DIR       = "/scratch/aniluchavez/ConvoDATAS/EmbedCache"
SPIKE_ROOT      = "/scratch/aniluchavez/ConvoDATAS/SpikeWindows"
TRANSCRIPT_ROOT = "/scratch/aniluchavez/ConvoDATAS/Transcripts"
RESULTS_ROOT    = "/scratch/aniluchavez/ConvoDATAS/SemanticGLM/clusterwise"


def _load_module(rel_path, name):
    """Load a neural_encoding/*.py file directly by path, bypassing
    neural_encoding/__init__.py (which eagerly imports regression.py --
    broken under this env's Python 3.12: a dataclass field has a mutable
    numpy-array default, which dataclasses now rejects). Matches the
    convention scripts/semantic_glm.py and scripts/run_reliability.py
    already use for the same reason."""
    path = os.path.join(PROJECT_ROOT, rel_path)
    if name in sys.modules:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_cluster_mod = _load_module("neural_encoding/cluster_analysis.py", "nn_cluster_analysis")

report_cluster_balance = _cluster_mod.report_cluster_balance
run_clusterwise_cosine_distance_bootstrap = _cluster_mod.run_clusterwise_cosine_distance_bootstrap


# fit_poisson_ridge_beta_sklearn / select_alpha_cv / poisson_ll_numpy, copied
# verbatim from neural_encoding/reliability.py (not aliased from a dynamically
# loaded module like the functions above) -- run_clusterwise_cosine_distance's
# run_poisson_ridge callback runs inside joblib.Parallel, and a process-pool
# worker can only unpickle a function by reference if it's a normal top-level
# function in a *really* importable module. A function aliased from a module
# that only exists because _load_module() stuffed it into this process's
# sys.modules doesn't qualify ("No module named 'nn_reliability'" in the
# worker). Top-level functions defined directly in this script do qualify,
# since this script runs as __main__ and cloudpickle special-cases functions
# from __main__ to serialize by value.

def poisson_ll_numpy(y_true, mu_pred):
    y_true = np.asarray(y_true, dtype=float)
    mu_pred = np.clip(np.asarray(mu_pred, dtype=float), 1e-10, None)
    return float(np.sum(y_true * np.log(mu_pred) - mu_pred - gammaln(y_true + 1)))


def fit_poisson_ridge_beta_sklearn(X, y, alpha, *, fit_intercept=True, max_iter=1000):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    model = PoissonRegressor(alpha=float(alpha), fit_intercept=fit_intercept, max_iter=max_iter)
    model.fit(X, y)
    return model.coef_.astype(float, copy=True)


def select_alpha_cv(X, y, alphas, *, n_splits=5, random_state=0, standardize=True):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = X.shape[0]
    if n < max(10, 2 * n_splits):
        return float(alphas[0])

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits_list = list(kf.split(X, y))

    best_alpha = float(alphas[0])
    best_score = -np.inf
    for a in alphas:
        ll_list = []
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

# ── config (matches scripts/semantic_glm.py defaults) ──────────────────────────
MODEL_TAG    = "llama-3.1-8b"
CONTEXT_TAG  = "_ctx200"  # matches what's actually on disk in EmbedCache --
                          # no patient has a bare (no-context-tag) llama-3.1-8b
                          # cache file; all are *_ctx200_word_emb_layers.npy
LAYER        = 18
N_COMPONENTS = 20  # ceiling; adaptive_n_components truncates further per-cluster
                    # when even this is too many dimensions for that cluster's
                    # trial count (see run_clusterwise_cosine_distance_bootstrap's
                    # docstring -- semantic_glm.py already guards against this
                    # for its own whole-condition fits, this mirrors that rule)
MIN_SPIKES   = 5
ALPHAS       = np.logspace(-3, 3, 30)
WINDOW_TAG   = "tshift-150_tlen500_oshift+200_olen500"   # "fixed" window convention
N_BOOTSTRAP          = 20  # resampled-fit repeats for the larger side per cluster
N_HALFSPLIT_REPEATS  = 5   # repeated 50/50 splits per side for reliability

# ── patients (verbatim from scripts/semantic_glm.py) ────────────────────────────
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


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--patient", type=str, default=None,
                    help="Run only this patient_ID; runs all if omitted")
    p.add_argument("--region", type=str, default=None,
                    help="Run only this region; runs all if omitted")
    p.add_argument("--layer", type=int, default=LAYER)
    p.add_argument("--n_components", type=int, default=N_COMPONENTS)
    p.add_argument("--n_jobs", type=int, default=8)
    p.add_argument("--no_balance", action="store_true",
                    help="Fit self/other on their full natural trial counts "
                         "(no within-cluster balancing/bootstrap). Writes to "
                         "a separate '_nobalance' results root since self and "
                         "other are fit on different N -- not directly "
                         "comparable to the balanced output.")
    p.add_argument("--cross_cluster_cap", type=str, default=None,
                    help="Cap every cluster's self/other target size to make "
                         "clusters directly comparable (bootstrap-resampling "
                         "whichever side(s) exceed the cap, same averaging "
                         "trick as the self/other balance itself). Pass "
                         "'min_kept' to cap at the smallest kept cluster's "
                         "natural size, or an integer for an explicit cap. "
                         "Writes to a separate '_capN'/'_capminkept' results "
                         "root. Ignored if --no_balance is set.")
    return p.parse_args()


# ── data-loading helpers, copied verbatim from scripts/semantic_glm.py ─────────

def load_spike_matrix(spike_dir, speaker, region):
    spk_dir = os.path.join(spike_dir, speaker)
    if not os.path.isdir(spk_dir):
        return None
    cands = [f for f in os.listdir(spk_dir)
             if f.lower().startswith(region.lower()) and f.endswith("_spike_counts.npy")]
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
                assign[i] = col
                break
    mask_self = assign == "Speaker1"
    mask_other = np.array([(a is not None and a != "Speaker1") for a in assign], dtype=bool)
    return assign, mask_self, mask_other, dir_membership


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


def find_spike_dir(patient):
    d = os.path.join(SPIKE_ROOT, f"output_{patient}_english_only_{WINDOW_TAG}")
    return d if os.path.isdir(d) else None


def load_cluster_ids(patient_ID, n_words_expected):
    """FinalClusterID from the canonical Transcripts xlsx, aligned to the
    embedding-cache row order via the same to_numeric/dropna/sort transform
    extract_embeddings_ctx200.py applies before writing the cache, then
    hard-checked against the cache's row count.

    Some patients' "New" filename is a stale/broken symlink (target since
    deleted); extract_control_features.py already tolerates a "Newest"
    suffix variant via regex, so check both filenames here too."""
    suffix = "Newest" if patient_ID == "PTYFA_task25" else "New"
    path = os.path.join(
        TRANSCRIPT_ROOT,
        f"{patient_ID}_filtered_used_rows_withNP_withClusterID{suffix}.xlsx")
    if not os.path.exists(path):
        alt_path = os.path.join(
            TRANSCRIPT_ROOT, f"{patient_ID}_filtered_used_rows_withNP_withClusterIDNewest.xlsx")
        if os.path.exists(alt_path):
            path = alt_path
        else:
            print(f"  {patient_ID}: no transcript/cluster file at {path} (or Newest variant)", flush=True)
            return None
    df = pd.read_excel(path)
    if "FinalClusterID" not in df.columns or "onset" not in df.columns:
        print(f"  {patient_ID}: missing FinalClusterID/onset column in {path}", flush=True)
        return None
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df = df.dropna(subset=["onset"]).sort_values("onset").reset_index(drop=True)
    if len(df) != n_words_expected:
        print(f"  [WARN] {patient_ID}: cluster-ID rows ({len(df)}) != "
              f"embedding-cache rows ({n_words_expected}) -- skipping cluster IDs", flush=True)
        return None
    return df["FinalClusterID"].values


# ── per-patient/region data builder ─────────────────────────────────────────────

def build_patient_region_data(cfg, region, layer, n_components):
    patient_ID = cfg["patient_ID"]
    patient = cfg["patient"]

    npy_path = os.path.join(EMBED_DIR, f"{patient_ID}_{MODEL_TAG}{CONTEXT_TAG}_word_emb_layers.npy")
    spike_dir = find_spike_dir(patient)
    if not os.path.exists(npy_path) or spike_dir is None:
        print(f"  {patient_ID}: missing embeddings or spike dir -- skip", flush=True)
        return None

    try:
        spk_assignment, mask_self, mask_other, dir_membership = load_speaker_assignment(spike_dir)
    except FileNotFoundError as e:
        print(f"  {patient_ID}: {e} -- skip", flush=True)
        return None

    embedding_cache = np.load(npy_path, mmap_mode="r")
    # Most caches contain every hidden layer: (layer, word, feature). Some
    # purpose-built caches contain only the selected layer: (word, feature).
    X_layer_raw = (
        embedding_cache[layer] if embedding_cache.ndim == 3 else embedding_cache
    ).astype(np.float32)

    cluster_ids_full = load_cluster_ids(patient_ID, X_layer_raw.shape[0])
    if cluster_ids_full is None:
        return None

    cond_data = {}
    for cond, mask in [("self", mask_self), ("other", mask_other)]:
        Y_mat = load_condition_ordered(spike_dir, cond, region, spk_assignment, dir_membership)
        if Y_mat is None or Y_mat.ndim < 2:
            cond_data[cond] = None
            continue
        if Y_mat.shape[0] != int(mask.sum()):
            print(f"  {region}/{cond}: row mismatch -- skip", flush=True)
            cond_data[cond] = None
            continue
        valid = ~np.isnan(Y_mat).any(axis=1)
        cond_data[cond] = (Y_mat[valid].astype(np.float32), mask, valid)

    if cond_data.get("self") is None or cond_data.get("other") is None:
        print(f"  {patient_ID}/{region}: missing self or other data -- skip", flush=True)
        return None

    Y_self_raw, mask_s, valid_s = cond_data["self"]
    Y_other_raw, mask_o, valid_o = cond_data["other"]

    spike_ok = (Y_self_raw.sum(0) >= MIN_SPIKES) & (Y_other_raw.sum(0) >= MIN_SPIKES)
    if spike_ok.sum() == 0:
        print(f"  {patient_ID}/{region}: 0 neurons pass spike filter -- skip", flush=True)
        return None

    # Joint per-patient PCA on the full word set, matching semantic_glm.py's
    # --reliability branch (PCA fit once, self/other split out afterward).
    X_layer_pca = PCA(n_components=n_components).fit_transform(
        np.asarray(X_layer_raw, dtype=np.float64))

    X_self = StandardScaler().fit_transform(X_layer_pca[mask_s][valid_s])
    X_other = StandardScaler().fit_transform(X_layer_pca[mask_o][valid_o])
    Y_self = Y_self_raw[:, spike_ok].astype(np.float64)
    Y_other = Y_other_raw[:, spike_ok].astype(np.float64)

    cluster_self = cluster_ids_full[mask_s][valid_s]
    cluster_other = cluster_ids_full[mask_o][valid_o]

    print(f"  {patient_ID}/{region}: self={X_self.shape} other={X_other.shape} "
          f"neurons={int(spike_ok.sum())}", flush=True)

    return dict(
        X_self=X_self, X_other=X_other, Y_self=Y_self, Y_other=Y_other,
        metadata_self=pd.DataFrame({"ClusterID": cluster_self}),
        metadata_other=pd.DataFrame({"ClusterID": cluster_other}),
        mask_self=mask_s, valid_self=valid_s,
        mask_other=mask_o, valid_other=valid_o,
    )


# ── beta-fit adapter: the actual "swap in the updated GLM" step ────────────────

def make_beta_fit_adapter(alphas):
    def _adapter(X, Y, patient_id, neuron_idx, region_name, n_semantic_dims,
                 results_dir, fast_beta_only=True, save_results=False, offset=None):
        # offset is accepted but ignored -- sklearn's PoissonRegressor has no
        # offset/exposure parameter. Use make_beta_fit_adapter_gpu (in
        # cluster_glm_worddur_offset.py) for offset-aware fits.
        y = np.asarray(Y)[:, neuron_idx]
        alpha = select_alpha_cv(X, y, alphas)
        coef = fit_poisson_ridge_beta_sklearn(X, y, alpha)
        return pd.DataFrame([coef], columns=[f"beta_{i}" for i in range(len(coef))])
    return _adapter


def main():
    args = _args()
    run_poisson_ridge = make_beta_fit_adapter(ALPHAS)

    cross_cluster_cap = None
    if args.cross_cluster_cap is not None and not args.no_balance:
        cross_cluster_cap = (args.cross_cluster_cap if args.cross_cluster_cap == "min_kept"
                              else int(args.cross_cluster_cap))

    if args.no_balance:
        results_root = RESULTS_ROOT + "_nobalance"
    elif cross_cluster_cap is not None:
        tag = "minkept" if cross_cluster_cap == "min_kept" else str(cross_cluster_cap)
        results_root = RESULTS_ROOT + f"_cap{tag}"
    else:
        results_root = RESULTS_ROOT

    patients = [p for p in PATIENTS if args.patient is None or p["patient_ID"] == args.patient]

    for cfg in patients:
        patient_ID = cfg["patient_ID"]
        regions = [r for r in cfg["region_ranges"] if args.region is None or r == args.region]

        for region in regions:
            print(f"\n{'='*60}\n  {patient_ID} / {region}", flush=True)

            data = build_patient_region_data(cfg, region, args.layer, args.n_components)
            if data is None:
                continue

            # Diagnostic only -- no longer pre-balanced via minimal_balancing
            # (that discarded trials globally to fix the cluster x condition
            # chi-square test; run_clusterwise_cosine_distance_bootstrap below
            # balances self/other per-cluster on its own, without needing the
            # raw input pre-trimmed, so skipping it keeps far more data).
            report_cluster_balance(
                data["metadata_self"], data["metadata_other"], patient_ID, region)

            # Default (process-pool/loky) backend: run_poisson_ridge and the
            # other functions it calls are now plain top-level functions in
            # this script (not aliased from a dynamically sys.modules-stuffed
            # fake module), so cloudpickle can ship them to worker processes
            # by value -- real multi-core parallelism instead of GIL-bound
            # threads, with each worker's own BLAS calls kept single-threaded
            # (env vars above) so n_jobs workers don't oversubscribe the box.
            run_clusterwise_cosine_distance_bootstrap(
                X_self=data["X_self"], X_other=data["X_other"],
                Y_self=data["Y_self"], Y_other=data["Y_other"],
                metadata_self=data["metadata_self"], metadata_other=data["metadata_other"],
                cluster_column="ClusterID",
                region_name=region, patient_id=patient_ID,
                n_components=args.n_components,
                results_root=results_root,
                run_poisson_ridge=run_poisson_ridge,
                min_trials_per_condition=10,
                balance_self_other=not args.no_balance,
                cross_cluster_cap=cross_cluster_cap,
                adaptive_n_components=True,
                cv_splits_for_pc_rule=5,
                min_n_comp=2,
                n_bootstrap=N_BOOTSTRAP,
                n_halfsplit_repeats=N_HALFSPLIT_REPEATS,
                compute_half_splits=True,
                n_jobs=args.n_jobs,
                print_trial_counts=True,
            )

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
