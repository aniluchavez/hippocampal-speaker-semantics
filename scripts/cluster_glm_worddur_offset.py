#!/usr/bin/env python3
"""
cluster_glm_worddur_offset.py

Quick check of whether word duration confounds the clusterwise self-vs-other
cosine-distance result. Two changes relative to cluster_glm_legacy_replication.py:

  1. Spike counting switches from the fixed window
     (tshift-150_tlen500_oshift+200_olen500) to the worddur window -- same
     trial count/order on disk (confirmed: (652, 77) for both, for
     PTYEU_task147/hippocampus), just a duration-scaled window per word
     instead of a fixed-length one.
  2. Word duration enters the Poisson fit as a log-exposure offset
     (log(word_dur_ms), matching semantic_glm.py's WORDDUR_MODE formula
     exactly), via a GPU L-BFGS fitter -- sklearn's PoissonRegressor (used
     everywhere else this session) has no offset/exposure parameter at all,
     so this can't be done with the existing fitter.

Uses the legacy (single-draw) balancing pipeline, not the bootstrap one --
this is a quick check of whether duration matters before deciding whether
it's worth threading the offset through the bootstrap resampling logic too.

n_jobs is forced to 1: CUDA contexts aren't fork-safe, and joblib's default
process-pool backend forks. For a single-patient/region smoke test this is
fine; a real batch run would need a 'spawn'-based or threading backend if
parallelism is wanted.

Usage:
    cd /scratch/aniluchavez/hippocampal-speaker-semantics
    python3 -u scripts/cluster_glm_worddur_offset.py [--patient PID] [--region NAME]
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.special import gammaln

sys.path.insert(0, os.path.dirname(__file__))
import cluster_glm_reliability as glm  # noqa: E402

RESULTS_ROOT = "/scratch/aniluchavez/ConvoDATAS/SemanticGLM/clusterwise_worddur_offset"
N_COMPONENTS = 10  # matches cluster_glm_legacy_replication.py's choice -- this
                    # is about isolating the duration-offset effect, not the
                    # PC-count effect already investigated separately

_cluster_mod = glm._cluster_mod
report_cluster_balance = _cluster_mod.report_cluster_balance
minimal_balancing = _cluster_mod.minimal_balancing
run_clusterwise_cosine_distance = _cluster_mod.run_clusterwise_cosine_distance

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ETA_CLIP = 20.0
LBFGS_ITER = 200
LBFGS_TOL = 1e-6
LBFGS_HISTORY = 10


# ── worddur-mode data loading (copied/adapted, not imported -- load_condition_ordered
#    in cluster_glm_reliability.py resolves load_spike_matrix via its own module
#    globals, so importing it would still hit the fixed-window spike files) ────

def find_spike_dir_worddur(patient):
    d = os.path.join(glm.SPIKE_ROOT, f"output_{patient}_english_only_worddur")
    return d if os.path.isdir(d) else None


def load_spike_matrix_worddur(spike_dir, speaker, region):
    spk_dir = os.path.join(spike_dir, speaker)
    if not os.path.isdir(spk_dir):
        return None
    cands = [f for f in os.listdir(spk_dir)
             if f.lower().startswith(region.lower()) and f.endswith("_worddur_spike_counts.npy")]
    return np.load(os.path.join(spk_dir, cands[0])) if cands else None


def load_condition_ordered_worddur(spike_dir, cond, region, spk_assignment, dir_membership):
    if cond == "self":
        return load_spike_matrix_worddur(spike_dir, "Speaker1", region)
    other_spks = sorted([c for c in dir_membership if c != "Speaker1"])
    mats = {s: load_spike_matrix_worddur(spike_dir, s, region) for s in other_spks}
    dir_pos = {s: 0 for s in other_spks}
    rows = []
    n = len(spk_assignment)
    for i in range(n):
        for s in other_spks:
            if dir_membership[s][i]:
                spk = spk_assignment[i]
                if spk == s:
                    rows.append(mats[s][dir_pos[s]])
                dir_pos[s] += 1
    return np.vstack(rows) if rows else None


def load_word_dur_ms(patient_ID, spike_dir, n_words_expected):
    """word_dur (ms), aligned to the same row order as the embedding cache via
    the identical dropna(onset)/sort(onset) transform load_cluster_ids uses."""
    cands = [f for f in os.listdir(spike_dir) if f.endswith("_with_regress_dur.xlsx")]
    if not cands:
        return None
    df = pd.read_excel(os.path.join(spike_dir, cands[0]))
    if "word_dur" not in df.columns or "onset" not in df.columns:
        return None
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df = df.dropna(subset=["onset"]).sort_values("onset").reset_index(drop=True)
    if len(df) != n_words_expected:
        print(f"  [WARN] {patient_ID}: word_dur rows ({len(df)}) != "
              f"embedding-cache rows ({n_words_expected}) -- skipping duration", flush=True)
        return None
    return df["word_dur"].values.astype(np.float64)


def build_patient_region_data_worddur(cfg, region, layer, n_components):
    """Same as glm.build_patient_region_data, but pointed at the worddur spike
    directory/files, and additionally returns offset_self/offset_other =
    log(word_dur_ms) (NaN/non-positive durations filled with 500ms, matching
    semantic_glm.py's WORDDUR_MODE default-fill exactly)."""
    patient_ID = cfg["patient_ID"]
    patient = cfg["patient"]

    npy_path = os.path.join(glm.EMBED_DIR, f"{patient_ID}_{glm.MODEL_TAG}{glm.CONTEXT_TAG}_word_emb_layers.npy")
    spike_dir = find_spike_dir_worddur(patient)
    if not os.path.exists(npy_path) or spike_dir is None:
        print(f"  {patient_ID}: missing embeddings or worddur spike dir -- skip", flush=True)
        return None

    try:
        spk_assignment, mask_self, mask_other, dir_membership = glm.load_speaker_assignment(spike_dir)
    except FileNotFoundError as e:
        print(f"  {patient_ID}: {e} -- skip", flush=True)
        return None

    X_layer_raw = np.load(npy_path, mmap_mode="r")[layer].astype(np.float32)

    cluster_ids_full = glm.load_cluster_ids(patient_ID, X_layer_raw.shape[0])
    if cluster_ids_full is None:
        return None

    dur_full = load_word_dur_ms(patient_ID, spike_dir, X_layer_raw.shape[0])
    if dur_full is None:
        return None
    dur_full = np.where(np.isnan(dur_full) | (dur_full <= 0), 500.0, dur_full)

    cond_data = {}
    for cond, mask in [("self", mask_self), ("other", mask_other)]:
        Y_mat = load_condition_ordered_worddur(spike_dir, cond, region, spk_assignment, dir_membership)
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

    spike_ok = (Y_self_raw.sum(0) >= glm.MIN_SPIKES) & (Y_other_raw.sum(0) >= glm.MIN_SPIKES)
    if spike_ok.sum() == 0:
        print(f"  {patient_ID}/{region}: 0 neurons pass spike filter -- skip", flush=True)
        return None

    X_layer_pca = PCA(n_components=n_components).fit_transform(
        np.asarray(X_layer_raw, dtype=np.float64))

    X_self = StandardScaler().fit_transform(X_layer_pca[mask_s][valid_s])
    X_other = StandardScaler().fit_transform(X_layer_pca[mask_o][valid_o])
    Y_self = Y_self_raw[:, spike_ok].astype(np.float64)
    Y_other = Y_other_raw[:, spike_ok].astype(np.float64)

    cluster_self = cluster_ids_full[mask_s][valid_s]
    cluster_other = cluster_ids_full[mask_o][valid_o]

    offset_self = np.log(dur_full[mask_s][valid_s])
    offset_other = np.log(dur_full[mask_o][valid_o])

    print(f"  {patient_ID}/{region}: self={X_self.shape} other={X_other.shape} "
          f"neurons={int(spike_ok.sum())} (worddur)", flush=True)

    return dict(
        X_self=X_self, X_other=X_other, Y_self=Y_self, Y_other=Y_other,
        metadata_self=pd.DataFrame({"ClusterID": cluster_self}),
        metadata_other=pd.DataFrame({"ClusterID": cluster_other}),
        offset_self=offset_self, offset_other=offset_other,
    )


# ── offset-aware GPU Poisson-ridge fitter ───────────────────────────────────────
# Copied (not imported) from semantic_glm.py's gpu_fit_per_neuron, simplified to
# a single neuron / single alpha / single offset call -- semantic_glm.py runs
# its full analysis loop at import time so can't be imported directly, same
# reason fit_poisson_ridge_beta_sklearn was copied verbatim into
# cluster_glm_reliability.py. COLLAB_LOSS is hardcoded True here (matching
# semantic_glm.py's COLLAB_LOSS = WORDDUR_MODE, since this fitter is only ever
# used in worddur mode).

def poisson_ll_numpy(y_true, mu_pred):
    y_true = np.asarray(y_true, dtype=float)
    mu_pred = np.clip(np.asarray(mu_pred, dtype=float), 1e-10, None)
    return float(np.sum(y_true * np.log(mu_pred) - mu_pred - gammaln(y_true + 1)))


def _lbfgs_poisson_ridge_fit(X, y, alpha, offset=None, max_iter=LBFGS_ITER):
    Xt = torch.tensor(np.asarray(X, dtype=np.float32), device=DEVICE)
    yt = torch.tensor(np.asarray(y, dtype=np.float32), device=DEVICE)
    n = Xt.shape[0]
    off = (torch.tensor(np.asarray(offset, dtype=np.float32), device=DEVICE)
           if offset is not None else None)

    Xi = torch.cat([torch.ones(n, 1, device=DEVICE), Xt], dim=-1)
    beta = torch.zeros(Xi.shape[1], device=DEVICE, requires_grad=True)
    opt = torch.optim.LBFGS([beta], max_iter=max_iter, line_search_fn="strong_wolfe",
                             tolerance_grad=LBFGS_TOL, tolerance_change=LBFGS_TOL,
                             history_size=LBFGS_HISTORY)

    def closure():
        opt.zero_grad()
        eta = Xi @ beta
        if off is not None:
            eta = eta + off
        eta = torch.clamp(eta, -ETA_CLIP, ETA_CLIP)
        loss = (torch.exp(eta) - yt * eta).sum() + alpha * beta[1:].pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    b = beta.detach().cpu().numpy()
    return b[1:], float(b[0])  # coef (p,), intercept


def fit_poisson_ridge_beta_gpu(X, y, alpha, offset=None):
    coef, _ = _lbfgs_poisson_ridge_fit(X, y, float(alpha), offset=offset)
    return coef


def select_alpha_cv_gpu(X, y, alphas, offset=None, *, n_splits=5, random_state=0, standardize=True):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = X.shape[0]
    if n < max(10, 2 * n_splits):
        return float(alphas[0])
    off_full = np.asarray(offset, dtype=float) if offset is not None else None

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits_list = list(kf.split(X, y))

    best_alpha, best_score = float(alphas[0]), -np.inf
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
            off_tr = off_full[tr] if off_full is not None else None
            off_va = off_full[va] if off_full is not None else None
            coef, intercept = _lbfgs_poisson_ridge_fit(Xtr_s, ytr, float(a), offset=off_tr)
            eta_va = Xva_s @ coef + intercept
            if off_va is not None:
                eta_va = eta_va + off_va
            mu_va = np.exp(np.clip(eta_va, -ETA_CLIP, ETA_CLIP))
            ll_list.append(poisson_ll_numpy(yva, mu_va))
        score = float(np.mean(ll_list)) if ll_list else -np.inf
        if score > best_score:
            best_score, best_alpha = score, float(a)
    return best_alpha


def make_beta_fit_adapter_gpu(alphas):
    def _adapter(X, Y, patient_id, neuron_idx, region_name, n_semantic_dims,
                 results_dir, fast_beta_only=True, save_results=False, offset=None):
        y = np.asarray(Y)[:, neuron_idx]
        alpha = select_alpha_cv_gpu(X, y, alphas, offset=offset)
        coef = fit_poisson_ridge_beta_gpu(X, y, alpha, offset=offset)
        return pd.DataFrame([coef], columns=[f"beta_{i}" for i in range(len(coef))])
    return _adapter


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--patient", type=str, default=None)
    p.add_argument("--region", type=str, default=None)
    p.add_argument("--exclude-clusters", type=int, nargs="*", default=[])
    p.add_argument("--hard-balance", action="store_true")
    p.add_argument("--results-root", type=str, default=RESULTS_ROOT)
    return p.parse_args()


def main():
    args = _args()
    run_poisson_ridge = make_beta_fit_adapter_gpu(glm.ALPHAS)

    patients = [p for p in glm.PATIENTS if args.patient is None or p["patient_ID"] == args.patient]

    for cfg in patients:
        patient_ID = cfg["patient_ID"]
        regions = [r for r in cfg["region_ranges"] if args.region is None or r == args.region]

        for region in regions:
            print(f"\n{'='*60}\n  {patient_ID} / {region}", flush=True)

            data = build_patient_region_data_worddur(cfg, region, glm.LAYER, N_COMPONENTS)
            if data is None:
                continue

            report_cluster_balance(
                data["metadata_self"], data["metadata_other"], patient_ID, region)

            # minimal_balancing drops rows and reset_index(drop=True)s the
            # result, so its output index no longer maps to row positions in
            # data["X_self"]/data["offset_self"] whenever it actually
            # downsamples a cluster. Track original row positions through the
            # call so X/Y/offset can all be re-sliced to match (see the same
            # fix in cluster_glm_legacy_replication.py).
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
            offset_self_bal = data["offset_self"][orig_ix_self]
            X_other_bal = data["X_other"][orig_ix_other]
            Y_other_bal = data["Y_other"][orig_ix_other]
            offset_other_bal = data["offset_other"][orig_ix_other]
            metadata_self_bal = metadata_self_bal[["ClusterID"]].reset_index(drop=True)
            metadata_other_bal = metadata_other_bal[["ClusterID"]].reset_index(drop=True)

            if args.exclude_clusters:
                keep_s = ~metadata_self_bal["ClusterID"].isin(args.exclude_clusters).to_numpy()
                keep_o = ~metadata_other_bal["ClusterID"].isin(args.exclude_clusters).to_numpy()
                X_self_bal, Y_self_bal = X_self_bal[keep_s], Y_self_bal[keep_s]
                offset_self_bal = offset_self_bal[keep_s]
                metadata_self_bal = metadata_self_bal.loc[keep_s].reset_index(drop=True)
                X_other_bal, Y_other_bal = X_other_bal[keep_o], Y_other_bal[keep_o]
                offset_other_bal = offset_other_bal[keep_o]
                metadata_other_bal = metadata_other_bal.loc[keep_o].reset_index(drop=True)
                print(f"  Excluded clusters: {args.exclude_clusters}", flush=True)

            run_clusterwise_cosine_distance(
                X_self=X_self_bal, X_other=X_other_bal,
                Y_self=Y_self_bal, Y_other=Y_other_bal,
                metadata_self=metadata_self_bal, metadata_other=metadata_other_bal,
                cluster_column="ClusterID",
                region_name=region, patient_id=patient_ID,
                n_components=N_COMPONENTS,
                results_root=args.results_root,
                run_poisson_ridge=run_poisson_ridge,
                offset_self=offset_self_bal, offset_other=offset_other_bal,
                balance_function_words=True,
                compute_half_splits=True,
                n_jobs=1,
                cap_large_clusters=False,
                min_trials_per_condition=10,
                print_trial_counts=True,
                max_size_ratio=2,
                cap_reference="median",
                balance_all_clusters=True,
                soft_balance=not args.hard_balance,
            )

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
