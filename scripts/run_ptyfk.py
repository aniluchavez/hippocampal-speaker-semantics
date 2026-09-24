#!/usr/bin/env python3
"""
Runs the full pipeline (regression + reliability) for PTYFK_task40 only.
Speaker labels in the embedding CSV were corrected (SPK1<->SPK2 swapped)
before running this script.

Usage:
    cd /scratch/aniluchavez/hippocampal-speaker-semantics
    nohup python3 -u scripts/run_ptyfk.py > logs/ptyfk_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    echo "PID: $!"
"""

import os, sys, pickle, warnings, importlib.util
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from scipy.special import gammaln
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import KFold
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT             = "/scratch/aniluchavez/hippocampal-speaker-semantics"
EMBED_CACHE_DIR          = "/scratch/aniluchavez/ConvoDATAS/EmbedCache"
BERT_EMBED_DIR           = "/scratch/aniluchavez/ConvoDATAS/BERTEmbeds"
SPIKE_WINDOW_OUTPUT_ROOT = "/scratch/aniluchavez/ConvoDATAS/SpikeWindows"

RESULTS_ROOT = "/projects/bhayden/anilu/Language_docs/RegressionRESULTSLLAMA31_L18/allwords"
if RESULTS_ROOT.startswith("/projects") and not os.path.isdir("/projects"):
    RESULTS_ROOT = "/scratch/aniluchavez/RegressionRESULTSLLAMA31_L18/allwords"

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── config ─────────────────────────────────────────────────────────────────────
MODEL_TAG       = "llama-3.1-8b"
LLAMA_LAYER     = 18
N_COMPONENTS    = 30
TARGET_SPEAKER  = "SPK1"
N_ITERATIONS    = 10
N_X_SHUFFLE_NULLS = 20
N_JOBS          = 10
N_JOBS_RELIABILITY  = 10
N_NULLS_RELIABILITY = 100
ALPHAS          = np.logspace(-3, 2, 10)
SPEAKERS        = [f"Speaker{i}" for i in range(1, 13)]

SPIKE_SAMPLE_RATE = 1000
SPIKE_VALUE_MODE  = "counts"
SPIKE_WINDOW_PRESETS = {
    "self":  {"speaker": "Speaker1", "start_offset_ms": -150, "window_length_ms": 500},
    "other": {"start_offset_ms": 200, "window_length_ms": 500},
}

PATIENT = {
    "patient_ID": "PTYFK_task40",
    "patient":    "ptYFK_task40",
    "region_ranges": {"hippocampus": [(1,16),(25,40)], "ACC": [(49,56)]},
}

# ── reliability module ─────────────────────────────────────────────────────────
def load_reliability_module():
    path = os.path.join(PROJECT_ROOT, "neural_encoding", "reliability.py")
    name = "nn_reliability"
    if name in sys.modules:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# ── embedding helpers ──────────────────────────────────────────────────────────
def load_and_reduce_llama_embeddings(patient_ID):
    bert_csv = os.path.join(BERT_EMBED_DIR, f"{patient_ID}_words_english_only",
                            f"{patient_ID}_aligned_embeddings_withNP.csv")
    df_meta = pd.read_csv(bert_csv)[["Word", "Speaker", "RowIndex"]].reset_index(drop=True)
    npy_path = os.path.join(EMBED_CACHE_DIR, f"{patient_ID}_{MODEL_TAG}_word_emb_layers.npy")
    arr = np.load(npy_path)
    emb = arr[LLAMA_LAYER].astype(np.float32)
    if emb.shape[0] != len(df_meta):
        raise ValueError(f"embedding rows ({emb.shape[0]}) != metadata rows ({len(df_meta)})")
    pcs = PCA(n_components=N_COMPONENTS).fit_transform(emb)
    pcs_df = pd.DataFrame(pcs, columns=[f"PC{i+1}" for i in range(N_COMPONENTS)])
    print(f"  Llama layer {LLAMA_LAYER}, emb {emb.shape} → PCA {pcs.shape}", flush=True)
    return df_meta, pcs_df

def build_feature_matrix_core(pcs_df, durations):
    pcs = pcs_df.values
    dur = durations.values.reshape(-1, 1)
    X = np.hstack([pcs, dur, pcs * dur])
    return StandardScaler().fit_transform(X)

def get_self_and_other_features(patient_ID, duration_file):
    df_meta, pcs_df = load_and_reduce_llama_embeddings(patient_ID)
    dur_df = pd.read_excel(duration_file) if duration_file.endswith(".xlsx") else pd.read_csv(duration_file)
    df_meta["regress_dur"] = dur_df["regress_dur"].values
    mask_self  = df_meta["Speaker"] == TARGET_SPEAKER
    mask_other = df_meta["Speaker"] != TARGET_SPEAKER
    X_self  = build_feature_matrix_core(pcs_df[mask_self].reset_index(drop=True),
                                        df_meta.loc[mask_self,  "regress_dur"].reset_index(drop=True))
    X_other = build_feature_matrix_core(pcs_df[mask_other].reset_index(drop=True),
                                        df_meta.loc[mask_other, "regress_dur"].reset_index(drop=True))
    print(f"  X_self: {X_self.shape}  X_other: {X_other.shape}", flush=True)
    return X_self, X_other, df_meta

# ── spike helpers ──────────────────────────────────────────────────────────────
def get_spike_window_dir(patient):
    sc = SPIKE_WINDOW_PRESETS["self"]
    oc = SPIKE_WINDOW_PRESETS["other"]
    tag = (f"tshift{sc['start_offset_ms']:+d}_tlen{sc['window_length_ms']}"
           f"_oshift{oc['start_offset_ms']:+d}_olen{oc['window_length_ms']}")
    return os.path.join(SPIKE_WINDOW_OUTPUT_ROOT, f"output_{patient}_english_only_{tag}")

def get_duration_file(patient):
    return os.path.join(get_spike_window_dir(patient), f"{patient}_with_regress_dur.xlsx")

def load_spike_data(regions, base_dir):
    spike_data = {}
    for speaker in SPEAKERS:
        try:
            folders = os.listdir(base_dir)
        except FileNotFoundError:
            continue
        folder = next((f for f in folders if f.strip().lower() == speaker.lower()), None)
        if not folder:
            continue
        spike_data[speaker] = {}
        spk_path = os.path.join(base_dir, folder)
        for region in regions:
            cands = [f for f in os.listdir(spk_path)
                     if f.lower().startswith(region.lower()) and f.endswith(".npy")]
            if cands:
                spike_data[speaker][region] = np.load(os.path.join(spk_path, cands[0]))
                print(f"  Loaded {speaker}-{region}: {spike_data[speaker][region].shape}", flush=True)
    return spike_data

def build_Y_matrices(df_meta, spike_data, regions):
    tag_map = {f"SPK{spk.replace('Speaker','').strip()}": spk.strip() for spk in spike_data}
    region_Ys = {}
    for region in regions:
        Y_self, Y_other = [], []
        counters = {spk: 0 for spk in spike_data}
        for _, row in df_meta.iterrows():
            tag = row["Speaker"]
            spk = tag_map.get(tag)
            if spk is None or region not in spike_data.get(spk, {}):
                continue
            idx = counters[spk]
            if idx >= spike_data[spk][region].shape[0]:
                continue
            rd = spike_data[spk][region][idx, :]
            (Y_self if tag == TARGET_SPEAKER else Y_other).append(rd)
            counters[spk] += 1
        print(f"  {region}: Y_self={len(Y_self)}  Y_other={len(Y_other)}", flush=True)
        region_Ys[region] = {
            "self":  np.vstack(Y_self)  if Y_self  else np.empty((0,)),
            "other": np.vstack(Y_other) if Y_other else np.empty((0,)),
        }
    return region_Ys

def clean_XY(X, Y, label=""):
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"{label}: X rows ({X.shape[0]}) != Y rows ({Y.shape[0]})")
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1))
    return X[mask], Y[mask], mask

# ── regression helpers (module-level so joblib can pickle them) ────────────────
def _safe_int(x):
    return int(x[0]) if isinstance(x, (list, tuple, np.ndarray)) else int(x)

def _poisson_ll(y_true, mu):
    mu = np.clip(mu, 1e-10, None)
    return float(np.sum(y_true * np.log(mu) - mu - gammaln(y_true + 1)))

def _effective_dof(Xm, mu, alpha):
    mu = np.clip(mu, 1e-8, None)
    Xw = Xm * np.sqrt(mu)[:, None]
    _, s, _ = np.linalg.svd(Xw, full_matrices=False)
    s2 = s ** 2
    return float(np.sum(s2 / (s2 + alpha)))

def _choose_alpha_cv(X_train, y_train, alphas, seed):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    best_alpha, best_ll = alphas[0], -np.inf
    for alpha in alphas:
        scores = []
        for tr, va in kf.split(X_train):
            m = PoissonRegressor(alpha=alpha, max_iter=1000).fit(X_train[tr], y_train[tr])
            scores.append(_poisson_ll(y_train[va], m.predict(X_train[va])))
        avg = np.mean(scores) if scores else -np.inf
        if avg > best_ll:
            best_ll, best_alpha = avg, alpha
    return best_alpha

def _make_split(n, n_features, base_ts, iter_idx):
    ts = base_ts
    for _ in range(4):
        sp = int((1 - ts) * n)
        offset = (iter_idx * int(n * ts)) % n
        idx_range = np.roll(np.arange(n), -offset)
        train_idx, test_idx = idx_range[:sp], idx_range[sp:]
        if len(test_idx) > (n_features + 1):
            return train_idx, test_idx, ts
        ts = min(0.6, ts + 0.1)
    return train_idx, test_idx, ts


# ── regression ─────────────────────────────────────────────────────────────────
def run_poisson_ridge(X, Y, patient_id, neuron_idx, region_name,
                      n_semantic_dims, n_iterations=N_ITERATIONS,
                      test_size=0.3, n_shuf_x=N_X_SHUFFLE_NULLS, alpha_grid=None):
    if alpha_grid is None:
        alpha_grid = ALPHAS
    neuron_idx = _safe_int(neuron_idx)
    n_words, n_neurons = Y.shape
    if neuron_idx >= n_neurons:
        raise IndexError(f"neuron_idx {neuron_idx} >= {n_neurons}")
    y = Y[:, neuron_idx].astype(float)
    X = np.asarray(X, dtype=float)
    if np.std(y) == 0 or np.all(y == 0):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    results_per_iter, coef_pval_rows, deviance_records = [], [], []
    coef_accumulator, support_flags = [], set()

    for iter_idx in range(n_iterations):
        train_idx, test_idx, used_ts = _make_split(n_words, X.shape[1], test_size, iter_idx)
        X_train, y_train = X[train_idx], y[train_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]
        best_alpha = _choose_alpha_cv(X_train, y_train, alpha_grid, seed=_safe_int(iter_idx))
        model = PoissonRegressor(alpha=best_alpha, max_iter=1000).fit(X_train, y_train)
        pred_train = model.predict(X_train)
        pred_test  = model.predict(X_test)
        ll_real = _poisson_ll(y_test, pred_test)
        ll_null = _poisson_ll(y_test, np.full_like(y_test, np.mean(y_test)))
        pseudo_r2 = 1 - (ll_real / ll_null)
        r_train, _ = pearsonr(y_train, np.clip(pred_train, 1e-10, None))
        r_test, corr_p = pearsonr(y_test, np.clip(pred_test, 1e-10, None))
        edf = _effective_dof(X_train, np.clip(pred_train, 1e-10, None), best_alpha)
        coef_accumulator.append(model.coef_)
        deviance_vals = 2 * (y_test * np.log((y_test + 1e-10) / np.clip(pred_test, 1e-10, None))
                             - (y_test - pred_test))
        for i, word_idx in enumerate(test_idx):
            deviance_records.append({"neuron": neuron_idx, "word_index": _safe_int(word_idx),
                                     "iteration": _safe_int(iter_idx), "deviance": float(deviance_vals[i])})
        ll_xshuf_list, edf_shuf_list = [], []
        for _ in range(n_shuf_x):
            Xs = X_train.copy()
            for c in range(min(n_semantic_dims, Xs.shape[1])):
                np.random.shuffle(Xs[:, c])
            ms = PoissonRegressor(alpha=best_alpha, max_iter=1000).fit(Xs, y_train)
            ll_xshuf_list.append(_poisson_ll(y_test, ms.predict(X_test)))
            edf_shuf_list.append(_effective_dof(Xs, np.clip(ms.predict(Xs), 1e-10, None), best_alpha))
        ll_xshuf = float(np.mean(ll_xshuf_list))
        edf_shuf = float(np.mean(edf_shuf_list))
        try:
            glm = sm.GLM(y_train, sm.add_constant(X_train, has_constant="add"),
                         family=sm.families.Poisson()).fit()
            pvals = glm.pvalues[1:]
            for p_idx, (coef, pval) in enumerate(zip(model.coef_, pvals)):
                coef_pval_rows.append({"neuron": neuron_idx, "predictor_index": _safe_int(p_idx),
                                       "iteration": _safe_int(iter_idx),
                                       "coefficient": float(coef), "p_value": float(pval)})
                if pval < 0.05:
                    support_flags.add(p_idx)
        except Exception:
            pass
        results_per_iter.append({
            "neuron": neuron_idx, "iteration": _safe_int(iter_idx),
            "best_alpha": float(best_alpha), "test_size_used": float(used_ts),
            "ll_real": float(ll_real), "ll_null": float(ll_null), "ll_xshuf": float(ll_xshuf),
            "ll_diff_xshuf": float(ll_real - ll_xshuf),
            "pseudo_r2": float(pseudo_r2), "pseudo_r2_shuf": float(1 - (ll_xshuf / ll_null)),
            "corr_train": float(r_train), "corr_test": float(r_test), "corr_p": float(corr_p),
            "edf": float(edf), "edf_shuf": float(edf_shuf),
            "AIC": float(2 * edf - 2 * ll_real), "AIC_xshuf": float(2 * edf_shuf - 2 * ll_xshuf),
            "p_val_ll_diff_xshuf": float(np.mean(np.array(ll_xshuf_list) >= ll_real))
        })

    df = pd.DataFrame(results_per_iter)
    summary_df = df.groupby("neuron").median(numeric_only=True).reset_index()
    coef_df_all = pd.DataFrame(coef_pval_rows)
    sig_rows = []
    if not coef_df_all.empty:
        for p_idx, grp in coef_df_all.groupby("predictor_index"):
            if grp["p_value"].median() < 0.05:
                sig_rows.append({"neuron": neuron_idx, "predictor_index": _safe_int(p_idx),
                                 "mean_coefficient": float(grp["coefficient"].mean()),
                                 "median_p_value": float(grp["p_value"].median()),
                                 "times_significant": int((grp["p_value"] < 0.05).sum()),
                                 "ridge_support": int(p_idx in support_flags)})
    final_sig_df = pd.DataFrame(sig_rows)
    mean_coef = np.mean(coef_accumulator, axis=0) if coef_accumulator else np.array([])
    ridge_df = pd.DataFrame([{"neuron": neuron_idx, "predictor_index": _safe_int(p_idx),
                               "mean_ridge_coef": float(v), "ridge_support": int(p_idx in support_flags)}
                              for p_idx, v in enumerate(mean_coef)])
    coef_df = (pd.DataFrame([mean_coef], index=[neuron_idx],
                             columns=[f"Predictor_{i}" for i in range(len(mean_coef))])
               if len(mean_coef) > 0 else pd.DataFrame())
    return df, summary_df, final_sig_df, ridge_df, coef_df


def run_all_conditions(X_dict, Y_dict, patient_id, region_list, n_jobs=N_JOBS):
    results_dict = {}
    for region in region_list:
        for condition in ["self", "other"]:
            key = f"{region}_{condition}"
            X = X_dict.get(key)
            Y = Y_dict.get(key)
            if X is None or Y is None:
                print(f"  Skipping {key}: no data", flush=True)
                continue
            print(f"  Running {patient_id} — {key} ({Y.shape[1]} neurons)", flush=True)
            outputs = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
                delayed(run_poisson_ridge)(X, Y, patient_id, nid, key, N_COMPONENTS)
                for nid in range(Y.shape[1])
            )
            all_df, all_summary, all_sig, all_ridge, all_coef = [], [], [], [], []
            for nid, out in enumerate(outputs):
                if not (isinstance(out, tuple) and len(out) == 5):
                    continue
                df, summary_df, sig_df, ridge_df, coef_df = out
                try:
                    coef_df.index = [nid]
                except Exception:
                    pass
                all_df.append(df); all_summary.append(summary_df)
                all_sig.append(sig_df); all_ridge.append(ridge_df); all_coef.append(coef_df)
            results_dict[key] = {
                "df":          pd.concat(all_df,      ignore_index=True) if all_df      else pd.DataFrame(),
                "summary":     pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame(),
                "significant": pd.concat(all_sig,     ignore_index=True) if all_sig     else pd.DataFrame(),
                "ridge":       pd.concat(all_ridge,   ignore_index=True) if all_ridge   else pd.DataFrame(),
                "coef":        pd.concat(all_coef)                        if all_coef    else pd.DataFrame(),
            }
    return results_dict


def compute_self_other_beta_corr(results, region):
    df_self  = results.get(f"{region}_self",  {}).get("coef")
    df_other = results.get(f"{region}_other", {}).get("coef")
    if df_self is None or df_other is None or df_self.empty or df_other.empty:
        return pd.DataFrame()
    common = df_self.index.intersection(df_other.index)
    df_self  = df_self.loc[common].sort_index()
    df_other = df_other.loc[common].sort_index()
    corrs = [pearsonr(df_self.loc[i], df_other.loc[i])[0] for i in df_self.index]
    return pd.DataFrame({"neuron_index": df_self.index, "true_corr": corrs, "region": region.upper()})


# ── main ────────────────────────────────────────────────────────────────────────
def main():
    patient_ID    = PATIENT["patient_ID"]
    patient       = PATIENT["patient"]
    region_ranges = PATIENT["region_ranges"]
    regions       = list(region_ranges.keys())
    patient_dir   = os.path.join(RESULTS_ROOT, patient_ID)
    os.makedirs(patient_dir, exist_ok=True)

    print(f"RESULTS_ROOT: {RESULTS_ROOT}", flush=True)
    print(f"\n{'='*55}", flush=True)
    print(f"Patient: {patient_ID}  |  regions: {regions}", flush=True)
    print(f"{'='*55}", flush=True)

    spike_base_dir = get_spike_window_dir(patient)
    duration_file  = get_duration_file(patient)

    # ── regression ──
    reg_pkl = os.path.join(patient_dir, "ALL_CONDITIONS_RESULTS.pkl")
    if os.path.exists(reg_pkl):
        print(f"  Loading existing regression results: {reg_pkl}", flush=True)
        with open(reg_pkl, "rb") as f:
            results = pickle.load(f)
    else:
        print("  Running regression...", flush=True)
        X_self, X_other, df_meta = get_self_and_other_features(patient_ID, duration_file)
        spike_data = load_spike_data(regions, spike_base_dir)
        region_Ys  = build_Y_matrices(df_meta, spike_data, regions)

        X_dict, Y_dict = {}, {}
        for region in regions:
            for cond in ["self", "other"]:
                key = f"{region}_{cond}"
                Y = region_Ys.get(region, {}).get(cond)
                X = X_self if cond == "self" else X_other
                if Y is None or Y.ndim < 2 or Y.shape[0] == 0:
                    continue
                try:
                    Xc, Yc, _ = clean_XY(X, Y, label=key)
                except ValueError as e:
                    print(f"  {key}: {e}", flush=True)
                    continue
                if Xc.shape[0] > 0:
                    X_dict[key] = Xc
                    Y_dict[key] = Yc
                    print(f"  {key}: X={Xc.shape}  Y={Yc.shape}", flush=True)

        results = run_all_conditions(X_dict, Y_dict, patient_ID, regions)
        with open(reg_pkl, "wb") as f:
            pickle.dump(results, f)
        print(f"  Regression saved: {reg_pkl}", flush=True)

        for region in regions:
            corr_df = compute_self_other_beta_corr(results, region)
            if corr_df.empty:
                continue
            corr_df["patient"] = patient_ID
            corr_df.to_csv(os.path.join(patient_dir, f"{region.upper()}_SELF_OTHER_CORRELATION.csv"), index=False)
            print(f"  Beta corr saved: {region} ({len(corr_df)} neurons)", flush=True)

    # ── reliability ──
    print("\n  Running reliability...", flush=True)

    # rebuild X/Y (needed even if regression was loaded)
    X_self, X_other, df_meta = get_self_and_other_features(patient_ID, duration_file)
    spike_data = load_spike_data(regions, spike_base_dir)
    region_Ys  = build_Y_matrices(df_meta, spike_data, regions)
    X_dict, Y_dict = {}, {}
    for region in regions:
        for cond in ["self", "other"]:
            key = f"{region}_{cond}"
            Y = region_Ys.get(region, {}).get(cond)
            X = X_self if cond == "self" else X_other
            if Y is None or Y.ndim < 2 or Y.shape[0] == 0:
                continue
            try:
                Xc, Yc, _ = clean_XY(X, Y, label=key)
            except ValueError as e:
                print(f"  {key}: {e}", flush=True)
                continue
            if Xc.shape[0] > 0:
                X_dict[key] = Xc
                Y_dict[key] = Yc

    rel_mod = load_reliability_module()
    ReliabilityConfig = rel_mod.ReliabilityConfig
    run_beta_reliability_all_neurons = rel_mod.run_beta_reliability_all_neurons
    cfg_rel = ReliabilityConfig(
        n_null=N_NULLS_RELIABILITY,
        alphas=tuple(float(a) for a in ALPHAS),
        n_jobs=N_JOBS_RELIABILITY,
    )

    for region in regions:
        ks, ko = f"{region}_self", f"{region}_other"
        if ks not in X_dict or ko not in X_dict:
            print(f"  {region}: missing self/other data; skipping reliability", flush=True)
            continue
        rel_path = os.path.join(patient_dir, f"{region.upper()}_RELIABILITY_RESULTS.pkl")
        if os.path.exists(rel_path):
            print(f"  {region}: reliability already exists, skipping", flush=True)
            continue
        print(f"  Running reliability: {region}", flush=True)
        rel = run_beta_reliability_all_neurons(
            X_self=X_dict[ks], X_other=X_dict[ko],
            Y_self=Y_dict[ks], Y_other=Y_dict[ko],
            cfg=cfg_rel, verbose=True,
        )
        with open(rel_path, "wb") as f:
            pickle.dump(rel, f)
        print(f"  Reliability saved: {rel_path}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
