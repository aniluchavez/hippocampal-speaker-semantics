from joblib import parallel_backend
from joblib import Parallel, delayed
import os
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed, parallel_backend
from scipy.stats import spearmanr, wilcoxon
from scipy.special import gammaln
from tqdm import tqdm
from poisson_ridge_torch import (
    train_poisson_model,
    predict_poisson_model,
    select_best_alpha_torch,
    calculate_effective_dof_poisson
)

def run_torch_poisson_with_nulls(X_scaled, y, scaler=None, prepare_features=None,
                                 n_semantic_dims=30, n_perms=100, neuron_idx=0,
                                 alphas=None, random_state=42, verbose=False,
                                 X_base=None,shuffle_only_embeddings=False):
    import torch
    import numpy as np
    from scipy.stats import spearmanr
    from scipy.special import gammaln
    import pandas as pd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not torch.is_tensor(X_scaled):
        X_torch = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    else:
        X_torch = X_scaled.to(device)

    y_torch = torch.tensor(y, dtype=torch.float32, device=device)

    # === Select best alpha
    best_alpha = select_best_alpha_torch(X_torch, y_torch, alphas)

    # === Fit model
    model = train_poisson_model(X_torch, y_torch, alpha=best_alpha, device=device)
    pred = predict_poisson_model(model, X_torch).cpu().numpy()
    pred = np.clip(pred, 1e-10, None)
    ll_real = np.sum(y * np.log(pred) - pred - gammaln(y + 1))
    pearson_corr = np.corrcoef(pred, y)[0, 1]
    spearman_corr, spearman_p = spearmanr(pred, y)

    # === Null models: skip if n_perms == 0
    if n_perms == 0:
        ll_shuf_list = []
        pval_ll = np.nan
    else:
        ll_shuf_list = []
        X_raw_np = X_base if X_base is not None else X_scaled.cpu().numpy()
        for _ in range(n_perms):
            if shuffle_only_embeddings:
        # === Split X_raw into pcs, durations
                pcs = X_raw_np[:, :n_semantic_dims]
                durations = X_raw_np[:, n_semantic_dims].reshape(-1, 1)

                # === Shuffle embeddings (pcs) only
                shuffled_pcs = pcs[np.random.permutation(pcs.shape[0])]

                # === Recompute interaction term
                shuffled_interaction = shuffled_pcs * durations

                # === Rebuild design matrix and rescale
                X_null_raw = np.hstack([shuffled_pcs, durations, shuffled_interaction])
                X_null_scaled = scaler.transform(X_null_raw)
            else:
                # === Shuffle full rows of X
                X_null_scaled = X_scaled[np.random.permutation(X_scaled.shape[0])].cpu().numpy()

            X_shuf_torch = torch.tensor(X_null_scaled, dtype=torch.float32, device=device)

            null_model = train_poisson_model(X_shuf_torch, y_torch, alpha=best_alpha, device=device)
            null_pred = predict_poisson_model(null_model, X_shuf_torch).cpu().numpy()
            null_pred = np.clip(null_pred, 1e-10, None)
            ll_shuf = np.sum(y * np.log(null_pred) - null_pred - gammaln(y + 1))
            ll_shuf_list.append(ll_shuf)

    pval_ll = np.mean(np.array(ll_shuf_list) >= ll_real)

    # === Package results
    summary_df = pd.DataFrame([{
        "neuron": neuron_idx,
        "ll_diff": ll_real - np.mean(ll_shuf_list) if ll_shuf_list else np.nan,
        "ll_shuf_list": ll_shuf_list if ll_shuf_list else np.nan,
        "ll_real":ll_real,
        "pval_xsem": pval_ll,
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr,
        "spearman_p": spearman_p,
        "best_alpha": best_alpha,
        "y_nonzero": int(np.count_nonzero(y)),
        "y_mean": float(np.mean(y)),
        "y_std": float(np.std(y))
    }])

    return summary_df, ll_real, ll_real, ll_shuf_list, spearman_corr, spearman_corr, spearmanr(pred, y)[0]


def run_condition_parallel_v2(X, Y, W, region, condition, patient_id,
                              results_dir=None, alphas=None,
                              n_shuffles=100, X_base=None, scaler=None,
                              prepare_features=None, n_semantic_dims=30,
                              use_parallel=True, save_full_per_neuron=True,
                              n_jobs=None):

    from tqdm import tqdm
    from functools import partial

    n_neurons = Y.shape[1]
    n_jobs = n_jobs or os.cpu_count() - 1

    print(f"🚀 Running {region}-{condition} on {n_neurons} neurons using {n_jobs} parallel jobs")

    def run_one(neuron_idx):
        y = Y[:, neuron_idx]
        if torch.is_tensor(y):
            y = y.cpu().numpy()

        if np.std(y) == 0 or np.all(y == 0):
            print(f"[SKIP] Neuron {neuron_idx}: zero variance or all zeros")
            return pd.DataFrame()

        try:
            df_main, *_ = run_torch_poisson_with_nulls(
                X_scaled=X.cpu().numpy(),
                y=y,
                X_base=X_base,
                scaler=scaler,
                prepare_features=prepare_features,
                n_semantic_dims=n_semantic_dims,
                n_perms=n_shuffles,
                neuron_idx=neuron_idx,
                alphas=alphas,
                random_state=neuron_idx,
                verbose=False
            )
            df_main["neuron"] = neuron_idx
            return df_main
        except Exception as e:
            print(f"[ERROR] Neuron {neuron_idx} failed: {e}")
            return pd.DataFrame()

    if use_parallel:
        results = Parallel(n_jobs=n_jobs)(
            delayed(run_one)(i) for i in range(n_neurons)
        )
    else:
        results = [run_one(i) for i in tqdm(range(n_neurons), desc=f"{region}-{condition}")]

    # Filter out empty results
    results = [r for r in results if not r.empty]

    if not results:
        print(f"[WARN] All neurons failed or were skipped for {region}-{condition}")
        return pd.DataFrame(), [], [], []

    full_df = pd.concat(results, ignore_index=True)

    if results_dir and save_full_per_neuron:
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, f"{region}_{condition}_per_neuron.csv")
        full_df.to_csv(out_path, index=False)
        print(f"✅ Saved per-neuron results to {out_path}")

    return full_df, [], [], []


def run_all_conditions_v2(X_dict, Y_dict, W_dict, patient_id, regions,
                          conditions=["self", "other"],
                          alphas=None, results_dir="./results",
                          rawX_dict=None, scaler_dict=None, featfunc_dict=None,
                          n_shuffles=100, use_parallel=True, n_semantic_dims=30,
                          save_full_per_neuron=True, device=None, return_outputs=True,
                          n_jobs=1):

    all_main, all_xsem, all_xfull, all_yshuf = [], [], [], []
    all_sxsem, all_sxfull, all_syshuf = [], [], [], []
    failed_keys = []

    def run_condition_wrapper(region, condition):
        key = f"{region}_{condition}"
        if key not in X_dict or key not in Y_dict:
            print(f"[SKIP] Missing {key}")
            return key, None

        X, Y = X_dict[key], Y_dict[key]
        if X.shape[0] == 0 or Y.shape[0] == 0:
            print(f"[SKIP] {key}: No data")
            return key, None

        print(f"🏃 Running: {patient_id} — {region}-{condition}")
        m, xs, xf, ys, sx, sxf, sy = run_condition_parallel_v2(
            X, Y, W_dict.get(key), region, condition, patient_id,
            results_dir=results_dir,
            alphas=alphas,
            n_shuffles=n_shuffles,
            use_parallel=use_parallel,
            X_base=rawX_dict.get(key),
            scaler=scaler_dict.get(key),
            prepare_features=featfunc_dict.get(key),
            n_semantic_dims=n_semantic_dims,
            save_full_per_neuron=save_full_per_neuron,
            n_jobs=n_jobs  # ✅ for internal neuron-level parallelism
        )

        for df in [m, xs, xf, ys, sx, sxf, sy]:
            df["region"] = region
            df["condition"] = condition
            df["patient_id"] = patient_id

        return key, (m, xs, xf, ys, sx, sxf, sy)

    # === Run all region-condition combos in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_condition_wrapper)(region, condition)
        for region in regions for condition in conditions
    )

    for key, res in results:
        if res is None:
            failed_keys.append(key)
            continue
        m, xs, xf, ys, sx, sxf, sy = res
        all_main.append(m)
        all_xsem.append(xs)
        all_xfull.append(xf)
        all_yshuf.append(ys)
        all_sxsem.append(sx)
        all_sxfull.append(sxf)
        all_syshuf.append(sy)

    # === Save results
    if save_full_per_neuron and results_dir is not None:
        save_dir = os.path.join(results_dir, patient_id)
        os.makedirs(save_dir, exist_ok=True)
        pd.concat(all_main, ignore_index=True).to_csv(os.path.join(save_dir, "summary_main.csv"), index=False)
        pd.concat(all_xsem, ignore_index=True).to_csv(os.path.join(save_dir, "summary_llh_xsem.csv"), index=False)
        pd.concat(all_xfull, ignore_index=True).to_csv(os.path.join(save_dir, "summary_llh_xfull.csv"), index=False)
        pd.concat(all_yshuf, ignore_index=True).to_csv(os.path.join(save_dir, "summary_llh_yshuf.csv"), index=False)
        pd.concat(all_sxsem, ignore_index=True).to_csv(os.path.join(save_dir, "summary_spearman_xsem.csv"), index=False)
        pd.concat(all_sxfull, ignore_index=True).to_csv(os.path.join(save_dir, "summary_spearman_xfull.csv"), index=False)
        pd.concat(all_syshuf, ignore_index=True).to_csv(os.path.join(save_dir, "summary_spearman_yshuf.csv"), index=False)
        print(f"✅ Saved all outputs to {save_dir}")

    if return_outputs:
        return (
            pd.concat(all_main, ignore_index=True),
            pd.concat(all_xsem, ignore_index=True),
            pd.concat(all_xfull, ignore_index=True),
            pd.concat(all_yshuf, ignore_index=True),
            pd.concat(all_sxsem, ignore_index=True),
            pd.concat(all_sxfull, ignore_index=True),
            pd.concat(all_syshuf, ignore_index=True),
        )
    else:
        return None

    
def run_one_shift(shift, meta_df, spike_data, region_list, target_speaker, patient_id,
                  results_dir, n_components, alphas, n_shuffles, use_parallel,
                  use_noise_embeddings, save_full_per_neuron, build_XY_fn):
    import pandas as pd
    import numpy as np
    from scipy.stats import wilcoxon

    all_rows = []

    if shift < 0:
        X_meta = meta_df.iloc[-shift:].reset_index(drop=True)
        Y_meta = meta_df.iloc[:len(X_meta)].reset_index(drop=True)
    elif shift > 0:
        X_meta = meta_df.iloc[:-shift].reset_index(drop=True)
        Y_meta = meta_df.iloc[shift:].reset_index(drop=True)
    else:
        X_meta = meta_df.copy()
        Y_meta = meta_df.copy()

    if use_noise_embeddings:
        for i in range(len(X_meta)):
            X_meta.at[i, "Parsed_Embedding"] = np.random.randn(n_components).tolist()

    for region in region_list:
        print(f"[Shift {shift}] Building XY for region: {region}")
        data_dict = build_XY_fn(
            X_meta=X_meta,
            Y_meta=Y_meta,
            spike_data=spike_data,
            region=region,
            target_speaker=target_speaker,
            n_components=n_components,
            return_weights=True
        )

        if "self" not in data_dict and "other" not in data_dict:
            print(f"[Shift {shift}] Detected pooled format → wrapping in 'all'")
            data_dict = {"all": data_dict}

        for condition in data_dict:
            sub = data_dict[condition]
            X, Y, W = sub["X_scaled"], sub["Y"], sub["weights"]

            if X.shape[0] == 0 or Y.shape[0] == 0:
                print(f"[Shift {shift}] SKIP: {region}-{condition} has empty data")
                continue

            df_main, *_ = run_condition_parallel_v2(
                X=X, Y=Y, W=W,
                region=region,
                condition=condition,
                patient_id=patient_id,
                results_dir=None,
                alphas=alphas,
                n_shuffles=n_shuffles,
                X_base=sub["X_raw"],
                scaler=sub["scaler"],
                prepare_features=sub["prepare_features"],
                n_semantic_dims=n_components,
                use_parallel=use_parallel,
                save_full_per_neuron=save_full_per_neuron
            )

            if df_main.empty:
                print(f"[Shift {shift}] WARN: empty df_main for {region}-{condition}")
                continue

            df_main["temporal_shift"] = shift
            df_main["region"] = region
            df_main["condition"] = condition
            df_main["patient_id"] = patient_id
            all_rows.append(df_main)

    return all_rows

from joblib import Parallel, delayed

def sweep_temporal_llh(meta_df, spike_data, region_list, target_speaker, patient_id,
                       shift_range=range(-20, 21), n_components=30, alphas=None,
                       build_XY_fn=None, use_parallel=True, n_jobs=4, n_shuffles=0):
    def run_one_shift(shift):
        rows = []

        # Align X and Y
        if shift < 0:
            X_meta = meta_df.iloc[-shift:].reset_index(drop=True)
            Y_meta = meta_df.iloc[:len(X_meta)].reset_index(drop=True)
        elif shift > 0:
            X_meta = meta_df.iloc[:-shift].reset_index(drop=True)
            Y_meta = meta_df.iloc[shift:].reset_index(drop=True)
        else:
            X_meta = meta_df.copy()
            Y_meta = meta_df.copy()

        for region in region_list:
            data_dict = build_XY_fn(
                X_meta=X_meta,
                Y_meta=Y_meta,
                spike_data=spike_data,
                region=region,
                target_speaker=target_speaker,
                n_components=n_components,
                return_weights=False
            )

            if "self" not in data_dict and "other" not in data_dict:
                data_dict = {"all": data_dict}

            for condition in data_dict:
                sub = data_dict[condition]
                X, Y = sub["X_scaled"], sub["Y"]

                if X.shape[0] == 0 or Y.shape[0] == 0:
                    continue

                df_main, *_ = run_condition_parallel_v2(
                    X=X, Y=Y, W=None,
                    region=region,
                    condition=condition,
                    patient_id=patient_id,
                    results_dir=None,
                    alphas=alphas,
                    n_shuffles=n_shuffles,  # <-- ADD THIS
                    X_base=sub["X_raw"],
                    scaler=sub["scaler"],
                    prepare_features=sub["prepare_features"],
                    n_semantic_dims=n_components,
                    use_parallel=use_parallel,
                    save_full_per_neuron=False
                )

                if df_main.empty:
                    continue

                df_main["temporal_shift"] = shift
                df_main["region"] = region
                df_main["condition"] = condition
                df_main["patient_id"] = patient_id
                rows.append(df_main)    
                print(f"[SHIFT={shift}] Region={region}, Condition={condition}, N={X.shape[0]} trials, Y={Y.shape}")

        return rows

    print(f"🚀 Running sweep across {len(shift_range)} shifts with n_jobs={n_jobs}")
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_one_shift)(shift) for shift in shift_range
    )

    return pd.concat([r for shift_rows in results for r in shift_rows], ignore_index=True) if results else pd.DataFrame()

def run_temporal_shift_regressions_continuous(meta_df, spike_data, region_list, target_speaker,
                                              patient_id, results_dir, shift_range=range(-7, 5),
                                              n_components=30, alphas=None, n_shuffles=100,
                                              use_parallel=True, use_noise_embeddings=False,
                                              save_full_per_neuron=True, device=None,
                                              build_XY_fn=None, n_jobs=4):
    import numpy as np
    import os
    import pandas as pd
    from scipy.stats import wilcoxon
    from joblib import Parallel, delayed

    if build_XY_fn is None:
        raise ValueError("You must provide build_XY_fn.")

    def run_one_shift(shift):
        all_rows = []

        if shift < 0:
            X_meta = meta_df.iloc[-shift:].reset_index(drop=True)
            Y_meta = meta_df.iloc[:len(X_meta)].reset_index(drop=True)
        elif shift > 0:
            X_meta = meta_df.iloc[:-shift].reset_index(drop=True)
            Y_meta = meta_df.iloc[shift:].reset_index(drop=True)
        else:
            X_meta = meta_df.copy()
            Y_meta = meta_df.copy()

        if use_noise_embeddings:
            print(f"[Shift {shift}] ⚠️ Injecting noise into embeddings")
            for i in range(len(X_meta)):
                X_meta.at[i, "Parsed_Embedding"] = np.random.randn(n_components).tolist()

        for region in region_list:
            print(f"[Shift {shift}] Building XY for region: {region}")
            data_dict = build_XY_fn(
                X_meta=X_meta,
                Y_meta=Y_meta,
                spike_data=spike_data,
                region=region,
                target_speaker=target_speaker,
                n_components=n_components,
                return_weights=True
            )

            if "self" not in data_dict and "other" not in data_dict:
                print(f"[Shift {shift}] Detected pooled format → wrapping in 'all'")
                data_dict = {"all": data_dict}

            for condition in data_dict:
                sub = data_dict[condition]
                X, Y, W = sub["X_scaled"], sub["Y"], sub["weights"]

                if X.shape[0] == 0 or Y.shape[0] == 0:
                    print(f"[Shift {shift}] SKIP: {region}-{condition} has empty data")
                    continue

                print(f"[DEBUG] shift={shift}, region={region}, condition={condition}, X={X.shape}, Y={Y.shape}")

                df_main, *_ = run_condition_parallel_v2(
                    X=X, Y=Y, W=W,
                    region=region,
                    condition=condition,
                    patient_id=patient_id,
                    results_dir=None,
                    alphas=alphas,
                    n_shuffles=n_shuffles,
                    X_base=sub["X_raw"],
                    scaler=sub["scaler"],
                    prepare_features=sub["prepare_features"],
                    n_semantic_dims=n_components,
                    use_parallel=use_parallel,
                    save_full_per_neuron=save_full_per_neuron
                )

                if df_main.empty:
                    print(f"[Shift {shift}] WARN: empty df_main for {region}-{condition}")
                    continue

                df_main["temporal_shift"] = shift
                df_main["region"] = region
                df_main["condition"] = condition
                df_main["patient_id"] = patient_id
                all_rows.append(df_main)

        return all_rows

    def aggregate_shift_summary_with_significance(all_df):
        LLH_THRESH = 0.0
        PVALUE_THRESH = 0.05
        all_df["significant_llh"] = (all_df["ll_diff"] > LLH_THRESH) & (all_df["pval_xsem"] < PVALUE_THRESH)
        all_df["significant_corr"] = (all_df["spearman_corr"] > 0) & (all_df["spearman_p"] < PVALUE_THRESH)
        group_cols = ["region", "condition", "temporal_shift"]
        return (
            all_df
            .groupby(group_cols)
            .agg(
                median_spearman_corr=("spearman_corr", "median"),
                median_pearson_corr=("pearson_corr", "median"),
                median_ll_diff=("ll_diff", "median"),
                n_neurons=("neuron", "count"),
                n_sig_llh=("significant_llh", "sum"),
                n_sig_corr=("significant_corr", "sum"),
                p_spearman_vs_zero=("spearman_corr", lambda x: wilcoxon(x, alternative="greater").pvalue if len(x) >= 5 else np.nan),
                p_ll_diff_vs_zero=("ll_diff", lambda x: wilcoxon(x, alternative="greater").pvalue if len(x) >= 5 else np.nan)
            ).reset_index()
        )

    # === Run all shifts in parallel
    print(f"🚀 Running {len(shift_range)} shifts in parallel with n_jobs={n_jobs}")
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_one_shift)(shift) for shift in shift_range
    )

    all_summary_rows = [df for shift_rows in results for df in shift_rows]
    print(f"[CHECK] Collected {len(all_summary_rows)} summary tables before concatenation.")

    if not all_summary_rows:
        raise RuntimeError(f"[FAIL] No data to save for patient {patient_id} — no usable regressions across all shifts.")

    all_df = pd.concat(all_summary_rows, ignore_index=True)
    save_dir = os.path.join(results_dir, patient_id)
    os.makedirs(save_dir, exist_ok=True)

    if save_full_per_neuron:
        all_df.to_csv(os.path.join(save_dir, "summary_all_shifts_per_neuron.csv"), index=False)

    summary = aggregate_shift_summary_with_significance(all_df)
    summary.to_csv(os.path.join(save_dir, "temporal_summary_statistics.csv"), index=False)
    print(f"✅ Saved summary to {save_dir}")

    return summary, all_df
