# my_regression_code
import pandas as pd
import numpy as np
import ast
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from joblib import delayed, Parallel
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import PoissonRegressor
import numpy as np
from joblib import Parallel, delayed
from scipy.special import gammaln
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter("ignore", PerfectSeparationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=PerfectSeparationWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="divide by zero encountered in log")

# Loading of the BERT/WORD2VEC embeddings and appies PCA to reduce it and user can input number of PCs
def load_and_reduce_embeddings(embedding_csv_path, n_components=100):
    df = pd.read_csv(embedding_csv_path)
    df['Parsed_Embedding'] = df['Embedding'].apply(ast.literal_eval)
    embedding_matrix = np.array(df['Parsed_Embedding'].tolist())

    if 'onset' in df.columns:
        df = df.sort_values(by='onset').reset_index(drop=True)

    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(embedding_matrix)
    pcs_df = pd.DataFrame(pcs, columns=[f'PC{i+1}' for i in range(n_components)])

    return df.reset_index(drop=True), pcs_df

def build_feature_matrix(pcs_df, df_metadata, duration_file, target_speaker="SPK1"):
    if duration_file.endswith(".xlsx"):
        durations_df = pd.read_excel(duration_file)
    else:
        durations_df = pd.read_csv(duration_file)

    durations_df = durations_df.reset_index(drop=True)
    durations = durations_df.loc[df_metadata.index, 'regress_dur'].reset_index(drop=True)
    # durations = durations_df.loc[df_metadata.index, 'Duration'].reset_index(drop=True)   

    if target_speaker is not None:
        mask = df_metadata['Speaker'] == target_speaker
    else:
        mask = df_metadata['Speaker'] != "SPK1"

    pcs_sub = pcs_df[mask.values].reset_index(drop=True)
    durations_sub = durations[mask.values].reset_index(drop=True)

    if pcs_sub.shape[0] != durations_sub.shape[0]:
        raise ValueError(f"Row mismatch: {pcs_sub.shape[0]} PCs vs {durations_sub.shape[0]} durations")

    pcs = pcs_sub.values
    durations_arr = durations_sub.values.reshape(-1, 1)
    interactions = pcs * durations_arr
    X_full = np.hstack([pcs, durations_arr, interactions])

    if X_full.shape[0] == 0:
        raise ValueError("No samples found for selected speaker subset (likely a mismatch in Speaker labels).")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)
    return X_scaled

def split_feature_matrices(embedding_csv_path, duration_csv_path, n_components=100, target_speaker="SPK1"):
    df_metadata, pcs_df = load_and_reduce_embeddings(embedding_csv_path, n_components)
    X_self = build_feature_matrix(pcs_df, df_metadata, duration_csv_path, target_speaker=target_speaker)
    X_other = build_feature_matrix(pcs_df, df_metadata, duration_csv_path, target_speaker=None)
    return X_self, X_other, df_metadata


def build_Y_matrices_from_raw_spikes(df_metadata, spike_data, regions, target_speaker="SPK1"):
    speakers_in_data = list(spike_data.keys())
    speaker_tag_map = {f"SPK{spk.replace('Speaker','').strip()}": spk.strip() for spk in speakers_in_data}

    region_Ys = {}

    for region in regions:
        print(f"\n🔍 Processing region: {region}")
        Y_self, Y_other = [], []
        W_self, W_other = [], []

        kept_self_indices = []
        kept_other_indices = []

        speaker_counters = {spk: 0 for spk in speakers_in_data}

        for idx, row in df_metadata.iterrows():
            spk = row['Speaker']
            word = row['Word']
            speaker_name = speaker_tag_map.get(spk)

            if speaker_name is None or region not in spike_data[speaker_name]:
                continue

            current_idx = speaker_counters[speaker_name]
            max_rows = spike_data[speaker_name][region].shape[0]

            if current_idx >= max_rows:
                continue

            row_data = spike_data[speaker_name][region][current_idx, :]

            if spk == target_speaker:
                Y_self.append(row_data)
                W_self.append(word)
                kept_self_indices.append(idx)
            else:
                Y_other.append(row_data)
                W_other.append(word)
                kept_other_indices.append(idx)

            speaker_counters[speaker_name] += 1

        print(f"✅ Y_self rows: {len(Y_self)}")
        print(f"✅ Y_other rows: {len(Y_other)}")

        region_Ys[region] = {
            'self': np.vstack(Y_self) if Y_self else np.empty((0,)),
            'other': np.vstack(Y_other) if Y_other else np.empty((0,)),
            'word_ref_self': W_self,
            'word_ref_other': W_other,
            'indices_self': kept_self_indices,
            'indices_other': kept_other_indices
        }

    return region_Ys

def build_Y_matrices_from_raw_spikes_direct(metadata, spikes, region_cells, speaker_events, region_list, target_speaker="SPK1"):
    """
    Build region_Ys dictionary directly from raw spike data and aligned speaker events,
    matching speaker labels in the metadata to keys in speaker_events (e.g., "Speaker1", "Speaker2").
    """
    region_Ys = {}

    for region in region_list:
        print(f"\n🔍 Building Y matrices for region: {region}")
        Y_self, Y_other = [], []
        W_self, W_other = [], []
        kept_self_indices = []
        kept_other_indices = []

        # Track how many events we’ve pulled per speaker
        speaker_counters = {spk: 0 for spk in speaker_events.keys()}

        for idx, row in metadata.iterrows():
            spk = row["Speaker"]
            word = row["Word"]
            speaker_name = f"Speaker{spk.replace('SPK', '')}"

            if speaker_name not in speaker_events or region not in region_cells:
                continue

            spk_idx = speaker_counters[speaker_name]
            region_indices = region_cells[region]
            if spk_idx >= len(speaker_events[speaker_name]):
                continue

            onset, offset = speaker_events[speaker_name][spk_idx, 1:3]
            spike_chunk = spikes[int(onset):int(offset), region_indices]
            spike_sum = np.sum(spike_chunk, axis=0)  # neuron count vector

            if spk == target_speaker:
                Y_self.append(spike_sum)
                W_self.append(word)
                kept_self_indices.append(idx)
            else:
                Y_other.append(spike_sum)
                W_other.append(word)
                kept_other_indices.append(idx)

            speaker_counters[speaker_name] += 1

        print(f"✅ Y_self rows: {len(Y_self)}")
        print(f"✅ Y_other rows: {len(Y_other)}")

        region_Ys[region] = {
            'self': np.vstack(Y_self) if Y_self else np.empty((0,)),
            'other': np.vstack(Y_other) if Y_other else np.empty((0,)),
            'word_ref_self': W_self,
            'word_ref_other': W_other,
            'indices_self': kept_self_indices,
            'indices_other': kept_other_indices
        }

    return region_Ys



def get_self_and_other_features(embedding_file, duration_file, n_components=100, target_speaker="SPK1"):
    df_metadata, pcs_df = load_and_reduce_embeddings(embedding_file, n_components)
    print(f"✅ EMBEDDING CSV rows: {len(df_metadata)} unique words")
    print("✅ Speakers in CSV:", df_metadata['Speaker'].unique())
    # Make sure durations match the original metadata
    if duration_file.endswith(".xlsx"):
        durations_df = pd.read_excel(duration_file)
    else:
        durations_df = pd.read_csv(duration_file)
    print(f"✅ DURATION file rows: {len(durations_df)}")
    durations_df = durations_df.reset_index(drop=True)
    df_metadata = df_metadata.reset_index(drop=True)

    # Attach duration column
    # df_metadata["Duration"] = durations_df["Duration"]
    df_metadata["regress_dur"] = durations_df["regress_dur"]
    # Separate
    mask_self = df_metadata['Speaker'] == target_speaker
    mask_other = df_metadata['Speaker'] != target_speaker

    # Filter everything
    pcs_self = pcs_df[mask_self].reset_index(drop=True)
    pcs_other = pcs_df[mask_other].reset_index(drop=True)
    # dur_self = df_metadata.loc[mask_self, "Duration"].reset_index(drop=True)
    # dur_other = df_metadata.loc[mask_other, "Duration"].reset_index(drop=True)
    dur_self = df_metadata.loc[mask_self, "regress_dur"].reset_index(drop=True)
    dur_other = df_metadata.loc[mask_other, "regress_dur"].reset_index(drop=True)

    words_self = df_metadata.loc[mask_self, "Word"].reset_index(drop=True)
    words_other = df_metadata.loc[mask_other, "Word"].reset_index(drop=True)

    # Build final matrices
    X_self = build_feature_matrix_core(pcs_self, dur_self)
    X_other = build_feature_matrix_core(pcs_other, dur_other)

    return X_self, X_other, df_metadata



def build_feature_matrix_core(pcs_df, durations):
    pcs = pcs_df.values
    durations_arr = durations.values.reshape(-1, 1)
    interactions = pcs * durations_arr
    X_full = np.hstack([pcs, durations_arr, interactions])

    if X_full.shape[0] == 0:
        raise ValueError("No samples found for selected speaker subset.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)
    return X_scaled


def clean_nan_XY_per_region(X_self, X_other, region_Ys):
    print("\n✅ Checking and cleaning NaNs in all X/Y conditions (joint dropping)...\n")

    cleaned_data = {}

    for region in region_Ys:
        cleaned_data[region] = {}

        for condition in ["self", "other"]:
            print(f"🔎 {region} - {condition}")

            X = X_self if condition == "self" else X_other
            Y = region_Ys[region][condition]

            if X.shape[0] != Y.shape[0]:
                print(f"⚠️  Skipping {region}_{condition}: shape mismatch before cleaning (X={X.shape[0]}, Y={Y.shape[0]})")
                continue

            nan_mask_X = np.isnan(X).any(axis=1)
            nan_mask_Y = np.isnan(Y).any(axis=1)
            joint_mask = ~(nan_mask_X | nan_mask_Y)

            n_before = X.shape[0]
            n_after = joint_mask.sum()
            n_dropped = n_before - n_after

            if n_dropped > 0:
                print(f"⚠️  Dropping {n_dropped} rows with NaNs in X or Y")
            else:
                print(f"✅ No NaNs detected")

            X_clean = X[joint_mask]
            Y_clean = Y[joint_mask]
            kept_indices = np.where(joint_mask)[0]

            cleaned_data[region][condition] = {
                "X": X_clean,
                "Y": Y_clean,
                "indices": kept_indices
            }

            region_Ys[region][condition] = Y_clean  # optional: keep in sync if reused

            print(f"✅ Cleaned shapes: X={X_clean.shape}, Y={Y_clean.shape}\n")

    return cleaned_data


def run_poisson_ridge(
    X, Y,
    patient_id,
    neuron_idx,
    region_name,
    n_semantic_dims,
    results_dir="/projects/bhayden/anilu/Language_docs/RegressionRESULTS",
    n_iterations=10,
    test_size=0.3,
    use_clusters=False,
    fast_beta_only=False,
):
    import os
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import PoissonRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import r2_score
    from scipy.stats import pearsonr
    from scipy.special import gammaln
    import statsmodels.api as sm
    import warnings
    from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.simplefilter("ignore", PerfectSeparationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=PerfectSeparationWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    if use_clusters:
        results_dir = os.path.join(results_dir, "clusterwise")
    else:
        results_dir = os.path.join(results_dir, "allwords")

    n_words, n_neurons = Y.shape
    n_predictors = X.shape[1]

    print(f"✅ Detected {n_semantic_dims} semantic embedding dimensions (excluding duration/interactions)")

    all_results = []
    significant_predictors = []
    summary_predictors = []
    ridge_coeff_accumulator = {neuron_idx: []}
    ridge_support_flags = {neuron_idx: set()}

    y_true_all = []
    y_pred_all = []
    train_lls = []
    test_lls = []
    train_corrs = []
    test_corrs = []

    y = Y[:, neuron_idx]
    neuron_predictor_records = []

    # # === Define your alpha grid
    # param_grid = {
    #     'alpha': np.unique(np.concatenate([
    #         [0.1],
    #         np.logspace(0, 2.5, 10),
    #         [1000.0],
    #         np.logspace(-2, 4, 20)
    #     ]))
    # }
    param_grid = {'alpha': np.logspace(-3, 3, 30)}

    model = PoissonRegressor(max_iter=1000)
    grid = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_poisson_deviance', n_jobs=1)
    grid.fit(X, y)
    best_alpha = grid.best_params_['alpha']
    print(f"✅ Best alpha selected: {best_alpha}")

    if fast_beta_only:
        final_model = PoissonRegressor(alpha=best_alpha, max_iter=1000)
        final_model.fit(X, y)
        coefs = final_model.coef_
        coef_df = pd.DataFrame([coefs], columns=[f"beta_{i}" for i in range(len(coefs))])
        coef_df.insert(0, "neuron", neuron_idx)
        coef_df.insert(1, "region", region_name)
        return coef_df

    for iter_idx in range(n_iterations):
        adaptive_test_size = test_size
        split_point = int((1 - adaptive_test_size) * n_words)
        offset = (iter_idx * int(n_words * adaptive_test_size)) % n_words
        idx_range = np.roll(np.arange(n_words), -offset)
        train_idx = idx_range[:split_point]
        test_idx = idx_range[split_point:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        n, k = len(y_test), X_test.shape[1]
        if k >= n - 1:
            adaptive_test_size = 0.4
            split_point = int((1 - adaptive_test_size) * n_words)
            offset = (iter_idx * int(n_words * adaptive_test_size)) % n_words
            idx_range = np.roll(np.arange(n_words), -offset)
            train_idx = idx_range[:split_point]
            test_idx = idx_range[split_point:]
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            n, k = len(y_test), X_test.shape[1]

        # === Real model
        ridge_model = PoissonRegressor(alpha=best_alpha, max_iter=1000)
        ridge_model.fit(X_train, y_train)
        pred_train = np.clip(ridge_model.predict(X_train), 1e-10, None)
        pred_test = np.clip(ridge_model.predict(X_test), 1e-10, None)

        ll_train = np.sum(y_train * np.log(pred_train) - pred_train - gammaln(y_train + 1))
        ll_real = np.sum(y_test * np.log(pred_test) - pred_test - gammaln(y_test + 1))
        r_train, _ = pearsonr(y_train, pred_train)
        r_test, corr_p = pearsonr(y_test, pred_test)
        r2 = r2_score(y_test, pred_test)
        adj_r2 = 1 - ((1 - r2) * (n - 1)) / (n - k - 1) if n > k + 1 else r2

        y_true_all.extend(y_test)
        y_pred_all.extend(pred_test)
        train_lls.append(ll_train)
        test_lls.append(ll_real)
        train_corrs.append(r_train)
        test_corrs.append(r_test)
        ridge_coeff_accumulator[neuron_idx].append(ridge_model.coef_)

        # === Y-shuffled null model
        y_train_shuf = np.random.permutation(y_train)
        null_model = PoissonRegressor(alpha=best_alpha, max_iter=1000)
        null_model.fit(X_train, y_train_shuf)
        null_pred = np.clip(null_model.predict(X_test), 1e-10, None)
        ll_shuf = np.sum(y_test * np.log(null_pred) - null_pred - gammaln(y_test + 1))
        corr_shuf, corr_p_shuf = pearsonr(y_test, null_pred)
        r2_shuf = r2_score(y_test, null_pred)
        adj_r2_shuf = 1 - ((1 - r2_shuf) * (n - 1)) / (n - k - 1) if n > k + 1 else r2_shuf
        ll_diff = ll_real - ll_shuf

        # === X-shuffled null model (only shuffling semantic embeddings)
        X_train_shuf = X_train.copy()
        X_test_shuf = X_test.copy()
        for col in range(n_semantic_dims):
            np.random.shuffle(X_train_shuf[:, col])
            np.random.shuffle(X_test_shuf[:, col])

        xshuf_model = PoissonRegressor(alpha=best_alpha, max_iter=1000)
        xshuf_model.fit(X_train_shuf, y_train)
        pred_xshuf = np.clip(xshuf_model.predict(X_test_shuf), 1e-10, None)
        ll_xshuf = np.sum(y_test * np.log(pred_xshuf) - pred_xshuf - gammaln(y_test + 1))
        ll_diff_xshuf = ll_real - ll_xshuf

        # === Record results
        all_results.append({
            'neuron': neuron_idx,
            'iteration': iter_idx,
            'best_alpha': best_alpha,
            'll_real': ll_real,
            'll_shuf': ll_shuf,
            'll_diff': ll_diff,
            'll_xshuf': ll_xshuf,
            'll_diff_xshuf': ll_diff_xshuf,
            'adj_r2': adj_r2,
            'adj_r2_shuf': adj_r2_shuf,
            'corr': r_test,
            'corr_p': corr_p,
            'corr_shuf': corr_shuf,
            'corr_p_shuf': corr_p_shuf
        })

        try:
            glm = sm.GLM(y_train, X_train, family=sm.families.Poisson())
            result = glm.fit()
            for idx, (coef, pval) in enumerate(zip(result.params, result.pvalues)):
                if pval < 0.05:
                    ridge_support_flags[neuron_idx].add(idx)
                    significant_predictors.append({
                        'neuron': neuron_idx,
                        'iteration': iter_idx,
                        'predictor_index': idx,
                        'coefficient': coef,
                        'p_value': pval
                    })
                neuron_predictor_records.append((idx, coef, pval))
        except:
            continue

    # === Post-processing results
    df_neuron = pd.DataFrame(neuron_predictor_records, columns=['predictor_index', 'coefficient', 'p_value'])
    grouped = df_neuron.groupby('predictor_index')
    mean_ridge_coef = np.mean(ridge_coeff_accumulator[neuron_idx], axis=0)

    for pred_idx, group in grouped:
        median_p = group['p_value'].median()
        if median_p < 0.05:
            ridge_support = 1 if pred_idx in ridge_support_flags[neuron_idx] else 0
            summary_predictors.append({
                'neuron': neuron_idx,
                'predictor_index': pred_idx,
                'mean_coefficient': group['coefficient'].mean(),
                'median_p_value': median_p,
                'times_significant': (group['p_value'] < 0.05).sum(),
                'ridge_support': ridge_support
            })

    df = pd.DataFrame(all_results)
    summary_df = df.groupby('neuron').agg({
        'll_real': 'mean',
        'll_shuf': 'mean',
        'll_diff': 'mean',
        'll_xshuf': 'mean',
        'll_diff_xshuf': 'mean',
        'adj_r2': 'mean',
        'adj_r2_shuf': 'mean',
        'corr': 'mean',
        'corr_shuf': 'mean',
        'corr_p': 'median',
        'corr_p_shuf': 'median',
        'best_alpha': 'mean'
    }).reset_index()

    final_sig_df = pd.DataFrame(summary_predictors)
    ridge_rows = []
    coef_matrix = []
    valid_neuron_indices = []

    for neuron_idx, coefs in ridge_coeff_accumulator.items():
        if len(coefs) > 0:
            mean_coef = np.mean(coefs, axis=0)
            if np.isscalar(mean_coef):
                mean_coef = np.array([mean_coef])
            coef_matrix.append(mean_coef)
            valid_neuron_indices.append(neuron_idx)
            for pred_idx, val in enumerate(mean_coef):
                support_flag = 1 if pred_idx in ridge_support_flags[neuron_idx] else 0
                ridge_rows.append({
                    'neuron': neuron_idx,
                    'predictor_index': pred_idx,
                    'mean_ridge_coef': val,
                    'ridge_support': support_flag
                })

    ridge_df = pd.DataFrame(ridge_rows)
    coef_matrix = np.vstack(coef_matrix)
    coef_df = pd.DataFrame(coef_matrix, index=valid_neuron_indices,
                           columns=[f'Predictor_{i}' for i in range(coef_matrix.shape[1])])

    output_dir = os.path.join(results_dir, f"{patient_id}_{region_name}")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "full_results.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, "summary_metrics.csv"), index=False)
    final_sig_df.to_csv(os.path.join(output_dir, "significant_predictors.csv"), index=False)
    ridge_df.to_csv(os.path.join(output_dir, "ridge_mean_coefs.csv"), index=False)
    coef_df.to_csv(os.path.join(output_dir, "neuron_by_predictor_coef_matrix.csv"))

    np.savez(os.path.join(output_dir, "diagnostics.npz"),
             y_true=np.array(y_true_all),
             y_pred=np.array(y_pred_all),
             train_lls=train_lls,
             test_lls=test_lls,
             train_corrs=train_corrs,
             test_corrs=test_corrs)

    print(f"✅ Results saved to: {output_dir}")
    return df, summary_df, final_sig_df, ridge_df, coef_df

import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.special import gammaln
from scipy.stats import pearsonr, spearmanr

def compute_effective_dof_ridge(X, alpha):
    try:
        XtX = X.T @ X
        I = np.eye(X.shape[1])
        H = X @ np.linalg.inv(XtX + alpha * I) @ X.T
        return np.trace(H)
    except np.linalg.LinAlgError:
        U, s, _ = np.linalg.svd(X, full_matrices=False)
        s_squared = s**2
        return np.sum(s_squared / (s_squared + alpha))

def run_poisson_ridge2(
    X_full, Y, neuron_idx, seed=42,
    test_size=0.2, alphas=np.logspace(-3, 3, 30), n_shuffles=100
):
    np.random.seed(seed)
    X_scaled = StandardScaler().fit_transform(X_full)
    y = Y[:, neuron_idx]

    if np.std(y) == 0 or np.all(y == 0) or not np.all(np.isfinite(y)):
        return None

    # === Inner CV to find best alpha
    kf_inner = KFold(n_splits=5, shuffle=True, random_state=seed)
    alpha_scores = {alpha: [] for alpha in alphas}
    for alpha in alphas:
        for train_idx, val_idx in kf_inner.split(X_scaled):
            model = PoissonRegressor(alpha=alpha, max_iter=1000)
            model.fit(X_scaled[train_idx], y[train_idx])
            pred = np.clip(model.predict(X_scaled[val_idx]), 1e-10, None)
            ll = np.sum(y[val_idx] * np.log(pred) - pred - gammaln(y[val_idx] + 1))
            alpha_scores[alpha].append(ll)

    avg_ll_per_alpha = {a: np.mean(s) for a, s in alpha_scores.items()}
    best_alpha = max(avg_ll_per_alpha, key=avg_ll_per_alpha.get)
    alpha_std = np.std(alpha_scores[best_alpha])

    # === Train/test split
    if test_size is None:
        X_train, y_train = X_scaled, y
        X_test, y_test = X_scaled, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=seed
        )

    # === Fit final model
    model_real = PoissonRegressor(alpha=best_alpha, max_iter=1000)
    model_real.fit(X_train, y_train)
    pred_real = np.clip(model_real.predict(X_test), 1e-10, None)

    ll_real = np.sum(y_test * np.log(pred_real) - pred_real - gammaln(y_test + 1))
    mu = np.mean(y_train)
    ll_null = np.sum(y_test * np.log(mu) - mu - gammaln(y_test + 1))
    pseudo_r2 = 1 - (ll_real / ll_null)
    pearson_corr = pearsonr(y_test, pred_real)[0]
    spearman_corr = spearmanr(y_test, pred_real)[0]

    # === Null distribution: X shuffling (semantic only)
    ll_shuf_list = []
    for _ in range(n_shuffles):
        X_shuf = X_scaled[np.random.permutation(len(X_scaled))]
        if test_size is None:
            Xs_train, Xs_test = X_shuf, X_shuf
        else:
            Xs_train, Xs_test, _, _ = train_test_split(
                X_shuf, y, test_size=test_size, random_state=seed
            )

        model_shuf = PoissonRegressor(alpha=best_alpha, max_iter=1000)
        model_shuf.fit(Xs_train, y_train)
        pred_shuf = np.clip(model_shuf.predict(Xs_test), 1e-10, None)
        ll_shuf = np.sum(y_test * np.log(pred_shuf) - pred_shuf - gammaln(y_test + 1))
        ll_shuf_list.append(ll_shuf)

    ll_xshuf_mean = np.mean(ll_shuf_list)
    ll_diff = ll_real - ll_xshuf_mean
    p_val_ll_xshuf = np.mean(np.array(ll_shuf_list) >= ll_real)

    edf = compute_effective_dof_ridge(X_train, best_alpha)
    aic = 2 * edf - 2 * ll_real
    bic = np.log(len(y_test)) * edf - 2 * ll_real

    return {
        "neuron": neuron_idx,
        "best_alpha": best_alpha,
        "alpha_ll_mean": avg_ll_per_alpha[best_alpha],
        "alpha_ll_std": alpha_std,
        "edf": edf,
        "aic": aic,
        "bic": bic,
        "ll_real": ll_real,
        "ll_xshuf_mean": ll_xshuf_mean,
        "ll_diff": ll_diff,
        "p_val_ll_xshuf": p_val_ll_xshuf,
        "pseudo_r2": pseudo_r2,
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr,
        "coef": model_real.coef_.tolist()
    }



### reliability part
# === TOP-LEVEL IMPORTS ===
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import PoissonRegressor
from scipy.stats import pearsonr

def get_best_alpha(X, y, alphas):
    """Select best alpha for Poisson ridge regression via CV."""
    model = PoissonRegressor(max_iter=1000)
    grid = GridSearchCV(model, {"alpha": alphas}, scoring="neg_mean_poisson_deviance", cv=5)
    grid.fit(X, y)
    return grid.best_params_["alpha"]

def fit_poisson_ridge_get_beta(X, y, alpha=None, alphas=np.logspace(-3, 3, 30)):
    """Fit Poisson ridge regression and return coefficients only."""
    if alpha is None:
        alpha = get_best_alpha(X, y, alphas)
    model = PoissonRegressor(alpha=alpha, max_iter=1000)
    model.fit(X, y)
    return model.coef_, alpha

def compute_beta_reliability_customridge(X_self, y_self, X_other, y_other, alphas, n_nulls=100):
    # === Split trials
    Xs1, Xs2, ys1, ys2 = train_test_split(X_self, y_self, test_size=0.5, random_state=0)
    Xo1, Xo2, yo1, yo2 = train_test_split(X_other, y_other, test_size=0.5, random_state=0)

    # === Fit ridge on each half (no saving)
    beta_s1, alpha_self = fit_poisson_ridge_get_beta(Xs1, ys1, alphas=alphas)
    beta_s2, _ = fit_poisson_ridge_get_beta(Xs2, ys2, alpha=alpha_self)

    beta_o1, alpha_other = fit_poisson_ridge_get_beta(Xo1, yo1, alphas=alphas)
    beta_o2, _ = fit_poisson_ridge_get_beta(Xo2, yo2, alpha=alpha_other)

    # === Compute reliability metrics
    r_self = pearsonr(beta_s1, beta_s2)[0]
    r_other = pearsonr(beta_o1, beta_o2)[0]
    r_cross = pearsonr(beta_s1, beta_o1)[0]

    # === Null distribution via Y-shuffling
    null_corrs = []
    for _ in range(n_nulls):
        yo_shuf = np.random.permutation(y_other)
        null_beta, _ = fit_poisson_ridge_get_beta(X_other, yo_shuf, alpha=alpha_other)
        null_corrs.append(pearsonr(beta_s1, null_beta)[0])

    return {
        'r_self': r_self,
        'r_other': r_other,
        'r_cross': r_cross,
        'null_distribution': null_corrs,
        'null_mean': np.mean(null_corrs),
        'null_std': np.std(null_corrs),
    }

# === Run for all neurons ===
from joblib import Parallel, delayed, parallel_backend

def run_beta_reliability_all_neurons(X_self, X_other, Y_self, Y_other,
                                     alphas, n_nulls=100,
                                     neuron_mode='all', n_jobs=-1):
    if neuron_mode == 'all':
        neuron_indices = range(Y_self.shape[1])
    else:
        neuron_indices = [neuron_mode]

    def run_single_neuron(neuron_idx):
        y_self = Y_self[:, neuron_idx]
        y_other = Y_other[:, neuron_idx]
        try:
            res = compute_beta_reliability_customridge(X_self, y_self, X_other, y_other,
                                           alphas=alphas, n_nulls=n_nulls)

            res['neuron'] = neuron_idx
            return res
        except Exception as e:
            print(f"Neuron {neuron_idx} failed: {e}")
            return None

    with parallel_backend("threading"):
        results = Parallel(n_jobs=n_jobs)(
            delayed(run_single_neuron)(i) for i in neuron_indices
        )

    return [r for r in results if r is not None]



# -------------------------
# Analysis helpers for reliability output
# -------------------------
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def _safe_pearson(a, b):
    try:
        if np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and (a.size>1) and (b.size>1):
            return pearsonr(a, b)[0]
    except Exception:
        pass
    return np.nan

def aggregate_null_distribution_from_neurons(reliability_results):
    """
    reliability_results: list of dicts (one per neuron) returned by run_beta_reliability_all_neurons.
      each dict must contain 'null_distribution' (list/array) and 'r_cross' (observed per neuron).
    Returns:
      R_null: array shape (min_nulls,) pooled nulls of mean r across neurons for each null iteration
      r_obs_vec: array of observed per-neuron r_cross
    """
    # gather per-neuron null lists and observed r_cross
    nulls_list = []
    r_obs = []
    for r in reliability_results:
        if r is None:
            continue
        r_obs.append(r.get('r_cross', np.nan))
        nd = r.get('null_distribution', None)
        if nd is None:
            nd = []
        nulls_list.append(np.asarray(nd, dtype=float))

    if len(nulls_list) == 0:
        return np.array([]), np.array(r_obs)

    # use minimum length across neurons to align iterations (simple & conservative)
    min_nulls = min([n.shape[0] for n in nulls_list if n.shape[0] > 0] + [0])
    if min_nulls == 0:
        # fall back to per-neuron null means
        per_neuron_null_means = [n.mean() if n.size>0 else np.nan for n in nulls_list]
        R_null = np.array(per_neuron_null_means)
        # not ideal — but keep consistent shape
    else:
        # build matrix (n_neurons x min_nulls)
        mat = np.vstack([n[:min_nulls] if n.shape[0] >= min_nulls else np.pad(n, (0, min_nulls-n.shape[0]), constant_values=np.nan)
                         for n in nulls_list])
        # compute mean across neurons for each null iteration (skip NaNs)
        R_null = np.nanmean(mat, axis=0)

    return np.asarray(R_null), np.asarray(r_obs)

from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import numpy as np

def compute_noise_ceiling_for_neuron(
    X_self, y_self, X_other, y_other,
    n_splits=200, alphas=None, random_state=None,
    reuse_alpha=False, alpha_self=None, alpha_other=None,
    clamp_negative_to_zero=True
):
    """
    Repeated half-split noise ceiling for one neuron.
    If reuse_alpha=True and alpha_self/alpha_other are None, caller should provide them (or compute outside).
    Returns:
      ceil_samples (n_splits,), summary dict {mean, median, ci_low, ci_high}
    """
    rng = np.random.default_rng(random_state)
    ceil_samples = np.full(n_splits, np.nan)

    for b in range(n_splits):
        try:
            # random half-splits
            Xs1, Xs2, ys1, ys2 = train_test_split(
                X_self, y_self, test_size=0.5, random_state=int(rng.integers(0, 2**31-1))
            )
            Xo1, Xo2, yo1, yo2 = train_test_split(
                X_other, y_other, test_size=0.5, random_state=int(rng.integers(0, 2**31-1))
            )

            if reuse_alpha:
                # must have been provided by caller (or precomputed)
                if alpha_self is None or alpha_other is None:
                    raise ValueError("When reuse_alpha=True, alpha_self and alpha_other must be provided.")
                beta_s1, _ = fit_poisson_ridge_get_beta(Xs1, ys1, alpha=alpha_self)
                beta_s2, _ = fit_poisson_ridge_get_beta(Xs2, ys2, alpha=alpha_self)
                beta_o1, _ = fit_poisson_ridge_get_beta(Xo1, yo1, alpha=alpha_other)
                beta_o2, _ = fit_poisson_ridge_get_beta(Xo2, yo2, alpha=alpha_other)
            else:
                # pick alpha per split (your current behaviour)
                beta_s1, alpha_s = fit_poisson_ridge_get_beta(Xs1, ys1, alpha=None, alphas=alphas)
                beta_s2, _ = fit_poisson_ridge_get_beta(Xs2, ys2, alpha=alpha_s, alphas=alphas)
                beta_o1, alpha_o = fit_poisson_ridge_get_beta(Xo1, yo1, alpha=None, alphas=alphas)
                beta_o2, _ = fit_poisson_ridge_get_beta(Xo2, yo2, alpha=alpha_o, alphas=alphas)

            r_s = _safe_pearson(beta_s1, beta_s2)
            r_o = _safe_pearson(beta_o1, beta_o2)

            if clamp_negative_to_zero:
                r_s = max(r_s, 0.0) if not np.isnan(r_s) else 0.0
                r_o = max(r_o, 0.0) if not np.isnan(r_o) else 0.0

            rs_sb = (2.0 * r_s) / (1.0 + r_s) if (1.0 + r_s) != 0 else 0.0
            ro_sb = (2.0 * r_o) / (1.0 + r_o) if (1.0 + r_o) != 0 else 0.0
            ceil_samples[b] = np.sqrt(rs_sb * ro_sb)
        except Exception:
            ceil_samples[b] = np.nan

    valid = ceil_samples[np.isfinite(ceil_samples)]
    if valid.size == 0:
        summary = {"mean": np.nan, "median": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    else:
        summary = {
            "mean": float(np.mean(valid)),
            "median": float(np.median(valid)),
            "ci_low": float(np.percentile(valid, 2.5)),
            "ci_high": float(np.percentile(valid, 97.5))
        }
    return ceil_samples, summary


def compute_noise_ceiling_all_neurons(
    X_self, X_other, Y_self, Y_other,
    neuron_indices=None, n_splits=200, n_jobs=8, alphas=None,
    reuse_alpha=True
):
    """
    Parallel compute per-neuron ceil_samples and summaries.
    Returns:
      ceil_matrix: shape (n_neurons, n_splits)
      summaries: list of per-neuron summary dicts
      ceil_means: array of mean per neuron (nan if failed)
    """
    if neuron_indices is None:
        neuron_indices = list(range(Y_self.shape[1]))

    # optional precompute alpha per neuron on full data (if reuse_alpha=True)
    cached_alphas = {}
    if reuse_alpha:
        for i in neuron_indices:
            y_s = Y_self[:, i]
            y_o = Y_other[:, i]
            # skip invalid / constant neurons
            if not np.all(np.isfinite(y_s)) or not np.all(np.isfinite(y_o)) or np.std(y_s)==0 or np.std(y_o)==0:
                cached_alphas[i] = (None, None)
                continue
            # get best alpha on full data (returns beta, alpha)
            _, a_s = fit_poisson_ridge_get_beta(X_self, y_s, alpha=None, alphas=alphas)
            _, a_o = fit_poisson_ridge_get_beta(X_other, y_o, alpha=None, alphas=alphas)
            cached_alphas[i] = (a_s, a_o)

    def _worker(i):
        y_s = Y_self[:, i]
        y_o = Y_other[:, i]
        if not np.all(np.isfinite(y_s)) or not np.all(np.isfinite(y_o)) or np.std(y_s)==0 or np.std(y_o)==0:
            return np.full(n_splits, np.nan), {"mean": np.nan, "median": np.nan, "ci_low": np.nan, "ci_high": np.nan}
        if reuse_alpha:
            a_s, a_o = cached_alphas.get(i, (None, None))
            if a_s is None or a_o is None:
                return np.full(n_splits, np.nan), {"mean": np.nan, "median": np.nan, "ci_low": np.nan, "ci_high": np.nan}
            samples, summary = compute_noise_ceiling_for_neuron(
                X_self, y_s, X_other, y_o,
                n_splits=n_splits, alphas=alphas, random_state=i+1,
                reuse_alpha=True, alpha_self=a_s, alpha_other=a_o
            )
        else:
            samples, summary = compute_noise_ceiling_for_neuron(
                X_self, y_s, X_other, y_o,
                n_splits=n_splits, alphas=alphas, random_state=i+1,
                reuse_alpha=False
            )
        return samples, summary

    results = Parallel(n_jobs=n_jobs)(
        delayed(_worker)(i) for i in neuron_indices
    )
    mats = [r[0] for r in results]
    sums = [r[1] for r in results]
    ceil_matrix = np.vstack(mats)
    ceil_means = np.array([s["mean"] for s in sums], dtype=float)
    return ceil_matrix, sums, ceil_means


def paired_bootstrap_difference(r_obs_vec, ceil_means, B=2000, random_state=0):
    """
    Paired bootstrap across neurons. Returns D_boot (Ceil - R) distribution and p-value for D <= 0.
    """
    rng = np.random.default_rng(random_state)
    n = len(r_obs_vec)
    D = np.zeros(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        R_samp = np.nanmean(r_obs_vec[idx])
        Ceil_samp = np.nanmean(ceil_means[idx])
        D[b] = Ceil_samp - R_samp
    D_ci = np.percentile(D, [2.5, 97.5])
    p_below = (np.sum(D <= 0) + 1) / (B + 1)
    return D, D_ci, p_below

def analyze_reliability_results(reliability_results, X_self, X_other, Y_self, Y_other,
                                n_nulls_to_use=None, n_ceiling_splits=500, ceil_n_jobs=8,
                                ceil_alphas=None, B_boot=2000, verbose=True):
    """
    Top-level analysis: takes list returned by run_beta_reliability_all_neurons and runs:
      - pooled null test (R_obs > R_null)
      - noise-ceiling paired bootstrap (R_obs < Ceil)
    Returns dict with stats and arrays for plotting.
    """
    # 1) get pooled null and observed r_i
    R_null, r_obs_vec = aggregate_null_distribution_from_neurons(reliability_results)
    r_obs_vec = np.asarray(r_obs_vec, dtype=float)
    # drop NaN neurons from observed vector (but keep mapping lengths consistent for ceiling)
    valid_mask = ~np.isnan(r_obs_vec)
    r_obs_clean = r_obs_vec[valid_mask]
    n_neurons_used = np.sum(valid_mask)
    if n_neurons_used == 0:
        raise RuntimeError("No valid neurons with observed cross-correlations to analyze.")

    R_obs = np.nanmean(r_obs_clean)

    # Standardize pooled null length if needed
    if R_null.size == 0:
        # fallback: per-neuron null means (if available)
        per_neuron_null_means = []
        for r in reliability_results:
            nd = r.get('null_distribution', [])
            per_neuron_null_means.append(np.nanmean(nd) if len(nd)>0 else np.nan)
        R_null = np.asarray(per_neuron_null_means)
        # if that gives single values per neuron, compute single-value distribution of length 1
        if R_null.size > 0:
            R_null = R_null[~np.isnan(R_null)]
    if n_nulls_to_use is not None and R_null.size > n_nulls_to_use:
        R_null = R_null[:n_nulls_to_use]

    # Null test p-value (one-sided)
    p_null = (np.sum(R_null >= R_obs) + 1) / (len(R_null) + 1) if R_null.size>0 else np.nan
    null_mean = np.nanmean(R_null) if R_null.size>0 else np.nan
    null_std = np.nanstd(R_null) if R_null.size>0 else np.nan

    # 2) noise ceiling: compute per-neuron ceil means (parallel)
    neuron_indices = [i for i in range(Y_self.shape[1])]
    ceil_matrix, ceil_means = compute_noise_ceiling_all_neurons(
        X_self, X_other, Y_self, Y_other,
        neuron_indices=neuron_indices, n_splits=n_ceiling_splits, n_jobs=ceil_n_jobs, alphas=ceil_alphas
    )

    # ensure same neuron mask as observed vector
    ceil_means_clean = ceil_means[valid_mask]

    # 3) paired bootstrap test (Ceil - R)
    D_boot, D_ci, p_below = paired_bootstrap_difference(r_obs_clean, ceil_means_clean, B=B_boot, random_state=1)

    # 4) assemble summary
    summary = {
        "n_neurons": int(n_neurons_used),
        "R_obs": float(R_obs),
        "R_null_mean": float(null_mean),
        "R_null_std": float(null_std),
        "p_null": float(p_null),
        "Ceil_mean": float(np.nanmean(ceil_means_clean)),
        "Ceil_std": float(np.nanstd(ceil_means_clean)),
        "paired_D_ci": (float(D_ci[0]), float(D_ci[1])),
        "p_below_ceiling": float(p_below),
        "D_boot": D_boot,
        "R_null": R_null,
        "r_obs_vec": r_obs_clean,
        "ceil_means_vec": ceil_means_clean,
        "ceil_matrix": ceil_matrix
    }

    # 5) quick plots for diagnostics (histogram + scatter)
    if verbose:
        fig, axs = plt.subplots(1, 2, figsize=(10,4))
        axs[0].hist(R_null, bins=40, alpha=0.8)
        axs[0].axvline(R_obs, color='r', lw=2, label=f"R_obs={R_obs:.3f}")
        axs[0].axvline(null_mean, color='k', lw=1, ls='--', label=f"null mean={null_mean:.3f}")
        axs[0].legend()
        axs[0].set_title("Pooled null of mean r")

        axs[1].scatter(ceil_means_clean, r_obs_clean, alpha=0.6)
        axs[1].plot([0,1],[0,1], 'k--', lw=1)
        axs[1].set_xlabel("Per-neuron noise ceiling (mean)")
        axs[1].set_ylabel("Observed r_cross")
        axs[1].set_title("Neuron-wise: ceil vs observed")
        plt.tight_layout()
        plt.show()

    return summary




