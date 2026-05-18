import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold

# Re-import everything after code state reset

# === Clean and prepare full feature matrix ===
def clean_and_prepare(X_embed, Y, duration, n_pcs):
    duration = duration.reshape(-1, 1)
    bad_rows = np.isnan(X_embed).any(axis=1) | np.isnan(Y).any(axis=1) | np.isnan(duration).ravel()

    X_embed_clean = X_embed[~bad_rows]
    Y_clean = Y[~bad_rows]
    dur_clean = duration[~bad_rows]

    pca = PCA(n_components=n_pcs)
    X_embed_pca = pca.fit_transform(X_embed_clean)

    interaction = dur_clean * X_embed_pca
    X_full = np.hstack([dur_clean, X_embed_pca, interaction])
    X_full_scaled = StandardScaler().fit_transform(X_full)

    return X_full_scaled, Y_clean, dur_clean, pca

def apply_temporal_shift(X_embed, Y, dur, shift):
    """
    Shift X_embed in time and align Y and dur accordingly.
    Positive shift = X_embed shifted into future = causal encoding.
    Negative shift = X_embed shifted into past = lagging encoding.
    """
    if shift == 0:
        return X_embed, Y, dur

    n = X_embed.shape[0]
    if shift > 0:
        X_shifted = X_embed[:-shift]
        Y_shifted = Y[shift:]
        dur_shifted = dur[shift:]
    elif shift < 0:
        shift = abs(shift)
        X_shifted = X_embed[shift:]
        Y_shifted = Y[:-shift]
        dur_shifted = dur[:-shift]

    return X_shifted, Y_shifted, dur_shifted

def run_model_for_config(neuron_idx, X_embed, dur, Y, test_size, n_pcs, alphas,
                         n_shuffles=50, seed=42, shift=None):
    from sklearn.linear_model import PoissonRegressor
    from sklearn.model_selection import KFold, train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.decomposition import PCA
    from scipy.special import gammaln
    from scipy.stats import pearsonr
    import numpy as np
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    np.random.seed(seed)

    if shift not in [None, 0]:
        X_embed, Y, dur = apply_temporal_shift(X_embed, Y, dur, shift)

    X_full, Y_clean, dur_clean, pca = clean_and_prepare(X_embed, Y, dur, n_pcs)
    y = Y_clean[:, neuron_idx]

    if np.std(y) == 0 or np.all(y == 0) or not np.all(np.isfinite(y)):
        return None

    # === Inner CV to select best alpha
    kf_inner = KFold(n_splits=5, shuffle=True, random_state=seed)
    alpha_scores = {alpha: [] for alpha in alphas}
    for alpha in alphas:
        for train_idx, val_idx in kf_inner.split(X_full):
            model = PoissonRegressor(alpha=alpha, max_iter=1000)
            model.fit(X_full[train_idx], y[train_idx])
            pred = np.clip(model.predict(X_full[val_idx]), 1e-10, None)
            ll = np.sum(y[val_idx] * np.log(pred) - pred - gammaln(y[val_idx] + 1))
            alpha_scores[alpha].append(ll)

    avg_ll_per_alpha = {a: np.mean(s) for a, s in alpha_scores.items()}
    best_alpha = max(avg_ll_per_alpha, key=avg_ll_per_alpha.get)
    alpha_std = np.std(alpha_scores[best_alpha])

    # === Outer split
    if test_size is None:
        X_train, y_train = X_full, y
        X_test, y_test = X_full, y
        test_label = "full"
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=test_size, random_state=seed)
        test_label = test_size

    # === Final model
    model_real = PoissonRegressor(alpha=best_alpha, max_iter=1000)
    model_real.fit(X_train, y_train)
    pred_real = np.clip(model_real.predict(X_test), 1e-10, None)
    ll_real = np.sum(y_test * np.log(pred_real) - pred_real - gammaln(y_test + 1))
    mu = np.mean(y_train)
    ll_null = np.sum(y_test * np.log(mu) - mu - gammaln(y_test + 1))
    pseudo_r2 = 1 - (ll_real / ll_null)
    pearson_corr = pearsonr(y_test, pred_real)[0]

    # === Effective degrees of freedom
    def compute_effective_dof_ridge(X, alpha):
        try:
            XtX = X.T @ X
            I = np.eye(X.shape[1])
            H = X @ np.linalg.inv(XtX + alpha * I) @ X.T
            return np.trace(H)
        except np.linalg.LinAlgError:
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            s_squared = s**2
            return np.sum(s_squared / (s_squared + alpha))
    
    edf = compute_effective_dof_ridge(X_train, best_alpha)
    aic = 2 * edf - 2 * ll_real
    bic = np.log(len(y_test)) * edf - 2 * ll_real

    # === Null model: X shuffled
    ll_shuf_list = []
    for _ in range(n_shuffles):
        X_embed_shuf = X_embed[np.random.permutation(len(X_embed))]
        X_embed_shuf_pca = pca.transform(X_embed_shuf)
        interaction = dur_clean * X_embed_shuf_pca
        X_shuf_full = np.hstack([dur_clean, X_embed_shuf_pca, interaction])
        X_shuf_scaled = StandardScaler().fit_transform(X_shuf_full)

        if test_size is None:
            Xs_train, Xs_test = X_shuf_scaled, X_shuf_scaled
        else:
            Xs_train, Xs_test, _, _ = train_test_split(X_shuf_scaled, y, test_size=test_size, random_state=seed)

        model_shuf = PoissonRegressor(alpha=best_alpha, max_iter=1000)
        model_shuf.fit(Xs_train, y_train)
        pred_shuf = np.clip(model_shuf.predict(Xs_test), 1e-10, None)
        ll_shuf = np.sum(y_test * np.log(pred_shuf) - pred_shuf - gammaln(y_test + 1))
        ll_shuf_list.append(ll_shuf)

    ll_xshuf_mean = np.mean(ll_shuf_list)
    ll_diff = ll_real - ll_xshuf_mean
    p_val_ll_xshuf = np.mean(np.array(ll_shuf_list) >= ll_real)

    # === Pseudo-R2 decomposition
    X_dur = X_full[:, [0]]
    X_sem = X_full[:, 1:1+n_pcs]
    X_inter = X_full[:, 1+n_pcs:]

    def fit_and_score(X_part):
        model = PoissonRegressor(alpha=best_alpha, max_iter=1000)
        model.fit(X_part, y)
        pred = np.clip(model.predict(X_part), 1e-10, None)
        ll = np.sum(y * np.log(pred) - pred - gammaln(y + 1))
        return 1 - (ll / ll_null)

    pseudo_r2_dur = fit_and_score(X_dur)
    pseudo_r2_dur_sem = fit_and_score(np.hstack([X_dur, X_sem]))

    # === Permutation pseudo-R2 nulls
    semantic_null_r2, interaction_null_r2 = [], []
    for _ in range(n_shuffles):
        X_sem_shuf = np.random.permutation(X_sem)
        X_inter_shuf = dur_clean * X_sem_shuf
        X_sem_model = np.hstack([X_dur, X_sem_shuf])
        X_full_model = np.hstack([X_dur, X_sem_shuf, X_inter_shuf])

        semantic_null_r2.append(fit_and_score(X_sem_model))
        interaction_null_r2.append(fit_and_score(X_full_model))

    semantic_gain_pval = np.mean(np.array(semantic_null_r2) >= pseudo_r2_dur_sem)
    interaction_gain_pval = np.mean(np.array(interaction_null_r2) >= pseudo_r2)

    return {
        "neuron": neuron_idx,
        "test_size": test_label,
        "n_pcs": n_pcs,
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
        "ll_real_per_word": ll_real / len(y_test),
        "pseudo_r2": pseudo_r2,
        "pseudo_r2_dur": pseudo_r2_dur,
        "pseudo_r2_dur_sem": pseudo_r2_dur_sem,
        "semantic_gain_pval": semantic_gain_pval,
        "interaction_gain_pval": interaction_gain_pval,
        "pearson_corr": pearson_corr,
        "shift": shift,
        "coef": model_real.coef_.tolist()
    }

## old model without aic##
# def run_model_for_config(neuron_idx, X_embed, dur, Y, test_size, n_pcs, alphas,
#                          n_shuffles=50, seed=42, shift=None):
#     from sklearn.linear_model import PoissonRegressor
#     from sklearn.model_selection import KFold, train_test_split
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.exceptions import ConvergenceWarning
#     from sklearn.decomposition import PCA
#     from scipy.special import gammaln
#     from scipy.stats import pearsonr
#     import numpy as np
#     import warnings
#     warnings.filterwarnings("ignore", category=UserWarning)
#     warnings.filterwarnings("ignore", category=RuntimeWarning)
#     warnings.filterwarnings("ignore", category=ConvergenceWarning)

#     np.random.seed(seed)

#     if shift not in [None, 0]:
#         X_embed, Y, dur = apply_temporal_shift(X_embed, Y, dur, shift)

#     X_full, Y_clean, dur_clean, pca = clean_and_prepare(X_embed, Y, dur, n_pcs)
#     y = Y_clean[:, neuron_idx]

#     if np.std(y) == 0 or np.all(y == 0) or not np.all(np.isfinite(y)):
#         return None

#     # === Inner CV to select best alpha
#     kf_inner = KFold(n_splits=5, shuffle=True, random_state=seed)
#     alpha_scores = {alpha: [] for alpha in alphas}
#     for alpha in alphas:
#         for train_idx, val_idx in kf_inner.split(X_full):
#             model = PoissonRegressor(alpha=alpha, max_iter=1000)
#             model.fit(X_full[train_idx], y[train_idx])
#             pred = np.clip(model.predict(X_full[val_idx]), 1e-10, None)
#             ll = np.sum(y[val_idx] * np.log(pred) - pred - gammaln(y[val_idx] + 1))
#             alpha_scores[alpha].append(ll)

#     avg_ll_per_alpha = {a: np.mean(s) for a, s in alpha_scores.items()}
#     best_alpha = max(avg_ll_per_alpha, key=avg_ll_per_alpha.get)
#     alpha_std = np.std(alpha_scores[best_alpha])

#     # === Outer split
#     if test_size is None:
#         X_train, y_train = X_full, y
#         X_test, y_test = X_full, y
#         test_label = "full"
#     else:
#         X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=test_size, random_state=seed)
#         test_label = test_size

#     # === Final model
#     model_real = PoissonRegressor(alpha=best_alpha, max_iter=1000)
#     model_real.fit(X_train, y_train)
#     pred_real = np.clip(model_real.predict(X_test), 1e-10, None)
#     ll_real = np.sum(y_test * np.log(pred_real) - pred_real - gammaln(y_test + 1))
#     mu = np.mean(y_train)
#     ll_null = np.sum(y_test * np.log(mu) - mu - gammaln(y_test + 1))
#     pseudo_r2 = 1 - (ll_real / ll_null)
#     pearson_corr = pearsonr(y_test, pred_real)[0]

#     # === Null model: X shuffled
#     ll_shuf_list = []
#     for _ in range(n_shuffles):
#         X_embed_shuf = X_embed[np.random.permutation(len(X_embed))]
#         X_embed_shuf_pca = pca.transform(X_embed_shuf)
#         interaction = dur_clean * X_embed_shuf_pca
#         X_shuf_full = np.hstack([dur_clean, X_embed_shuf_pca, interaction])
#         X_shuf_scaled = StandardScaler().fit_transform(X_shuf_full)

#         if test_size is None:
#             Xs_train, Xs_test = X_shuf_scaled, X_shuf_scaled
#         else:
#             Xs_train, Xs_test, _, _ = train_test_split(X_shuf_scaled, y, test_size=test_size, random_state=seed)

#         model_shuf = PoissonRegressor(alpha=best_alpha, max_iter=1000)
#         model_shuf.fit(Xs_train, y_train)
#         pred_shuf = np.clip(model_shuf.predict(Xs_test), 1e-10, None)
#         ll_shuf = np.sum(y_test * np.log(pred_shuf) - pred_shuf - gammaln(y_test + 1))
#         ll_shuf_list.append(ll_shuf)

#     ll_xshuf_mean = np.mean(ll_shuf_list)
#     ll_diff = ll_real - ll_xshuf_mean
#     p_val_ll_xshuf = np.mean(np.array(ll_shuf_list) >= ll_real)

#     # === Pseudo-R2 decomposition
#     X_dur = X_full[:, [0]]
#     X_sem = X_full[:, 1:1+n_pcs]
#     X_inter = X_full[:, 1+n_pcs:]

#     def fit_and_score(X_part):
#         model = PoissonRegressor(alpha=best_alpha, max_iter=1000)
#         model.fit(X_part, y)
#         pred = np.clip(model.predict(X_part), 1e-10, None)
#         ll = np.sum(y * np.log(pred) - pred - gammaln(y + 1))
#         return 1 - (ll / ll_null)

#     pseudo_r2_dur = fit_and_score(X_dur)
#     pseudo_r2_dur_sem = fit_and_score(np.hstack([X_dur, X_sem]))

#     # === Permutation pseudo-R2 nulls
#     semantic_null_r2, interaction_null_r2 = [], []
#     for _ in range(n_shuffles):
#         X_sem_shuf = np.random.permutation(X_sem)
#         X_inter_shuf = dur_clean * X_sem_shuf
#         X_sem_model = np.hstack([X_dur, X_sem_shuf])
#         X_full_model = np.hstack([X_dur, X_sem_shuf, X_inter_shuf])

#         semantic_null_r2.append(fit_and_score(X_sem_model))
#         interaction_null_r2.append(fit_and_score(X_full_model))

#     semantic_gain_pval = np.mean(np.array(semantic_null_r2) >= pseudo_r2_dur_sem)
#     interaction_gain_pval = np.mean(np.array(interaction_null_r2) >= pseudo_r2)

#     return {
#         "neuron": neuron_idx,
#         "test_size": test_label,
#         "n_pcs": n_pcs,
#         "best_alpha": best_alpha,
#         "alpha_ll_mean": avg_ll_per_alpha[best_alpha],
#         "alpha_ll_std": alpha_std,
#         "ll_real": ll_real,
#         "ll_xshuf_mean": ll_xshuf_mean,
#         "ll_diff": ll_diff,
#         "p_val_ll_xshuf": p_val_ll_xshuf,
#         "ll_real_per_word": ll_real / len(y_test),
#         "pseudo_r2": pseudo_r2,
#         "pseudo_r2_dur": pseudo_r2_dur,
#         "pseudo_r2_dur_sem": pseudo_r2_dur_sem,
#         "semantic_gain_pval": semantic_gain_pval,
#         "interaction_gain_pval": interaction_gain_pval,
#         "pearson_corr": pearson_corr,
#         "shift": shift,
#     }


#### This one does the sweeping of parameters ####
from itertools import product
from joblib import Parallel, delayed
import pandas as pd

def sweep_all_configs(X_embed, dur, Y, test_sizes, n_pcs_list, alpha_grid, n_shuffles=50, n_jobs=-1):
    n_neurons = Y.shape[1]
    configs = list(product(test_sizes, n_pcs_list))

    def run_all_for_neuron(neuron_idx):
        results = []
        for test_size, n_pcs in configs:
            result = run_model_for_config(
                neuron_idx=neuron_idx,
                X_embed=X_embed,
                dur=dur,
                Y=Y,
                test_size=test_size,
                n_pcs=n_pcs,
                alphas=alpha_grid,
                n_shuffles=n_shuffles,
                seed=42
            )
            results.append(result)
        return results

    all_results = Parallel(n_jobs=n_jobs)(
        delayed(run_all_for_neuron)(i) for i in range(n_neurons)
    )
    
    # Only keep valid per-neuron result lists
    valid_results = [neuron_results for neuron_results in all_results if neuron_results is not None]

    # Flatten and clean
    flat_results = [r for neuron_results in valid_results for r in neuron_results if r is not None]


    return pd.DataFrame(flat_results)



### This does temporal sweeps ###

def sweep_all_shifts(X_embed, dur, Y, shift_range, test_size, n_pcs, alpha_grid,
                     n_shuffles=50, n_jobs=-1):
    """
    Run regression across a range of temporal shifts and return per-neuron, per-shift results.
    """
    from joblib import Parallel, delayed
    from tqdm import tqdm

    n_neurons = Y.shape[1]

    def run_shift_for_neuron(neuron_idx):
        results = []
        for shift in shift_range:
            result = run_model_for_config(
                neuron_idx=neuron_idx,
                X_embed=X_embed,
                dur=dur,
                Y=Y,
                test_size=test_size,
                n_pcs=n_pcs,
                alphas=alpha_grid,
                n_shuffles=n_shuffles,
                seed=42,
                shift=shift
            )
            results.append(result)
        return results

    all_results = Parallel(n_jobs=n_jobs)(
        delayed(run_shift_for_neuron)(i) for i in tqdm(range(n_neurons))
    )

    flat_results = [r for neuron_results in all_results for r in neuron_results]
    return pd.DataFrame(flat_results)


