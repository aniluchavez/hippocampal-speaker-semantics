import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import os
import pickle
from collections import defaultdict
import re
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.special import gammaln
from scipy.stats import spearmanr 
from sklearn.model_selection import KFold
import math
from joblib import Parallel, delayed


def cleanX(X, Y, duration, n_shuffles=100):

    # Remove rows with NaNs
    remove = np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1)
    X = X[~remove, :]
    Y = Y[~remove, :]
    duration = duration[~remove]

    # Ensure duration is (n, 1)
    duration = np.squeeze(duration)
    duration = duration.reshape(-1, 1)

    # Create interactions
    interactions = X * duration

    # Construct original design matrix and z-score
    X_full = np.hstack([duration, X, interactions])
    X_full = StandardScaler().fit_transform(X_full)

    # Make shuffled versions
    Xs_list = []
    for _ in range(n_shuffles):
        perm = np.random.permutation(X.shape[0])
        X_shuff = X[perm, :]
        interactions_shuff = X_shuff * duration
        Xs = np.hstack([duration, X_shuff, interactions_shuff])
        Xs = StandardScaler().fit_transform(Xs)
        Xs_list.append(Xs)

    return X_full, Xs_list, Y


def extract_index(filename):
    match = re.search(r"(\d+)\.pkl$", filename)
    return int(match.group(1)) if match else -1


def write2csv(output_dir, key):
    all_results = []
    ridge_coeff_accumulator = {}
    ridge_deviance_accumulator = {}

    # Get sorted list of .pkl files by numeric index
    pkl_files = [f for f in os.listdir("neuron_outputs") if f.endswith(".pkl")]
    pkl_files.sort(key=extract_index)

    for path in pkl_files:
        full_path = os.path.join("neuron_outputs", path)
        with open(full_path, "rb") as f:
            data = pickle.load(f)
            all_results.extend(data["results"])
            ridge_coeff_accumulator[data["neuron_idx"]] = data["ridge_coefs"]
            ridge_deviance_accumulator[data["neuron_idx"]] = data["deviance_residuals"]
        os.remove(full_path)

    # Save summary CSV
    df = pd.DataFrame(all_results)
    df.to_csv(f"{output_dir}/{key}.csv", index=False)

    # === Sanitize coefficient and deviance dictionaries ===
    def sanitize_dict_length(d):
        min_len = min(len(v) for v in d.values())
        return {k: v[:min_len] for k, v in d.items()}

    if ridge_coeff_accumulator:
        sanitized_coefs = sanitize_dict_length(ridge_coeff_accumulator)
        pd.DataFrame(sanitized_coefs).to_csv(f"{output_dir}/{key}_Coeffs.csv", index=False)

    if ridge_deviance_accumulator:
        sanitized_devs = sanitize_dict_length(ridge_deviance_accumulator)
        pd.DataFrame(sanitized_devs).to_csv(f"{output_dir}/{key}_Deviance.csv", index=False)

    return


# Custom Poisson regression with L2 regularization using PyTorch
class PoissonRegressionTorch(nn.Module):
    def __init__(self, input_dim):
        super(PoissonRegressionTorch, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x):
        return torch.exp(self.linear(x))

def poisson_loss(y_pred, y_true):
    return torch.sum(y_pred - y_true * torch.log(y_pred + 1e-10))

def train_poisson_model(X, y, alpha=0.0, lr=0.01, epochs=300, tol=1e-4, patience=10):
    model = PoissonRegressionTorch(X.shape[1]).to(X.device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X)
        loss = poisson_loss(y_pred.squeeze(), y)
        l2_reg = sum(torch.norm(param) ** 2 for param in model.parameters())
        total_loss = loss + alpha * l2_reg
        total_loss.backward()
        optimizer.step()

        if total_loss.item() < best_loss - tol:
            best_loss = total_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return model

def predict_poisson_model(model, X):
    model.eval()
    with torch.no_grad():
        return model(X).squeeze()

def select_best_alpha_torch(X, y, alphas, k_folds=5):
    device = X.device
    kf = KFold(n_splits=k_folds)
    best_alpha = alphas[0]
    best_score = -np.inf

    for alpha in alphas:
        ll_scores = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            model = train_poisson_model(X_train, y_train, alpha=alpha)
            with torch.no_grad():
                y_pred = predict_poisson_model(model, X_val).cpu().numpy()
            y_val_np = y_val.cpu().numpy()
            y_pred = np.clip(y_pred, 1e-10, None)
            ll = np.sum(y_val_np * np.log(y_pred) - y_pred - gammaln(y_val_np + 1))
            ll_scores.append(ll)

        avg_ll = np.mean(ll_scores)
        if avg_ll > best_score:
            best_score = avg_ll
            best_alpha = alpha

    return best_alpha

def estimate_dof_poisson(model, X_torch, eps=1e-6):
    """
    Approximates effective degrees of freedom (edf) for ridge-penalized Poisson regression.
    """
    model.eval()
    with torch.no_grad():
        y_pred = torch.exp(model(X_torch))  # Apply exp to get Poisson means
        edf_approx = torch.sum(y_pred / (y_pred + eps)).item() / X_torch.shape[0]
    return edf_approx


def calculate_effective_dof_poisson(X_torch, y_pred_torch, lam):
    """
    Computes effective degrees of freedom (edf) for Poisson ridge regression.
    Falls back to pseudo-inverse approximation if Cholesky fails.
    """
    n_samples, n_features = X_torch.shape
    y_pred_torch = y_pred_torch.flatten()
    W_sqrt = torch.sqrt(y_pred_torch)

    WX = W_sqrt.unsqueeze(1) * X_torch
    XtWX = WX.T @ WX
    I = torch.eye(n_features, device=X_torch.device)
    XtWX_plus_lambdaI = XtWX + lam * I + 1e-6 * I

    try:
        # Try Cholesky
        L = torch.linalg.cholesky(XtWX_plus_lambdaI)
        XtWX_inv = torch.cholesky_inverse(L)
    except RuntimeError:
        # Fallback to pseudo-inverse
        XtWX_inv = torch.linalg.pinv(XtWX_plus_lambdaI)
        print("[WARN] Cholesky failed — using pseudo-inverse instead")

    # Compute EDF
    edf = torch.sum(torch.diagonal(torch.matmul(XtWX_inv, XtWX))).item()
    return edf


def process_neuron_torch(neuron_idx, Y, X, Xs_list, output_dir="neuron_outputs"):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        y = Y[:, neuron_idx]
        if np.std(y) == 0 or np.all(y == 0):
            return

        y_torch = torch.tensor(y, dtype=torch.float32).to(device)
        X_torch = torch.tensor(X, dtype=torch.float32).to(device)

        # Alpha selection
        coarse_alphas = np.logspace(-5, 3, 9)
        best_alpha_coarse = select_best_alpha_torch(X_torch, y_torch, coarse_alphas)
        log_best = math.log10(best_alpha_coarse)
        fine_alphas = np.logspace(log_best - 1, log_best + 1, 10)
        best_alpha = select_best_alpha_torch(X_torch, y_torch, fine_alphas)

        model = train_poisson_model(X_torch, y_torch, alpha=best_alpha)
        pred_test_torch = predict_poisson_model(model, X_torch)
        pred_test = pred_test_torch.cpu().numpy()
        pred_test = np.clip(pred_test, 1e-10, None)
        y_clipped = np.clip(y, 1e-10, None)
        pearson_r = np.corrcoef(y_clipped, pred_test)[0, 1]
        spearman_r = pd.Series(y_clipped).corr(pd.Series(pred_test), method="spearman")

        deviance_residuals = 2 * (y * np.log(y_clipped / pred_test) - (y - pred_test))
        edf = calculate_effective_dof_poisson(X_torch, pred_test_torch, best_alpha)

        n = len(y)
        y_mean = np.mean(y_clipped)
        null_pred = np.full_like(y_clipped, y_mean)
        null_deviance = 2 * np.sum(y * np.log(y_clipped / null_pred) - (y - null_pred))
        deviance = np.sum(deviance_residuals)

        ll_real = np.sum(y * np.log(pred_test) - pred_test - gammaln(y + 1))
        BIC = np.log(n) * edf - 2 * ll_real
        AIC = 2 * edf - 2 * ll_real
        pseudo_r2 = 1 - deviance / null_deviance

        ridge_coefs = model.linear.weight.detach().cpu().numpy().flatten()

        # Null models
        ll_shuf_list = []
        pearson_null_list = []
        spearman_null_list = []
        Xs_torch_list = [torch.tensor(Xs, dtype=torch.float32, device=device) for Xs in Xs_list]
        for Xs_torch in Xs_torch_list:
            null_model = train_poisson_model(Xs_torch, y_torch, alpha=best_alpha)
            null_pred = predict_poisson_model(null_model, Xs_torch).cpu().numpy()
            null_pred = np.clip(null_pred, 1e-10, None)
            ll_shuf = np.sum(y * np.log(null_pred) - null_pred - gammaln(y + 1))
            ll_shuf_list.append(ll_shuf)
            pearson_null = np.corrcoef(y_clipped, null_pred)[0, 1]
            spearman_null = spearmanr(y_clipped, null_pred).correlation
            pearson_null_list.append(pearson_null)
            spearman_null_list.append(spearman_null)

        ll_shuf_avg = np.mean(ll_shuf_list)
        pval_ll = np.mean(np.array(ll_shuf_list) >= ll_real)
        pearson_null_avg = np.mean(pearson_null_list)
        spearman_null_avg = np.mean(spearman_null_list)

        # Save
        results = [{
            'neuron': neuron_idx,
            'best_alpha': best_alpha,
            'll_real': ll_real,
            'll_shuf': ll_shuf_avg,
            'll_diff': ll_real - ll_shuf_avg,
            'pval_ll': pval_ll,
            'pseudo_r2': pseudo_r2,
            'pearson_r': pearson_r,
            'pearson_r_null': pearson_null_avg,
            'spearman_r': spearman_r,
            'spearman_r_null': spearman_null_avg,
            'AIC': AIC,
            'BIC': BIC,
        }]

        output_path = os.path.join(output_dir, f"neuron_{neuron_idx:03d}.pkl")
        with open(output_path, "wb") as f:
            pickle.dump({
                'neuron_idx': neuron_idx,
                'ridge_coefs': ridge_coefs,
                'results': results,
                'deviance_residuals': deviance_residuals,
            }, f)

        return output_path

    except Exception as e:
        print(f"[ERROR] Neuron {neuron_idx}: {e}")
        return None
