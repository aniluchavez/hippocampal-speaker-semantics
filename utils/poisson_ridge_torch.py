import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.special import gammaln
import warnings

warnings.filterwarnings("ignore")


class PoissonRegressionTorch(nn.Module):
    """
    Simple ridge-regularized Poisson regression model with log-link.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x):
        return torch.exp(self.linear(x))  # Poisson log-link


def poisson_loss(y_pred, y_true):
    """
    Compute the negative log-likelihood for Poisson regression.
    """
    return torch.sum(y_pred - y_true * torch.log(y_pred + 1e-10))


def train_poisson_model(X, y, alpha=0.0, lr=0.01, epochs=300, tol=1e-4, patience=10, device=None):
    """
    Train a Poisson regression model using L2 regularization (ridge).
    Uses early stopping based on loss improvement tolerance.
    """
    # Use given device or infer from X
    device = device or X.device
    X, y = X.to(device), y.to(device)

    model = PoissonRegressionTorch(X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")
    patience_counter = 0

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X).squeeze()
        loss = poisson_loss(y_pred, y)
        l2_penalty = sum(torch.norm(p) ** 2 for p in model.parameters())
        total_loss = loss + alpha * l2_penalty
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
    """
    Predict output from a trained Poisson model.
    """
    model.eval()
    with torch.no_grad():
        return model(X).squeeze()


def select_best_alpha_torch(X, y, alphas, k_folds=5):
    """
    Perform cross-validation to select best L2 alpha (regularization strength).
    """
    best_alpha, best_score = alphas[0], -np.inf
    kf = KFold(n_splits=k_folds)

    for alpha in alphas:
        scores = []
        for train_idx, val_idx in kf.split(X):
            model = train_poisson_model(X[train_idx], y[train_idx], alpha=alpha)
            pred = predict_poisson_model(model, X[val_idx]).cpu().numpy()
            target = y[val_idx].cpu().numpy()
            pred = np.clip(pred, 1e-10, None)
            ll = np.sum(target * np.log(pred) - pred - gammaln(target + 1))
            scores.append(ll)
        avg_ll = np.mean(scores)
        if avg_ll > best_score:
            best_score = avg_ll
            best_alpha = alpha

    return best_alpha


def calculate_effective_dof_poisson(X, y_pred, alpha):
    """
    Estimate the effective degrees of freedom for ridge Poisson regression.
    """
    W_sqrt = torch.sqrt(y_pred)
    WX = W_sqrt.unsqueeze(1) * X
    XtWX = WX.T @ WX
    XtWX_reg = XtWX + alpha * torch.eye(X.shape[1], device=X.device)
    L = torch.linalg.cholesky(XtWX_reg)
    XtWX_inv = torch.cholesky_inverse(L)
    edf = torch.trace(XtWX_inv @ XtWX).item()
    return edf
