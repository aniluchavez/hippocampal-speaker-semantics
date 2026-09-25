# ccgp_decode/models.py
from __future__ import annotations
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# optional XGBoost
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

@dataclass
class ModelSpec:
    name: str
    estimator: Any
    param_grid: Optional[Dict[str, Any]] = None
    use_pca: bool = False
    pca_var: float = 0.95
    needs_coef: bool = False
    preprocess: str = "standard"   # "standard" | "scale" | "none"

def make_pipeline(spec: ModelSpec) -> Pipeline:
    steps = []
    preprocess = getattr(spec, "preprocess", "standard")

    if preprocess in ("scale", "standard"):
        steps.append(("scaler", StandardScaler()))
    elif preprocess == "none":
        pass
    else:
        raise ValueError(f"Unknown preprocess='{preprocess}'")

    if getattr(spec, "use_pca", False):
        steps.append(("pca", PCA(n_components=spec.pca_var)))

    steps.append(("clf", spec.estimator))
    return Pipeline(steps)

class PoissonIndependentNB(BaseEstimator, ClassifierMixin):
    """
    Multiclass independent Poisson Naive Bayes for spike COUNTS (nonnegative).
    """
    def __init__(self, alpha=1e-3, class_prior="empirical"):
        self.alpha = float(alpha)
        self.class_prior = class_prior

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        if np.any(X < 0):
            raise ValueError("PoissonIndependentNB requires nonnegative counts.")

        self.classes_ = np.unique(y)
        K = len(self.classes_)
        n_features = X.shape[1]

        lambdas = np.zeros((K, n_features), dtype=float)
        priors = np.zeros(K, dtype=float)

        for i, c in enumerate(self.classes_):
            Xc = X[y == c]
            if Xc.shape[0] == 0:
                raise RuntimeError(f"Class {c} has zero samples in fit.")
            lambdas[i] = Xc.mean(axis=0) + self.alpha

            if self.class_prior == "uniform":
                priors[i] = 1.0 / K
            else:
                priors[i] = Xc.shape[0] / X.shape[0]

        self.lambdas_ = lambdas
        self.log_lambdas_ = np.log(lambdas)
        self.log_priors_ = np.log(priors + 1e-30)
        return self

    def _log_joint(self, X):
        X = np.asarray(X)
        return (X @ self.log_lambdas_.T) - np.sum(self.lambdas_, axis=1)[None, :] + self.log_priors_[None, :]

    def predict(self, X):
        scores = self._log_joint(X)
        idx = np.argmax(scores, axis=1)
        return self.classes_[idx]

    def predict_proba(self, X):
        scores = self._log_joint(X)
        scores = scores - scores.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

def default_model_specs(seed=0):
    specs = []

    specs.append(
        ModelSpec(
            name="mnlogreg_L2",
            estimator=LogisticRegression(
                solver="lbfgs",
                max_iter=5000,
                class_weight="balanced",
            ),
            param_grid={"clf__C": np.logspace(-2, 4, 9)},
            use_pca=True,
            needs_coef=True,
            preprocess="standard",
        )
    ) 
    specs.append(
        ModelSpec(
            name="mnlogreg_no_reg",
            estimator=LogisticRegression(
                solver="lbfgs",
                max_iter=5000,
                class_weight="balanced",
                C=1e6,   # effectively no regularization
            ),
            param_grid=None,  # no hyperparameter tuning
            use_pca=False,
            needs_coef=True,
            preprocess="standard",
        )
    )

    specs.append(ModelSpec(
        name="mnlogreg_L2_bal",
        estimator=LogisticRegression(
            solver="lbfgs",
            max_iter=10000,
            class_weight="balanced",
        ),
        param_grid={"clf__C": np.logspace(-4, 3, 15)},
        use_pca=False,
        needs_coef=True,
        preprocess="standard",
    ))

    # 2) L2, no class_weight (sometimes better when using balanced-acc scoring)
    specs.append(ModelSpec(
        name="mnlogreg_L2_unw",
        estimator=LogisticRegression(
            solver="lbfgs",
            max_iter=10000,
            class_weight=None,
        ),
        param_grid={"clf__C": np.logspace(-4, 3, 15)},
        use_pca=False,
        needs_coef=True,
        preprocess="standard",
    ))

    # 3) Elastic net (still linear; often helps in low-data/high-dim)
    specs.append(ModelSpec(
        name="mnlogreg_elasticnet",
        estimator=LogisticRegression(
            solver="saga",
            penalty="elasticnet",
            max_iter=3000,
            tol=1e-3,  
            class_weight="balanced",
        ),
        param_grid={
            "clf__C": np.logspace(-2, 1, 5),
            "clf__l1_ratio": [0.0, 0.2, 0.5, 0.8],
        },
        use_pca=False,
        needs_coef=True,
        preprocess="standard",
    ))



    specs.append(
        ModelSpec(
            name="poisson_indep_nb",
            estimator=PoissonIndependentNB(alpha=1e-3, class_prior="empirical"),
            param_grid={"clf__alpha": [1e-6, 1e-4, 1e-3, 1e-2]},
            use_pca=False,
            needs_coef=False,
            preprocess="none",
        )
    )

    specs.append(
    ModelSpec(
        name="linear_svm",
        estimator=LinearSVC(
            class_weight="balanced",
            max_iter=10000,
            dual=False,  # better when n_samples > n_features
        ),
        param_grid={"clf__C": np.logspace(-3, 3, 7)},
        use_pca=False,
        needs_coef=True,
        preprocess="standard",
    )
    )
    specs.append(
    ModelSpec(
        name="lda",
        estimator=LinearDiscriminantAnalysis(solver="lsqr",shrinkage='auto'),
        param_grid=None,
        use_pca=False,
        needs_coef=True,
        preprocess="standard",
    )
    )

 #### Start of non-linear stuff #####
    specs.append(
        ModelSpec(
            name="rf",
            estimator=RandomForestClassifier(
                n_estimators=600,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=1,
                random_state=seed,
            ),
            param_grid={
                "clf__n_estimators": [300, 600],
                "clf__max_depth": [None, 8, 16],
                "clf__min_samples_leaf": [1, 2, 5],
                "clf__max_features": ["sqrt", "log2", None],
            },
            preprocess="none",
        )
    )

    specs.append(
        ModelSpec(
            name="svm_rbf_pca95",
            estimator=SVC(kernel="rbf", class_weight="balanced", probability=False),
            param_grid={"clf__C": np.logspace(-2, 3, 6), "clf__gamma": ["scale", "auto"]},
            use_pca=True,
            pca_var=0.95,
            preprocess="scale",
        )
    )

    if _HAS_XGB:
        specs.append(
            ModelSpec(
                name="xgb_multiclass",
                estimator=XGBClassifier(
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    tree_method="hist",
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    min_child_weight=1.0,
                    n_jobs=1,
                    random_state=seed,
                ),
                param_grid={
                    "clf__max_depth": [3, 5],
                    "clf__n_estimators": [300, 600],
                    "clf__learning_rate": [0.05, 0.1],
                    "clf__subsample": [0.8],
                    "clf__colsample_bytree": [0.8],
                },
                preprocess="none",
            )
        )

    return specs