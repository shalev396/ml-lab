"""The two experiments: LogisticRegression (balanced) and the PyTorch MLP from ../model/model.py."""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression

import model as M  # ../model/model.py

from . import utils

LOGREG = "logistic_regression"
MLP = "pytorch_mlp"


def build_logreg(cfg) -> LogisticRegression:
    """Linear baseline on standardized flags; class_weight="balanced" offsets the ~4 % positives."""
    return LogisticRegression(max_iter=cfg.logreg_max_iter, class_weight="balanced", random_state=cfg.seed)


def build_mlp(cfg, feature_names: list[str] = M.FEATURES) -> M.PropensityMLP:
    """`model.PropensityMLP`: Linear/ReLU/Dropout per hidden layer -> 1 logit (random init)."""
    utils.set_seeds(cfg.seed, "torch")
    return M.PropensityMLP(feature_names=list(feature_names), hidden_units=list(cfg.hidden_units),
                           dropout=cfg.dropout)


def count_params(name: str, estimator) -> int:
    """Learned parameters: coefficients + intercept for the logistic regression, weights for the MLP."""
    if name == LOGREG:
        n_features = getattr(estimator, "n_features_in_", len(M.FEATURES))
        return int(n_features + 1)
    return utils.count_params(estimator)


def describe(name: str, estimator) -> str:
    if name == LOGREG:
        return f"LogisticRegression(class_weight='balanced', max_iter={estimator.max_iter})"
    dims = [len(estimator.feature_names), *estimator.hidden_units, 1]
    return "MLP " + " -> ".join(map(str, dims)) + " (ReLU + dropout)"
