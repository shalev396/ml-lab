"""Every candidate model of the experiment. The deployed pieces (MLP architecture, preprocessing)
come from model/model.py; the classic-ML variants and resampling pipelines are defined here.

Variants (same grid as the original ProjectPro brief):
    baseline_{logreg,rf}                      imbalanced data, no resampling
    nearmiss_{logreg,knn,dtree,rf,svc}        NearMiss-1 undersampling (kernel SVC only here: it is
                                              too slow on the full data)
    smote_{logreg,knn,dtree,rf}               SMOTE oversampling
    keras_mlp                                 Keras MLP with balanced class weights (see engine.py)
Resampling sits inside an imblearn Pipeline, so it only ever touches the rows the model is fit on.
"""
from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import NearMiss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import model as M  # ../model/model.py (put on sys.path by src/__init__.py)

from .config import Config

MLP_NAME = "keras_mlp"


def classifiers(cfg: Config) -> dict:
    return {
        "logreg": LogisticRegression(max_iter=cfg.logreg_max_iter, random_state=cfg.seed),
        "knn": KNeighborsClassifier(n_neighbors=cfg.knn_neighbors, n_jobs=-1),
        "dtree": DecisionTreeClassifier(random_state=cfg.seed),
        "rf": RandomForestClassifier(n_estimators=cfg.rf_n_estimators, n_jobs=-1, random_state=cfg.seed),
        # Platt-scaled RBF SVM (the sklearn>=1.9 replacement for SVC(probability=True))
        "svc": CalibratedClassifierCV(SVC(random_state=cfg.seed), method="sigmoid", ensemble=False),
    }


def samplers(cfg: Config) -> dict:
    return {
        "nearmiss": NearMiss(version=1),
        "smote": SMOTE(k_neighbors=cfg.smote_k_neighbors, random_state=cfg.seed),
    }


def build_variants(cfg: Config) -> dict:
    """{name: unfitted estimator} for every scikit-learn / imbalanced-learn candidate."""
    grid = {"baseline": ("logreg", "rf"),
            "nearmiss": ("logreg", "knn", "dtree", "rf", "svc"),
            "smote": ("logreg", "knn", "dtree", "rf")}
    variants = {}
    for sampling, names in grid.items():
        for name in names:
            clf = classifiers(cfg)[name]  # fresh instances for every variant
            variants[f"{sampling}_{name}"] = clf if sampling == "baseline" else Pipeline(
                [("sampler", samplers(cfg)[sampling]), ("clf", clf)])
    return variants


def build_mlp(cfg: Config):
    """The deployed MLP architecture from model.py, sized by the config (compiled in engine.py)."""
    return M.build_mlp(len(M.FEATURE_COLUMNS), cfg.hidden_units, cfg.dropout)


def for_inference(estimator):
    """Drop the resampler: a fitted resampling pipeline predicts with its final classifier alone."""
    return estimator.named_steps["clf"] if isinstance(estimator, Pipeline) else estimator


def describe(name: str, estimator) -> str:
    """Human-readable one-liner for cards and logs, e.g. 'RandomForestClassifier (100 trees) + SMOTE'."""
    if name == MLP_NAME:
        widths = [estimator.input_shape[-1]] + [layer.units for layer in estimator.layers if hasattr(layer, "units")]
        return f"Keras MLP {'-'.join(map(str, widths))} (balanced class weights)"
    clf = for_inference(estimator)
    text = type(clf).__name__
    if isinstance(clf, CalibratedClassifierCV):
        text = f"{type(clf.estimator).__name__} (Platt-calibrated)"
    if hasattr(clf, "n_estimators"):
        text += f" ({clf.n_estimators} trees)"
    if isinstance(estimator, Pipeline):
        text += f" + {type(estimator.named_steps['sampler']).__name__}"
    return text


def count_params(name: str, estimator) -> int | None:
    """Learned parameters for linear models and the MLP; None for trees / kNN / kernel SVMs."""
    if name == MLP_NAME:
        return int(estimator.count_params())
    clf = for_inference(estimator)
    if hasattr(clf, "coef_"):
        return int(clf.coef_.size + clf.intercept_.size)
    return None
