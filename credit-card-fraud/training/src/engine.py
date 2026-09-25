"""Training + evaluation steps, called one by one from the notebook:

    prepare_features -> fit_variants / compile_mlp + fit_mlp -> evaluate_all -> select

Protocol (the test split never influences a choice):
    1. fit every candidate on the train split (60 %)
    2. validation split (20 %): per-model decision threshold = max F1; model score = PR-AUC
    3. export the best scikit-learn variant + the Keras MLP; default = higher validation PR-AUC
    4. report test-split (20 %) metrics of every variant at its own tuned threshold
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import keras
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import model as M  # ../model/model.py (put on sys.path by src/__init__.py)

from .config import Config
from .data_setup import Splits
from .model_builder import MLP_NAME


# --------------------------------------------------------------------------- features
def prepare_features(splits: Splits):
    """Fit the RobustScaler on the train rows only; return (scaler, x, y) with x/y keyed by split."""
    scaler = M.make_scaler().fit(splits.train[M.SCALED_COLUMNS])
    x = {name: M.to_features(part, scaler) for name, part in splits.items()}
    y = {name: part[M.TARGET].to_numpy() for name, part in splits.items()}
    return scaler, x, y


# --------------------------------------------------------------------------- metrics
def binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    """Threshold-free ranking metrics (PR-AUC, ROC-AUC) + label metrics at `threshold`."""
    pred = prob >= threshold
    return {
        "pr_auc": float(average_precision_score(y_true, prob)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
    }


def confusion(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, prob >= threshold, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def best_f1_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    """The probability cut-off that maximises F1 on (validation) data; `prob >= t` means fraud."""
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[int(np.argmax(f1[:-1]))])  # the last PR point has no threshold


@dataclass
class Evaluation:
    """One candidate: threshold + selection score from validation, a single report on test."""
    name: str
    threshold: float
    val: dict[str, float]
    test: dict[str, float]
    val_prob: np.ndarray = field(repr=False)
    test_prob: np.ndarray = field(repr=False)


def evaluate(name: str, estimator, x: dict[str, np.ndarray], y: dict[str, np.ndarray]) -> Evaluation:
    """Tune the threshold on validation, then score validation and test at that threshold."""
    val_prob, test_prob = (M.positive_proba(estimator, x[s]) for s in ("val", "test"))
    threshold = best_f1_threshold(y["val"], val_prob)
    return Evaluation(name, threshold, binary_metrics(y["val"], val_prob, threshold),
                      binary_metrics(y["test"], test_prob, threshold), val_prob, test_prob)


# --------------------------------------------------------------------------- Keras
def compile_mlp(model: keras.Model, cfg: Config) -> keras.Model:
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg.lr), loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(curve="PR", name="pr_auc"), keras.metrics.AUC(name="roc_auc")])
    return model


def fit_mlp(model: keras.Model, x_train: np.ndarray, y_train: np.ndarray, cfg: Config) -> keras.callbacks.History:
    """Balanced class weights + early stopping on a stratified slice of the TRAIN split, so the
    validation split stays untouched for model selection and threshold tuning (like every variant)."""
    x_fit, x_stop, y_fit, y_stop = train_test_split(x_train, y_train, test_size=cfg.early_stopping_split,
                                                    stratify=y_train, random_state=cfg.seed)
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_fit)  # ~0.5 legit, ~290 fraud
    stop = keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=cfg.patience,
                                         restore_best_weights=True)
    return model.fit(x_fit, y_fit.astype(np.float32), validation_data=(x_stop, y_stop.astype(np.float32)),
                     epochs=cfg.epochs, batch_size=cfg.batch_size,
                     class_weight={i: float(w) for i, w in enumerate(weights)}, callbacks=[stop], verbose=2)


# --------------------------------------------------------------------------- the experiment
def fit_variants(variants: dict, x: dict[str, np.ndarray], y: dict[str, np.ndarray]) -> dict:
    """Fit every scikit-learn / imblearn candidate on the train split; returns {name: fitted estimator}."""
    fitted = {}
    for name, estimator in variants.items():
        tic = time.perf_counter()
        fitted[name] = estimator.fit(x["train"], y["train"])
        print(f"  fit {name:<16} {time.perf_counter() - tic:6.1f} s")
    return fitted


def evaluate_all(fitted: dict, x: dict[str, np.ndarray], y: dict[str, np.ndarray]) -> dict[str, Evaluation]:
    """Threshold tuned on validation, then validation + test metrics for every candidate."""
    evals = {name: evaluate(name, estimator, x, y) for name, estimator in fitted.items()}
    for e in sorted(evals.values(), key=lambda e: e.val["pr_auc"], reverse=True):
        print(f"  {e.name:<16} val PR-AUC {e.val['pr_auc']:.4f}  F1 {e.val['f1']:.4f} @ {e.threshold:.3f}"
              f"  | test PR-AUC {e.test['pr_auc']:.4f}  F1 {e.test['f1']:.4f}")
    return evals


def select(evals: dict[str, Evaluation]) -> tuple[str, list[str]]:
    """Model selection on VALIDATION PR-AUC only. Returns (default model, exported models): the best
    scikit-learn variant and the Keras MLP are both exported; the default is the higher of the two."""
    sklearn_best = max((n for n in evals if n != MLP_NAME), key=lambda n: evals[n].val["pr_auc"])
    exported = [sklearn_best, MLP_NAME] if MLP_NAME in evals else [sklearn_best]
    best = max(exported, key=lambda n: evals[n].val["pr_auc"])
    print(f"[select] default model: {best} (val PR-AUC {evals[best].val['pr_auc']:.4f}, "
          f"threshold {evals[best].threshold:.3f}); exported: {', '.join(exported)}")
    return best, exported


def comparison_table(evals: dict[str, Evaluation]):
    """Every variant ranked by validation PR-AUC (the selection metric), with its test metrics."""
    import pandas as pd

    rows = {e.name: {"val_pr_auc": e.val["pr_auc"], "threshold": e.threshold,
                     **{f"test_{k}": v for k, v in e.test.items()}} for e in evals.values()}
    return pd.DataFrame(rows).T.sort_values("val_pr_auc", ascending=False)
