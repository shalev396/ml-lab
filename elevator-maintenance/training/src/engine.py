"""Fitting + evaluation: RandomForest on features, Keras nets on raw windows (class-weighted). The decision
threshold is tuned for max F1 on VALIDATION; metrics are reported on the untouched TEST segment."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score)

from .config import Config
from .data_setup import SPLITS, Windows
from .model_builder import RF


def class_weight(y_train: np.ndarray) -> dict[int, float]:
    """{0: 1, 1: neg/pos} (capped at 25): the standard imbalance correction for the Keras nets."""
    pos = max(1, int(y_train.sum()))
    return {0: 1.0, 1: float(min((len(y_train) - pos) / pos, 25.0))}


def fit_forest(forest, feats: dict[str, np.ndarray], windows: Windows) -> float:
    """Fit on the train windows' features; returns the wall time in seconds."""
    start = time.perf_counter()
    forest.fit(feats["train"], windows.y["train"])
    return time.perf_counter() - start


def fit_keras(net, windows: Windows, cfg: Config, verbose: int = 2) -> tuple[dict, float]:
    """Adam + binary cross-entropy, class weights, early stopping on validation PR-AUC (best weights kept).
    Returns (history dict, wall time in seconds)."""
    import keras

    net.compile(optimizer=keras.optimizers.Adam(cfg.lr), loss="binary_crossentropy",
                metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")])
    stop = keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=cfg.patience,
                                         restore_best_weights=True)
    start = time.perf_counter()
    history = net.fit(windows.scaled["train"], windows.y["train"],
                      validation_data=(windows.scaled["val"], windows.y["val"]),
                      epochs=cfg.epochs, batch_size=cfg.batch_size, class_weight=class_weight(windows.y["train"]),
                      callbacks=[stop], verbose=verbose)
    return history.history, time.perf_counter() - start


def probabilities(name: str, model, windows: Windows, feats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """P(failure) per split for any of the three experiments."""
    if name == RF:
        return {s: model.predict_proba(feats[s])[:, 1] for s in SPLITS}
    return {s: model.predict(windows.scaled[s], batch_size=1024, verbose=0).ravel() for s in SPLITS}


def best_f1_threshold(y_true, prob, lo: float = 0.05, hi: float = 0.95) -> float:
    """Threshold with max F1 on (validation) data, restricted to [lo, hi] so a few extreme
    probabilities cannot push the operating point to a degenerate 0/1; 0.5 if nothing qualifies."""
    y_true = np.asarray(y_true).astype(int)
    if not y_true.any():
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    valid = np.flatnonzero((thresholds >= lo) & (thresholds <= hi))
    return float(thresholds[valid[np.argmax(f1[valid])]]) if valid.size else 0.5


def classification_metrics(y_true, prob, threshold: float) -> dict[str, float]:
    """F1 / precision / recall / accuracy at the threshold + threshold-free PR-AUC and ROC-AUC."""
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(prob) >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    both = len(np.unique(y_true)) == 2
    return {"f1": float(f1), "precision": float(precision), "recall": float(recall),
            "accuracy": float((pred == y_true).mean()),
            "pr_auc": float(average_precision_score(y_true, prob)) if both else 0.0,
            "roc_auc": float(roc_auc_score(y_true, prob)) if both else 0.0}


def confusion(y_true, prob, threshold: float) -> dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(np.asarray(y_true).astype(int), (np.asarray(prob) >= threshold).astype(int),
                                      labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


@dataclass
class Evaluation:
    name: str
    threshold: float
    val: dict
    test: dict
    prob: dict            # {split: P(failure) per window}
    train_time_s: float
    params: int


def evaluate(name: str, prob: dict[str, np.ndarray], windows: Windows, cfg: Config, train_time_s: float,
             params: int) -> Evaluation:
    """Tune the threshold on validation, then score validation and test at that threshold."""
    thr = best_f1_threshold(windows.y["val"], prob["val"], cfg.threshold_min, cfg.threshold_max)
    return Evaluation(name, thr, classification_metrics(windows.y["val"], prob["val"], thr),
                      classification_metrics(windows.y["test"], prob["test"], thr), prob, train_time_s, params)


def select_best(evals: dict[str, Evaluation]) -> str:
    """Model selection on VALIDATION only: highest F1, ties broken by PR-AUC. The test split never decides."""
    return max(evals.values(), key=lambda e: (round(e.val["f1"], 6), e.val["pr_auc"])).name


def comparison_table(evals: dict[str, Evaluation]):
    """Every experiment side by side (validation F1 selects, test is reported)."""
    import pandas as pd

    return pd.DataFrame({e.name: {"val_f1": e.val["f1"], "threshold": e.threshold,
                                  **{f"test_{k}": v for k, v in e.test.items()},
                                  "params": e.params, "train_time_s": e.train_time_s}
                         for e in evals.values()}).T
