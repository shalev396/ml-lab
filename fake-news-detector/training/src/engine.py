"""Keras compile / fit helpers and the evaluation used for model selection and the report."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import keras
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)

from .config import Config


def compile_model(model, cfg: Config):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg.lr), loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return model


class EpochTimer(keras.callbacks.Callback):
    """Wall time of every epoch."""

    def __init__(self):
        super().__init__()
        self.times: list[float] = []

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.perf_counter() - self._t0)


def train(model, X_train, y_train, X_val, y_val, cfg: Config, verbose: int = 2) -> dict:
    """Fit with early stopping on val_loss (best weights restored). Returns the Keras history
    plus `epoch_s` (seconds per epoch)."""
    timer = EpochTimer()
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.patience, restore_best_weights=True)
    hist = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=cfg.epochs,
                     batch_size=cfg.batch_size, callbacks=[stop, timer], verbose=verbose)
    return {**{k: [float(v) for v in vals] for k, vals in hist.history.items()},
            "epoch_s": [round(t, 2) for t in timer.times]}


def scores(y_true, prob, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(prob) >= threshold).astype(int)
    return {"accuracy": float(accuracy_score(y_true, pred)), "f1": float(f1_score(y_true, pred)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred)), "roc_auc": float(roc_auc_score(y_true, prob))}


def confusion(y_true, prob, threshold: float = 0.5) -> dict[str, int]:
    """Rows = actual (real, fake), columns = predicted; positive class = fake."""
    tn, fp, fn, tp = confusion_matrix(np.asarray(y_true).astype(int), (np.asarray(prob) >= threshold).astype(int),
                                      labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


@dataclass
class Evaluation:
    name: str
    params: int
    val: dict[str, float]
    test: dict[str, float]
    test_prob: np.ndarray = field(repr=False)
    history: dict = field(default_factory=dict, repr=False)

    @property
    def epochs_ran(self) -> int:
        return len(self.history.get("loss", []))


def predict_proba(model, X) -> np.ndarray:
    return model.predict(X, batch_size=512, verbose=0).ravel().astype(np.float64)


def evaluate(name: str, model, X_val, y_val, X_test, y_test, history: dict | None = None) -> Evaluation:
    """Validation scores (select the model) and test scores (reported once) at threshold 0.5."""
    test_prob = predict_proba(model, X_test)
    return Evaluation(name=name, params=int(model.count_params()), val=scores(y_val, predict_proba(model, X_val)),
                      test=scores(y_test, test_prob), test_prob=test_prob, history=history or {})


def select_best(evals: dict[str, Evaluation]) -> str:
    """Highest validation accuracy (ties: validation ROC-AUC). The test split plays no part."""
    return max(evals, key=lambda n: (evals[n].val["accuracy"], evals[n].val["roc_auc"]))
