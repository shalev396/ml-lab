"""Training + evaluation: logistic-regression fit, the PyTorch train loop (early stopping on validation
ROC-AUC), threshold tuning, metrics and permutation importance. Device-agnostic (cuda / mps / cpu)."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)

from .model_builder import LOGREG


# --------------------------------------------------------------------------- training
def fit_logreg(logreg, splits):
    """Fit on the standardized train split; returns (fitted model, seconds)."""
    start = time.perf_counter()
    x, y = splits.xy("train")
    logreg.fit(x, y)
    return logreg, time.perf_counter() - start


def pos_weight(y_train: np.ndarray) -> float:
    """negatives / positives: weights the rare buyers as much as all non-buyers in the BCE loss."""
    return float((y_train == 0).sum() / max((y_train == 1).sum(), 1))


def train_step(model, loader, loss_fn, optimizer, device) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = loss_fn(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(x)
        n += len(x)
    return total / n


@torch.inference_mode()
def eval_step(model, loader, loss_fn, device) -> tuple[float, np.ndarray]:
    model.eval()
    total, n, probs = 0.0, 0, []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += loss_fn(logits, y).item() * len(x)
        n += len(x)
        probs.append(torch.sigmoid(logits).float().cpu().numpy().ravel())
    return total / n, np.concatenate(probs)


def train_mlp(model, train_loader, val_loader, y_val: np.ndarray, cfg, device, weight: float) -> dict:
    """AdamW + pos_weight BCE; keeps the weights of the epoch with the best validation ROC-AUC."""
    model.to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history = {"train_loss": [], "val_loss": [], "val_roc_auc": []}
    best_auc, best_state, best_epoch, since_best = -np.inf, None, 0, 0
    start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_step(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_prob = eval_step(model, val_loader, loss_fn, device)
        val_auc = float(roc_auc_score(y_val, val_prob))
        for key, value in zip(history, (train_loss, val_loss, val_auc)):
            history[key].append(round(float(value), 6))
        improved = val_auc > best_auc + 1e-6
        if improved:
            best_auc, best_state, best_epoch, since_best = val_auc, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            since_best += 1
        print(f"epoch {epoch:>2}/{cfg.epochs} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
              f"val_roc_auc {val_auc:.5f}{' *' if improved else ''}")
        if since_best >= cfg.patience:
            print(f"early stopping: no validation ROC-AUC gain for {cfg.patience} epochs")
            break
    model.load_state_dict(best_state)
    model.eval()
    history.update(best_epoch=best_epoch, best_val_roc_auc=round(best_auc, 6),
                   seconds=round(time.perf_counter() - start, 1), pos_weight=round(weight, 4))
    return history


@torch.inference_mode()
def predict_mlp(model, x: np.ndarray, device, batch_size: int = 16384) -> np.ndarray:
    """P(order) for a standardized feature matrix."""
    model.eval()
    out = []
    for i in range(0, len(x), batch_size):
        t = torch.as_tensor(x[i:i + batch_size], dtype=torch.float32, device=device)
        out.append(torch.sigmoid(model(t)).float().cpu().numpy().ravel())
    return np.concatenate(out).astype(np.float64)


def predict_proba(name: str, estimator, x: np.ndarray, device="cpu") -> np.ndarray:
    return estimator.predict_proba(x)[:, 1] if name == LOGREG else predict_mlp(estimator, x, device)


# --------------------------------------------------------------------------- evaluation
def tune_threshold(y_true, y_prob) -> float:
    """Threshold with the highest F1 on the given (validation) split."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[int(np.nanargmax(f1[:-1]))])  # the last PR point has no threshold


def binary_metrics(y_true, y_prob, threshold: float) -> dict[str, float]:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float((y_pred == np.asarray(y_true)).mean()),
    }


def confusion(y_true, y_prob, threshold: float) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_prob) >= threshold
    return {"tn": int((~y_true & ~y_pred).sum()), "fp": int((~y_true & y_pred).sum()),
            "fn": int((y_true & ~y_pred).sum()), "tp": int((y_true & y_pred).sum())}


@dataclass
class Evaluation:
    """One experiment: threshold tuned on validation, metrics on validation and (once) on test."""
    name: str
    threshold: float
    val: dict
    test: dict
    val_prob: np.ndarray = field(repr=False)
    test_prob: np.ndarray = field(repr=False)


def evaluate(name: str, estimator, splits, device="cpu") -> Evaluation:
    x_val, y_val = splits.xy("val")
    x_test, y_test = splits.xy("test")
    val_prob = predict_proba(name, estimator, x_val, device)
    test_prob = predict_proba(name, estimator, x_test, device)
    thr = tune_threshold(y_val, val_prob)
    return Evaluation(name, thr, binary_metrics(y_val, val_prob, thr), binary_metrics(y_test, test_prob, thr),
                      val_prob, test_prob)


def select_best(evals: dict[str, Evaluation]) -> str:
    """Default model = highest validation PR-AUC (the test split plays no part in the choice)."""
    return max(evals.values(), key=lambda e: e.val["pr_auc"]).name


def comparison_table(evals: dict[str, Evaluation], params: dict[str, int]):
    import pandas as pd

    rows = {e.name: {"params": params[e.name], "threshold": e.threshold, "val_pr_auc": e.val["pr_auc"],
                     **{f"test_{k}": v for k, v in e.test.items()}} for e in evals.values()}
    return pd.DataFrame(rows).T.sort_values("val_pr_auc", ascending=False)


def permutation_importance(name: str, estimator, splits, device="cpu", repeats: int = 3,
                           seed: int = 42) -> dict[str, float]:
    """Mean test ROC-AUC drop when one flag is shuffled (model-agnostic)."""
    import model as M

    x, y = splits.xy("test", scaled=False)
    rng = np.random.default_rng(seed)
    score = lambda raw: roc_auc_score(y, predict_proba(name, estimator, splits.scaler.transform(raw), device))  # noqa: E731
    base = score(x)
    drops = {}
    for j, feature in enumerate(M.FEATURES):
        values = []
        for _ in range(repeats):
            shuffled = x.copy()
            rng.shuffle(shuffled[:, j])
            values.append(base - score(shuffled))
        drops[feature] = round(float(np.mean(values)), 6)
    return drops


def logreg_coefficients(logreg) -> dict[str, float]:
    import model as M

    return {f: round(float(c), 4) for f, c in zip(M.FEATURES, logreg.coef_.ravel())}

