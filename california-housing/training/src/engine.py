"""Fit / predict / evaluate helpers for every experiment (device-agnostic PyTorch loop for the MLP).

Predictions go through the same preprocessing as `model.Predictor` (`model.to_matrix`, the train-split
scaler for the MLP, floor at $0), so the numbers here are exactly what the Space serves.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from . import utils


# --------------------------------------------------------------------------- scikit-learn / XGBoost
def fit_sklearn(model, X: np.ndarray, y: np.ndarray) -> dict:
    """Fit a scikit-learn regressor; returns {"train_time_s"}."""
    start = time.perf_counter()
    model.fit(X, y)
    return {"train_time_s": round(time.perf_counter() - start, 2)}


def fit_xgboost(model, X_tr, y_tr, X_val, y_val) -> dict:
    """Fit with early stopping on validation RMSE, then trim the booster to the best round so the
    saved `xgb.ubj` predicts exactly like the in-memory model. Returns history + best round."""
    start = time.perf_counter()
    model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], verbose=False)
    evals = model.evals_result()
    best = int(model.best_iteration)
    booster = model.get_booster()[: best + 1]
    return {
        "booster": booster,
        "best_iteration": best,
        "n_rounds": len(evals["validation_1"]["rmse"]),
        "history": {"train_rmse": evals["validation_0"]["rmse"], "val_rmse": evals["validation_1"]["rmse"]},
        "train_time_s": round(time.perf_counter() - start, 2),
    }


# --------------------------------------------------------------------------- PyTorch MLP
def make_loaders(X_tr, y_tr, X_val, y_val, batch_size: int, device) -> tuple[DataLoader, DataLoader]:
    """Standardized float32 tensors -> shuffled train loader + ordered validation loader."""
    def dataset(X, y):
        return TensorDataset(torch.tensor(np.asarray(X), dtype=torch.float32),
                             torch.tensor(np.asarray(y), dtype=torch.float32).unsqueeze(1))

    kwargs = dict(num_workers=utils.num_workers(), pin_memory=device.type == "cuda")
    return (DataLoader(dataset(X_tr, y_tr), batch_size=batch_size, shuffle=True, **kwargs),
            DataLoader(dataset(X_val, y_val), batch_size=batch_size, shuffle=False, **kwargs))


def train_step(model, loader, loss_fn, optimizer, device) -> float:
    """One epoch; returns the mean training MSE."""
    model.train()
    total, n = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        loss = loss_fn(model(X), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(X)
        n += len(X)
    return total / n


@torch.inference_mode()
def eval_step(model, loader, loss_fn, device) -> float:
    """Mean MSE over a loader (eval mode, no dropout)."""
    model.eval()
    total, n = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        total += loss_fn(model(X), y).item() * len(X)
        n += len(X)
    return total / n


def train_mlp(model, train_loader, val_loader, cfg, device) -> dict:
    """AdamW + MSE with early stopping on validation loss; the best epoch's weights are restored."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.MSELoss()
    history = {"train_loss": [], "val_loss": []}
    best_val, best_state, best_epoch, since_best = float("inf"), None, 0, 0
    start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_step(model, train_loader, loss_fn, optimizer, device)
        val_loss = eval_step(model, val_loader, loss_fn, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_val - 1e-6:
            best_val, best_epoch, since_best = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            since_best += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"  epoch {epoch:>3} | train MSE {train_loss:.4f} | val MSE {val_loss:.4f}")
        if since_best >= cfg.patience:
            print(f"  early stop at epoch {epoch} (no val improvement for {cfg.patience} epochs)")
            break
    model.load_state_dict(best_state)
    model.eval()
    print(f"  best epoch {best_epoch}: val MSE {best_val:.4f}")
    return {"history": history, "best_epoch": best_epoch, "epochs_run": len(history["train_loss"]),
            "train_time_s": round(time.perf_counter() - start, 2)}


# --------------------------------------------------------------------------- prediction + metrics
@torch.inference_mode()
def predict(name: str, model, X: np.ndarray, scaler=None, device="cpu") -> np.ndarray:
    """Unscaled (n, 11) features -> predicted MedHouseVal in $100k (floored at 0, as in model.Predictor)."""
    if name == "mlp":
        x = torch.as_tensor(scaler.transform(X), dtype=torch.float32, device=device)
        model.eval()
        pred = model.to(device)(x)[:, 0].float().cpu().numpy()
    elif hasattr(model, "inplace_predict"):   # trimmed XGBoost Booster
        pred = model.inplace_predict(X)
    else:                                      # scikit-learn regressors (unscaled features)
        pred = model.predict(X)
    return np.maximum(np.asarray(pred, dtype=np.float64), 0.0)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE / MAE in $100k (the target unit), R², and RMSE / MAE in dollars."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": float(r2_score(y_true, y_pred)),
            "rmse_usd": round(rmse * 100_000, 0), "mae_usd": round(mae * 100_000, 0)}


def evaluate(models: dict, X: np.ndarray, y: np.ndarray, scaler, device="cpu") -> dict[str, dict]:
    """{name: metrics} for every model on one split."""
    return {name: regression_metrics(y, predict(name, m, X, scaler, device)) for name, m in models.items()}


def select_best(val_metrics: dict[str, dict], candidates=("xgboost", "mlp")) -> str:
    """Default model = lowest validation RMSE among the exported models (the test split is never used)."""
    return min(candidates, key=lambda name: val_metrics[name]["rmse"])
