"""Keras compile/fit, one-step evaluation on the original $ scale, and the multi-step (recursive)
evaluation that mirrors what the Space does."""
from __future__ import annotations

import time

import keras
import numpy as np
import pandas as pd

import model as M  # ../model/model.py (put on sys.path by src/__init__.py)

from . import utils
from .config import Config
from .data_setup import Track


# --------------------------------------------------------------------------- metrics
def forecast_metrics(actual, pred) -> dict[str, float]:
    actual, pred = np.asarray(actual, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    err = actual - pred
    return {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
            "mape": float(np.mean(np.abs(err / actual)) * 100.0)}


# --------------------------------------------------------------------------- training
def compile_model(net: keras.Model, cfg: Config) -> keras.Model:
    net.compile(optimizer=keras.optimizers.Adam(cfg.lr), loss="mse", metrics=["mae"])
    return net


def fit(net: keras.Model, track: Track, cfg: Config, verbose: int = 0) -> dict:
    """Adam/MSE on the scaled target; early stopping + LR halving on validation loss (best weights kept)."""
    utils.set_seeds(cfg.seed, "tensorflow")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(1, cfg.patience // 2)),
    ]
    start = time.perf_counter()
    hist = net.fit(track.X["train"], track.y["train"], validation_data=(track.X["val"], track.y["val"]),
                   epochs=cfg.epochs, batch_size=cfg.batch_size, callbacks=callbacks, shuffle=True,
                   verbose=verbose)
    out = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    out["seconds"] = time.perf_counter() - start
    out["best_epoch"] = int(np.argmin(out["val_loss"])) + 1
    return out


# --------------------------------------------------------------------------- evaluation
def evaluate(net: keras.Model, track: Track) -> dict:
    """One-step-ahead predictions on validation + test, back on the $ scale."""
    out = {}
    for split in ("val", "test"):
        pred = track.to_prices(net.predict(track.X[split], verbose=0, batch_size=1024), split)
        out[split] = forecast_metrics(track.actual(split), pred)
        out[f"{split}_pred"] = pred
    return out


def evaluate_naive(track: Track) -> dict:
    """Persistence baseline: prediction(t) = close(t-1)."""
    out = {}
    for split in ("val", "test"):
        pred = track.close[track.idx[split] - 1]
        out[split] = forecast_metrics(track.actual(split), pred)
        out[f"{split}_pred"] = pred
    return out


def select_best(evals: dict[str, dict], candidates) -> str:
    """Lowest VALIDATION RMSE among the deployable variants (the test split never picks a model)."""
    return min(candidates, key=lambda name: evals[name]["val"]["rmse"])


def comparison_table(evals: dict[str, dict], params: dict[str, int | None]) -> pd.DataFrame:
    rows = {name: {"val_rmse": e["val"]["rmse"], "test_rmse": e["test"]["rmse"], "test_mae": e["test"]["mae"],
                   "test_mape": e["test"]["mape"], "params": params.get(name)} for name, e in evals.items()}
    return pd.DataFrame(rows).T.sort_values("val_rmse")


def horizon_eval(net: keras.Model, track: Track, cfg: Config) -> dict:
    """Recursive multi-step forecasts from many test origins, exactly like `Predictor.predict`.

    Origin o = a test day used as the cutoff: the model sees closes[o-lookback .. o] and forecasts
    closes[o+1 .. o+H]; naive repeats closes[o]. Returns the MAE per step for both.
    """
    close, H, L = track.close, cfg.eval_horizon, cfg.lookback
    first = max(int(track.idx["test"][0]) - 1, L)
    origins = np.arange(first, len(close) - H, cfg.eval_stride)
    windows = np.stack([M.standardize(M.log_returns(close[o - L:o + 1]), track.ret_mu, track.ret_sigma)
                        for o in origins])
    scaled = M.recursive_forecast(net, windows, H)
    pred = M.to_prices(close[origins], scaled, track.ret_mu, track.ret_sigma)
    actual = np.stack([close[o + 1:o + H + 1] for o in origins])
    naive = np.repeat(close[origins][:, None], H, axis=1)
    mae_model = np.abs(pred - actual).mean(axis=0)
    mae_naive = np.abs(naive - actual).mean(axis=0)
    return {"horizon": H, "n_origins": int(len(origins)), "stride": cfg.eval_stride,
            "mae_by_step": mae_model.tolist(), "naive_mae_by_step": mae_naive.tolist(),
            "mae_mean": float(mae_model.mean()), "naive_mae_mean": float(mae_naive.mean()),
            "first_origin": str(track.dates[origins[0]].date()),
            "last_origin": str(track.dates[origins[-1]].date())}
