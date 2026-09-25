"""Fitting, walk-forward evaluation, model selection and the multi-step backtest.

Protocol (per index, nothing shuffled):
- every model is fit on the FIT window only (ARIMA order by AIC, Holt-Winters smoothing, VAR lag,
  NN weights; the NNs early-stop on the validation window);
- one-step-ahead predictions are walked forward over validation + test with the parameters frozen,
  each step seeing the actual closes up to the day before (the R course's `accuracy()` setting);
- the deployed method of each index (rule fixed before looking at the test window): rank the classical
  models by VALIDATION MAPE; among those within `select_tolerance` (2%, relative) of the best, deploy the
  one with the fewest fitted parameters (ties -> lower validation MAPE). Test metrics are never used;
- the test window is reported once. A multi-step backtest then replays exactly what the Space does
  (`model.forecast`, refit/filter on the history up to each origin, `backtest_horizon` days ahead).
"""
from __future__ import annotations

import time
import warnings

import keras
import numpy as np
import pandas as pd

import model as M

from .model_builder import classical_params
from .config import Config
from .data_setup import Splits

NAIVE = "Naive persistence"


# --------------------------------------------------------------------------- metrics
def accuracy(actual, pred, insample) -> dict:
    """The R `forecast::accuracy()` table: ME, RMSE, MAE, MPE, MAPE (%), MASE.
    MASE scales the MAE by the in-sample one-step naive MAE (non-seasonal)."""
    actual, pred = np.asarray(actual, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    err = actual - pred
    scale = float(np.mean(np.abs(np.diff(np.asarray(insample, dtype=np.float64)))))
    return {"ME": float(err.mean()), "RMSE": float(np.sqrt(np.mean(err ** 2))), "MAE": float(np.abs(err).mean()),
            "MPE": float(np.mean(100 * err / actual)), "MAPE": float(np.mean(100 * np.abs(err) / np.abs(actual))),
            "MASE": float(np.abs(err).mean() / scale)}


# --------------------------------------------------------------------------- fitting
def select_arima(values: np.ndarray, cfg: Config):
    """(p, d, q) grid with d fixed by the ADF test; lowest AIC wins. Returns (order, results, AIC table)."""
    rows, best = [], None
    for p in range(cfg.arima_max_p + 1):
        for q in range(cfg.arima_max_q + 1):
            if p == q == 0:
                continue
            try:
                res = M.fit_arima(values, (p, cfg.arima_d, q))
            except Exception as err:  # a failed fit just drops out of the grid
                print(f"[arima] ({p},{cfg.arima_d},{q}) failed: {err}")
                continue
            rows.append({"p": p, "d": cfg.arima_d, "q": q, "AIC": float(res.aic),
                         "converged": bool(res.mle_retvals.get("converged", True))})
            if best is None or res.aic < best[1].aic:
                best = ((p, cfg.arima_d, q), res)
    table = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    return best[0], best[1], table


def fit_classical(splits: Splits, index: str, cfg: Config) -> dict:
    """ARIMA grid + Holt-Winters on the fit window of one index."""
    values = splits.fit[index].to_numpy()
    start = time.perf_counter()
    order, arima, aic = select_arima(values, cfg)
    hw = M.fit_holt_winters(values, cfg.hw_trend)
    print(f"[{index}] ARIMA{order} (AIC {arima.aic:,.1f}, {len(aic)} fits) | Holt-Winters alpha "
          f"{hw.params['smoothing_level']:.3f} beta {hw.params['smoothing_trend']:.3f} | "
          f"{time.perf_counter() - start:.1f}s")
    return {"arima_order": order, "arima": arima, "aic_table": aic, "hw": hw}


def fit_var(splits: Splits, cfg: Config):
    """VAR on all four indices; lag by AIC (at least 1). Returns (results, lag)."""
    from statsmodels.tsa.api import VAR

    values = splits.fit[list(M.INDICES)].to_numpy(dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lag = max(1, int(VAR(values).select_order(cfg.var_maxlags).aic))
    print(f"[VAR] lag order by AIC: {lag}")
    return M.fit_var(values, lag), lag


def fit_nn(net: keras.Model, data: dict, cfg: Config, verbose: int = 0) -> dict:
    """Adam + MSE on scaled closes, early stopping on the validation window. Returns the history."""
    reshape = len(net.input_shape) == 3
    prep = (lambda x: x[..., None]) if reshape else (lambda x: x)
    net.compile(optimizer=keras.optimizers.Adam(cfg.lr), loss="mse", metrics=["mae"])
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.patience, restore_best_weights=True)
    hist = net.fit(prep(data["X_train"]), data["y_train"], validation_data=(prep(data["X_val"]), data["y_val"]),
                   epochs=cfg.epochs, batch_size=cfg.batch_size, callbacks=[stop], verbose=verbose, shuffle=True)
    return {k: [float(v) for v in vals] for k, vals in hist.history.items()}


# --------------------------------------------------------------------------- one-step walk-forward
def one_step_arima(values: np.ndarray, start: int, order, params) -> np.ndarray:
    """Fixed coefficients; the Kalman filter absorbs each actual close before the next forecast."""
    res = M.filter_arima(values, order, params)
    return np.asarray(res.predict(start=start, end=len(values) - 1), dtype=np.float64)


def one_step_holt_winters(values: np.ndarray, start: int, hw, trend: str) -> np.ndarray:
    """Smoothing parameters and initial state frozen from the fit window, recursion run over all data."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    p = hw.params
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ExponentialSmoothing(values, trend=trend, initialization_method="known",
                                   initial_level=p["initial_level"], initial_trend=p["initial_trend"]).fit(
            smoothing_level=p["smoothing_level"], smoothing_trend=p["smoothing_trend"], optimized=False)
    return np.asarray(res.fittedvalues, dtype=np.float64)[start:]


def one_step_var(var_res, lag: int, values2d: np.ndarray, start: int, target: int) -> np.ndarray:
    """Fixed VAR coefficients, each forecast from the last `lag` ACTUAL rows."""
    return np.array([var_res.forecast(values2d[t - lag: t], steps=1)[0, target]
                     for t in range(start, len(values2d))])


def one_step_nn(net: keras.Model, data: dict) -> np.ndarray:
    X = data["X_eval"][..., None] if len(net.input_shape) == 3 else data["X_eval"]
    scaled = np.asarray(net(X, training=False), dtype=np.float64).reshape(-1, 1)  # direct call: no retracing
    return data["scaler"].inverse_transform(scaled).ravel()


def one_step_predictions(splits: Splits, index: str, fit: dict, var_fit: tuple, nets: dict | None,
                         nn_data: dict | None, cfg: Config) -> dict[str, np.ndarray]:
    """{variant: predictions for every validation + test day} for one index (`nets` may be empty)."""
    values = splits.frame[index].to_numpy(dtype=np.float64)
    start = splits.eval_start
    var_res, lag = var_fit
    order = fit["arima_order"]
    preds = {
        "ARIMA({},{},{})".format(*order): one_step_arima(values, start, order, fit["arima"].params),
        "Holt-Winters": one_step_holt_winters(values, start, fit["hw"], cfg.hw_trend),
        f"VAR({lag})": one_step_var(var_res, lag, splits.frame[list(M.INDICES)].to_numpy(np.float64), start,
                                    M.INDICES.index(index)),
    }
    for name, net in (nets or {}).items():
        preds[name] = one_step_nn(net, nn_data)
    preds[NAIVE] = values[start - 1: -1]
    return preds


VARIANT_METHOD = {"ARIMA": "arima", "Holt-Winters": "holt_winters", "VAR": "var"}  # name prefix -> model.py method


def method_of(variant: str) -> str | None:
    """model.py method id of a variant name, None for the neural nets and the naive baseline."""
    return next((m for prefix, m in VARIANT_METHOD.items() if variant.startswith(prefix)), None)


def evaluate(splits: Splits, index: str, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    """Validation MAPE (selection) + the full R accuracy table on the test window, per variant."""
    values = splits.frame[index].to_numpy(dtype=np.float64)
    start, n_val = splits.eval_start, splits.n_val
    val_actual, test_actual = values[start: start + n_val], values[start + n_val:]
    rows = {}
    for name, pred in preds.items():
        val = accuracy(val_actual, pred[:n_val], values[:start])
        test = accuracy(test_actual, pred[n_val:], values[: start + n_val])
        rows[name] = {"val_MAPE": val["MAPE"], **test}
    return pd.DataFrame(rows).T


def classical_param_counts(table: pd.DataFrame, fit: dict, var_lag: int, cfg: Config) -> dict[str, int]:
    """{classical variant: number of fitted parameters} for one index."""
    settings = _settings(fit, var_lag, cfg)
    return {name: classical_params(method_of(name), settings) for name in table.index if method_of(name)}


def select_deployed(table: pd.DataFrame, params: dict[str, int], tolerance: float) -> str:
    """Parsimony rule, validation window only (the test columns are never read):
    candidates = classical variants whose validation MAPE is within `tolerance` (relative) of the best one;
    deploy the candidate with the fewest fitted parameters, ties broken by the lower validation MAPE."""
    val = {name: float(table.loc[name, "val_MAPE"]) for name in table.index if method_of(name)}
    best = min(val.values())
    candidates = [name for name, mape in val.items() if mape <= best * (1 + tolerance)]
    return min(candidates, key=lambda name: (params[name], val[name]))


def _settings(fit: dict, var_lag: int, cfg: Config) -> dict:
    return {"holt_winters": {"trend": cfg.hw_trend},
            "arima": {"order": [int(o) for o in fit["arima_order"]],
                      "params": [float(p) for p in np.asarray(fit["arima"].params)]},
            "var": {"lag": int(var_lag)}}


def serving_spec(fit: dict, var_lag: int, deployed_variant: str, cfg: Config) -> dict:
    """One entry of config.json["models"]: the deployed method + the settings of every classical method."""
    return {"deployed": method_of(deployed_variant), **_settings(fit, var_lag, cfg)}


# --------------------------------------------------------------------------- multi-step backtest
def backtest(splits: Splits, index: str, spec: dict, cfg: Config) -> dict:
    """Replays the Space: from origins every `backtest_step` days of the test window, forecast
    `backtest_horizon` days with `model.forecast` (history up to the origin only) for every classical
    method and the naive baseline. Returns {"table": DataFrame (MAE, MAPE per method), "paths": [...]}."""
    frame, h = splits.frame, cfg.backtest_horizon
    first = splits.n_fit + splits.n_val
    origins = list(range(first, len(frame) - h + 1, cfg.backtest_step))
    errors = {name: [] for name in [*(M.describe(m, spec) for m in M.METHODS), NAIVE]}
    paths = []
    for origin in origins:
        past, actual = frame.iloc[:origin], frame[index].to_numpy()[origin: origin + h]
        path = {"origin": frame.index[origin - 1], "dates": frame.index[origin: origin + h], "actual": actual,
                "naive": np.full(h, past[index].iloc[-1])}
        for method in M.METHODS:
            fc = M.forecast(past, index, h, spec, method)
            errors[M.describe(method, spec)].append(np.abs(fc - actual) / actual)
            path[method] = fc
        errors[NAIVE].append(np.abs(path["naive"] - actual) / actual)
        paths.append(path)
    table = {}
    for name, errs in errors.items():
        rel = np.concatenate(errs)
        abs_err = np.concatenate([e * p["actual"] for e, p in zip(errs, paths)])
        table[name] = {"MAE": float(abs_err.mean()), "MAPE": float(100 * rel.mean())}
    return {"table": pd.DataFrame(table).T, "paths": paths, "origins": len(origins), "horizon": h}
