"""The five forecasters that are compared.

Classical (deployable, code in ../model/model.py): ARIMA, Holt-Winters, VAR.
Neural (training-only baselines, Keras 3, float32): an MLP on 20 lagged values and an LSTM on the
same 20-step window. They are compared but not deployed: they lose to the classical models.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # keep TensorFlow's C++ start-up logs quiet

import keras  # noqa: E402

import model as M  # noqa: E402

from .config import Config  # noqa: E402

CLASSICAL = {
    "arima": "ARIMA(p,1,q): (p, q) in 0..3 picked by AIC on the fit window; one-step forecasts by "
             "Kalman-filter updates with fixed coefficients",
    "holt_winters": "Holt-Winters exponential smoothing, additive trend, no seasonality (4 fitted parameters)",
    "var": "VAR(k) on all four indices, k picked by AIC (max 12); OLS",
}
NEURAL = ("MLP", "LSTM")


def build_mlp(cfg: Config) -> keras.Model:
    """Dense ReLU MLP on the last `window` scaled closes -> next close."""
    inputs = keras.Input(shape=(cfg.window,), name="lags")
    x = inputs
    for units in cfg.mlp_hidden:
        x = keras.layers.Dense(units, activation="relu")(x)
        x = keras.layers.Dropout(cfg.dropout)(x)
    return keras.Model(inputs, keras.layers.Dense(1, name="next_close")(x), name="mlp_lags")


def build_lstm(cfg: Config) -> keras.Model:
    """LSTM over the (window, 1) sequence of scaled closes -> next close."""
    inputs = keras.Input(shape=(cfg.window, 1), name="sequence")
    x = keras.layers.LSTM(cfg.lstm_units)(inputs)
    x = keras.layers.Dropout(cfg.dropout)(x)
    return keras.Model(inputs, keras.layers.Dense(1, name="next_close")(x), name="lstm_window")


BUILDERS = {"MLP": build_mlp, "LSTM": build_lstm}


def classical_params(method: str, spec: dict, n_series: int = len(M.INDICES)) -> int:
    """Number of fitted parameters of a classical forecaster."""
    if method == "holt_winters":
        return 4 if spec["holt_winters"]["trend"] else 2   # smoothing level/trend + initial level/trend
    if method == "arima":
        return len(spec["arima"]["params"])                # AR + MA coefficients + noise variance
    if method == "var":
        lag = int(spec["var"]["lag"])
        return lag * n_series * n_series + n_series         # lag matrices + intercepts
    raise ValueError(method)
