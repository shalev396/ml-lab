"""Every variant of the experiment. The deployed architecture comes from ../model/model.py
(`M.build_lstm`); the SimpleRNN and stacked-LSTM baselines exist only for the comparison."""
from __future__ import annotations

import keras

import model as M  # ../model/model.py (put on sys.path by src/__init__.py)

from .config import Config

# variant -> the data track it trains on (see data_setup.prepare_tracks)
TRACKS = {
    "simple_rnn_levels": "levels",
    "lstm_levels": "levels",
    "lstm_returns": "returns",
    "stacked_lstm_multi": "multivariate",
}
DEPLOYABLE = ("lstm_returns",)  # the Space forecasts recursively from closes only -> univariate returns model
NAIVE = "naive_persistence"


def describe(cfg: Config) -> dict[str, str]:
    return {
        "simple_rnn_levels": f"SimpleRNN({cfg.rnn_units}) on MinMax-scaled closes",
        "lstm_levels": f"LSTM({cfg.lstm_units}) on MinMax-scaled closes",
        "lstm_returns": f"LSTM({cfg.lstm_units}) on standardized log-returns",
        "stacked_lstm_multi": f"{cfg.num_layers}x LSTM({cfg.lstm_units}) on Close/Volume/RSI/EMA20/EMA50",
        NAIVE: "tomorrow = today (no model)",
    }


def build_simple_rnn(cfg: Config, n_features: int = 1) -> keras.Model:
    return keras.Sequential([
        keras.layers.Input(shape=(cfg.lookback, n_features)),
        keras.layers.SimpleRNN(cfg.rnn_units),
        keras.layers.Dropout(cfg.dropout),
        keras.layers.Dense(1, dtype="float32"),
    ], name="simple_rnn_levels")


def build_stacked_lstm(cfg: Config, n_features: int) -> keras.Model:
    layers: list = [keras.layers.Input(shape=(cfg.lookback, n_features))]
    for i in range(cfg.num_layers):
        layers += [keras.layers.LSTM(cfg.lstm_units, return_sequences=i < cfg.num_layers - 1),
                   keras.layers.Dropout(cfg.dropout)]
    layers.append(keras.layers.Dense(1, dtype="float32"))
    return keras.Sequential(layers, name="stacked_lstm_multi")


def build_deployed(cfg: Config) -> keras.Model:
    """The architecture served by the Space: `model.build_lstm` on 1 feature (log-returns)."""
    return M.build_lstm(cfg.lookback, cfg.lstm_units, cfg.dropout, n_features=1)


def build_variant(name: str, cfg: Config, n_features: int) -> keras.Model:
    if name == "simple_rnn_levels":
        return build_simple_rnn(cfg, n_features)
    if name == "lstm_levels":
        return M.build_lstm(cfg.lookback, cfg.lstm_units, cfg.dropout, n_features, name="lstm_levels")
    if name == "lstm_returns":
        return build_deployed(cfg)
    if name == "stacked_lstm_multi":
        return build_stacked_lstm(cfg, n_features)
    raise KeyError(name)


def build_variants(cfg: Config, tracks: dict) -> dict[str, keras.Model]:
    """{variant: uncompiled float32 Keras model}, each sized for the feature count of its track."""
    return {name: build_variant(name, cfg, tracks[track].n_features) for name, track in TRACKS.items()}
