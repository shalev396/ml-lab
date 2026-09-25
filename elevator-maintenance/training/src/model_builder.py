"""The three experiments: the deployed RandomForest (from model/model.py) and two Keras comparison nets.

The Keras models are compared in the notebook and on the model card but are NOT exported: the forest
wins on validation, and serving it keeps the Space free of TensorFlow.
"""
from __future__ import annotations

import os

import model as M

from .config import Config

RF, CNN, LSTM = "RandomForest", "CNN-1D", "LSTM"


def build_random_forest(cfg: Config):
    """RandomForestClassifier(300 trees, class_weight="balanced") on the 63 window features."""
    return M.build_forest(cfg.rf_estimators, cfg.seed, cfg.n_jobs)


def _keras():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # keep TensorFlow's C++ start-up logs quiet
    import keras

    return keras


def build_cnn1d(cfg: Config, n_channels: int = len(M.SENSORS)):
    """1D-CNN on raw standardised (window, channels) readings: Conv-Pool-Conv-GlobalMax-Dense-Dropout-sigmoid."""
    keras = _keras()
    layers = [keras.Input(shape=(cfg.window, n_channels))]
    for i, filters in enumerate(cfg.cnn_filters):
        layers.append(keras.layers.Conv1D(filters, cfg.cnn_kernel, padding="same", activation="relu"))
        if i < len(cfg.cnn_filters) - 1:
            layers.append(keras.layers.MaxPooling1D(2))
    layers += [keras.layers.GlobalMaxPooling1D(),
               keras.layers.Dense(cfg.dense_units, activation="relu"),
               keras.layers.Dropout(cfg.dropout),
               keras.layers.Dense(1, activation="sigmoid")]
    return keras.Sequential(layers, name="cnn1d_raw_windows")


def build_lstm(cfg: Config, n_channels: int = len(M.SENSORS)):
    """LSTM on raw standardised (window, channels) readings: LSTM-Dense-Dropout-sigmoid."""
    keras = _keras()
    return keras.Sequential([keras.Input(shape=(cfg.window, n_channels)),
                             keras.layers.LSTM(cfg.lstm_units),
                             keras.layers.Dense(cfg.dense_units, activation="relu"),
                             keras.layers.Dropout(cfg.dropout),
                             keras.layers.Dense(1, activation="sigmoid")], name="lstm_raw_windows")


def build_all(cfg: Config) -> dict:
    """{name: untrained model} for every experiment, in training order."""
    return {RF: build_random_forest(cfg), CNN: build_cnn1d(cfg), LSTM: build_lstm(cfg)}


def params(name: str, model) -> int:
    """Keras: trainable + non-trainable weights. RandomForest: total tree nodes (after fitting)."""
    return M.tree_nodes(model) if name == RF else int(model.count_params())


def describe(name: str, cfg: Config) -> str:
    if name == RF:
        return f"RandomForest ({cfg.rf_estimators} trees, balanced class weights) on 63 engineered features"
    if name == CNN:
        filters = "-".join(str(f) for f in cfg.cnn_filters)
        return f"Keras 1D-CNN (Conv1D {filters}, kernel {cfg.cnn_kernel}) on raw standardised windows"
    return f"Keras LSTM ({cfg.lstm_units} units) on raw standardised windows"
