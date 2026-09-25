"""Builds every experiment variant from model/model.py (the deployed architecture lives there)."""
from __future__ import annotations

import model as M  # ../model/model.py

from .config import Config


def build(variant: str, cfg: Config):
    """Uncompiled Keras model for one variant with the run's shared hyperparameters."""
    return M.build_model(variant, **cfg.model_kwargs())


def describe(variant: str, cfg: Config) -> str:
    """Short human-readable architecture string (model card / Space cards)."""
    core = M.VARIANTS[variant] + f"({cfg.units})"
    return (f"Embedding({cfg.max_tokens:,}, {cfg.embedding_dim}) -> {core} -> Dropout({cfg.dropout}) "
            f"-> Dense({cfg.dense_units}, relu) -> Dense(1, sigmoid)")
