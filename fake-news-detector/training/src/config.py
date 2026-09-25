"""Every tunable of the fake-news-detector run in one dataclass."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from . import utils


@dataclass
class Config:
    # --- data --------------------------------------------------------------------------------
    seed: int = 42
    kaggle_dataset: str = "clmentbisaillon/fake-and-real-news-dataset"
    hf_fallback: str = "GonzaloA/fake_news"   # used only when the Kaggle download fails
    subsample: int = 20_000                   # balanced articles used (0 = all ~39k deduplicated)
    test_fraction: float = 0.15               # of the (subsampled) dataset
    val_fraction: float = 0.15                # of what remains after the test split
    use_stemming: bool = False                # PorterStemmer after stopword removal

    # --- vectorizer --------------------------------------------------------------------------
    max_tokens: int = 20_000
    sequence_length: int = 300

    # --- architecture (shared by every variant) ----------------------------------------------
    variants: tuple[str, ...] = ("simple_rnn", "lstm", "gru", "bilstm")
    embedding_dim: int = 100
    units: int = 64
    dense_units: int = 32
    dropout: float = 0.3

    # --- training ----------------------------------------------------------------------------
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 4
    patience: int = 2                         # early stopping on val_loss (best weights restored)

    # --- run ---------------------------------------------------------------------------------
    smoke: bool = field(default_factory=lambda: os.getenv("SMOKE_TEST", "0") == "1")

    def __post_init__(self):
        self.variants = tuple(self.variants)
        if self.smoke:  # same code path on a tiny slice: every variant, 1 epoch, small vocab/model
            self.subsample = 800
            self.max_tokens = 5_000
            self.sequence_length = 100
            self.embedding_dim = 32
            self.units = 16
            self.epochs = 1
            self.patience = 1

    def model_kwargs(self) -> dict:
        """Keyword arguments of model.build_model() (except the variant)."""
        return dict(max_tokens=self.max_tokens, sequence_length=self.sequence_length,
                    embedding_dim=self.embedding_dim, units=self.units,
                    dense_units=self.dense_units, dropout=self.dropout)

    @property
    def model_dir(self) -> Path:
        """Where export writes the model repo; smoke runs never touch the real model/."""
        return utils.OUTPUTS_DIR / "smoke" / "model" if self.smoke else utils.MODEL_DIR

    @property
    def outputs_dir(self) -> Path:
        return utils.OUTPUTS_DIR / "smoke" if self.smoke else utils.OUTPUTS_DIR
