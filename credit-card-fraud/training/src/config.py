"""Every tunable of the credit-card-fraud run in one dataclass (the notebook builds it in section 2)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from . import utils


@dataclass
class Config:
    # --- data: stratified 60/20/20 train/val/test split --------------------------------------
    seed: int = 42
    test_size: float = 0.2            # fraction of all rows held out for the final test report
    val_size: float = 0.2             # fraction of all rows used for model selection + threshold tuning
    smoke_rows: int = 20_000          # stratified subsample size in smoke mode

    # --- scikit-learn / imbalanced-learn variants --------------------------------------------
    logreg_max_iter: int = 2000
    knn_neighbors: int = 5
    rf_n_estimators: int = 100
    smote_k_neighbors: int = 5

    # --- Keras MLP ------------------------------------------------------------------------
    hidden_units: tuple[int, ...] = (64, 32, 16)
    dropout: float = 0.3
    lr: float = 1e-3
    batch_size: int = 2048
    epochs: int = 30
    patience: int = 5                 # early stopping on the inner split's PR-AUC
    early_stopping_split: float = 0.15  # fraction of the TRAIN split held out for early stopping

    # --- run ------------------------------------------------------------------------------
    smoke: bool = field(default_factory=lambda: os.getenv("SMOKE_TEST", "0") == "1")

    def __post_init__(self):
        self.hidden_units = tuple(self.hidden_units)
        if self.smoke:  # tiny, fast end-to-end run: same code path, small data + small models
            self.smoke_rows = min(self.smoke_rows, 20_000)
            self.rf_n_estimators = min(self.rf_n_estimators, 20)
            self.smote_k_neighbors = min(self.smote_k_neighbors, 3)
            self.hidden_units = (16, 8)
            self.batch_size = 512
            self.epochs = 2
            self.patience = 1

    @property
    def model_dir(self) -> Path:
        """Where export writes the model repo; smoke runs never touch the real model/."""
        return utils.OUTPUTS_DIR / "smoke" / "model" if self.smoke else utils.MODEL_DIR

    @property
    def outputs_dir(self) -> Path:
        """Run artifacts that are not part of the model repo (training curves, logs)."""
        return utils.OUTPUTS_DIR / "smoke" if self.smoke else utils.OUTPUTS_DIR
