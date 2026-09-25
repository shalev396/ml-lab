"""Every tunable of the purchase-propensity run, in one dataclass.

Defaults reproduce the original project: stratified 68/12/20 train/val/test split (seed 42),
LogisticRegression(class_weight="balanced") on standardized flags vs a PyTorch MLP 64-32 with
dropout 0.2, pos_weight BCE, AdamW and early stopping on validation ROC-AUC. Decision thresholds
are tuned for max F1 on validation; the default model is the one with the best validation PR-AUC.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from . import utils


@dataclass
class Config:
    # --- data --------------------------------------------------------------------------------
    seed: int = 42
    test_size: float = 0.2            # share of all sessions held out for the final test report
    val_size: float = 0.15            # share of the remaining sessions used for early stopping,
                                      # threshold tuning and model selection (-> 68/12/20)
    smoke_rows: int = 25_000          # stratified subsample in smoke mode

    # --- LogisticRegression --------------------------------------------------------------------
    logreg_max_iter: int = 2000

    # --- PyTorch MLP ---------------------------------------------------------------------------
    hidden_units: tuple[int, ...] = (64, 32)
    dropout: float = 0.2
    epochs: int = 30
    batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5                 # early stopping: epochs without a validation ROC-AUC gain

    # --- evaluation ----------------------------------------------------------------------------
    importance_repeats: int = 3       # permutation-importance shuffles per feature (MLP)

    # --- run -----------------------------------------------------------------------------------
    smoke: bool = field(default_factory=lambda: os.getenv("SMOKE_TEST", "0") == "1")

    def __post_init__(self):
        self.hidden_units = tuple(int(u) for u in self.hidden_units)
        if self.smoke:  # same code path on a small subset with a tiny MLP (well under a minute on CPU)
            self.hidden_units = (16,)
            self.epochs = 3
            self.patience = 2
            self.batch_size = 2048
            self.importance_repeats = 1

    @property
    def model_dir(self) -> Path:
        """Where export writes the model repo; smoke runs never touch the real model/."""
        return utils.OUTPUTS_DIR / "smoke" / "model" if self.smoke else utils.MODEL_DIR

    @property
    def outputs_dir(self) -> Path:
        """Run artifacts that are not part of the model repo (segment tables, extra plots)."""
        return utils.OUTPUTS_DIR / "smoke" if self.smoke else utils.OUTPUTS_DIR
