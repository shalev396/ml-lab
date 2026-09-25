"""Every tunable of the California housing run, in one dataclass.

Defaults reproduce the original experiment: 80/20 train/test split (seed 42) with 15 % of the train part
held out as validation; LinearRegression, RandomForest (200 trees), XGBoost (hist, early stopping on
validation) and a PyTorch MLP 11-256-128-64-1 (AdamW, early stopping on validation loss).
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

from . import utils


@dataclass
class Config:
    # --- data / splits
    seed: int = 42
    test_size: float = 0.2            # held-out test split (reported once, never used for choices)
    val_size: float = 0.15            # carved from the train part: early stopping + model selection
    subset_rows: int = 0              # 0 = all 20,640 districts; smoke uses a random sample
    # --- CPU threads for RandomForest / XGBoost (-1 = all cores; env N_JOBS overrides, e.g. on a shared machine)
    n_jobs: int = field(default_factory=lambda: int(os.getenv("N_JOBS", "-1")))
    # --- RandomForest (baseline, evaluated but not exported: the fitted forest is ~100s of MB)
    rf_n_estimators: int = 200
    # --- XGBoost
    xgb_n_estimators: int = 2000
    xgb_lr: float = 0.05
    xgb_max_depth: int = 6
    xgb_subsample: float = 0.8
    xgb_colsample: float = 0.8
    xgb_early_stopping_rounds: int = 50
    # --- PyTorch MLP
    hidden_units: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.15
    epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10                # early stopping on validation MSE (best weights restored)
    # --- run mode (SMOKE_TEST=1): 3,000 rows, tiny models, exports to outputs/smoke/model
    smoke: bool = field(default_factory=lambda: os.getenv("SMOKE_TEST", "0") == "1")

    def __post_init__(self):
        self.hidden_units = tuple(self.hidden_units)
        if self.smoke:
            self.subset_rows = self.subset_rows or 3000
            self.rf_n_estimators = 30
            self.xgb_n_estimators = 60
            self.xgb_early_stopping_rounds = 10
            self.hidden_units = (32, 16)
            self.epochs = 3
            self.patience = 3

    @property
    def data_dir(self) -> Path:
        return utils.DATA_DIR

    @property
    def outputs_dir(self) -> Path:
        """Scratch outputs (gitignored); smoke runs keep everything under outputs/smoke/."""
        return utils.OUTPUTS_DIR / "smoke" if self.smoke else utils.OUTPUTS_DIR

    @property
    def model_dir(self) -> Path:
        """Where export() writes weights + card. Smoke runs never touch the real model/ repo."""
        return utils.OUTPUTS_DIR / "smoke" / "model" if self.smoke else utils.MODEL_DIR
