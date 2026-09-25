"""Every tunable of the elevator-maintenance run in one dataclass."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import model as M  # ../model/model.py (put on sys.path by src/__init__.py)

from . import utils


@dataclass
class Config:
    # --- data: a SYNTHETIC month of minutely readings (31 x 1440 = 44,640 rows, 11 sensors) --------
    days: int = 31
    start_timestamp: str = "2024-01-01"
    n_episodes: int = 5               # degradation -> failure -> maintenance-reset episodes
    seed: int = 42                    # seeds the generator AND every model

    # --- windowing (fixed by the deployed model: model.WINDOW / LABEL_LAG / FFT_TOP_K) ----------
    window: int = M.WINDOW            # minutes per input window
    label_lag: int = M.LABEL_LAG      # predict Status this many minutes after the window ends
    fft_top_k: int = M.FFT_TOP_K
    stride: int = 5                   # minutes between consecutive window ends
    train_frac: float = 0.70          # time-based split; no window/label straddles a boundary
    val_frac: float = 0.15

    # --- RandomForest (deployed) -------------------------------------------------------------------
    rf_estimators: int = 300
    n_jobs: int = -1

    # --- Keras comparison models (raw standardised windows) -------------------------------------
    cnn_filters: tuple[int, ...] = (64, 128)
    cnn_kernel: int = 5
    lstm_units: int = 64
    dense_units: int = 64
    dropout: float = 0.3
    epochs: int = 8
    batch_size: int = 128
    lr: float = 1e-3
    patience: int = 3                 # early stopping on validation PR-AUC

    # --- decision threshold search (max F1 on validation) ----------------------------------------
    threshold_min: float = 0.05
    threshold_max: float = 0.95

    # --- run -----------------------------------------------------------------------------------------
    smoke: bool = field(default_factory=lambda: os.getenv("SMOKE_TEST", "0") == "1")

    def __post_init__(self):
        self.cnn_filters = tuple(self.cnn_filters)
        if self.smoke:  # same code path, fewer windows and tiny models (well under a minute on CPU)
            self.stride = max(self.stride, 20)
            self.rf_estimators = min(self.rf_estimators, 50)
            self.cnn_filters = (8, 16)
            self.lstm_units = 8
            self.dense_units = 8
            self.epochs = 1
            self.patience = 1

    @property
    def model_dir(self) -> Path:
        """Where export writes the model repo; smoke runs never touch the real model/."""
        return utils.OUTPUTS_DIR / "smoke" / "model" if self.smoke else utils.MODEL_DIR

    @property
    def outputs_dir(self) -> Path:
        """Run artifacts that are not part of the model repo."""
        return utils.OUTPUTS_DIR / "smoke" if self.smoke else utils.OUTPUTS_DIR

    @property
    def examples_dir(self) -> Path:
        """The Space's example CSVs (smoke runs write a copy under outputs/smoke/)."""
        return self.outputs_dir / "examples" if self.smoke else utils.SPACE_DIR / "examples"
