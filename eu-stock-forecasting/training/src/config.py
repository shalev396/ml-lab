"""Every tunable of the eu-stock-forecasting run in one dataclass."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import model as M  # ../model/model.py (put on sys.path by src/__init__.py)

from . import utils


@dataclass
class Config:
    # --- data: chronological fit / validation / test split, nothing is shuffled -----------------
    indices: tuple[str, ...] = M.INDICES
    target: str = "DAX"               # headline series: primary metric, card plots
    val_days: int = 100               # validation window (selects the deployed method, early-stops the nets)
    test_days: int = 100              # final holdout, reported once
    seed: int = 42

    # --- classical models -----------------------------------------------------------------
    arima_max_p: int = 3              # ARIMA(p, d, q) grid, p, q in 0..3, picked by AIC on the fit window
    arima_max_q: int = 3
    arima_d: int = 1                  # from the ADF test: levels have a unit root, first differences don't
    hw_trend: str = "add"             # Holt-Winters additive trend, no seasonality
    var_maxlags: int = 12             # VAR lag order picked by AIC up to this
    select_tolerance: float = 0.02    # deploy the fewest-parameter classical model within 2% (relative) of the best val MAPE

    # --- neural nets (compared, not deployed) ----------------------------------------------
    window: int = 20                  # lag window of the MLP and LSTM inputs
    mlp_hidden: tuple[int, ...] = (64, 32)
    lstm_units: int = 64
    dropout: float = 0.1
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 60
    patience: int = 8                 # early stopping on the validation window

    # --- multi-step backtest (what the Space serves) -----------------------------------------
    backtest_horizon: int = 20        # business days ahead
    backtest_step: int = 10           # a new forecast origin every N days of the test window

    # --- serving (written to config.json) -------------------------------------------------
    history_days: int = 250           # history points returned by Predictor.predict
    max_horizon: int = 120
    min_history: int = 100

    # --- run ------------------------------------------------------------------------------
    smoke: bool = field(default_factory=lambda: os.getenv("SMOKE_TEST", "0") == "1")
    smoke_rows: int = 400             # smoke: fit on the last 400 rows only

    def __post_init__(self):
        self.indices = tuple(self.indices)
        self.mlp_hidden = tuple(self.mlp_hidden)
        if self.smoke:  # tiny, fast end-to-end run: same code path, small data + small models
            self.val_days = self.test_days = 30
            self.arima_max_p = self.arima_max_q = 1
            self.var_maxlags = 4
            self.window = 10
            self.mlp_hidden = (8,)
            self.lstm_units = 8
            self.epochs = 1
            self.patience = 1
            self.backtest_horizon = 10
            self.backtest_step = 15

    @property
    def nn_indices(self) -> tuple[str, ...]:
        """Indices the neural nets are trained on (smoke: only the target, TensorFlow fits are the slow part)."""
        return (self.target,) if self.smoke else self.indices

    @property
    def model_dir(self) -> Path:
        """Where export writes the model repo; smoke runs never touch the real model/."""
        return utils.OUTPUTS_DIR / "smoke" / "model" if self.smoke else utils.MODEL_DIR

    @property
    def outputs_dir(self) -> Path:
        """Run artifacts that are not part of the model repo (AIC grid, tables)."""
        return utils.OUTPUTS_DIR / "smoke" if self.smoke else utils.OUTPUTS_DIR
