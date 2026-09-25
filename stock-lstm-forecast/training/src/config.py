"""Every tunable of the stock-lstm-forecast run in one dataclass."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import utils


@dataclass
class Config:
    # --- data ---------------------------------------------------------------------------------
    ticker: str = "AAPL"
    start: str = "2015-01-01"         # first day downloaded from yfinance (cached in training/data/)
    lookback: int = 60                # sliding-window length (trading days)
    train_frac: float = 0.70          # chronological split: 70 % train / 15 % validation / 15 % test
    val_frac: float = 0.15

    # --- hand-rolled indicators (multivariate variant) -----------------------------------------
    rsi_period: int = 14
    ema_short: int = 20
    ema_long: int = 50

    # --- models ---------------------------------------------------------------------------------
    rnn_units: int = 64               # SimpleRNN variant
    lstm_units: int = 64              # every LSTM variant (the deployed one included)
    num_layers: int = 2               # depth of the stacked multivariate LSTM
    dropout: float = 0.2

    # --- training (Adam + MSE, early stopping on validation loss) -------------------------------
    epochs: int = 40
    batch_size: int = 32
    lr: float = 1e-3
    patience: int = 6
    seed: int = 42

    # --- multi-step evaluation (what the Space does: recursive forecasts from a cutoff) --------
    eval_horizon: int = 20            # business days forecast from each test origin
    eval_stride: int = 5              # a new forecast origin every N test days

    # --- smoke run ------------------------------------------------------------------------------
    smoke: bool = field(default_factory=lambda: os.getenv("SMOKE_TEST", "0") == "1")
    smoke_rows: int = 400             # last N trading days only

    def __post_init__(self):
        if self.smoke:  # same code path, tiny data + tiny models: finishes in well under a minute
            self.epochs = 1
            self.rnn_units = self.lstm_units = 8
            self.num_layers = 1
            self.patience = 1
            self.eval_horizon = 5

    @property
    def model_dir(self) -> Path:
        """Where export writes the model repo; smoke runs never touch the real model/."""
        return utils.OUTPUTS_DIR / "smoke" / "model" if self.smoke else utils.MODEL_DIR

    @property
    def outputs_dir(self) -> Path:
        return utils.OUTPUTS_DIR / "smoke" if self.smoke else utils.OUTPUTS_DIR

    def to_dict(self) -> dict:
        return asdict(self)
