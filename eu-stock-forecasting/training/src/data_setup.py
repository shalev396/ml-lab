"""EuStockMarkets: download -> cache in training/data -> chronological splits -> NN lag windows.

1,860 daily closing prices (1991-1998) of DAX (Germany), SMI (Switzerland), CAC (France) and
FTSE (UK), from R's `datasets` package. The R ts object has no calendar (start 1991.496,
frequency 260), so a synthetic business-day index from 1991-07-01 is attached (`model.business_days`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import model as M

from . import utils
from .config import Config

CACHE = utils.DATA_DIR / "EuStockMarkets.csv"
SOURCE_NOTE = utils.DATA_DIR / "DATA_SOURCE.txt"
FALLBACK_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/EuStockMarkets.csv"


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep the four index columns (case-insensitive) and attach the business-day index."""
    lookup = {str(c).upper().strip(): c for c in raw.columns}
    missing = [ix for ix in M.INDICES if ix not in lookup]
    if missing:
        raise ValueError(f"columns missing from the source: {missing} (got {list(raw.columns)})")
    frame = raw[[lookup[ix] for ix in M.INDICES]].astype("float64").dropna()
    frame.columns = list(M.INDICES)
    frame.index = M.business_days(len(frame))
    return frame


def load_eustocks() -> pd.DataFrame:
    """Cached CSV -> statsmodels `get_rdataset` -> the Rdatasets CSV mirror. Raises if all fail."""
    if CACHE.is_file():
        frame = M.read_frame(CACHE)
        print(f"[data] cache {CACHE.name}: {len(frame):,} rows")
        return frame
    utils.DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import statsmodels.api as sm

        frame = _normalize(sm.datasets.get_rdataset("EuStockMarkets", "datasets").data)
        source = "statsmodels get_rdataset('EuStockMarkets', 'datasets')"
    except Exception as err:  # network / Rdatasets hiccup -> direct CSV
        print(f"[data] get_rdataset failed ({err}); trying {FALLBACK_URL}")
        frame = _normalize(pd.read_csv(FALLBACK_URL))
        source = f"Rdatasets CSV {FALLBACK_URL}"
    frame.to_csv(CACHE, float_format="%.2f")
    SOURCE_NOTE.write_text(f"{source}\nBusiness-day index synthesised from {M.INDEX_START} "
                           "(R ts started at fractional year 1991.496, frequency 260).\n", encoding="utf-8")
    print(f"[data] {source} -> {CACHE.name} ({len(frame):,} rows)")
    return frame


@dataclass
class Splits:
    """Chronological split of `frame`: rows [0, n_fit) fit, [n_fit, n_fit+n_val) validation, rest test."""
    frame: pd.DataFrame
    n_fit: int
    n_val: int
    n_test: int

    @property
    def fit(self) -> pd.DataFrame:
        return self.frame.iloc[: self.n_fit]

    @property
    def val(self) -> pd.DataFrame:
        return self.frame.iloc[self.n_fit: self.n_fit + self.n_val]

    @property
    def test(self) -> pd.DataFrame:
        return self.frame.iloc[self.n_fit + self.n_val:]

    @property
    def eval_start(self) -> int:
        """First row that is predicted (validation + test are walked forward together)."""
        return self.n_fit

    def summary(self) -> pd.DataFrame:
        rows = {name: part for name, part in (("fit", self.fit), ("validation", self.val), ("test", self.test))}
        return pd.DataFrame({"rows": {k: len(v) for k, v in rows.items()},
                             "from": {k: f"{v.index[0]:%Y-%m-%d}" for k, v in rows.items()},
                             "to": {k: f"{v.index[-1]:%Y-%m-%d}" for k, v in rows.items()}})


def make_splits(frame: pd.DataFrame, cfg: Config) -> Splits:
    """Last `test_days` = test, the `val_days` before them = validation, the rest = fit.
    Smoke runs keep only the last `smoke_rows` rows."""
    if cfg.smoke:
        frame = frame.tail(cfg.smoke_rows)
    n_fit = len(frame) - cfg.val_days - cfg.test_days
    if n_fit < 2 * cfg.window + 20:
        raise ValueError(f"not enough rows ({len(frame)}) for the split")
    return Splits(frame, n_fit, cfg.val_days, cfg.test_days)


def adf_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Augmented Dickey-Fuller test on the levels and on the first differences of every index."""
    import warnings

    from statsmodels.tsa.stattools import adfuller

    rows = {}
    for ix in frame.columns:
        with warnings.catch_warnings():  # statsmodels 0.15 announces a new return type; values are the same
            warnings.simplefilter("ignore", FutureWarning)
            level, diff = adfuller(frame[ix].to_numpy()), adfuller(np.diff(frame[ix].to_numpy()))
        rows[ix] = {"ADF (levels)": level[0], "p (levels)": level[1],
                    "ADF (1st diff)": diff[0], "p (1st diff)": diff[1]}
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- neural-net inputs
def make_lag_windows(values: np.ndarray, window: int):
    """X[i] = values[t-window:t], y[i] = values[t] for every t >= window; also returns the t's."""
    X = np.stack([values[t - window: t] for t in range(window, len(values))]).astype("float32")
    return X, values[window:].astype("float32"), np.arange(window, len(values))


def prepare_nn_data(splits: Splits, index: str, cfg: Config) -> dict:
    """MinMax scaling fit on the fit window only, then windows whose TARGET day falls in the fit /
    validation / test window. Inputs always come from actual history (one-step-ahead, walk-forward)."""
    values = splits.frame[index].to_numpy(dtype="float64")
    scaler = MinMaxScaler().fit(values[: splits.n_fit].reshape(-1, 1))
    X, y, t = make_lag_windows(scaler.transform(values.reshape(-1, 1)).ravel(), cfg.window)
    fit, val = t < splits.n_fit, (t >= splits.n_fit) & (t < splits.n_fit + splits.n_val)
    test = t >= splits.n_fit + splits.n_val
    return {"X_train": X[fit], "y_train": y[fit], "X_val": X[val], "y_val": y[val],
            "X_test": X[test], "y_test": y[test], "X_eval": X[val | test], "scaler": scaler}
