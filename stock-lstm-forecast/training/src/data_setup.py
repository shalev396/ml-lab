"""Download -> cache -> indicators -> sliding windows -> chronological splits.

Source: yfinance daily OHLCV for `cfg.ticker` from `cfg.start`, cached in training/data/<TICKER>.csv
and reused on every later run. Fallback when yfinance is unreachable on a fresh clone: a seeded
geometric-Brownian-motion series, clearly labelled in training/data/DATA_SOURCE.txt.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import model as M  # ../model/model.py (put on sys.path by src/__init__.py)

from . import utils
from .config import Config

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


# --------------------------------------------------------------------------- download + cache
def _synthetic_prices(cfg: Config) -> pd.DataFrame:
    """Offline fallback: seeded GBM OHLCV (only used when yfinance fails and nothing is cached)."""
    rng = np.random.default_rng(cfg.seed)
    dates = pd.bdate_range(cfg.start, periods=2900)
    n = len(dates)
    close = 25.0 * np.exp(np.cumsum(rng.normal(0.0007, 0.018, n)))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    volume = rng.lognormal(17.8, 0.35, n).astype(np.int64)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
                        index=pd.DatetimeIndex(dates, name="Date"))


def _download(cfg: Config) -> tuple[pd.DataFrame, str]:
    try:
        import yfinance as yf

        raw = yf.download(cfg.ticker, start=cfg.start, progress=False, auto_adjust=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw[OHLCV].dropna()
        if len(raw) < 300:
            raise RuntimeError(f"yfinance returned only {len(raw)} rows")
        return raw, f"yfinance: {cfg.ticker} daily OHLCV from {cfg.start}"
    except Exception as exc:  # offline / API change -> documented synthetic fallback
        print(f"[data] WARNING: yfinance failed ({exc}); using the SYNTHETIC GBM fallback")
        return _synthetic_prices(cfg), "SYNTHETIC geometric Brownian motion (yfinance unavailable)"


def load_prices(cfg: Config) -> pd.DataFrame:
    """Daily OHLCV, loaded from training/data/<TICKER>.csv when cached, else downloaded and cached.
    Smoke runs keep only the last `cfg.smoke_rows` days."""
    cache = utils.DATA_DIR / f"{cfg.ticker}.csv"
    if cache.is_file():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        df, source = _download(cfg)
        df.index.name = "Date"
        utils.DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache)
        (utils.DATA_DIR / "DATA_SOURCE.txt").write_text(source + "\n", encoding="utf-8")
    df = df[OHLCV].astype("float64").sort_index()
    return df.tail(cfg.smoke_rows) if cfg.smoke else df


def data_source() -> str:
    path = utils.DATA_DIR / "DATA_SOURCE.txt"
    return path.read_text(encoding="utf-8").strip() if path.is_file() else "unknown"


# --------------------------------------------------------------------------- indicators
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index with Wilder's smoothing (no pandas_ta dependency)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return (100.0 - 100.0 / (1.0 + gain / loss)).rename(f"RSI{period}")


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean().rename(f"EMA{span}")


def add_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Close, Volume, RSI14, EMA20, EMA50; the RSI warm-up rows are dropped. Every track uses this
    same trimmed frame so all variants are scored on exactly the same test days."""
    out = df[["Close", "Volume"]].copy()
    out[f"RSI{cfg.rsi_period}"] = rsi(df["Close"], cfg.rsi_period)
    out[f"EMA{cfg.ema_short}"] = ema(df["Close"], cfg.ema_short)
    out[f"EMA{cfg.ema_long}"] = ema(df["Close"], cfg.ema_long)
    return out.dropna()


# --------------------------------------------------------------------------- windows + splits
@dataclass
class Track:
    """One input/target representation: windows per split + how to turn predictions back into $."""
    name: str
    kind: str                           # "levels" | "returns" | "multivariate"
    X: dict[str, np.ndarray]            # split -> (n, lookback, n_features)
    y: dict[str, np.ndarray]            # split -> (n,) scaled target
    idx: dict[str, np.ndarray]          # split -> row index (into `close`) of each window's label
    close: np.ndarray                   # the trimmed close series ($)
    dates: pd.DatetimeIndex
    scaler: object = None               # levels / multivariate: MinMax scaler of the target
    ret_mu: float = 0.0                 # returns: train mean / std of the log-returns
    ret_sigma: float = 1.0
    feature_names: list[str] = field(default_factory=lambda: ["Close"])

    @property
    def n_features(self) -> int:
        return int(self.X["train"].shape[-1])

    def to_prices(self, scaled_pred: np.ndarray, split: str) -> np.ndarray:
        """One-step predictions on `split` -> original $ scale."""
        pred = np.asarray(scaled_pred, dtype=np.float64).reshape(-1)
        if self.kind == "returns":
            # one step per row: last_close (n,) and returns (n, 1) -> prices (n, 1)
            return M.to_prices(self.close[self.idx[split] - 1], pred[:, None], self.ret_mu, self.ret_sigma)[:, 0]
        return self.scaler.inverse_transform(pred.reshape(-1, 1)).ravel()

    def actual(self, split: str) -> np.ndarray:
        return self.close[self.idx[split]]


def split_bounds(n: int, cfg: Config) -> tuple[int, int]:
    """Row indices where validation and test start (chronological, nothing shuffled)."""
    return int(n * cfg.train_frac), int(n * (cfg.train_frac + cfg.val_frac))


def _windows(features: np.ndarray, target: np.ndarray, lookback: int, first_label: int, bounds):
    """X[i] = features[t-lookback:t], y[i] = target[t]; each window belongs to the split of its label t."""
    labels = np.arange(max(lookback, first_label), len(features))
    X = np.stack([features[t - lookback:t] for t in labels]).astype(np.float32)
    y = target[labels].astype(np.float32)
    i_val, i_test = bounds
    masks = {"train": labels < i_val, "val": (labels >= i_val) & (labels < i_test), "test": labels >= i_test}
    return ({s: X[m] for s, m in masks.items()}, {s: y[m] for s, m in masks.items()},
            {s: labels[m] for s, m in masks.items()})


def prepare_levels(feats: pd.DataFrame, cfg: Config) -> Track:
    """Close price levels, MinMax-scaled with a scaler fit on the train rows only."""
    close = feats["Close"].to_numpy(dtype=np.float64)
    bounds = split_bounds(len(close), cfg)
    scaler = MinMaxScaler().fit(close[:bounds[0], None])
    scaled = scaler.transform(close[:, None])
    X, y, idx = _windows(scaled, scaled[:, 0], cfg.lookback, cfg.lookback, bounds)
    return Track("levels", "levels", X, y, idx, close, feats.index, scaler=scaler)


def prepare_returns(feats: pd.DataFrame, cfg: Config) -> Track:
    """Daily log-returns, standardized with the train mean/std (the deployed representation).
    Returns are roughly stationary, so the network never sees targets outside its training range."""
    close = feats["Close"].to_numpy(dtype=np.float64)
    bounds = split_bounds(len(close), cfg)
    rets = np.zeros(len(close), dtype=np.float32)
    rets[1:] = M.log_returns(close)            # rets[t] = ln(C_t / C_{t-1}); rets[0] is a filler
    mu, sigma = float(rets[1:bounds[0]].mean()), float(rets[1:bounds[0]].std()) or 1.0
    scaled = M.standardize(rets, mu, sigma)
    # first label lookback+1: no window contains the rets[0] filler
    X, y, idx = _windows(scaled[:, None], scaled, cfg.lookback, cfg.lookback + 1, bounds)
    return Track("returns", "returns", X, y, idx, close, feats.index, ret_mu=mu, ret_sigma=sigma)


def prepare_multivariate(feats: pd.DataFrame, cfg: Config) -> Track:
    """Close, Volume, RSI, EMA20, EMA50 (MinMax on train) -> next close (its own MinMax on train)."""
    values = feats.to_numpy(dtype=np.float64)
    close = feats["Close"].to_numpy(dtype=np.float64)
    bounds = split_bounds(len(close), cfg)
    features = MinMaxScaler().fit(values[:bounds[0]]).transform(values)
    target_scaler = MinMaxScaler().fit(close[:bounds[0], None])
    target = target_scaler.transform(close[:, None])[:, 0]
    X, y, idx = _windows(features, target, cfg.lookback, cfg.lookback, bounds)
    return Track("multivariate", "multivariate", X, y, idx, close, feats.index, scaler=target_scaler,
                 feature_names=list(feats.columns))


def prepare_tracks(df: pd.DataFrame, cfg: Config) -> dict[str, Track]:
    feats = add_indicators(df, cfg)
    return {t.name: t for t in (prepare_levels(feats, cfg), prepare_returns(feats, cfg),
                                prepare_multivariate(feats, cfg))}


def align_test(tracks: dict[str, Track]) -> np.ndarray:
    """Test label rows shared by every track (the returns track starts one row later)."""
    common = None
    for track in tracks.values():
        common = track.idx["test"] if common is None else np.intersect1d(common, track.idx["test"])
    return common


def split_summary(track: Track) -> pd.DataFrame:
    rows = {}
    for split, labels in track.idx.items():
        rows[split] = {"windows": len(labels), "first label": track.dates[labels[0]].date(),
                       "last label": track.dates[labels[-1]].date(),
                       "close min $": round(float(track.close[labels].min()), 2),
                       "close max $": round(float(track.close[labels].max()), 2)}
    return pd.DataFrame(rows).T
