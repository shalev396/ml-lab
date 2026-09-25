"""Data: a SEEDED SYNTHETIC month of elevator sensor readings -> time-based splits -> sliding windows.

Why synthetic: the brief this project follows (ProjectPro, "time series project for elevator predictive
maintenance with IoT sensor data") describes one month of MINUTELY readings from 11 sensors with a
binary `Status` target. The public Kaggle set usually linked to it
(`shivamb/elevator-predictive-maintenance-dataset`) has 112,001 rows of `ID, revolutions, humidity,
vibration, x1..x5` and NO failure/status label, so it cannot train a failure classifier. The generator
below produces a spec-matched month instead: daily usage seasonality drives all sensors, and five
degradation episodes (hours of drift, then a failure, then a maintenance reset) provide the positives.
Everything learned here is about this generator, not about real elevators.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import model as M

from . import utils
from .config import Config

CACHE = utils.DATA_DIR / "elevator.csv"
SOURCE_FILE = utils.DATA_DIR / "DATA_SOURCE.txt"
SOURCE = ("SYNTHETIC generator, spec-matched to the article: 31 days x 1440 min, 11 sensors, degradation "
          "episodes + maintenance resets (Kaggle set lacks a usable binary Status label)")
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- generator
def generate_synthetic(cfg: Config) -> pd.DataFrame:
    """Seeded synthetic month: one row per minute, 11 sensors + Status (1 = degradation/failure)."""
    rng = np.random.default_rng(cfg.seed)
    n = cfg.days * 1440
    index = pd.date_range(cfg.start_timestamp, periods=n, freq="min", name="timestamp")
    hour = index.hour.to_numpy() + index.minute.to_numpy() / 60.0
    dow = index.dayofweek.to_numpy()

    # daily usage profile: morning / lunch / evening peaks, quiet nights and weekends
    usage = (0.08
             + 0.55 * np.exp(-((hour - 8.5) ** 2) / (2 * 1.6 ** 2))
             + 0.40 * np.exp(-((hour - 13.0) ** 2) / (2 * 2.0 ** 2))
             + 0.60 * np.exp(-((hour - 18.0) ** 2) / (2 * 1.8 ** 2)))
    usage = np.clip(usage * np.where(dow >= 5, 0.35, 1.0) + rng.normal(0, 0.04, n), 0.0, 1.4)

    df = pd.DataFrame(index=index)
    df["temperature"] = 21.0 + 5.0 * usage + 1.5 * np.sin(2 * np.pi * (hour - 14) / 24) + rng.normal(0, 0.3, n)
    df["humidity"] = np.clip(45.0 + 6.0 * np.sin(2 * np.pi * (hour - 4) / 24) + rng.normal(0, 2.0, n), 15, 95)
    df["vibration_rms"] = np.clip(0.45 + 0.90 * usage + rng.normal(0, 0.06, n), 0.05, None)
    spikes = (rng.random(n) < 0.002) * rng.uniform(1.0, 3.0, n)
    df["vibration_peak"] = df["vibration_rms"] * 2.6 + np.abs(rng.normal(0, 0.25, n)) + spikes
    df["motor_current"] = np.clip(7.5 + 9.0 * usage + rng.normal(0, 0.4, n), 0.5, None)
    df["motor_rpm"] = np.clip(1500.0 * np.clip(usage, 0.05, 1.0) + rng.normal(0, 30.0, n), 0, None)
    df["door_cycles"] = rng.poisson(np.clip(3.2 * usage, 0.05, None))
    df["load_kg"] = np.clip(620.0 * usage * rng.uniform(0.2, 1.0, n), 0, 680)
    df["acoustic_db"] = 52.0 + 14.0 * usage + rng.normal(0, 1.2, n)
    df["power_kw"] = np.clip(2.2 + 6.5 * usage + rng.normal(0, 0.3, n), 0.2, None)

    decay = 0.00045                                   # oil: slow decay, refilled at each maintenance
    oil = 100.0 - decay * np.arange(n)

    # Episode anchors: train, val AND test each contain at least one full episode.
    status = np.zeros(n, dtype=np.int64)
    b1d, b2d = cfg.days * cfg.train_frac, cfg.days * (cfg.train_frac + cfg.val_frac)
    anchors = list(np.linspace(2.5, b1d - 1.4, max(1, cfg.n_episodes - 2)))
    if cfg.n_episodes >= 2:
        anchors.append(b1d + (b2d - b1d) / 2.0)
    if cfg.n_episodes >= 3:
        anchors.append(b2d + (cfg.days - b2d) / 2.0)
    for anchor in anchors:
        start = int((anchor + rng.uniform(-0.35, 0.35)) * 1440)
        pre_len = int(rng.integers(6 * 60, 14 * 60))      # 6-14 h of gradual drift
        fail_len = int(rng.integers(45, 120))             # then the failure itself
        end = min(start + pre_len + fail_len, n - 1)
        ramp = np.zeros(end - start)
        ramp[:pre_len] = np.linspace(0.0, 1.0, min(pre_len, len(ramp)))
        ramp[pre_len:] = 1.0
        rows = df.index[start:end]
        df.loc[rows, "vibration_rms"] += 1.8 * ramp
        df.loc[rows, "vibration_peak"] += 4.5 * ramp + np.abs(rng.normal(0, 0.5, end - start)) * ramp
        df.loc[rows, "motor_current"] += 5.0 * ramp
        df.loc[rows, "temperature"] += 7.0 * ramp
        df.loc[rows, "acoustic_db"] += 9.0 * ramp
        df.loc[rows, "motor_rpm"] -= 90.0 * ramp
        df.loc[rows, "power_kw"] += 2.0 * ramp
        oil[start:end] -= np.linspace(0.0, 7.0, end - start)   # accelerated oil loss
        status[start:end] = 1
        if end < n - 1:                                         # maintenance reset: fresh oil
            oil[end:] = 100.0 - decay * np.arange(n - end)

    df["oil_level"] = np.clip(oil, 0, 100)
    df["Status"] = status
    df = df[M.SENSORS + ["Status"]].astype(np.float64).round(4)
    df["Status"] = df["Status"].astype(np.int64)
    return df


def load_elevator(cfg: Config) -> pd.DataFrame:
    """Cached month (training/data/elevator.csv) or a fresh run of the seeded generator."""
    if CACHE.is_file():
        df = pd.read_csv(CACHE, index_col=0, parse_dates=True)
        print(f"[data] cache {CACHE.name}: {len(df):,} rows | source: {SOURCE_FILE.read_text(encoding='utf-8').strip()}"
              if SOURCE_FILE.is_file() else f"[data] cache {CACHE.name}: {len(df):,} rows")
        return df
    df = generate_synthetic(cfg)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE)
    SOURCE_FILE.write_text(SOURCE + "\n", encoding="utf-8")
    print(f"[data] generated {len(df):,} synthetic rows -> {CACHE.name}")
    return df


def episodes(df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(start, end) of every run of Status == 1."""
    status = df["Status"].to_numpy()
    edges = np.flatnonzero(np.diff(np.concatenate([[0], status, [0]])))
    return [(df.index[a], df.index[b - 1]) for a, b in zip(edges[::2], edges[1::2])]


# --------------------------------------------------------------------------- windows
@dataclass
class Windows:
    """Per split: raw (n, window, 11) readings, standardised copies for the Keras nets, labels,
    window-end times (features) and label times (plots). `scaler` is fit on the train rows only."""
    scaler: StandardScaler
    boundaries: tuple[int, int]
    n_rows: int
    raw: dict = field(default_factory=dict)
    scaled: dict = field(default_factory=dict)
    y: dict = field(default_factory=dict)
    ends: dict = field(default_factory=dict)
    label_times: dict = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({s: {"windows": len(self.y[s]), "failure windows": int(self.y[s].sum()),
                                 "failure share": round(float(self.y[s].mean()), 4),
                                 "from": self.label_times[s][0], "to": self.label_times[s][-1]}
                             for s in SPLITS}).T


def build_windows(df: pd.DataFrame, cfg: Config) -> Windows:
    """Time-based 70/15/15 split of the rows, then sliding windows INSIDE each segment: a window
    [end-59, end] and its label at end+label_lag both lie in one segment (nothing straddles a boundary)."""
    values = df[M.SENSORS].to_numpy(dtype=np.float64)
    status = df["Status"].to_numpy(dtype=np.int64)
    n = len(df)
    b1, b2 = int(n * cfg.train_frac), int(n * (cfg.train_frac + cfg.val_frac))
    scaler = StandardScaler().fit(values[:b1])
    scaled = scaler.transform(values).astype(np.float32)
    offsets = np.arange(-(cfg.window - 1), 1)
    out = Windows(scaler=scaler, boundaries=(b1, b2), n_rows=n)
    for name, (lo, hi) in zip(SPLITS, ((0, b1), (b1, b2), (b2, n))):
        ends = np.arange(lo + cfg.window - 1, hi - cfg.label_lag, cfg.stride)
        gather = ends[:, None] + offsets[None, :]
        out.raw[name] = values[gather]
        out.scaled[name] = scaled[gather]
        out.y[name] = status[ends + cfg.label_lag]
        out.ends[name] = df.index[ends]
        out.label_times[name] = df.index[ends + cfg.label_lag]
    return out


def features(windows: Windows, cfg: Config) -> dict[str, np.ndarray]:
    """The 63 engineered features per split, computed by model.window_features (same code as serving)."""
    return {s: M.window_features(windows.raw[s], windows.ends[s], M.SENSORS, cfg.fft_top_k) for s in SPLITS}


# --------------------------------------------------------------------------- Space examples
def export_examples(df: pd.DataFrame, cfg: Config) -> dict[str, str]:
    """Cut healthy / degrading / failing 60-minute windows from the TEST segment into cfg.examples_dir."""
    b2 = int(len(df) * (cfg.train_frac + cfg.val_frac))
    test = df.iloc[b2:]
    runs = [(a, b) for a, b in episodes(test)]
    if not runs:
        raise RuntimeError("the test segment has no failure episode to cut examples from")
    start, end = max(runs, key=lambda r: r[1] - r[0])
    i_start, i_end = test.index.get_loc(start), test.index.get_loc(end) + 1
    ramp = i_end - i_start
    fail_end = i_end - cfg.label_lag - 5                                     # label still inside the failure
    picks = {
        "failing": slice(fail_end - cfg.window, fail_end),                   # the failure itself
        "degrading": slice(i_start + ramp // 2 - cfg.window, i_start + ramp // 2),   # halfway up the drift
    }
    status = test["Status"].to_numpy()
    margin = 240                                                             # healthy: midday, far from any episode
    for i in range(margin + cfg.window, len(test) - margin, 30):
        if test.index[i].hour == 12 and test.index[i].dayofweek < 5 and not status[i - margin:i + margin].any():
            picks["healthy"] = slice(i - cfg.window, i)
            break
    cfg.examples_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, sl in picks.items():
        path = cfg.examples_dir / f"{name}.csv"
        test.iloc[sl][M.SENSORS].to_csv(path, index_label="timestamp")
        written[name] = path.as_posix()
    return written
