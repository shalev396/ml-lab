"""California housing: download -> cache in training/data -> feature engineering -> splits.

Primary source: `sklearn.datasets.fetch_california_housing` (StatLib, 20,640 districts, 1990 census).
Fallback: the raw `housing.csv` from the "Hands-On Machine Learning" repo (same districts, raw totals),
converted to the same 8 per-household features. The CSV cache is reused on every later run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import model as M

CSV_NAME = "california_housing.csv"
FALLBACK_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"


def download(data_dir: str | Path) -> Path:
    """Cache the dataset as CSV (columns: 8 raw features + MedHouseVal) and return its path."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / CSV_NAME
    if path.is_file():
        print(f"cached: {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
        return path
    try:
        from sklearn.datasets import fetch_california_housing

        frame = fetch_california_housing(as_frame=True, data_home=str(data_dir / "sklearn")).frame
        print("downloaded via sklearn.datasets.fetch_california_housing")
    except Exception as err:  # noqa: BLE001 - any network/source failure -> documented fallback
        print(f"sklearn download failed ({err!r}); using the fallback CSV")
        frame = _from_raw_csv(pd.read_csv(FALLBACK_URL))
    frame[[*M.RAW_FEATURES, M.TARGET]].to_csv(path, index=False)
    print(f"saved: {path.name} ({len(frame):,} rows)")
    return path


def _from_raw_csv(raw: pd.DataFrame) -> pd.DataFrame:
    """housing.csv (district totals) -> sklearn's per-household features; rows without bedrooms dropped."""
    raw = raw.dropna(subset=["total_bedrooms"])
    return pd.DataFrame({
        "MedInc": raw["median_income"],
        "HouseAge": raw["housing_median_age"],
        "AveRooms": raw["total_rooms"] / raw["households"],
        "AveBedrms": raw["total_bedrooms"] / raw["households"],
        "Population": raw["population"],
        "AveOccup": raw["population"] / raw["households"],
        "Latitude": raw["latitude"],
        "Longitude": raw["longitude"],
        M.TARGET: raw["median_house_value"] / M.TARGET_UNIT_USD,
    }).reset_index(drop=True)


def load_frame(cfg) -> pd.DataFrame:
    """Raw districts (8 features + target); smoke runs keep a random `cfg.subset_rows` sample."""
    frame = pd.read_csv(download(cfg.data_dir))
    if cfg.subset_rows and cfg.subset_rows < len(frame):
        frame = frame.sample(cfg.subset_rows, random_state=cfg.seed).reset_index(drop=True)
    return frame


def with_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Frame with the 3 engineered ratios added (computed by model.engineer, as at inference)."""
    feats = M.engineer(frame)
    return frame.assign(**{name: feats[name] for name in M.ENGINEERED_FEATURES})


def split(frame: pd.DataFrame, cfg) -> dict[str, pd.DataFrame]:
    """80/20 train/test, then 15 % of train as validation -> {"train", "val", "test"} raw frames."""
    train_full, test = train_test_split(frame, test_size=cfg.test_size, random_state=cfg.seed)
    train, val = train_test_split(train_full, test_size=cfg.val_size, random_state=cfg.seed)
    splits = {"train": train, "val": val, "test": test}
    print("  ".join(f"{k}={len(v):,}" for k, v in splits.items()), f"features={len(M.FEATURES)}")
    return {k: v.reset_index(drop=True) for k, v in splits.items()}


def xy(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Raw frame -> (unscaled 11-column feature matrix, target in $100k)."""
    return M.to_matrix(frame), frame[M.TARGET].to_numpy(dtype=np.float64)


# City centres for the Space examples: one real test-split district near each.
EXAMPLE_CITIES = {"San Francisco": (37.77, -122.42), "Los Angeles": (34.05, -118.24),
                  "San Diego": (32.72, -117.16), "Sacramento": (38.58, -121.49), "Fresno": (36.74, -119.79)}


def example_districts(test: pd.DataFrame) -> pd.DataFrame:
    """The test-split district closest to each city in EXAMPLE_CITIES (for space/examples/)."""
    rows = []
    for city, (lat, lon) in EXAMPLE_CITIES.items():
        dist = (test["Latitude"] - lat) ** 2 + (test["Longitude"] - lon) ** 2
        rows.append({"city": city, **test.loc[dist.idxmin(), [*M.RAW_FEATURES, M.TARGET]].to_dict()})
    return pd.DataFrame(rows).round(4)
