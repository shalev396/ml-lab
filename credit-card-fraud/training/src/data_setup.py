"""Download, cache and split the ULB credit-card fraud dataset.

Primary source: Kaggle `mlg-ulb/creditcardfraud` via kagglehub (anonymous download).
Fallback:       OpenML `data_id=1597` (same 284,807 rows) via scikit-learn.
Cache:          training/data/creditcard.csv (skipped when present).
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import model as M  # ../model/model.py (put on sys.path by src/__init__.py)

from . import utils
from .config import Config

KAGGLE_DATASET = "mlg-ulb/creditcardfraud"
OPENML_ID = 1597
CSV_NAME = "creditcard.csv"


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def items(self):
        return {"train": self.train, "val": self.val, "test": self.test}.items()


def download(data_dir: Path = utils.DATA_DIR) -> Path:
    """Return the cached CSV, downloading it on first use (kagglehub, then OpenML)."""
    csv_path = Path(data_dir) / CSV_NAME
    if csv_path.is_file():
        return csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import kagglehub

        print(f"[data] downloading {KAGGLE_DATASET} with kagglehub ...")
        source = next(Path(kagglehub.dataset_download(KAGGLE_DATASET)).rglob(CSV_NAME))
        shutil.copyfile(source, csv_path)
    except Exception as err:  # no Kaggle access / network hiccup: same data from OpenML
        print(f"[data] kagglehub failed ({err!r}); falling back to OpenML data_id={OPENML_ID} ...")
        from sklearn.datasets import fetch_openml

        frame = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto").frame
        frame[M.TARGET] = frame[M.TARGET].astype(str).str.strip("'").astype(float).astype(int)
        frame.to_csv(csv_path, index=False)
    print(f"[data] cached {csv_path.name} ({csv_path.stat().st_size / 1e6:.0f} MB)")
    return csv_path


def load_dataframe(cfg: Config) -> pd.DataFrame:
    """All transactions (exact duplicates dropped so no row lands in two splits); tiny subset when smoke."""
    df = pd.read_csv(download())[[*M.INPUT_COLUMNS, M.TARGET]]
    df[M.TARGET] = df[M.TARGET].astype(int)
    df = df.drop_duplicates().reset_index(drop=True)
    if cfg.smoke and cfg.smoke_rows < len(df):
        df, _ = train_test_split(df, train_size=cfg.smoke_rows, stratify=df[M.TARGET], random_state=cfg.seed)
        df = df.reset_index(drop=True)
    return df


def split(df: pd.DataFrame, cfg: Config) -> Splits:
    """Stratified train / validation / test split (fractions of all rows: 1 - val - test, val, test)."""
    rest, test = train_test_split(df, test_size=cfg.test_size, stratify=df[M.TARGET], random_state=cfg.seed)
    train, val = train_test_split(rest, test_size=cfg.val_size / (1 - cfg.test_size),
                                  stratify=rest[M.TARGET], random_state=cfg.seed)
    return Splits(*(part.reset_index(drop=True) for part in (train, val, test)))


def load_splits(cfg: Config) -> Splits:
    splits = split(load_dataframe(cfg), cfg)
    for name, part in splits.items():
        print(f"[data] {name:<5} {len(part):>7,} rows  {int(part[M.TARGET].sum()):>4} frauds "
              f"({part[M.TARGET].mean():.3%})")
    return splits
