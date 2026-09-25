"""Data: download -> cache in training/data -> metadata -> stratified catalog / val / test split.

Primary source: Kaggle `paramaggarwal/fashion-product-images-small` (44k Myntra product photos,
60x80 px, plus `styles.csv` with gender / masterCategory / subCategory / articleType labels),
downloaded anonymously with kagglehub. `styles.csv` is copied into `training/data/`; the jpgs stay
in the kagglehub cache and `training/data/kaggle_path.txt` points at them (no 570 MB duplicate).

Fallback (if Kaggle is unreachable): CIFAR-10 via `keras.datasets`, written to
`training/data/cifar_images/` so the same file-based pipeline runs; the class name doubles as
every label. Retrieval numbers on the fallback are not comparable with the fashion numbers.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from . import utils
from .config import Config

CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
COLUMNS = ["id", "image_path", "name", "gender", "master_category", "sub_category", "article_type", "base_colour"]
RENAME = {"productDisplayName": "name", "masterCategory": "master_category", "subCategory": "sub_category",
          "articleType": "article_type", "baseColour": "base_colour"}
LABELS = ("master_category", "article_type")   # the two granularities retrieval is scored on


@dataclass
class Splits:
    index: pd.DataFrame   # the searchable catalog
    val: pd.DataFrame     # held-out queries: pick the variant
    test: pd.DataFrame    # held-out queries: reported once
    source: str

    def items(self):
        return {"index": self.index, "val": self.val, "test": self.test}.items()

    def sizes(self) -> dict[str, int]:
        return {name: len(part) for name, part in self.items()}


# --------------------------------------------------------------------------- download
def download_fashion(cfg: Config) -> tuple[Path, Path]:
    """(styles.csv in training/data, images dir). kagglehub only runs when the cache is missing."""
    styles_local = utils.DATA_DIR / "styles.csv"
    pointer = utils.DATA_DIR / "kaggle_path.txt"
    if styles_local.is_file() and pointer.is_file():
        images_dir = Path(pointer.read_text(encoding="utf-8").strip())
        if images_dir.is_dir():
            print(f"[data] cached: {styles_local.name} + images in the kagglehub cache")
            return styles_local, images_dir

    import kagglehub

    root = Path(kagglehub.dataset_download(cfg.kaggle_dataset))
    images_dir = next((p for p in sorted(root.rglob("images")) if p.is_dir()), None)
    if images_dir is None:
        raise FileNotFoundError(f"no images/ folder under {root}")
    utils.DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(next(root.rglob("styles.csv")), styles_local)
    pointer.write_text(str(images_dir), encoding="utf-8")
    print(f"[data] downloaded {cfg.kaggle_dataset} (images stay in the kagglehub cache)")
    return styles_local, images_dir


def load_fashion(cfg: Config) -> pd.DataFrame:
    styles_csv, images_dir = download_fashion(cfg)
    df = pd.read_csv(styles_csv, encoding="utf-8", on_bad_lines="skip")  # a few rows have stray commas
    present = {entry.name for entry in os.scandir(images_dir)}
    df = df[(df["id"].astype(str) + ".jpg").isin(present)].rename(columns=RENAME)
    df = df.dropna(subset=["master_category", "article_type"]).copy()
    df["image_path"] = [str(images_dir / f"{i}.jpg") for i in df["id"]]
    df["name"] = df["name"].fillna("(unnamed product)")
    for col in ("gender", "sub_category", "base_colour"):
        df[col] = df[col].fillna("unknown")
    return df[COLUMNS].reset_index(drop=True)


def load_cifar_fallback(cfg: Config) -> pd.DataFrame:
    import keras

    (x, y), _ = keras.datasets.cifar10.load_data()
    out = utils.DATA_DIR / "cifar_images"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(min(cfg.fallback_images, len(x))):
        path = out / f"{i}.png"
        if not path.is_file():
            Image.fromarray(x[i]).save(path)
        cls = CIFAR_CLASSES[int(y[i][0])]
        rows.append({"id": i, "image_path": str(path), "name": f"CIFAR-10 {cls} #{i}", "gender": "unknown",
                     "master_category": cls, "sub_category": cls, "article_type": cls, "base_colour": "unknown"})
    return pd.DataFrame(rows, columns=COLUMNS)


def load_metadata(cfg: Config) -> tuple[pd.DataFrame, str]:
    """Fashion metadata (one row per product photo that exists on disk) and a source tag."""
    try:
        df, source = load_fashion(cfg), f"kaggle:{cfg.kaggle_dataset}"
    except Exception as err:  # documented fallback, loudly reported
        print(f"[data] FALLBACK to CIFAR-10: fashion dataset unavailable ({type(err).__name__}: {err})")
        df, source = load_cifar_fallback(cfg), "keras:cifar10 (fallback)"
    print(f"[data] {len(df):,} images · {df.master_category.nunique()} master categories · "
          f"{df.article_type.nunique()} article types  ({source})")
    return df, source


# --------------------------------------------------------------------------- split
def _readable(df: pd.DataFrame) -> pd.DataFrame:
    ok = []
    for path in df["image_path"]:
        try:
            with Image.open(path) as img:
                img.verify()
            ok.append(True)
        except Exception:
            ok.append(False)
    if not all(ok):
        print(f"[data] dropped {len(ok) - sum(ok)} unreadable image(s)")
    return df[np.asarray(ok, dtype=bool)].reset_index(drop=True)


def _strata(labels: pd.Series) -> pd.Series:
    """Stratification key: a label seen only once cannot be stratified, so it joins the largest label."""
    counts = labels.value_counts()
    return labels.where(labels.map(counts) >= 2, counts.index[0])


def split(cfg: Config, df: pd.DataFrame, source: str) -> Splits:
    """Stratified (by master category) catalog / val / test. Queries are never in the catalog."""
    counts = df["master_category"].value_counts()
    df = df[df["master_category"].isin(counts[counts >= cfg.min_category_count].index)].reset_index(drop=True)
    n_query = cfg.n_val + cfg.n_test
    index, queries = train_test_split(df, train_size=min(cfg.index_size, len(df) - n_query), test_size=n_query,
                                      stratify=_strata(df["master_category"]), random_state=cfg.seed)
    val, test = train_test_split(queries, train_size=cfg.n_val, test_size=cfg.n_test,
                                 stratify=_strata(queries["master_category"]), random_state=cfg.seed)
    splits = Splits(_readable(index), _readable(val), _readable(test), source)
    print(f"[data] split {splits.sizes()} (stratified by master category, seed {cfg.seed})")
    return splits


def load_images(paths) -> list[Image.Image]:
    """Decode every photo once (RGB, in memory: 60x80 px each, ~120 MB for 9,000)."""
    images = []
    for path in paths:
        with Image.open(path) as img:
            images.append(img.convert("RGB"))
    return images


def label_table(splits: Splits, label: str = "master_category") -> pd.DataFrame:
    """Products per label and split (for the EDA table)."""
    return pd.DataFrame({name: part[label].value_counts() for name, part in splits.items()}).fillna(0).astype(int)


def write_examples(test: pd.DataFrame, out_dir: Path, n: int = 4) -> list[Path]:
    """One held-out test photo per master category (largest categories first) -> out_dir/<id>_<category>.jpg.
    These photos are not in the catalog, so the demo searches with truly unseen images."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    order = test["master_category"].value_counts().index
    picks = [test[test["master_category"] == cat].iloc[0] for cat in order[:n]]
    paths = []
    for row in picks:
        path = out_dir / f"{row['id']}_{str(row['master_category']).lower().replace(' ', '_')}.jpg"
        with Image.open(row["image_path"]) as img:
            img.convert("RGB").save(path, "JPEG", quality=92)
        paths.append(path)
    return paths
