"""Data: download UTKFace -> cache in training/data/utkface -> train/validation split -> DataLoaders.

Dataset: UTKFace (https://susanqq.github.io/UTKFace/) as published on the Hugging Face Hub by
`nu-delta/utkface`: 23,705 aligned+cropped face photos (200x200) labelled with age (1-116), gender
and ethnicity, stored as 3 parquet shards (~1 GB). UTKFace is for non-commercial research only.

Primary source: the parquet files of the dataset repo. Fallback: the Hub's auto-converted parquet
branch of the same dataset (`refs/convert/parquet`), a separate set of files.
"""
from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

import model as M  # ../model/model.py, put on sys.path by src/__init__.py

from . import utils
from .config import Config

GENDER_TO_INDEX = {"male": 0, "female": 1}          # same order as model.DEFAULT_GENDER_LABELS


def parquet_files(cfg: Config) -> list[Path]:
    root = cfg.data_dir
    return sorted(root.glob("data/train-*.parquet")) or sorted(root.glob("convert/default/train/*.parquet"))


def download_data(cfg: Config) -> list[Path]:
    """Fetch the parquet shards once; later calls reuse the cache in training/data/utkface/."""
    files = parquet_files(cfg)
    if files:
        return files
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(cfg.dataset_id, repo_type="dataset", local_dir=cfg.data_dir,
                          allow_patterns=["data/*.parquet", "README.md"])
    except Exception as err:  # Hub hiccup / renamed files -> the auto-converted parquet branch
        print(f"primary download failed ({err}); trying refs/convert/parquet")
        snapshot_download(cfg.dataset_id, repo_type="dataset", revision="refs/convert/parquet",
                          local_dir=cfg.data_dir / "convert", allow_patterns=["default/train/*.parquet"])
    files = parquet_files(cfg)
    if not files:
        raise RuntimeError(f"no UTKFace parquet files found under {cfg.data_dir}")
    return files


def load_table(cfg: Config, files: list[Path]) -> pa.Table:
    """Columns image (struct bytes/path), age, gender. Smoke runs read only the first shard and keep
    a random `cfg.subset` rows of it; full runs read every shard in order (matching the original
    `load_dataset(...)["train"]` row order, which the train/validation split depends on)."""
    columns = ["image", "age", "gender"]
    if cfg.subset:
        table = pq.read_table(files[0], columns=columns)
        rng = np.random.default_rng(cfg.seed)
        keep = np.sort(rng.choice(len(table), size=min(cfg.subset, len(table)), replace=False))
        return table.take(pa.array(keep))
    return pa.concat_tables([pq.read_table(f, columns=columns) for f in files])


def split_indices(n: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(train_idx, val_idx). Identical to `datasets.Dataset.train_test_split(test_size=val_frac,
    seed=seed)`, which produced the split of the deployed model."""
    n_val = math.ceil(val_frac * n)
    perm = np.random.default_rng(seed).permutation(n)
    return perm[n_val:], perm[:n_val]


def labels(table: pa.Table) -> tuple[np.ndarray, np.ndarray]:
    """(ages int array, gender index array 0=male 1=female)."""
    ages = table.column("age").to_numpy().astype(np.int64)
    genders = np.array([GENDER_TO_INDEX[g.strip().lower()] for g in table.column("gender").to_pylist()])
    return ages, genders


def compute_age_representatives(ages: np.ndarray, edges=M.DEFAULT_AGE_BIN_EDGES) -> list[float]:
    """Representative age of each bin = mean true age of the training faces in it (empty bins fall
    back to the bin midpoint). Stored in config.json; the headline age is sum(p_i * rep_i)."""
    defaults = M.default_representatives(edges)
    bins = np.array([M.age_to_bin(int(a), edges) for a in ages])
    return [float(ages[bins == i].mean()) if (bins == i).any() else defaults[i] for i in range(len(edges))]


def train_transform():
    """Eval preprocessing (resize 288 bicubic, center-crop 288, ImageNet mean/std) + light augmentation:
    horizontal flip and mild colour jitter."""
    ev = M.get_transform()
    return T.Compose([
        T.Resize(ev.resize_size, interpolation=ev.interpolation),
        T.CenterCrop(ev.crop_size),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=ev.mean, std=ev.std),
    ])


class UTKFaceDataset(Dataset):
    """Decodes the JPEG bytes lazily; yields (image, age_bin, gender, true_age)."""

    def __init__(self, images: list[bytes], ages: np.ndarray, genders: np.ndarray, transform,
                 edges=M.DEFAULT_AGE_BIN_EDGES):
        self.images, self.ages, self.genders, self.transform = images, ages, genders, transform
        self.bins = np.array([M.age_to_bin(int(a), edges) for a in ages])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        with Image.open(io.BytesIO(self.images[i])) as img:
            x = self.transform(img.convert("RGB"))
        return x, int(self.bins[i]), int(self.genders[i]), int(self.ages[i])

    def image(self, i: int) -> Image.Image:
        with Image.open(io.BytesIO(self.images[i])) as img:
            return img.convert("RGB")


def build_datasets(cfg: Config, table: pa.Table) -> tuple[UTKFaceDataset, UTKFaceDataset]:
    """(train_ds, val_ds): 90/10 random split; training faces get augmentation if cfg.augment."""
    images = table.column("image").combine_chunks().field("bytes").to_pylist()
    ages, genders = labels(table)
    train_idx, val_idx = split_indices(len(images), cfg.val_frac, cfg.seed)
    pick = lambda idx: ([images[i] for i in idx], ages[idx], genders[idx])  # noqa: E731
    train_ds = UTKFaceDataset(*pick(train_idx), train_transform() if cfg.augment else M.get_transform())
    val_ds = UTKFaceDataset(*pick(val_idx), M.get_transform())
    return train_ds, val_ds


def create_dataloaders(cfg: Config, train_ds: Dataset, val_ds: Dataset, device) -> tuple[DataLoader, DataLoader]:
    common = dict(batch_size=cfg.batch_size, num_workers=utils.num_workers(),
                  pin_memory=(torch.device(device).type == "cuda"))
    train_loader = DataLoader(train_ds, shuffle=True, generator=torch.Generator().manual_seed(cfg.seed), **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    return train_loader, val_loader


def summary(train_ds: UTKFaceDataset, val_ds: UTKFaceDataset) -> dict:
    return {
        "n_train": len(train_ds), "n_val": len(val_ds),
        "age_mean_train": round(float(train_ds.ages.mean()), 2),
        "age_median_train": float(np.median(train_ds.ages)),
        "age_min": int(min(train_ds.ages.min(), val_ds.ages.min())),
        "age_max": int(max(train_ds.ages.max(), val_ds.ages.max())),
        "female_share_train": round(float(train_ds.genders.mean()), 4),
    }
