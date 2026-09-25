"""Data: Food-101 from the Hub -> cache in training/data -> datasets -> DataLoaders.

Dataset: [Food-101](https://huggingface.co/datasets/ethz/food101), 101 dishes, 750 train and
250 test photos per dish (75,750 / 25,250). The Hub copy calls the official test split
"validation". Two ways to get it:

- `source="hub"` (full runs): `datasets.load_dataset` downloads all parquet shards once (~5 GB)
  into `training/data/hf/`.
- `source="slice"` (smoke runs, quick checks): only a few 100-image parquet row groups, spread
  evenly over the split, are read over HTTP range requests and cached in `training/data/slices/`.
  The Hub files are grouped by class, so a slice covers only as many dishes as it has chunks.

Fallback when the Hub is unreachable: `torchvision.datasets.Food101` (the original ETH Zurich
tarball, ~5 GB) into `training/data/food-101/`.

Label order: the model uses the alphabetical class order of torchvision's Food101 (the order the
deployed checkpoint was trained with). The Hub dataset lists `cheesecake` before `cheese_plate`,
so Hub labels are remapped by name, never by index.
"""
from __future__ import annotations

import io
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from . import utils
from .config import Config

HUB_SPLIT = {"train": "train", "test": "validation"}   # our name -> name on the Hub


# --------------------------------------------------------------------------- datasets
class FoodImages(Dataset):
    """Common interface of every source: `labels` (ints in model class order), `image(i)` (PIL, RGB)
    and `__getitem__` -> (transformed tensor, label)."""

    labels: list[int]
    transform = None

    def image(self, i: int) -> Image.Image:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        img = self.image(i)
        return (self.transform(img) if self.transform else img), self.labels[i]


class HubImages(FoodImages):
    """A split of the fully downloaded Hub dataset (optionally a subset of its rows)."""

    def __init__(self, hf_ds, label_map: list[int], indices: list[int] | None = None, transform=None):
        self.hf_ds, self.transform = hf_ds, transform
        self.indices = list(range(len(hf_ds))) if indices is None else list(indices)
        all_labels = list(hf_ds["label"])
        self.labels = [label_map[all_labels[i]] for i in self.indices]

    def image(self, i: int) -> Image.Image:
        return self.hf_ds[self.indices[i]]["image"].convert("RGB")


class BytesImages(FoodImages):
    """Encoded images held in memory (a cached slice)."""

    def __init__(self, blobs: list[bytes], labels: list[int], transform=None):
        self.blobs, self.labels, self.transform = blobs, list(labels), transform

    def image(self, i: int) -> Image.Image:
        return Image.open(io.BytesIO(self.blobs[i])).convert("RGB")


class TorchvisionImages(FoodImages):
    """Fallback: torchvision's Food101 (already in alphabetical class order)."""

    def __init__(self, tv_ds, indices: list[int] | None = None, transform=None):
        self.tv_ds, self.transform = tv_ds, transform
        self.indices = list(range(len(tv_ds))) if indices is None else list(indices)
        self.labels = [tv_ds._labels[i] for i in self.indices]

    def image(self, i: int) -> Image.Image:
        with Image.open(self.tv_ds._image_files[self.indices[i]]) as img:
            return img.convert("RGB")


# --------------------------------------------------------------------------- class names
def hub_class_names(cfg: Config) -> list[str]:
    """Class names in the Hub dataset's own label order."""
    from datasets import load_dataset_builder

    return list(load_dataset_builder(cfg.dataset_id, cache_dir=str(cfg.data_dir / "hf")).info.features["label"].names)


def class_names_and_map(cfg: Config) -> tuple[list[str], list[int]]:
    """(model class order = alphabetical, map Hub label index -> model label index)."""
    hub_names = hub_class_names(cfg)
    class_names = sorted(hub_names)
    return class_names, [class_names.index(name) for name in hub_names]


# --------------------------------------------------------------------------- sources
def _subset(n_total: int, n: int, seed: int) -> list[int] | None:
    if not n or n >= n_total:
        return None
    return sorted(np.random.default_rng(seed).choice(n_total, size=n, replace=False).tolist())


def load_hub(cfg: Config, label_map: list[int]) -> tuple[FoodImages, FoodImages]:
    """Full download of the Hub dataset (cached; later runs reuse it)."""
    from datasets import load_dataset

    dsd = load_dataset(cfg.dataset_id, cache_dir=str(cfg.data_dir / "hf"))
    train, test = dsd[HUB_SPLIT["train"]], dsd[HUB_SPLIT["test"]]
    return (HubImages(train, label_map, _subset(len(train), cfg.train_images, cfg.seed)),
            HubImages(test, label_map, _subset(len(test), cfg.test_images, cfg.seed + 1)))


def load_slice(cfg: Config, split: str, n_images: int, n_chunks: int, label_map: list[int]) -> BytesImages:
    """`n_images` photos from `n_chunks` parquet row groups spread evenly over the Hub split.

    Only those row groups are downloaded (HTTP range requests); the slice is cached as one small
    parquet file under training/data/slices/.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    cache = cfg.data_dir / "slices" / f"{split}-{n_images}-{n_chunks}chunks-seed{cfg.seed}.parquet"
    if cache.is_file():
        table = pq.read_table(cache)
        return BytesImages(table.column("image").to_pylist(), table.column("label").to_pylist())

    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/{cfg.dataset_id}/**/{HUB_SPLIT[split]}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files for split {HUB_SPLIT[split]!r} in {cfg.dataset_id}")
    handles = [pq.ParquetFile(fs.open(path, "rb")) for path in files]
    groups = [(f, g) for f, pf in enumerate(handles) for g in range(pf.metadata.num_row_groups)]
    picks = np.unique(np.linspace(0, len(groups) - 1, num=min(n_chunks, len(groups))).round().astype(int))
    per_chunk = -(-n_images // len(picks))                     # ceil
    rng = np.random.default_rng(cfg.seed)
    blobs, labels = [], []
    for k in picks:
        f, g = groups[k]
        table = handles[f].read_row_group(g, columns=["image", "label"])
        rows = rng.choice(table.num_rows, size=min(per_chunk, table.num_rows), replace=False)
        images, hub_labels = table.column("image").to_pylist(), table.column("label").to_pylist()
        for r in sorted(rows.tolist()):
            blobs.append(images[r]["bytes"])
            labels.append(label_map[hub_labels[r]])
    blobs, labels = blobs[:n_images], labels[:n_images]
    cache.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"image": pa.array(blobs, pa.binary()), "label": pa.array(labels, pa.int32())}), cache)
    return BytesImages(blobs, labels)


def load_torchvision(cfg: Config) -> tuple[FoodImages, FoodImages]:
    """Fallback source: the original Food-101 tarball via torchvision (~5 GB, cached)."""
    from torchvision import datasets

    train = datasets.Food101(root=str(cfg.data_dir), split="train", download=True)
    test = datasets.Food101(root=str(cfg.data_dir), split="test", download=True)
    return (TorchvisionImages(train, _subset(len(train), cfg.train_images, cfg.seed)),
            TorchvisionImages(test, _subset(len(test), cfg.test_images, cfg.seed + 1)))


# --------------------------------------------------------------------------- public API
def train_transform(eval_transform, augment: bool = True):
    """Training preprocessing: TrivialAugmentWide (on the PIL image) + the eval preprocessing."""
    return T.Compose([T.TrivialAugmentWide(), eval_transform]) if augment else eval_transform


def build_datasets(cfg: Config, eval_transform) -> tuple[FoodImages, FoodImages, list[str]]:
    """(train_ds, test_ds, class_names) with the train/eval transforms attached."""
    try:
        class_names, label_map = class_names_and_map(cfg)
        if cfg.source == "slice":
            chunks = cfg.slice_chunks
            train_ds = load_slice(cfg, "train", cfg.train_images, chunks, label_map)
            test_ds = load_slice(cfg, "test", cfg.test_images, chunks, label_map)
        else:
            train_ds, test_ds = load_hub(cfg, label_map)
    except Exception as err:  # Hub unreachable -> the documented fallback
        print(f"Hugging Face Hub source failed ({type(err).__name__}: {err}); falling back to torchvision Food101")
        train_ds, test_ds = load_torchvision(cfg)
        class_names = train_ds.tv_ds.classes
    train_ds.transform = train_transform(eval_transform, cfg.augment)
    test_ds.transform = eval_transform
    return train_ds, test_ds, class_names


def create_dataloaders(cfg: Config, eval_transform, device) -> tuple[DataLoader, DataLoader, list[str]]:
    """(train_loader, test_loader, class_names). Train is shuffled with a seeded generator."""
    train_ds, test_ds, class_names = build_datasets(cfg, eval_transform)
    workers = utils.num_workers()
    common = dict(batch_size=cfg.batch_size, num_workers=workers,
                  pin_memory=(torch.device(device).type == "cuda"), persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, generator=torch.Generator().manual_seed(cfg.seed), **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, test_loader, class_names


def class_counts(ds: FoodImages, class_names: list[str]) -> dict[str, int]:
    """{class_name: number of images} (every class listed, zeros included)."""
    counts = Counter(ds.labels)
    return {name: counts[i] for i, name in enumerate(class_names)}
