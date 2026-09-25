"""Data: download -> cache in training/data -> ImageFolder datasets -> DataLoaders.

Dataset: pizza / steak / sushi images from Food-101 (ethz/food101), a random 20 % sample
per class prepared for the PyTorch Deep Learning course (mrdbourke/pytorch-deep-learning):
450 train (pizza 154 / steak 146 / sushi 150) and 150 test (46 / 58 / 46) images, laid out as
`train/<class>/*.jpg` + `test/<class>/*.jpg`.
"""
import shutil
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets

from . import utils
from .config import Config

SPLITS = ("train", "test")


def download_data(cfg: Config) -> Path:
    """Fetch and extract the image zip once; later calls reuse the cache in training/data/."""
    root = cfg.data_dir
    if all((root / split).is_dir() for split in SPLITS):
        return root
    root.mkdir(parents=True, exist_ok=True)
    archive = root.parent / f"{root.name}.zip"
    partial = root.parent / f"{root.name}.zip.part"
    errors = []
    for url in (cfg.data_url, cfg.data_fallback_url):
        try:
            print(f"downloading {url}")
            with urllib.request.urlopen(url, timeout=60) as response, open(partial, "wb") as f:
                shutil.copyfileobj(response, f)
            partial.replace(archive)   # only a complete download gets the final name
            break
        except OSError as err:          # URLError / HTTPError / timeouts are all OSErrors
            errors.append(f"{url}: {err}")
    else:
        raise RuntimeError("could not download the dataset:\n  " + "\n  ".join(errors))
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(root)
    archive.unlink()
    return root


def _first_per_class(ds: datasets.ImageFolder, n: int) -> Subset:
    """Deterministic tiny subset: the first `n` files of every class (ImageFolder order is sorted)."""
    taken, keep = Counter(), []
    for idx, label in enumerate(ds.targets):
        if taken[label] < n:
            taken[label] += 1
            keep.append(idx)
    return Subset(ds, keep)


def build_datasets(cfg: Config, transform) -> tuple[Dataset, Dataset, list[str]]:
    """(train_ds, test_ds, class_names). Both splits use the same (eval) transform, as in the recipe."""
    root = download_data(cfg)
    train_ds = datasets.ImageFolder(root / "train", transform=transform)
    test_ds = datasets.ImageFolder(root / "test", transform=transform)
    if train_ds.classes != test_ds.classes:
        raise ValueError(f"class folders differ: {train_ds.classes} vs {test_ds.classes}")
    class_names = train_ds.classes
    if cfg.subset_per_class:
        train_ds = _first_per_class(train_ds, cfg.subset_per_class)
        test_ds = _first_per_class(test_ds, cfg.subset_per_class)
    return train_ds, test_ds, class_names


def create_dataloaders(cfg: Config, transform, device) -> tuple[DataLoader, DataLoader, list[str]]:
    """(train_loader, test_loader, class_names). Train is shuffled with a seeded generator."""
    train_ds, test_ds, class_names = build_datasets(cfg, transform)
    common = dict(batch_size=cfg.batch_size, num_workers=utils.num_workers(),
                  pin_memory=(torch.device(device).type == "cuda"))
    train_loader = DataLoader(train_ds, shuffle=True,
                              generator=torch.Generator().manual_seed(cfg.seed), **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, test_loader, class_names


def class_counts(ds: Dataset, class_names: list[str]) -> dict[str, int]:
    """{class_name: number of images} for an ImageFolder or a Subset of one."""
    targets = ds.dataset.targets if isinstance(ds, Subset) else ds.targets
    indices = ds.indices if isinstance(ds, Subset) else range(len(targets))
    counts = Counter(targets[i] for i in indices)
    return {name: counts[i] for i, name in enumerate(class_names)}


def image_paths(ds: Dataset) -> list[Path]:
    """File path of every sample, in dataset order (for EDA / error analysis)."""
    samples = ds.dataset.samples if isinstance(ds, Subset) else ds.samples
    indices = ds.indices if isinstance(ds, Subset) else range(len(samples))
    return [Path(samples[i][0]) for i in indices]
