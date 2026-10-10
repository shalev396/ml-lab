"""Data for training: download the dataset, turn it into memory-mapped Arrow files, serve batches.

  download()        snapshot of shalev396/calorie-lens-data into training/data/ (smoke: the mini build)
  to_arrow()        once per session: the parquet shards -> one uncompressed Arrow IPC file per
                    (table, split) with just the JPEG / PNG bytes. Memory-mapped, so every DataLoader
                    worker reads it zero-copy (no per-worker copy of a list of bytes objects)
  Labels            per-dish targets: grams of each of the K ingredients and the five totals
  N5kImages         one sample = one image: decode (libjpeg-turbo) + fit into the canvas, uint8
  MixtureSampler    a fixed number of samples per epoch, overhead_share of them overhead scans
  gpu_augment()     batched on the GPU: rotation / flips / shift for overhead scans (same transform on
                    the depth map), flip + mild zoom for side frames, mild colour jitter
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

import model as M

from . import utils

SPLITS = {"train": "train", "val": "validation", "test": "test"}
DEPTH_SCALE = 1e-4                  # depth_raw.png units -> metres
DEPTH_VALID = (1, 6000)             # raw readings outside (0, 0.6 m] are no-reading / background


def download(cfg) -> Path:
    """The dataset folder: the mini build for smoke runs, else a Hub snapshot (skips files present)."""
    if cfg.smoke:
        if not (cfg.data_dir / "dishes.parquet").exists():
            raise FileNotFoundError(f"{cfg.data_dir} has no mini build: run the notebook's data cell with BUILD_DATA=1")
        return cfg.data_dir
    from huggingface_hub import snapshot_download

    snapshot_download(cfg.dataset_id, repo_type="dataset", local_dir=cfg.data_dir,
                      allow_patterns=["*.parquet", "*.csv", "README.md"])
    if not (cfg.data_dir / "dishes.parquet").exists():
        raise FileNotFoundError(f"{cfg.dataset_id} has no label tables yet (a build uploads them last): "
                                "run the notebook's data cell with BUILD_DATA=1")
    return cfg.data_dir


def published(dataset_id: str) -> bool:
    """True once a full build is on the Hub: its label tables are the last files it uploads."""
    from huggingface_hub import HfApi

    return HfApi().file_exists(dataset_id, "dishes.parquet", repo_type="dataset")


def to_arrow(data_dir: Path) -> dict[tuple[str, str], Path]:
    """(table, split) -> Arrow IPC file with dish_id + raw image bytes (+ depth bytes). Built once."""
    out = {}
    for table in ("overhead", "side"):
        for split, fname in SPLITS.items():
            shards = sorted((data_dir / table).glob(f"{fname}-*.parquet"))
            dst = data_dir / "arrow" / f"{table}_{split}.arrow"
            out[(table, split)] = dst
            if dst.exists() or not shards:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp, writer = dst.with_suffix(".tmp"), None
            with pa.OSFile(str(tmp), "wb") as sink:
                for shard in shards:
                    t = pq.read_table(shard)
                    cols = {"dish_id": t["dish_id"], "image": pc.struct_field(t["image"], "bytes")}
                    if table == "overhead":
                        cols["depth"] = pc.struct_field(t["depth"], "bytes")
                    else:
                        cols["camera"], cols["frame"] = t["camera"], t["frame"]
                    batch = pa.table(cols)
                    writer = writer or pa.ipc.new_file(sink, batch.schema)
                    writer.write_table(batch)
                writer.close()
            tmp.replace(dst)
    return out


class FrameStore:
    """Read-only, memory-mapped Arrow table; opened lazily so it is never pickled into workers."""

    def __init__(self, path: Path):
        self.path, self._t = Path(path), None

    @property
    def table(self) -> pa.Table:
        if self._t is None:
            self._t = pa.ipc.open_file(pa.memory_map(str(self.path), "r")).read_all()
        return self._t

    def __getstate__(self):
        return {"path": self.path, "_t": None}

    def __len__(self) -> int:
        return self.table.num_rows

    def get(self, col: str, i: int) -> bytes:
        return self.table.column(col)[i].as_py()

    def column(self, col: str) -> list:
        return self.table.column(col).to_pylist()


@dataclass
class Labels:
    """Per-dish targets, indexed by dish row (dish_ids[i])."""
    dish_ids: list
    index: dict             # dish_id -> row
    grams: np.ndarray       # (D, K) float32, grams of each ingredient
    totals: np.ndarray      # (D, 5) float32: kcal, mass, fat, carb, protein (corrected)
    orig_totals: np.ndarray # (D, 5) as logged by Nutrition5k
    split: np.ndarray       # (D,) train | val | test
    test_clean: np.ndarray  # (D,) bool
    group: np.ndarray       # (D,) plate group
    ingredients: pd.DataFrame

    @property
    def densities(self) -> np.ndarray:
        return self.ingredients[list(M.NUTRIENTS)].to_numpy(np.float32)


def load_labels(data_dir: Path) -> Labels:
    dishes = pd.read_parquet(data_dir / "dishes.parquet")
    items = pd.read_parquet(data_dir / "items.parquet")
    ing = pd.read_csv(data_dir / "ingredients.csv").sort_values("k").reset_index(drop=True)
    index = {d: i for i, d in enumerate(dishes["dish_id"])}
    grams = np.zeros((len(dishes), len(ing)), np.float32)
    np.add.at(grams, (items["dish_id"].map(index).to_numpy(), items["k"].to_numpy()), items["grams"].to_numpy())
    cols = list(M.TOTALS)
    arr = lambda c, dtype=None: np.array(dishes[c].to_numpy(dtype), copy=True)    # writable (pandas CoW views are not)
    return Labels(list(dishes["dish_id"]), index, grams, arr(cols, np.float32),
                  arr([f"orig_{c}" for c in cols], np.float32), arr("split"), arr("test_clean", bool), arr("group"), ing)


def fit_canvas(img: torch.Tensor, size, fill, mode: str = "bilinear") -> torch.Tensor:
    """(C, h, w) -> (C, H, W): resize to fit (aspect kept, model.fit_box), pad centred with `fill`
    (one value per channel)."""
    left, top, nw, nh = M.fit_box(img.shape[-1], img.shape[-2], size)
    if mode == "nearest":
        small = F.interpolate(img[None].float(), size=(nh, nw), mode="nearest")[0].to(img.dtype)
    else:
        small = F.interpolate(img[None].float(), size=(nh, nw), mode="bilinear", antialias=True,
                              align_corners=False)[0].round().clamp(0, 255).to(img.dtype)
    out = torch.tensor(fill, dtype=img.dtype).view(-1, 1, 1).expand(img.shape[0], *size).clone()
    out[:, top:top + nh, left:left + nw] = small
    return out


_FILL = tuple(round(255 * m) for m in M.MEAN)


class N5kImages(Dataset):
    """Samples = (table, row) pairs of one split; returns uint8 canvas + targets (+ metric depth)."""

    def __init__(self, stores: dict, rows: list[tuple[str, int, int]], labels: Labels, size, depth: bool):
        self.stores, self.rows, self.labels, self.size, self.depth = stores, rows, labels, tuple(size), depth

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        from torchvision.io import decode_jpeg

        table, row, dish = self.rows[i]
        store = self.stores[table]
        raw = torch.frombuffer(bytearray(store.get("image", row)), dtype=torch.uint8)
        img = decode_jpeg(raw)
        canvas = fit_canvas(img, self.size, _FILL)
        overhead = table == "overhead"
        item = {"image": canvas, "overhead": overhead, "dish": dish,
                "grams": torch.from_numpy(self.labels.grams[dish]), "totals": torch.from_numpy(self.labels.totals[dish])}
        if self.depth:
            if overhead:
                from PIL import Image

                d = np.asarray(Image.open(io.BytesIO(store.get("depth", row))), dtype=np.int32)
                d = np.where((d >= DEPTH_VALID[0]) & (d <= DEPTH_VALID[1]), d, 0).astype(np.int16)
                item["depth"] = fit_canvas(torch.from_numpy(d)[None], self.size, (0,), mode="nearest")[0]
            else:
                item["depth"] = torch.zeros(self.size, dtype=torch.int16)
        return item


class MixtureSampler(Sampler):
    """samples_per_epoch indices per epoch, overhead_share from the overhead pool (with replacement:
    every overhead scan repeats under fresh rotations), the rest from the side-frame pool."""

    def __init__(self, overhead: list[int], side: list[int], n: int, overhead_share: float, seed: int):
        self.overhead, self.side, self.n, self.share, self.seed, self.epoch = overhead, side, n, overhead_share, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        g = np.random.default_rng((self.seed, self.epoch))
        share = self.share if self.side and self.overhead else float(bool(self.overhead))
        n_over = int(round(self.n * share))
        idx = np.concatenate([g.choice(self.overhead, n_over) if n_over else np.zeros(0, int),
                              g.choice(self.side, self.n - n_over) if self.n - n_over else np.zeros(0, int)])
        g.shuffle(idx)
        return iter(idx.tolist())


@dataclass
class Loaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    labels: Labels
    sampler: MixtureSampler


def build_loaders(cfg, device, micro_batch: int) -> Loaders:
    """Train (mixture sampler, uint8 canvases, augmentation later on the GPU), val and test (every image)."""
    data_dir = download(cfg)
    files = to_arrow(data_dir)
    labels = load_labels(data_dir)
    stores = {k: FrameStore(p) for k, p in files.items() if p.exists()}

    def rows(split: str) -> tuple[list, list[int], list[int]]:
        out, over, side = [], [], []
        for table in ("overhead", "side"):
            store = stores.get((table, split))
            if store is None:
                continue
            for r, d in enumerate(store.column("dish_id")):
                (over if table == "overhead" else side).append(len(out))
                out.append((table, r, labels.index[d]))
        return out, over, side

    pin = device.type == "cuda"
    kw = dict(num_workers=cfg.workers, pin_memory=pin, persistent_workers=cfg.workers > 0,
              prefetch_factor=4 if cfg.workers > 0 else None)
    made = {}
    for split in SPLITS:
        r, over, side = rows(split)
        split_stores = {t: stores[(t, split)] for t in ("overhead", "side") if (t, split) in stores}
        made[split] = (N5kImages(split_stores, r, labels, cfg.image_size, cfg.depth_aux and split != "test"), over, side)
    ds, over, side = made["train"]
    sampler = MixtureSampler(over, side, cfg.samples_per_epoch, cfg.overhead_share, cfg.seed)
    train = DataLoader(ds, batch_size=micro_batch, sampler=sampler, drop_last=True, **kw)
    val = DataLoader(made["val"][0], batch_size=2 * micro_batch, shuffle=False, **kw)
    test = DataLoader(made["test"][0], batch_size=2 * micro_batch, shuffle=False, **kw)
    return Loaders(train, val, test, labels, sampler)


# --------------------------------------------------------------------------- GPU augmentation
def _affine(n: int, overhead: torch.Tensor, gen: torch.Generator, device, aspect: float) -> torch.Tensor:
    """(N, 2, 3) affine_grid matrices. Overhead: any rotation, flips, +-4 % shift, no zoom (the fixed
    camera height makes apparent size a mass cue). Side: horizontal flip, zoom-in up to 1/0.85."""
    u = torch.rand(n, 6, generator=gen, device=device)
    theta = torch.where(overhead, u[:, 0] * 2 * math.pi, torch.zeros_like(u[:, 0]))
    zoom = torch.where(overhead, torch.ones_like(u[:, 1]), 0.85 + 0.15 * u[:, 1])
    fx = torch.where(u[:, 2] < 0.5, -1.0, 1.0)
    fy = torch.where(overhead & (u[:, 3] < 0.5), -1.0, 1.0)
    shift = torch.where(overhead, 0.04, 1.0 - zoom)[:, None] * (2 * u[:, 4:6] - 1)
    c, s = torch.cos(theta), torch.sin(theta)
    # rotation in pixel space on a non-square canvas: scale x by aspect before / after rotating
    mat = torch.stack([torch.stack([c * zoom * fx, -s * zoom * fy / aspect, shift[:, 0]], -1),
                       torch.stack([s * zoom * fx * aspect, c * zoom * fy, shift[:, 1]], -1)], 1)
    return mat


def gpu_augment(batch: dict, train: bool, gen: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
    """uint8 canvases (already on the GPU) -> float images in [0, 1] (+ depth in metres, 0 = invalid)."""
    x = batch["image"].float() / 255.0
    depth = batch.get("depth")
    depth = None if depth is None else depth.float() * DEPTH_SCALE
    if not train:
        return x, depth
    n, device = x.shape[0], x.device
    overhead = batch["overhead"].to(device).bool()
    mat = _affine(n, overhead, gen, device, x.shape[-1] / x.shape[-2])
    grid = F.affine_grid(mat, list(x.shape), align_corners=False)
    fill = torch.tensor(M.MEAN, device=device).view(1, 3, 1, 1)
    x = F.grid_sample(x - fill, grid, mode="bilinear", padding_mode="zeros", align_corners=False) + fill
    if depth is not None:
        depth = F.grid_sample(depth[:, None], grid, mode="nearest", padding_mode="zeros", align_corners=False)[:, 0]
    u = torch.rand(n, 3, 1, 1, 1, generator=gen, device=device)
    x = x * (0.85 + 0.3 * u[:, 0])                                    # brightness
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * (0.85 + 0.3 * u[:, 1]) + mean                     # contrast
    gray = x.mean(dim=1, keepdim=True)
    x = (x - gray) * (0.85 + 0.3 * u[:, 2]) + gray                     # saturation (no hue: colour is identity)
    return x.clamp(0, 1), depth


def summary(loaders: Loaders) -> dict:
    """Images and dishes per split (what the run actually trains / validates / tests on)."""
    out = {}
    for split, loader in (("train", loaders.train), ("val", loaders.val), ("test", loaders.test)):
        rows = loader.dataset.rows
        over = sum(1 for t, _, _ in rows if t == "overhead")
        out[split] = {"dishes": len({d for _, _, d in rows}), "overhead_scans": over, "side_frames": len(rows) - over}
    out["ingredients"] = int(loaders.labels.grams.shape[1])
    return out


def save_examples(cfg, out_dir: Path, n_overhead: int = 3, n_side: int = 1, seed: int = 0) -> list[Path]:
    """A few test-split photos as JPEG files (Space examples / inference checks), named by dish."""
    files = to_arrow(download(cfg))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng, paths = np.random.default_rng(seed), []
    for table, n in (("overhead", n_overhead), ("side", n_side)):
        path = files.get((table, "test"))
        if n == 0 or path is None or not path.exists():
            continue
        store = FrameStore(path)
        for r in rng.choice(len(store), min(n, len(store)), replace=False):
            view = "overhead" if table == "overhead" else f"side_{store.get('camera', int(r))}{store.get('frame', int(r))}"
            p = out_dir / f"{store.get('dish_id', int(r))}_{view}.jpg"
            p.write_bytes(store.get("image", int(r)))
            paths.append(p)
    return paths
