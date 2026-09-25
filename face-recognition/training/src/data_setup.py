"""Data: LFW download -> every photo aligned by our FaceDetector + 5 landmarks -> cached crops -> splits.

Dataset: Labeled Faces in the Wild (http://vis-www.cs.umass.edu/lfw/), the "funneled" version:
13,233 photos (250 x 250) of 5,749 people. sklearn downloads it once (~233 MB, into
training/data/sklearn_lfw). Every photo is aligned with model.FaceAligner (the Predictor's code
path, with the detector trained in this run) and the 128 x 128 crops are cached in training/data/lfw_aligned_128.npz (~650 MB).

Splits (all seeded, see config.py):
  * the 42 people with >= 25 photos keep the original 75/25 split -> train / test; 15% of their
    training photos become the validation split used for every choice;
  * 10% of the other people with >= 2 photos are "unseen": never trained on, they measure how
    well the embedder generalises to faces it has never seen;
  * the embedder trains on the 42 people's training photos + every other person's photos.
"""
from __future__ import annotations

import shutil
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import model as M

from .config import Config

# sklearn downloads lfw-funneled.tgz from figshare. Fallback: the same archive from the LFW site.
LFW_FALLBACK_URL = "http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz"
ALIGN_ID = "face-detector+5-point-template"   # cache key: how the crops were made (+ detector fingerprint)


def lfw_dir(cfg: Config) -> Path:
    """training/data/sklearn_lfw/lfw_home/lfw_funneled, downloaded on first use."""
    folder = cfg.lfw_home / "lfw_home" / "lfw_funneled"
    if folder.is_dir():
        return folder
    try:   # sklearn's downloader (figshare); only the 42-person subset is loaded, the archive has everyone
        fetch_lfw_people(data_home=str(cfg.lfw_home), min_faces_per_person=cfg.min_faces_per_person,
                         download_if_missing=True)
    except OSError as err:          # URLError / HTTPError / checksum problems
        print(f"sklearn download failed ({err}); trying the LFW website")
        folder.parent.mkdir(parents=True, exist_ok=True)
        archive = folder.parent / "lfw-funneled.tgz"
        with urllib.request.urlopen(LFW_FALLBACK_URL, timeout=120) as response, open(archive, "wb") as f:
            shutil.copyfileobj(response, f)
        with tarfile.open(archive) as tar:
            tar.extractall(folder.parent, filter="data")
        archive.unlink()
    return folder


def list_photos(cfg: Config) -> tuple[list[Path], np.ndarray]:
    """Every LFW photo (sorted by person, then file) and its person's name."""
    paths, names = [], []
    for person in sorted(p for p in lfw_dir(cfg).iterdir() if p.is_dir()):
        for f in sorted(person.glob("*.jpg")):
            paths.append(f)
            names.append(person.name.replace("_", " "))
    paths, names = np.array(paths, dtype=object), np.array(names)
    if cfg.smoke:
        keep = smoke_subset(cfg, names)
        paths, names = paths[keep], names[keep]
    return list(paths), names


def smoke_subset(cfg: Config, names: np.ndarray) -> np.ndarray:
    """Deterministic tiny subset: 6 of the 42 people (12 photos each) + 60 people with 2-4 photos."""
    uniq, counts = np.unique(names, return_counts=True)
    known = [n for n, c in zip(uniq, counts) if c >= cfg.min_faces_per_person][:6]
    others = [n for n, c in zip(uniq, counts) if 2 <= c <= 4][:60]
    keep = [i for n in known for i in np.flatnonzero(names == n)[:12]]
    keep += [i for n in others for i in np.flatnonzero(names == n)]
    return np.array(sorted(keep))


def align_photos(cfg: Config, aligner: M.FaceAligner, detector_id: str, batch_size: int = 64) -> dict:
    """{"crops": uint8 (N, 128, 128, 3), "found": bool (N,), "names": (N,)} for every photo.

    LFW photos are all 250 x 250, so the detector runs on whole batches. `aligner.selection` should
    be "center": LFW is labelled by the person in the middle, not the largest face. Cached in
    cfg.crops_cache; recomputed when the photos, the alignment or the detector weights change.
    """
    paths, names = list_photos(cfg)
    cache = cfg.crops_cache
    if cache.is_file():
        blob = np.load(cache, allow_pickle=False)
        if (len(blob["names"]) == len(names) and str(blob["align"]) == f"{ALIGN_ID}:{detector_id}"
                and blob["crops"].shape[1] == cfg.crop_size):
            print(f"loaded {len(names)} aligned crops from {cache.name}")
            return {"crops": blob["crops"], "found": blob["found"], "names": blob["names"]}
        print("crop cache is stale -> recomputing")
    crops = np.zeros((len(paths), cfg.crop_size, cfg.crop_size, 3), np.uint8)
    found = np.zeros(len(paths), bool)
    start = time.perf_counter()
    for s in tqdm(range(0, len(paths), batch_size), desc="detection + alignment"):
        images = [Image.open(p).convert("RGB") for p in paths[s:s + batch_size]]
        faces, ok = aligner.align_batch(images)
        crops[s:s + len(faces)] = np.stack([np.asarray(f) for f in faces])
        found[s:s + len(faces)] = ok
    print(f"aligned {len(paths)} photos in {time.perf_counter() - start:.0f}s; no face in {int((~found).sum())}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, crops=crops, found=found, names=names, align=f"{ALIGN_ID}:{detector_id}")
    return {"crops": crops, "found": found, "names": names}


def lfw_detector_photos(paths: list[Path], splits: dict) -> dict[str, Path]:
    """LFW photos the detector may train on: everything except the recognition test split and the
    unseen people (the recognition evaluation never sees a photo the detector trained on)."""
    held = set(splits["test"].tolist()) | set(splits["unseen"].tolist())
    return {f"{p.parent.name}/{p.name}": p for i, p in enumerate(paths) if i not in held}


def make_splits(cfg: Config, names: np.ndarray) -> dict:
    """Index arrays of every split + the 42 class names and the embedder's training identities."""
    uniq, counts = np.unique(names, return_counts=True)
    known = sorted(str(n) for n, c in zip(uniq, counts) if c >= (cfg.min_faces_per_person if not cfg.smoke else 12))
    idx_known = np.flatnonzero(np.isin(names, known))
    y_known = np.array([known.index(n) for n in names[idx_known]])
    train, test = train_test_split(idx_known, test_size=cfg.test_size, stratify=y_known, random_state=cfg.seed)
    y_train = np.array([known.index(n) for n in names[train]])
    train, val = train_test_split(train, test_size=cfg.val_size, stratify=y_train, random_state=cfg.seed)

    rest = np.flatnonzero(~np.isin(names, known))
    r_uniq, r_counts = np.unique(names[rest], return_counts=True)
    multi = sorted(n for n, c in zip(r_uniq, r_counts) if c >= 2)
    rng = np.random.default_rng(cfg.seed)
    unseen_people = rng.choice(multi, size=max(2, int(len(multi) * cfg.unseen_fraction)), replace=False)
    unseen = rest[np.isin(names[rest], unseen_people)]
    extra = rest[~np.isin(names[rest], unseen_people)]

    embedder_train = np.concatenate([train, extra])
    identities = sorted(str(n) for n in set(names[embedder_train]))
    return {"class_names": known, "train": train, "val": val, "test": test, "unseen": unseen,
            "embedder_train": embedder_train, "embedder_identities": identities}


def labels(names: np.ndarray, idx: np.ndarray, classes: list[str]) -> np.ndarray:
    lookup = {n: i for i, n in enumerate(classes)}
    return np.array([lookup[n] for n in names[idx]], dtype=np.int64)


def split_summary(splits: dict, names: np.ndarray) -> dict[str, str]:
    return {k: f"{len(splits[k]):,} photos / {len(set(names[splits[k]])):,} people"
            for k in ("embedder_train", "train", "val", "test", "unseen")}
