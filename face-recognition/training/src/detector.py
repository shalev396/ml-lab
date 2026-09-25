"""Our face detector (model.FaceDetector): data, training and evaluation.

Data (all cached in training/data):
  * Open Images V7 (Google; annotations CC BY 4.0, images CC BY 2.0): every validation + test
    photo with a human-drawn "Human face" box, ~12,000 photos, resized to <= 800 px. The human
    boxes are the detection targets; "group of faces" boxes are ignored.
  * LFW photos (not the recognition test split, not the unseen people).
  * Landmarks: training/labels/landmark_labels.json.gz holds, for every one of these photos, the
    faces a reference detector found (box, score, 5 points). They were made once with MTCNN
    (facenet-pytorch); only these numbers are used here, no third-party code or weights. The 5
    points label the faces they agree with; faces found there that nobody boxed are ignored.
Training: random square crops / zoom-outs / 2x2 mosaics at 640 px, flips, colour jitter; per anchor
a face / not-face loss with online hard-negative mining (3 negatives per positive), box and
landmark regression. The checkpoint with the best AP on 600 held-out Open Images photos is kept.
Evaluation: AP at IoU 0.5 on the held-out photos and on WIDER FACE val (evaluation only: WIDER is
CC BY-NC-ND and never trained on). The reference detector's scores on the same photos, measured
once with the same evaluation code, are in config.REFERENCE_DETECTOR_AP.
"""
from __future__ import annotations

import copy
import csv
import gzip
import io
import json
import math
import random
import time
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_iou
from tqdm.auto import tqdm

import model as M

from . import utils
from .config import Config

OI_FACE = "/m/0dzct"
OI_URL = "https://storage.googleapis.com/openimages"
FLIP = [1, 0, 2, 4, 3]            # left eye <-> right eye, left mouth corner <-> right


# --------------------------------------------------------------------------- Open Images
def download_open_images(cfg: Config, workers: int = 32) -> dict:
    """{image id: {"w", "h", "faces": [[x1, y1, x2, y2, is_group], ...]}}; photos in data/openimages/images."""
    root = cfg.open_images_dir
    ann_path = root / "faces.json"
    if ann_path.is_file():
        ann = json.load(open(ann_path))
        return dict(list(ann.items())[: cfg.det_smoke_images]) if cfg.smoke else ann
    (root / "images").mkdir(parents=True, exist_ok=True)
    faces, rotated = defaultdict(list), set()
    for split, folder in (("validation", "v5"), ("test", "v5")):
        boxes_csv, rot_csv = root / f"{split}-annotations-bbox.csv", root / f"{split}-images-with-rotation.csv"
        if not boxes_csv.is_file():
            urllib.request.urlretrieve(f"{OI_URL}/{folder}/{split}-annotations-bbox.csv", boxes_csv)
        if not rot_csv.is_file():
            urllib.request.urlretrieve(f"{OI_URL}/2018_04/{split}/{split}-images-with-rotation.csv", rot_csv)
        rotated |= {r["ImageID"] for r in csv.DictReader(open(rot_csv, encoding="utf-8")) if r["Rotation"] not in ("", "0", "0.0")}
        for r in csv.DictReader(open(boxes_csv, encoding="utf-8")):
            if r["LabelName"] == OI_FACE:
                faces[(split, r["ImageID"])].append([float(r["XMin"]), float(r["YMin"]), float(r["XMax"]),
                                                      float(r["YMax"]), int(r["IsGroupOf"])])
    jobs = sorted((s, i, b) for (s, i), b in faces.items() if i not in rotated)   # rotation metadata: skipped
    if cfg.smoke:
        jobs = jobs[: cfg.det_smoke_images]

    def fetch(job):
        split, iid, boxes = job
        dst = root / "images" / f"{iid}.jpg"
        try:
            if not dst.is_file():
                data = urllib.request.urlopen(f"https://open-images-dataset.s3.amazonaws.com/{split}/{iid}.jpg", timeout=60).read()
                im = Image.open(io.BytesIO(data)).convert("RGB")
                im.thumbnail((800, 800), Image.BILINEAR)
                im.save(dst, quality=92)
            w, h = Image.open(dst).size
            return iid, {"w": w, "h": h, "faces": [[b[0] * w, b[1] * h, b[2] * w, b[3] * h, b[4]] for b in boxes]}
        except Exception:  # noqa: BLE001  (a missing photo is skipped)
            return iid, None

    ann = {}
    with ThreadPoolExecutor(workers) as pool:
        for iid, a in tqdm(pool.map(fetch, jobs), total=len(jobs), desc="Open Images photos"):
            if a is not None:
                ann[iid] = a
    if not cfg.smoke:
        json.dump(ann, open(ann_path, "w"))
    return ann


# --------------------------------------------------------------------------- landmark labels
def landmark_labels(cfg: Config) -> tuple[dict, dict]:
    """({Open Images id: faces}, {LFW "person/file.jpg": faces}); each face is
    [x1, y1, x2, y2, score, 10 landmark coords], from training/labels/landmark_labels.json.gz."""
    with gzip.open(cfg.landmark_labels, "rt", encoding="utf-8") as f:
        data = json.load(f)
    return data["openimages"], data["lfw"]


def _iou(a, b):
    x1, y1 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / np.maximum(area(a)[:, None] + area(b)[None] - inter, 1e-9)


def build_records(cfg: Config, oi_ann: dict, oi_labels: dict, lfw_paths: dict, lfw_labels: dict) -> list[tuple]:
    """Training records (path, boxes (G, 4), landmarks (G, 10), has_landmarks (G,), ignore (G,))."""
    recs = []
    for iid, a in oi_ann.items():
        gt = np.array([f[:4] for f in a["faces"]], dtype=np.float32).reshape(-1, 4)
        group = np.array([f[4] == 1 for f in a["faces"]], dtype=bool)
        t = np.array(oi_labels.get(iid, []), dtype=np.float32).reshape(-1, 15)
        ldm, has = np.zeros((len(gt), 10), np.float32), np.zeros(len(gt), bool)
        unlabelled = t[t[:, 4] > 0.95, :4] if len(t) else np.zeros((0, 4), np.float32)
        if len(t) and len(gt):
            ov = _iou(gt, t[:, :4])
            for g in range(len(gt)):
                j = int(ov[g].argmax())
                if ov[g, j] >= 0.4 and not group[g]:
                    ldm[g], has[g] = t[j, 5:], True
            unlabelled = t[(ov.max(0) < 0.3) & (t[:, 4] > 0.95), :4]   # faces found in the labels but nobody boxed
        n = len(unlabelled)
        recs.append((cfg.open_images_dir / "images" / f"{iid}.jpg", np.concatenate([gt, unlabelled]),
                     np.concatenate([ldm, np.zeros((n, 10), np.float32)]), np.concatenate([has, np.zeros(n, bool)]),
                     np.concatenate([group, np.ones(n, bool)])))
    keys = sorted(k for k in lfw_labels if k in lfw_paths)
    random.Random(cfg.seed).shuffle(keys)
    for key in keys[: cfg.det_lfw_photos]:
        t = np.array(lfw_labels[key], dtype=np.float32).reshape(-1, 15)
        sure = t[:, 4] >= 0.95
        recs.append((lfw_paths[key], t[:, :4], t[:, 5:], sure.copy(), ~sure))
    return recs


def split_records(cfg: Config, recs: list[tuple]) -> tuple[list, list]:
    """(train, held-out) records: `det_val_images` Open Images photos are held out for selection."""
    oi = [r for r in recs if cfg.open_images_dir in r[0].parents]
    val = random.Random(cfg.seed).sample(oi, min(cfg.det_val_images, len(oi) // 5))
    held = {str(r[0]) for r in val}
    return [r for r in recs if str(r[0]) not in held], val


# --------------------------------------------------------------------------- augmentation
def _load(rec):
    path, b, l, h, ig = rec
    with Image.open(path) as im:
        return im.convert("RGB"), b.copy(), l.copy(), h.copy(), ig.copy()


def _crop(img, b, l, h, ig, size, rng):
    """A random square crop (30-100% of the short side), or 20% of the time a zoom-out onto a bigger
    canvas, resized to `size`. Faces whose centre leaves the crop are dropped; tiny ones ignored."""
    W, H = img.size
    if rng.random() < 0.2:
        side = int(max(W, H) * rng.uniform(1.0, 2.0))
        x0, y0 = -rng.randint(0, side - W), -rng.randint(0, side - H)
    else:
        side = int(min(W, H) * rng.uniform(0.3, 1.0))
        x0, y0 = rng.randint(0, W - side), rng.randint(0, H - side)
    canvas = Image.new("RGB", (side, side), tuple(rng.randint(0, 255) for _ in range(3)))
    canvas.paste(img.crop((max(x0, 0), max(y0, 0), min(x0 + side, W), min(y0 + side, H))), (max(-x0, 0), max(-y0, 0)))
    s = size / side
    canvas = canvas.resize((size, size), Image.BILINEAR)
    b, l = (b - [x0, y0, x0, y0]) * s, (l - [x0, y0] * 5) * s
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    keep = (cx > 0) & (cy > 0) & (cx < size) & (cy < size)
    b, l, h, ig = np.clip(b[keep], 0, size), l[keep], h[keep], ig[keep]
    ig = ig | ((b[:, 3] - b[:, 1]) < 8)
    h = h & ((l.reshape(-1, 5, 2) >= 0) & (l.reshape(-1, 5, 2) < size)).all((1, 2))
    return canvas, b, l, h, ig


class DetectorData(Dataset):
    """Augmented training images: 30% are 2x2 mosaics of four crops (many small faces)."""

    def __init__(self, recs, size: int = 640, mosaic: float = 0.3, seed: int = 42):
        self.recs, self.size, self.mosaic, self.seed = recs, size, mosaic, seed

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        rng = random.Random(hash((self.seed, i, torch.initial_seed())))
        if rng.random() < self.mosaic:
            half = self.size // 2
            img, parts = Image.new("RGB", (self.size, self.size)), []
            for q, (dx, dy) in enumerate(((0, 0), (half, 0), (0, half), (half, half))):
                rec = self.recs[i] if q == 0 else self.recs[rng.randrange(len(self.recs))]
                im, b, l, h, ig = _crop(*_load(rec), half, rng)
                img.paste(im, (dx, dy))
                parts.append((b + [dx, dy, dx, dy], l + [dx, dy] * 5, h, ig))
            b, l, h, ig = (np.concatenate([p[k] for p in parts]) for k in range(4))
        else:
            img, b, l, h, ig = _crop(*_load(self.recs[i]), self.size, rng)
        if rng.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if len(b):
                b = np.stack([self.size - b[:, 2], b[:, 1], self.size - b[:, 0], b[:, 3]], 1)
            l = l.reshape(-1, 5, 2)[:, FLIP].copy()
            l[..., 0] = self.size - l[..., 0]
            l = l.reshape(-1, 10)
        for enhance in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
            if rng.random() < 0.8:
                img = enhance(img).enhance(rng.uniform(0.6, 1.4))
        x = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1)
        return x, {"boxes": torch.as_tensor(b, dtype=torch.float32).reshape(-1, 4),
                   "ldm": torch.as_tensor(l, dtype=torch.float32).reshape(-1, 10),
                   "has_ldm": torch.as_tensor(h, dtype=torch.bool), "ignore": torch.as_tensor(ig, dtype=torch.bool)}


def collate(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


# --------------------------------------------------------------------------- loss + training
def detector_loss(out, targets, size, neg_ratio: int = 3, pos_iou: float = 0.35, neg_iou: float = 0.3):
    """(face/not-face, box, landmark) losses. Anchors with IoU >= pos_iou to a face are positives
    (every face gets at least its best anchor); anchors between neg_iou and pos_iou, or on an ignored
    face, count for nothing; of the rest, only the `neg_ratio` x #positives hardest negatives count."""
    logits, d_box, d_ldm = out
    anc = M.anchors(size, size, logits.device)
    anc_xyxy = torch.stack([anc[:, 0] - anc[:, 2] / 2, anc[:, 1] - anc[:, 3] / 2,
                            anc[:, 0] + anc[:, 2] / 2, anc[:, 1] + anc[:, 3] / 2], 1)
    cls_t, valid = torch.zeros_like(logits), torch.ones_like(logits, dtype=torch.bool)
    l_box = l_ldm = logits.sum() * 0
    n_box = n_ldm = 0
    for i, t in enumerate(targets):
        g = t["boxes"]
        if len(g) == 0:
            continue
        ov = box_iou(anc_xyxy, g)
        best, idx = ov.max(1)
        best_anchor = ov.argmax(0)
        best[best_anchor], idx[best_anchor] = 1.0, torch.arange(len(g), device=g.device)
        ign = t["ignore"][idx]
        pos = (best >= pos_iou) & ~ign
        valid[i] = ~((best >= neg_iou) & ((best < pos_iou) | ign))
        cls_t[i, pos] = 1.0
        if pos.any():
            tb, tl = M.encode_detections(g[idx[pos]], t["ldm"][idx[pos]], anc[pos])
            l_box = l_box + F.smooth_l1_loss(d_box[i, pos], tb, reduction="sum")
            n_box += int(pos.sum())
            has = t["has_ldm"][idx[pos]]
            if has.any():
                l_ldm = l_ldm + F.smooth_l1_loss(d_ldm[i, pos][has], tl[has], reduction="sum")
                n_ldm += int(has.sum())
    raw = F.binary_cross_entropy_with_logits(logits, cls_t, reduction="none")
    pos_mask, neg_mask = (cls_t > 0) & valid, (cls_t == 0) & valid
    n_pos = int(pos_mask.sum())
    neg = raw[neg_mask]
    neg = neg.topk(min(max(neg_ratio * n_pos, 16 * logits.shape[0]), neg.numel())).values
    return (raw[pos_mask].sum() + neg.sum()) / max(n_pos, 1), l_box / max(n_box, 1), l_ldm / max(n_ldm, 1)


def average_precision(results, min_height: float) -> float:
    """AP at IoU 0.5. results: [(gt (G, 5): x1 y1 x2 y2 ignore, detections (D, 5): x1 y1 x2 y2 score)].
    Faces shorter than min_height or marked ignore don't count; detections on them are skipped."""
    scores, hits, n_pos = [], [], 0
    for gt, dets in results:
        valid = (gt[:, 4] == 0) & ((gt[:, 3] - gt[:, 1]) >= min_height)
        n_pos += int(valid.sum())
        used = np.zeros(len(gt), bool)
        for d in dets[np.argsort(-dets[:, 4])]:
            ov = _iou(d[None, :4], gt[:, :4])[0] if len(gt) else np.zeros(0)
            match = np.where(valid & ~used & (ov >= 0.5))[0]
            if len(match):
                used[match[np.argmax(ov[match])]] = True
                scores.append(d[4]); hits.append(1)
            elif not ((ov >= 0.5) & ~valid).any():
                scores.append(d[4]); hits.append(0)
    order = np.argsort(-np.array(scores))
    tp = np.cumsum(np.array(hits)[order]) if hits else np.zeros(0)
    fp = np.cumsum(1 - np.array(hits)[order]) if hits else np.zeros(0)
    recall, precision = tp / max(n_pos, 1), tp / np.maximum(tp + fp, 1e-9)
    mrec, mpre = np.concatenate([[0], recall, [1]]), np.concatenate([[0], precision, [0]])
    for k in range(len(mpre) - 2, -1, -1):
        mpre[k] = max(mpre[k], mpre[k + 1])
    step = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[step + 1] - mrec[step]) * mpre[step + 1]))


SIZE_BANDS = {"large": 64, "medium": 32, "small": 16}     # minimum face height in pixels


def evaluate_detections(detect, items) -> dict[str, float]:
    """detect(PIL) -> (D, 5); items: [(image path, gt (G, 5))] -> {band: AP}."""
    results = []
    for path, gt in items:
        with Image.open(path) as im:
            results.append((gt, np.asarray(detect(im.convert("RGB")), dtype=np.float64).reshape(-1, 5)))
    return {band: average_precision(results, h) for band, h in SIZE_BANDS.items()}


def detector_fn(det: M.FaceDetector, score: float = 0.02):
    def detect(img):
        boxes, scores, _ = det.detect([img], score=score)[0]
        return np.hstack([boxes, scores[:, None]]) if len(boxes) else np.zeros((0, 5))
    return detect


def held_out_items(val_recs):
    return [(r[0], np.hstack([r[1], r[4][:, None].astype(np.float64)])) for r in val_recs]


def train_detector(det: M.FaceDetector, train_recs, val_recs, cfg: Config, device, log_every: int = 1):
    """Trains `det` in place; keeps the epoch with the best held-out AP on medium faces (>= 32 px)."""
    workers = cfg.det_workers
    dl = DataLoader(DetectorData(train_recs, cfg.det_size, cfg.det_mosaic, cfg.seed), batch_size=cfg.det_batch_size,
                    shuffle=True, num_workers=workers, collate_fn=collate, drop_last=True,
                    persistent_workers=workers > 0, generator=torch.Generator().manual_seed(cfg.seed))
    opt = torch.optim.AdamW(det.parameters(), lr=cfg.det_lr, weight_decay=5e-4)
    total, warm = cfg.det_epochs * len(dl), len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))
    scaler = utils.grad_scaler(device)
    items = held_out_items(val_recs)
    history = {k: [] for k in ("loss_face", "loss_box", "loss_landmarks", "val_epoch", "val_large", "val_medium", "val_small")}
    best, best_state, start = -1.0, None, time.perf_counter()
    for epoch in range(1, cfg.det_epochs + 1):
        det.train()
        sums, n = np.zeros(3), 0
        for x, t in dl:
            x = x.to(device, non_blocking=True).float().div_(255)
            t = [{k: v.to(device) for k, v in ti.items()} for ti in t]
            with utils.autocast(device):
                out = det(x)
            lc, lb, ll = detector_loss(tuple(o.float() for o in out), t, cfg.det_size)
            loss = lc + lb + cfg.det_landmark_weight * ll
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            sums += [lc.item(), lb.item(), ll.item()]
            n += 1
        for k, v in zip(("loss_face", "loss_box", "loss_landmarks"), sums / max(n, 1)):
            history[k].append(float(v))
        if epoch % cfg.det_val_every == 0 or epoch == cfg.det_epochs:
            det.eval()
            ap = evaluate_detections(detector_fn(det), items)
            history["val_epoch"].append(epoch)
            for band, v in ap.items():
                history[f"val_{band}"].append(v)
            if ap["medium"] > best:
                best, best_state, history["best_epoch"] = ap["medium"], copy.deepcopy(det.state_dict()), epoch
            print(f"epoch {epoch:3d}/{cfg.det_epochs} | loss face {history['loss_face'][-1]:.3f} box {history['loss_box'][-1]:.3f} "
                  f"landmarks {history['loss_landmarks'][-1]:.3f} | held-out AP large {ap['large']:.3f} medium {ap['medium']:.3f} "
                  f"small {ap['small']:.3f} | {time.perf_counter() - start:.0f}s", flush=True)
    det.load_state_dict(best_state)
    det.eval()
    print(f"kept epoch {history['best_epoch']} (best held-out AP, medium faces)")
    return history


# --------------------------------------------------------------------------- WIDER FACE (evaluation only)
def wider_items(cfg: Config, limit: int | None = None) -> list:
    """WIDER FACE val images + boxes ([x1, y1, x2, y2, invalid]), downloaded once from the Hub
    (CUHK-CSE/wider_face, CC BY-NC-ND 4.0: used for evaluation only)."""
    root = cfg.wider_dir
    if not (root / "WIDER_val").is_dir():
        from huggingface_hub import hf_hub_download

        for f in ("data/WIDER_val.zip", "data/wider_face_split.zip"):
            zipfile.ZipFile(hf_hub_download("CUHK-CSE/wider_face", f, repo_type="dataset")).extractall(root)
    lines = (root / "wider_face_split/wider_face_val_bbx_gt.txt").read_text().splitlines()
    items, i = [], 0
    while i < len(lines):
        name, n = lines[i], int(lines[i + 1])
        rows = [list(map(int, lines[i + 2 + k].split()[:8])) for k in range(max(n, 1))]
        i += 2 + max(n, 1)
        gt = np.array([[r[0], r[1], r[0] + r[2], r[1] + r[3], r[7]] for r in rows if n and r[2] > 0 and r[3] > 0],
                      dtype=np.float64).reshape(-1, 5)
        items.append((root / "WIDER_val/images" / name, gt))
    return items[:limit] if limit else items
