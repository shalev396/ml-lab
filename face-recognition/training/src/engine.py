"""Training + evaluation. Device-agnostic (CUDA -> MPS -> CPU); AMP only on CUDA.

Stage 1 trains the embedder with a CosFace margin loss over every training identity; augmentation
runs batched on the device (one random affine + colour jitter + erasing per face). Stage 2 trains
the identity MLP on frozen embeddings. Both keep the epoch with the best validation score.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, top_k_accuracy_score
from torch import nn

from . import utils


# --------------------------------------------------------------------------- augmentation
def augment(faces: torch.Tensor, size: int) -> torch.Tensor:
    """(N, 3, crop, crop) uint8 faces -> (N, 3, size, size) float 0..255, randomly transformed:
    zoom (crop area 70-100%), shift, rotation +-10 deg (half the faces), horizontal flip, brightness /
    contrast / saturation +-30% (80%), grayscale (10%) and one erased rectangle (25%)."""
    n, dev = faces.shape[0], faces.device
    rand = lambda lo, hi: torch.empty(n, device=dev).uniform_(lo, hi)
    x = F.interpolate(faces.float() / 255, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    angle = rand(-10, 10) * math.pi / 180 * (torch.rand(n, device=dev) < 0.5)
    scale = rand(0.84, 1.0)
    flip = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    tx, ty = rand(-1, 1) * (1 - scale), rand(-1, 1) * (1 - scale)
    cos, sin = torch.cos(angle) * scale, torch.sin(angle) * scale
    theta = torch.stack([torch.stack([cos * flip, -sin, tx], 1), torch.stack([sin * flip, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, (n, 3, size, size), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    gray = lambda t: 0.299 * t[:, :1] + 0.587 * t[:, 1:2] + 0.114 * t[:, 2:]
    jitter = (torch.rand(n, 1, 1, 1, device=dev) < 0.8).float()
    brightness, contrast, saturation = (1 + jitter * rand(-0.3, 0.3).view(n, 1, 1, 1) for _ in range(3))
    x = x * brightness
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * contrast + mean
    g = gray(x)
    x = (x - g) * saturation + g
    to_gray = (torch.rand(n, 1, 1, 1, device=dev) < 0.1).float()
    x = (x * (1 - to_gray) + gray(x) * to_gray).clamp(0, 1)

    erase = torch.rand(n, device=dev) < 0.25
    area, aspect = rand(0.02, 0.15) * size * size, rand(0.5, 2.0)
    h, w = (area * aspect).sqrt().clamp(max=size - 1), (area / aspect).sqrt().clamp(max=size - 1)
    y0, x0 = rand(0, 1) * (size - h), rand(0, 1) * (size - w)
    yy, xx = torch.arange(size, device=dev).view(1, -1, 1), torch.arange(size, device=dev).view(1, 1, -1)
    mask = ((yy >= y0.view(-1, 1, 1)) & (yy < (y0 + h).view(-1, 1, 1)) & (xx >= x0.view(-1, 1, 1))
            & (xx < (x0 + w).view(-1, 1, 1)) & erase.view(-1, 1, 1)).unsqueeze(1)
    x = torch.where(mask, torch.rand_like(x), x)
    return x * 255


# --------------------------------------------------------------------------- stage 1: embedder
class CosFace(nn.Module):
    """Large-margin cosine loss head (Wang et al. 2018): logits = s * (cos(theta) - m on the true class)."""

    def __init__(self, embedding_dim: int, n_classes: int, scale: float = 30.0, margin: float = 0.35):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_classes, embedding_dim) * 0.01)
        self.scale, self.margin = scale, margin

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cos = F.normalize(embeddings) @ F.normalize(self.weight).t()
        return (cos - self.margin * F.one_hot(targets, cos.shape[1])) * self.scale


@torch.inference_mode()
def embed(net, crops: torch.Tensor, idx: np.ndarray, device, batch_size: int = 256) -> np.ndarray:
    """L2-normalised, flip-averaged embeddings of crops[idx] (exactly FaceRecognizer.embed)."""
    net.eval()
    out = []
    for s in range(0, len(idx), batch_size):
        faces = crops[torch.as_tensor(idx[s:s + batch_size], device=crops.device)].to(device)
        with utils.autocast(device):
            out.append(net.embed(faces).float().cpu())
    return torch.cat(out).numpy()


def centroid_accuracy(train_emb, train_y, eval_emb, eval_y) -> float:
    """Nearest class-mean accuracy (cosine): a head-free check of how well identities separate."""
    classes = np.unique(train_y)
    centroids = np.stack([train_emb[train_y == c].mean(0) for c in classes])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    return float((classes[(eval_emb @ centroids.T).argmax(1)] == eval_y).mean())


def pair_distances(embeddings: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cosine distance and same-person flag for every unordered pair of samples."""
    e = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    i, j = np.triu_indices(len(e), k=1)
    return 1.0 - np.einsum("nd,nd->n", e[i], e[j]), labels[i] == labels[j]


def verify_auc(embeddings: np.ndarray, labels: np.ndarray) -> float:
    dist, same = pair_distances(embeddings, labels)
    return float(roc_auc_score(same, -dist)) if same.any() and (~same).any() else float("nan")


def train_embedder(member, spec: dict, crops: torch.Tensor, names: np.ndarray, splits: dict, cfg, device,
                   log_every: int = 5):
    """CosFace training of one ensemble member (model.FaceEmbedNet); returns (history, best epoch).
    `spec` is its cfg.embedders entry (epochs, pretrained).

    After every epoch: nearest-centroid accuracy on the 42 people's train and validation photos
    and the verification ROC AUC on the unseen people. The weights of the epoch with the best
    `val_acc + 0.1 * unseen_auc` are restored at the end (the test split is never used).
    """
    idx = splits["embedder_train"]
    ident = {n: i for i, n in enumerate(splits["embedder_identities"])}
    targets = torch.as_tensor([ident[n] for n in names[idx]], device=device)
    classes = splits["class_names"]
    y_train = np.array([classes.index(n) for n in names[splits["train"]]])
    y_val = np.array([classes.index(n) for n in names[splits["val"]]])
    unseen_names = names[splits["unseen"]]

    net, epochs = member, spec["epochs"]
    cosface = CosFace(net.neck[1].out_features, len(ident), cfg.cosface_scale, cfg.cosface_margin).to(device)
    optimizer = torch.optim.AdamW([
        {"params": net.backbone.parameters(), "lr": cfg.lr * (cfg.backbone_lr_mult if spec["pretrained"] else 1.0)},
        {"params": list(net.neck.parameters()) + list(cosface.parameters()), "lr": cfg.lr},
    ], weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(idx) / cfg.batch_size)
    total, warmup = steps_per_epoch * epochs, steps_per_epoch * cfg.warmup_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / max(1, warmup))
                                                  * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    scaler = utils.grad_scaler(device)
    history = {k: [] for k in ("train_loss", "train_acc", "val_acc", "unseen_auc")}
    best_score, best_state, best_epoch = -1.0, None, 0
    generator = np.random.default_rng(cfg.seed)
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        net.train()
        cosface.train()
        total_loss, seen = 0.0, 0
        for batch in np.array_split(generator.permutation(len(idx)), steps_per_epoch):
            faces = crops[torch.as_tensor(idx[batch], device=crops.device)].to(device)
            x = net.normalize(augment(faces, net.input_size))
            y = targets[torch.as_tensor(batch, device=device)]
            with utils.autocast(device):
                loss = F.cross_entropy(cosface(net.features(x).float(), y), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item() * len(batch)
            seen += len(batch)
        e_train = embed(net, crops, splits["train"], device)
        e_val = embed(net, crops, splits["val"], device)
        e_unseen = embed(net, crops, splits["unseen"], device)
        row = {"train_loss": total_loss / seen, "train_acc": centroid_accuracy(e_train, y_train, e_train, y_train),
               "val_acc": centroid_accuracy(e_train, y_train, e_val, y_val),
               "unseen_auc": verify_auc(e_unseen, unseen_names)}
        for k, v in row.items():
            history[k].append(v)
        score = row["val_acc"] + 0.1 * row["unseen_auc"]
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in net.state_dict().items()})
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(f"epoch {epoch:3d}/{epochs} | loss {row['train_loss']:.3f} | train acc {row['train_acc']:.4f} | "
                  f"val acc {row['val_acc']:.4f} | unseen AUC {row['unseen_auc']:.4f} | "
                  f"{time.perf_counter() - start:.0f}s")
    net.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    print(f"kept epoch {best_epoch} (best validation score)")
    return history, best_epoch


# --------------------------------------------------------------------------- stage 2: identity head
def train_head(net, emb: dict, y: dict, cfg, device):
    """Fit net.head on frozen train embeddings; keep the epoch with the lowest validation loss
    (validation accuracy saturates at 100% within a few epochs, when the head is still
    under-confident; the loss keeps improving). Returns the history (train/val loss + accuracy)."""
    torch.manual_seed(cfg.seed)
    for layer in net.head:
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
    head = net.head
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.head_lr, weight_decay=cfg.head_weight_decay)
    xt, yt = torch.as_tensor(emb["train"], device=device), torch.as_tensor(y["train"], device=device)
    xv, yv = torch.as_tensor(emb["val"], device=device), torch.as_tensor(y["val"], device=device)
    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best, best_state, gen = (-1.0, float("inf")), None, torch.Generator().manual_seed(cfg.seed)
    for _ in range(cfg.head_epochs):
        head.train()
        for batch in torch.randperm(len(xt), generator=gen).split(cfg.head_batch_size):
            batch = batch.to(device)
            loss = F.cross_entropy(head(xt[batch]), yt[batch])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            lt, lv = head(xt), head(xv)
            row = {"train_loss": F.cross_entropy(lt, yt).item(), "train_acc": (lt.argmax(1) == yt).float().mean().item(),
                   "val_loss": F.cross_entropy(lv, yv).item(), "val_acc": (lv.argmax(1) == yv).float().mean().item()}
        for k, v in row.items():
            history[k].append(v)
        if row["val_loss"] < best[1]:
            best, best_state = (row["val_acc"], row["val_loss"]), copy.deepcopy(head.state_dict())
            history["best_epoch"] = len(history["val_acc"])
    head.load_state_dict(best_state)
    head.eval()
    print(f"head: kept epoch {history['best_epoch']} (lowest val loss {best[1]:.4f}, val acc {best[0]:.4f})")
    return history


@torch.inference_mode()
def head_proba(net, embeddings: np.ndarray, device) -> np.ndarray:
    net.eval()
    return net.classify(torch.as_tensor(embeddings, device=device)).softmax(1).float().cpu().numpy()


# --------------------------------------------------------------------------- metrics
def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Top-1 accuracy, macro F1 and top-3 accuracy from class probabilities."""
    labels = np.arange(y_prob.shape[1])
    y_pred = y_prob.argmax(1)
    return {
        "accuracy": float((y_pred == y_true).mean()),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "top3_accuracy": float(top_k_accuracy_score(y_true, y_prob, k=min(3, len(labels)), labels=labels)),
    }


def verification_metrics(embeddings: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    """Face verification on all pairs of the given faces with `distance < threshold` = match.

    Pairs are very imbalanced (few same-person pairs), so accuracy is reported class-balanced:
    the mean of the true-accept rate (same pairs accepted) and true-reject rate.
    """
    dist, same = pair_distances(embeddings, labels)
    accept = dist < threshold
    tar = float(accept[same].mean()) if same.any() else float("nan")
    trr = float((~accept[~same]).mean()) if (~same).any() else float("nan")
    return {
        "verify_roc_auc": float(roc_auc_score(same, -dist)) if same.any() and (~same).any() else float("nan"),
        "verify_balanced_accuracy": (tar + trr) / 2,
        "verify_true_accept_rate": tar,
        "verify_false_accept_rate": 1.0 - trr,
        "n_pairs_same": int(same.sum()),
        "n_pairs_different": int((~same).sum()),
    }


def tune_verify_threshold(embeddings: np.ndarray, labels: np.ndarray, grid) -> float:
    """Threshold from `grid` with the best balanced accuracy on the given (validation) pairs."""
    scores = {float(t): verification_metrics(embeddings, labels, t)["verify_balanced_accuracy"] for t in grid}
    return max(scores, key=scores.get)
