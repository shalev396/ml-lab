"""Train / evaluate loops (device-agnostic, AMP on CUDA only) and classification metrics."""
import copy
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from . import utils


def make_loss(label_smoothing: float) -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def make_optimizer(net: nn.Module, lr: float) -> torch.optim.Optimizer:
    """Adam over every trainable parameter (the whole network in the original full fine-tune)."""
    return torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr)


def train_one_epoch(model: nn.Module, loader: DataLoader, loss_fn, optimizer, device, scaler,
                    log_every: int = 200) -> tuple[float, float]:
    """One pass over `loader`. Returns (mean loss, accuracy) over all samples."""
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    start = time.perf_counter()
    for step, (x, y) in enumerate(loader, 1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with utils.autocast(device):
            logits = model(x)
            loss = loss_fn(logits, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(dim=1) == y).sum().item()
        seen += len(y)
        if log_every and (step % log_every == 0 or step == len(loader)):
            print(f"  batch {step}/{len(loader)} | loss {total_loss / seen:.4f} acc {correct / seen:.4f} "
                  f"| {time.perf_counter() - start:.0f}s", flush=True)
    return total_loss / seen, correct / seen


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn, device) -> dict:
    """Per-image evaluation of every sample in `loader`.

    Returns {"loss", "accuracy", "top5_accuracy", "y_true", "y_pred", "top5"} (numpy arrays).
    """
    device = torch.device(device)
    model.eval()
    total_loss, labels, top5 = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with utils.autocast(device):
            logits = model(x)
        logits = logits.float()
        total_loss += loss_fn(logits, y).item() * len(y)
        labels.append(y.cpu())
        top5.append(logits.topk(min(5, logits.shape[1]), dim=1).indices.cpu())
    y_true, top5 = torch.cat(labels).numpy(), torch.cat(top5).numpy()
    y_pred = top5[:, 0]
    return {"loss": total_loss / len(y_true), "accuracy": float((y_pred == y_true).mean()),
            "top5_accuracy": float((top5 == y_true[:, None]).any(axis=1).mean()),
            "y_true": y_true, "y_pred": y_pred, "top5": top5}


def train(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, loss_fn, optimizer,
          epochs: int, device, keep: str = "best_test_loss", log_every: int = 200):
    """Train for `epochs`, evaluating on `test_loader` after each one.

    `keep="best_test_loss"` (the original recipe) restores the weights of the epoch with the lowest
    test loss at the end; `keep="last"` keeps the final epoch. Food-101 has no validation split,
    so the original run selected on the test split (in its run the best epoch was also the last).

    Returns (history, seconds, best_epoch): history = {"train_loss", "train_acc", "test_loss", "test_acc"}.
    """
    device = torch.device(device)
    scaler = utils.grad_scaler(device)
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_loss, best_epoch, best_state = float("inf"), epochs, None
    start_all = time.perf_counter()
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device, scaler, log_every)
        test = evaluate(model, test_loader, loss_fn, device)
        for key, value in (("train_loss", train_loss), ("train_acc", train_acc),
                           ("test_loss", test["loss"]), ("test_acc", test["accuracy"])):
            history[key].append(value)
        print(f"epoch {epoch}/{epochs} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"test loss {test['loss']:.4f} acc {test['accuracy']:.4f} top5 {test['top5_accuracy']:.4f} | "
              f"{time.perf_counter() - start:.0f}s", flush=True)
        if keep == "best_test_loss" and test["loss"] < best_loss:
            best_loss, best_epoch = test["loss"], epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in _unwrap(model).state_dict().items()})
    if keep == "best_test_loss" and best_state is not None and best_epoch != epochs:
        _unwrap(model).load_state_dict(best_state)
        print(f"restored epoch {best_epoch} (lowest test loss {best_loss:.4f})")
    return history, time.perf_counter() - start_all, best_epoch


def _unwrap(model: nn.Module) -> nn.Module:
    """The module beneath a torch.compile wrapper."""
    return getattr(model, "_orig_mod", model)


def classification_metrics(result: dict) -> dict[str, float]:
    """Top-1 accuracy, top-5 accuracy and macro F1 over every image.

    Macro F1 averages over the classes that occur in y_true or y_pred (all 101 on the full test split).
    """
    return {
        "accuracy": float(result["accuracy"]),
        "top5_accuracy": float(result["top5_accuracy"]),
        "f1_macro": float(f1_score(result["y_true"], result["y_pred"], average="macro", zero_division=0)),
    }


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Share of each class's images predicted correctly (NaN for classes absent from y_true)."""
    totals = np.bincount(y_true, minlength=n_classes).astype(float)
    hits = np.bincount(y_true[y_true == y_pred], minlength=n_classes).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(totals > 0, hits / totals, np.nan)
