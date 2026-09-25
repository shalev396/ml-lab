"""Train / evaluate loops (device-agnostic, AMP on CUDA only) and classification metrics."""
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch import nn
from torch.utils.data import DataLoader

from . import utils


def make_optimizer(net: nn.Module, lr: float) -> torch.optim.Optimizer:
    """Adam over the trainable parameters only (the frozen backbone is left out)."""
    return torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr)


def train_one_epoch(model: nn.Module, loader: DataLoader, loss_fn, optimizer, device, scaler) -> tuple[float, float]:
    """One pass over `loader`. Returns (mean loss, accuracy) over all samples."""
    # Whole network in train mode, as in the original recipe: the frozen backbone's BatchNorm
    # running statistics still adapt to the food images and stochastic depth stays active.
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for x, y in loader:
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
    return total_loss / seen, correct / seen


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn, device) -> dict:
    """Returns {"loss", "accuracy", "y_true", "y_prob", "y_pred"} over every sample of `loader`."""
    device = torch.device(device)
    model.eval()
    total_loss, labels, probs = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with utils.autocast(device):
            logits = model(x)
        total_loss += loss_fn(logits.float(), y).item() * len(y)
        labels.append(y.cpu())
        probs.append(logits.float().softmax(dim=1).cpu())
    y_true, y_prob = torch.cat(labels).numpy(), torch.cat(probs).numpy()
    y_pred = y_prob.argmax(axis=1)
    return {"loss": total_loss / len(y_true), "accuracy": float((y_pred == y_true).mean()),
            "y_true": y_true, "y_prob": y_prob, "y_pred": y_pred}


def train(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, loss_fn, optimizer,
          epochs: int, device) -> tuple[dict[str, list[float]], float]:
    """Train for `epochs`, evaluating on `test_loader` after each one (the recipe has no val split
    and keeps the last epoch, so the test curve is only monitored, never used for selection).

    Returns (history, seconds): history = {"train_loss", "train_acc", "test_loss", "test_acc"} per epoch.
    """
    device = torch.device(device)
    scaler = utils.grad_scaler(device)
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    start_all = time.perf_counter()
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device, scaler)
        test = evaluate(model, test_loader, loss_fn, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test["loss"])
        history["test_acc"].append(test["accuracy"])
        print(f"epoch {epoch:2d}/{epochs} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"test loss {test['loss']:.4f} acc {test['accuracy']:.4f} | {time.perf_counter() - start:.1f}s")
    return history, time.perf_counter() - start_all


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict[str, float]:
    """accuracy, macro F1 and per-class accuracy (= recall: share of each class's images right)."""
    labels = list(range(len(class_names)))
    per_class = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        **{f"accuracy_{name}": float(acc) for name, acc in zip(class_names, per_class)},
    }
