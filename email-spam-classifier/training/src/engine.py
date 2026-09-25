"""Train / evaluate loops (device-agnostic, AMP on CUDA only), the baseline, and binary metrics."""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

import model as M

from . import model_builder, utils
from .config import Config


def _to(batch: dict, device) -> tuple[dict, torch.Tensor]:
    labels = batch["labels"].to(device, non_blocking=True)
    inputs = {k: batch[k].to(device, non_blocking=True) for k in ("input_ids", "attention_mask")}
    return inputs, labels


def train_one_epoch(net: nn.Module, loader: DataLoader, loss_fn, optimizer, scheduler, device, scaler) -> float:
    """One pass over `loader` (scheduler steps per batch). Returns the mean training loss."""
    net.train()
    total, seen = 0.0, 0
    for batch in loader:
        inputs, labels = _to(batch, device)
        with utils.autocast(device):
            logits = net(**inputs)
        loss = loss_fn(logits.float(), labels)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total += loss.item() * len(labels)
        seen += len(labels)
    return total / max(seen, 1)


@torch.inference_mode()
def evaluate(net: nn.Module, loader: DataLoader, loss_fn, device, threshold: float = 0.5) -> dict:
    """{"loss", "accuracy", "f1", "y_true", "y_prob"} over every email of `loader`."""
    net.eval()
    total, labels, probs = 0.0, [], []
    for batch in loader:
        inputs, y = _to(batch, device)
        with utils.autocast(device):
            logits = net(**inputs)
        total += loss_fn(logits.float(), y).item() * len(y)
        labels.append(y.cpu())
        probs.append(torch.sigmoid(logits.float()).cpu())
    y_true, y_prob = torch.cat(labels).numpy().astype(int), torch.cat(probs).numpy()
    y_pred = (y_prob >= threshold).astype(int)
    return {"loss": total / max(len(y_true), 1), "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)), "y_true": y_true, "y_prob": y_prob}


def fit(net: nn.Module, loaders: dict[str, DataLoader], cfg: Config, device, pos_weight: float) -> dict:
    """AdamW (differential LRs) + linear warmup/decay + BCE(pos_weight), early stopping on val loss.

    The best epoch's weights are restored into `net` at the end. Returns the per-epoch history
    {"train_loss", "val_loss", "val_acc", "val_f1", "best_epoch", "train_time_s"}.
    """
    net.to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.AdamW(model_builder.param_groups(net, cfg), weight_decay=cfg.weight_decay)
    total_steps = len(loaders["train"]) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg.warmup_ratio * total_steps), total_steps)
    scaler = utils.grad_scaler(device)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_loss, best_state, best_epoch, bad_epochs = float("inf"), None, 0, 0
    start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(net, loaders["train"], loss_fn, optimizer, scheduler, device, scaler)
        val = evaluate(net, loaders["val"], loss_fn, device, cfg.threshold)
        for key, value in (("train_loss", train_loss), ("val_loss", val["loss"]),
                           ("val_acc", val["accuracy"]), ("val_f1", val["f1"])):
            history[key].append(float(value))
        improved = val["loss"] < best_loss
        print(f"epoch {epoch}/{cfg.epochs} | train loss {train_loss:.4f} | val loss {val['loss']:.4f} "
              f"acc {val['accuracy']:.4f} f1 {val['f1']:.4f} | {time.perf_counter() - t0:.0f}s"
              + (" | best" if improved else ""))
        if improved:
            best_loss, best_epoch, bad_epochs = val["loss"], epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in net.state_dict().items()})
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"early stopping after epoch {epoch} (patience {cfg.patience})")
                break
    net.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["train_time_s"] = round(time.perf_counter() - start, 1)
    return history


def save_checkpoint(net: M.SpamClassifier, tokenizer, path) -> None:
    """Best weights + config.json + tokenizer: a folder `model.load()` can open."""
    net.save_pretrained(path)
    tokenizer.save_pretrained(path)


def predict_texts(predictor: M.Predictor, texts: list[str], batch_size: int = 64) -> np.ndarray:
    """P(spam) for every text through `Predictor`, i.e. the exact code path of the Space/endpoint."""
    return np.asarray(predictor.predict_proba(list(texts), batch_size=batch_size), dtype=float)


def train_baseline(cfg: Config, train: pd.DataFrame):
    """Fit the TF-IDF + logistic-regression baseline on the (cleaned) train split."""
    pipe = model_builder.build_baseline(cfg)
    start = time.perf_counter()
    pipe.fit(train["text"], train["label"])
    pipe.train_time_s = round(time.perf_counter() - start, 1)
    return pipe


def baseline_proba(pipe, texts) -> np.ndarray:
    return pipe.predict_proba(pd.Series([M.prepare_text(t) for t in texts]))[:, 1]


def binary_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    """accuracy, precision / recall / F1 of the spam class, ROC-AUC and average precision (PR-AUC)."""
    y_true, y_prob = np.asarray(y_true).astype(int), np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    both = len(np.unique(y_true)) == 2
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if both else float("nan"),
        "average_precision": float(average_precision_score(y_true, y_prob)) if both else float("nan"),
    }


def confusion(y_true, y_prob, threshold: float = 0.5) -> np.ndarray:
    """2x2 counts, rows = true (ham, spam), columns = predicted."""
    return confusion_matrix(np.asarray(y_true).astype(int), (np.asarray(y_prob) >= threshold).astype(int),
                            labels=[0, 1])
