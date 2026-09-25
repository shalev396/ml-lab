"""Train / evaluate loops (device-agnostic, AMP on CUDA only via utils) and age/gender metrics."""
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import model as M  # ../model/model.py, put on sys.path by src/__init__.py

from . import utils


def set_trunk_trainable(net: nn.Module, trainable: bool) -> None:
    for p in net.features.parameters():
        p.requires_grad = trainable


def train_one_epoch(net, loader: DataLoader, optimizer, device, scaler, gender_loss_weight: float = 1.0,
                    amp: bool = True) -> dict:
    """One pass over `loader`; loss = CE(age bins) + w * CE(gender). Returns mean losses."""
    net.train()
    ce = nn.CrossEntropyLoss()
    sums, seen = {"loss": 0.0, "age_loss": 0.0, "gender_loss": 0.0}, 0
    for x, age_bin, gender, _age in loader:
        x = x.to(device, non_blocking=True)
        age_bin, gender = age_bin.to(device), gender.to(device)
        with utils.autocast(device, enabled=amp):
            out = net(x)
            age_loss = ce(out["age"].float(), age_bin)
            gender_loss = ce(out["gender"].float(), gender)
            loss = age_loss + gender_loss_weight * gender_loss
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        n = len(x)
        sums["loss"] += loss.item() * n
        sums["age_loss"] += age_loss.item() * n
        sums["gender_loss"] += gender_loss.item() * n
        seen += n
    return {k: v / seen for k, v in sums.items()}


@torch.inference_mode()
def evaluate(net, loader: DataLoader, device, age_representatives: list[float], amp: bool = True) -> dict:
    """Predictions for every sample: true/expected/argmax ages, gender truth/prediction/p(female)."""
    net.eval()
    reps = torch.tensor(age_representatives, dtype=torch.float32)
    true_age, exp_age, argmax_age, g_true, g_prob = [], [], [], [], []
    for x, _bin, gender, age in loader:
        with utils.autocast(device, enabled=amp):
            out = net(x.to(device, non_blocking=True))
        age_p = out["age"].float().softmax(dim=1).cpu()
        g_p = out["gender"].float().softmax(dim=1).cpu()
        true_age.append(age)
        exp_age.append(M.expected_age(age_p, age_representatives))
        argmax_age.append(reps[age_p.argmax(dim=1)])
        g_true.append(gender)
        g_prob.append(g_p[:, 1])
    cat = lambda xs: torch.cat(xs).numpy()  # noqa: E731
    g_prob = cat(g_prob)
    return {"true_age": cat(true_age).astype(float), "age_pred": cat(exp_age), "age_pred_argmax": cat(argmax_age),
            "gender_true": cat(g_true), "gender_pred": (g_prob >= 0.5).astype(int), "gender_prob_female": g_prob}


def scores(true_age, age_pred, gender_true, gender_pred) -> dict:
    """age_mae (years), age_rmse, age_within5 (share with |error| <= 5 y), gender_accuracy."""
    err = np.abs(np.asarray(age_pred, float) - np.asarray(true_age, float))
    return {
        "age_mae": float(err.mean()),
        "age_rmse": float(np.sqrt((err ** 2).mean())),
        "age_within5": float((err <= 5).mean()),
        "gender_accuracy": float((np.asarray(gender_pred) == np.asarray(gender_true)).mean()),
    }


def mae_by_age_group(true_age, age_pred, edges=(0, 13, 20, 30, 40, 50, 60, 70, 80, 200)) -> dict:
    """{'0-12': MAE, ...} for readable age groups (None for empty groups)."""
    true_age, age_pred = np.asarray(true_age), np.asarray(age_pred)
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (true_age >= lo) & (true_age < hi)
        label = f"{lo}+" if hi >= 200 else f"{lo}-{hi - 1}"
        out[label] = {"mae": float(np.abs(age_pred[m] - true_age[m]).mean()) if m.any() else None, "n": int(m.sum())}
    return out


def fit(net, train_loader: DataLoader, val_loader: DataLoader, cfg, device, age_representatives) -> dict:
    """Train for cfg.epochs, validating after each; keeps the weights of the epoch with the lowest
    validation age MAE (loaded back into `net` at the end). Returns the per-epoch history."""
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    scaler = utils.grad_scaler(device, enabled=cfg.amp)
    history = {k: [] for k in ("train_loss", "age_loss", "gender_loss", "val_age_mae", "val_within5",
                               "val_gender_accuracy", "epoch_s")}
    best = {"mae": float("inf"), "epoch": 0, "state": None}
    for epoch in range(1, cfg.epochs + 1):
        if cfg.freeze_epochs and epoch == cfg.freeze_epochs + 1:
            set_trunk_trainable(net, True)
            print("unfreezing the trunk")
        start = time.perf_counter()
        tr = train_one_epoch(net, train_loader, optimizer, device, scaler, cfg.gender_loss_weight, cfg.amp)
        ev = evaluate(net, val_loader, device, age_representatives, cfg.amp)
        sc = scores(ev["true_age"], ev["age_pred"], ev["gender_true"], ev["gender_pred"])
        seconds = time.perf_counter() - start
        history["train_loss"].append(tr["loss"])
        history["age_loss"].append(tr["age_loss"])
        history["gender_loss"].append(tr["gender_loss"])
        history["val_age_mae"].append(sc["age_mae"])
        history["val_within5"].append(sc["age_within5"])
        history["val_gender_accuracy"].append(sc["gender_accuracy"])
        history["epoch_s"].append(seconds)
        print(f"epoch {epoch:2d}/{cfg.epochs} | loss {tr['loss']:.4f} | val age MAE {sc['age_mae']:.2f} y | "
              f"±5y {sc['age_within5']:.1%} | gender acc {sc['gender_accuracy']:.1%} | {seconds:.0f}s")
        if sc["age_mae"] < best["mae"]:
            best = {"mae": sc["age_mae"], "epoch": epoch,
                    "state": {k: v.detach().to("cpu", copy=True) for k, v in net.state_dict().items()}}
    net.load_state_dict(best["state"])
    history["best_epoch"] = best["epoch"]
    print(f"best epoch {best['epoch']} (val age MAE {best['mae']:.2f} y) restored")
    return history
