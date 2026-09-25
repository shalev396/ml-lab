"""Export a trained AgeGenderNet as a Hugging Face model repo folder + the card's plots.

`save_model_files()` writes a loadable folder (weights + config via PyTorchModelHubMixin, plus
model.py / handler.py / requirements.txt / face_detector.safetensors copied from model/ when the target is not
model/ itself). `export()` adds metrics.json (STANDARD §4) and assets/*.png, then refreshes the
metrics in README.md. Smoke runs export to training/outputs/smoke/model, never to model/.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from . import utils

MODEL_NAME = "EfficientNet-B2 multi-task (90 age bins + gender), FaceDetector face detection"
TASK = "age-estimation + gender-classification (face images)"
DATASET = "UTKFace (nu-delta/utkface on the Hugging Face Hub), 90/10 random train/validation split, seed 42"
CODE_FILES = ("model.py", "handler.py", "requirements.txt", "LICENSE", "face_detector.safetensors")

# Chart colours (validated categorical slots + one-hue sequential ramp, see the dataviz palette).
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#a3a29d"
BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f8fe", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])


def build_metrics(scores: dict[str, float], *, data: dict, params: int, train_time_s: float | None,
                  device: str, smoke: bool, source: str, trained_at: str | None = None,
                  comparison: dict | None = None, extra: dict | None = None) -> dict:
    """metrics.json content (schema: STANDARD §4). `scores` = the deployed model's validation metrics."""
    metrics = {
        "model": MODEL_NAME,
        "task": TASK,
        "dataset": DATASET,
        "split": "validation",
        "primary_metric": {"name": "age_mae", "value": scores["age_mae"]},   # years, lower is better
        "metrics": scores,
        "data": data,
        "params": int(params),
        "train_time_s": None if train_time_s is None else round(train_time_s, 1),
        "device": device,
        "smoke": smoke,
        "source": source,
        "versions": utils.lib_versions("torch", "torchvision", "huggingface_hub", "safetensors", "numpy"),
        "trained_at": trained_at or utils.today(),
    }
    if comparison:
        metrics["comparison"] = comparison
    if extra:
        metrics.update(extra)
    return metrics


def save_model_files(net, model_dir: str | Path) -> Path:
    """Weights + config.json, and (outside model/) the code + detector needed to `model.load()` it."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if model_dir.resolve() != utils.MODEL_DIR.resolve():
        for name in CODE_FILES:
            shutil.copy2(utils.MODEL_DIR / name, model_dir / name)
    # model.safetensors + config.json (an existing README.md is kept). The mixin would record the
    # training-time `pretrained=True`; the saved config says False so `from_pretrained` never
    # downloads the ImageNet trunk just to overwrite it with these weights.
    config = {**getattr(net, "_hub_mixin_config", {}), "pretrained": False}
    net.save_pretrained(model_dir, config=config)
    return model_dir


def deployed_is_better(model_dir: str | Path, metrics: dict) -> bool:
    """True when `model_dir` already holds a model whose validation age MAE is <= this run's."""
    path = Path(model_dir) / "metrics.json"
    if not path.is_file():
        return False
    deployed = utils.load_json(path)
    old = (deployed.get("metrics") or {}).get("age_mae")
    return old is not None and not deployed.get("smoke") and metrics["metrics"]["age_mae"] >= old


def export(net, model_dir: str | Path, metrics: dict, *, evaluation: dict, history: dict | None = None,
           mae_groups: dict | None = None, overwrite_if_worse: bool = False) -> dict | None:
    """Write weights, config, metrics and plots to `model_dir`, then update its model card.

    Into the deployed model/ folder only when this run has a lower validation age MAE than the model
    already there (or `overwrite_if_worse=True`); otherwise nothing is written and None is returned."""
    model_dir = Path(model_dir)
    if (model_dir.resolve() == utils.MODEL_DIR.resolve() and not overwrite_if_worse
            and deployed_is_better(model_dir, metrics)):
        old = utils.load_json(model_dir / "metrics.json")["metrics"]["age_mae"]
        print(f"not exported: this run's age MAE {metrics['metrics']['age_mae']:.4f} is not lower than the "
              f"deployed model's {old:.4f} (set OVERWRITE_IF_WORSE = True to replace it anyway)")
        return None
    if model_dir.resolve() != utils.MODEL_DIR.resolve():
        model_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(utils.MODEL_DIR / "README.md", model_dir / "README.md")
    save_model_files(net, model_dir)
    write_assets(model_dir / "assets", metrics, evaluation, history, mae_groups)
    utils.save_json(metrics, model_dir / "metrics.json")
    utils.update_model_card(model_dir, metrics)
    print(f"exported -> {model_dir}")
    return metrics


def write_assets(assets: Path, metrics: dict, evaluation: dict, history: dict | None = None,
                 mae_groups: dict | None = None) -> list[Path]:
    paths = [
        plot_age_scatter(evaluation["true_age"], evaluation["age_pred"], assets / "age_pred_vs_true.png"),
        plot_gender_confusion(evaluation["gender_true"], evaluation["gender_pred"], assets / "gender_confusion.png"),
    ]
    if mae_groups:
        paths.append(plot_mae_by_group(mae_groups, assets / "age_error_by_group.png"))
    if metrics.get("comparison"):
        paths.append(plot_comparison(metrics["comparison"], assets / "comparison.png"))
    if history:
        paths.append(plot_history(history, assets / "training_curves.png"))
    return paths


# --------------------------------------------------------------------------- plots
def _style(ax, grid_axis: str | None = "both") -> None:
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)


def _save(fig: Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path


def plot_age_distribution(ages: np.ndarray, path: str | Path, title: str = "Age distribution") -> Path:
    fig = Figure(figsize=(7.0, 3.0), dpi=150, facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    _style(ax, "y")
    ax.hist(ages, bins=np.arange(0, 118, 2), color=BLUE, edgecolor=SURFACE, linewidth=0.5)
    ax.set_xlabel("age (years)", color=INK_MUTED)
    ax.set_ylabel("faces", color=INK_MUTED)
    ax.set_title(f"{title} ({len(ages):,} faces)", color=INK, fontsize=10, loc="left")
    return _save(fig, path)


def plot_age_scatter(true_age, age_pred, path: str | Path) -> Path:
    """Predicted (expected) age vs true age; the diagonal is a perfect prediction."""
    true_age, age_pred = np.asarray(true_age, float), np.asarray(age_pred, float)
    mae = np.abs(age_pred - true_age).mean()
    fig = Figure(figsize=(4.8, 4.4), dpi=150, facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    _style(ax)
    top = max(true_age.max(), age_pred.max()) + 3
    ax.plot([0, top], [0, top], color=GREY, linewidth=1, linestyle="--")
    ax.scatter(true_age, age_pred, s=6, color=BLUE, alpha=0.35, linewidths=0)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("true age (years)", color=INK_MUTED)
    ax.set_ylabel("predicted age (years)", color=INK_MUTED)
    ax.set_title(f"Validation: {len(true_age):,} faces, MAE {mae:.2f} y", color=INK, fontsize=10, loc="left")
    return _save(fig, path)


def plot_gender_confusion(y_true, y_pred, path: str | Path, labels=("male", "female")) -> Path:
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (np.asarray(y_true, int), np.asarray(y_pred, int)), 1)
    fig = Figure(figsize=(3.9, 3.5), dpi=150, facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    _style(ax, None)
    ax.imshow(cm, cmap=BLUES, vmin=0, vmax=max(cm.max(), 1))
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=12,
                    color="#ffffff" if cm[i, j] > cm.max() * 0.55 else INK, fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(n), labels)
    ax.set_yticks(range(n), labels)
    ax.set_xlabel("Predicted", color=INK_MUTED)
    ax.set_ylabel("True", color=INK_MUTED)
    acc = np.trace(cm) / max(cm.sum(), 1)
    ax.set_title(f"Gender: {np.trace(cm):,}/{cm.sum():,} correct ({acc:.1%})", color=INK, fontsize=10, loc="left")
    return _save(fig, path)


def plot_mae_by_group(groups: dict, path: str | Path) -> Path:
    names = [k for k, v in groups.items() if v["mae"] is not None]
    vals = [groups[k]["mae"] for k in names]
    counts = [groups[k]["n"] for k in names]
    fig = Figure(figsize=(7.0, 3.0), dpi=150, facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    _style(ax, "y")
    bars = ax.bar(names, vals, color=BLUE, width=0.7)
    for bar, v, c in zip(bars, vals, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f}\nn={c}", ha="center", va="bottom",
                fontsize=7.5, color=INK_MUTED)
    ax.set_ylim(0, max(vals) * 1.3 if vals else 1)
    ax.set_xlabel("true age group", color=INK_MUTED)
    ax.set_ylabel("MAE (years)", color=INK_MUTED)
    ax.set_title("Age error by age group (validation)", color=INK, fontsize=10, loc="left")
    return _save(fig, path)


def plot_comparison(comparison: dict, path: str | Path) -> Path:
    """Age MAE (lower is better) and gender accuracy of every evaluated variant, side by side."""
    names = list(comparison)
    short = [n.split(" (")[0] for n in names]
    fig = Figure(figsize=(8.4, 0.55 * len(names) + 1.4), dpi=150, facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, 2, sharey=True)
    for ax, key, title, fmt in ((axes[0], "age_mae", "Age MAE, years (lower is better)", "{:.2f}"),
                                (axes[1], "gender_accuracy", "Gender accuracy (higher is better)", "{:.1%}")):
        _style(ax, "x")
        vals = [comparison[n].get(key) for n in names]
        y = np.arange(len(names))
        colors = [BLUE if i == 0 else GREY for i in range(len(names))]
        ax.barh(y, [v or 0 for v in vals], color=colors, height=0.6)
        for yi, v in zip(y, vals):
            if v is not None:
                ax.text(v, yi, " " + fmt.format(v), va="center", fontsize=8.5, color=INK)
        ax.set_yticks(y, short)
        ax.invert_yaxis()
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xlim(0, max(v or 0 for v in vals) * 1.25 or 1)
    return _save(fig, path)


def plot_history(history: dict, path: str | Path) -> Path:
    """Training loss and validation age MAE / gender accuracy per epoch (small multiples)."""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig = Figure(figsize=(9.6, 3.0), dpi=150, facecolor=SURFACE, layout="constrained")
    specs = (("train_loss", "Training loss (age CE + gender CE)", ORANGE),
             ("val_age_mae", "Validation age MAE (years)", BLUE),
             ("val_gender_accuracy", "Validation gender accuracy", BLUE))
    for ax, (key, title, color) in zip(fig.subplots(1, 3), specs):
        _style(ax)
        ax.plot(epochs, history[key], color=color, linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1)
        best = history.get("best_epoch")
        if best:
            ax.axvline(best, color=GREY, linewidth=1, linestyle="--")
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("epoch", color=INK_MUTED)
        ax.set_xticks(epochs)
    return _save(fig, path)
