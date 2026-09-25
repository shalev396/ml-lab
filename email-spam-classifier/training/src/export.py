"""Export the spam classifier as a Hugging Face model repo folder.

Copies the best checkpoint (`model.safetensors` + `config.json` from PyTorchModelHubMixin, and
the tokenizer files), writes `metrics.json` (STANDARD §4) and the card plots in `assets/`, then
refreshes the metrics in `README.md`. Exporting anywhere other than model/ (e.g. smoke runs)
first copies the card, `model.py`, `handler.py` and `requirements.txt` there, so the folder is a
self-contained snapshot and the real model/ is never touched.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from sklearn.metrics import precision_recall_curve, roc_curve

from . import utils

MODEL_NAME = "DistilBERT (uncased) fine-tuned, last 2 blocks + linear head"
TASK = "binary-text-classification"
DATASET = "Enron-Spam (SetFit/enron_spam), cleaned, stratified 70/15/15 split"
COPIED_FROM_MODEL_DIR = ("README.md", "model.py", "handler.py", "requirements.txt")
CHECKPOINT_FILES = ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json")

# Chart colours: validated categorical slots 1-2 (dataviz reference palette) + a one-hue ramp.
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834"]
BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f8fe", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])


def build_metrics(scores: dict[str, float], *, data: dict, params: int, train_time_s: float | None,
                  device: str, smoke: bool, source: str, trained_at: str | None = None,
                  comparison: dict | None = None, history: dict | None = None,
                  confusion: np.ndarray | None = None, threshold: float = 0.5) -> dict:
    """metrics.json content (schema: STANDARD §4). `scores` holds the deployed model's test metrics."""
    metrics = {
        "model": MODEL_NAME,
        "task": TASK,
        "dataset": DATASET,
        "split": "test",
        "primary_metric": {"name": "f1", "value": scores["f1"]},
        "metrics": scores,
        "threshold": threshold,
        "data": data,
        "params": int(params),
        "train_time_s": None if train_time_s is None else round(float(train_time_s), 1),
        "device": device,
        "smoke": smoke,
        "source": source,
        "versions": utils.lib_versions("torch", "transformers", "tokenizers", "huggingface_hub",
                                       "safetensors", "sklearn", "numpy"),
        "trained_at": trained_at or utils.today(),
    }
    if confusion is not None:
        cm = np.asarray(confusion)
        metrics["confusion_matrix"] = {"labels": ["ham", "spam"], "rows": "true", "cols": "predicted",
                                       "counts": cm.tolist()}
    if comparison:
        metrics["comparison"] = comparison
    if history:
        metrics["history"] = history
    return metrics


def export(checkpoint_dir: str | Path, model_dir: str | Path, metrics: dict, *, y_true, y_prob,
           curves: dict[str, tuple] | None = None) -> dict:
    """Copy the checkpoint to `model_dir`, write metrics + plots, update its model card.

    `curves`: {variant: (y_true, y_prob)} for the ROC / PR plot (defaults to the deployed model).
    """
    checkpoint_dir, model_dir = Path(checkpoint_dir), Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if model_dir.resolve() != utils.MODEL_DIR.resolve():
        for name in COPIED_FROM_MODEL_DIR:
            shutil.copy2(utils.MODEL_DIR / name, model_dir / name)
    if checkpoint_dir.resolve() != model_dir.resolve():
        for name in CHECKPOINT_FILES:
            shutil.copy2(checkpoint_dir / name, model_dir / name)
    write_report(model_dir, metrics, y_true=y_true, y_prob=y_prob, curves=curves)
    print(f"exported -> {model_dir}")
    return metrics


def write_report(model_dir: str | Path, metrics: dict, *, y_true, y_prob, curves: dict | None = None) -> None:
    """metrics.json + assets/*.png + model-card refresh (weights are not touched)."""
    model_dir = Path(model_dir)
    assets = model_dir / "assets"
    threshold = metrics.get("threshold", 0.5)
    plot_confusion_matrix(y_true, y_prob, assets / "confusion_matrix.png", threshold)
    plot_roc_pr(curves or {metrics["model"]: (y_true, y_prob)}, assets / "roc_pr_curves.png")
    if metrics.get("comparison"):
        plot_comparison(metrics["comparison"], assets / "comparison.png")
    if metrics.get("history"):
        plot_history(metrics["history"], assets / "training_curves.png")
    utils.save_json(metrics, model_dir / "metrics.json")
    utils.update_model_card(model_dir, metrics)


# --------------------------------------------------------------------------- plots
def _style(ax, grid: bool = False) -> None:
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    if grid:
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)


def plot_confusion_matrix(y_true, y_prob, path: str | Path, threshold: float = 0.5) -> Path:
    """2x2 counts (rows = true, columns = predicted); the diagonal holds the correct predictions."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    cm = np.zeros((2, 2), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    fig = Figure(figsize=(4.4, 3.9), dpi=150, facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    _style(ax)
    ax.imshow(cm, cmap=BLUES, vmin=0, vmax=max(cm.max(), 1))
    for i in range(2):
        for j in range(2):
            dark = cm[i, j] > cm.max() * 0.55
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=13,
                    color="#ffffff" if dark else INK, fontweight="bold" if i == j else "normal")
    ax.set_xticks([0, 1], ["ham", "spam"])
    ax.set_yticks([0, 1], ["ham", "spam"])
    ax.set_xlabel("Predicted", color=INK_MUTED)
    ax.set_ylabel("True", color=INK_MUTED)
    errors = cm.sum() - np.trace(cm)
    ax.set_title(f"Test split: {errors} errors in {cm.sum():,} emails (threshold {threshold})",
                 color=INK, fontsize=9.5, loc="left")
    return _save(fig, path)


def plot_roc_pr(curves: dict[str, tuple], path: str | Path) -> Path:
    """ROC and precision-recall curves of every variant as two small multiples."""
    fig = Figure(figsize=(8.4, 3.6), dpi=150, facecolor=SURFACE, layout="constrained")
    ax_roc, ax_pr = fig.subplots(1, 2)
    for ax in (ax_roc, ax_pr):
        _style(ax, grid=True)
    for (name, (y_true, y_prob)), color in zip(curves.items(), SERIES):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ax_roc.plot(fpr, tpr, color=color, linewidth=2, label=name)
        ax_pr.plot(rec, prec, color=color, linewidth=2, label=name)
    ax_roc.set_title("ROC (spam = positive)", color=INK, fontsize=10, loc="left")
    ax_roc.set_xlabel("false positive rate (ham flagged as spam)", color=INK_MUTED, fontsize=9)
    ax_roc.set_ylabel("true positive rate", color=INK_MUTED, fontsize=9)
    ax_roc.set_xscale("symlog", linthresh=0.01)
    ax_roc.set_xlim(0, 1)
    ax_pr.set_title("Precision-recall (zoomed to the top-right corner)", color=INK, fontsize=10, loc="left")
    ax_pr.set_xlabel("recall", color=INK_MUTED, fontsize=9)
    ax_pr.set_ylabel("precision", color=INK_MUTED, fontsize=9)
    ax_pr.set_xlim(0.9, 1.001)
    ax_pr.set_ylim(0.9, 1.001)
    ax_pr.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left")
    return _save(fig, path)


def plot_comparison(comparison: dict[str, dict[str, float]], path: str | Path,
                    keys: tuple[str, ...] = ("accuracy", "precision", "recall", "f1")) -> Path:
    """Test error rate (1 - metric) per variant: a dot plot, so small gaps near 1.0 stay visible."""
    names = list(comparison)
    fig = Figure(figsize=(7.2, 0.9 + 0.55 * len(keys)), dpi=150, facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    _style(ax, grid=True)
    for k, (name, color) in enumerate(zip(names, SERIES)):
        errs = [100 * (1 - comparison[name][key]) for key in keys]
        ys = np.arange(len(keys)) + (k - (len(names) - 1) / 2) * 0.18
        ax.scatter(errs, ys, s=64, color=color, edgecolors=SURFACE, linewidths=2, zorder=3, label=name)
        for x, y in zip(errs, ys):
            ax.annotate(f"{x:.2f}%", (x, y), xytext=(7, 0), textcoords="offset points", va="center",
                        fontsize=8, color=INK_MUTED)
    ax.set_yticks(range(len(keys)), [f"1 - {key}" for key in keys])
    ax.invert_yaxis()
    ax.set_xlabel("error on the test split (%) - lower is better", color=INK_MUTED, fontsize=9)
    ax.set_xlim(0, max(100 * (1 - comparison[n][k]) for n in names for k in keys) * 1.25)
    ax.set_title("Experiments", color=INK, fontsize=10, loc="left", pad=22)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=len(names), borderaxespad=0.2, handletextpad=0.3)
    return _save(fig, path)


def plot_history(history: dict, path: str | Path) -> Path:
    """Loss (train vs val) and validation F1 per epoch, as two small multiples (never a dual axis)."""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig = Figure(figsize=(8.0, 3.2), dpi=150, facecolor=SURFACE, layout="constrained")
    ax_loss, ax_f1 = fig.subplots(1, 2)
    for ax in (ax_loss, ax_f1):
        _style(ax, grid=True)
        ax.set_xticks(epochs)
        ax.set_xlabel("epoch", color=INK_MUTED)
    marker = dict(linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1)
    ax_loss.plot(epochs, history["train_loss"], color=SERIES[0], label="train", **marker)
    ax_loss.plot(epochs, history["val_loss"], color=SERIES[1], label="val", **marker)
    ax_loss.set_title("BCE loss (pos_weight)", color=INK, fontsize=10, loc="left")
    ax_loss.legend(frameon=False, fontsize=9, labelcolor=INK)
    ax_f1.plot(epochs, history["val_f1"], color=SERIES[1], **marker)
    ax_f1.set_title("Validation F1 (spam)", color=INK, fontsize=10, loc="left")
    best = history.get("best_epoch")
    if best:
        for ax in (ax_loss, ax_f1):
            ax.axvline(best, color=INK_MUTED, linewidth=1, linestyle=":")
        ax_f1.annotate(f"best epoch (val loss): {best}", (best, ax_f1.get_ylim()[0]), xytext=(-6, 6),
                       textcoords="offset points", ha="right", fontsize=8, color=INK_MUTED)
    return _save(fig, path)


def _save(fig: Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path
