"""Compare the evaluated variants, pick the best, and export it as a Hugging Face model repo folder.

A *variant* is one set of weights evaluated on the test split in this run: the checkpoint that is
currently deployed in model/ and the model trained by the notebook. `select_best` keeps the
deployed one unless the new run is strictly better, so a retrain never silently makes things worse.

`export()` writes `model.safetensors` + `config.json` (PyTorchModelHubMixin) of the best variant,
`metrics.json` (STANDARD §4, with a `comparison` of every variant), the experiment plots in
`assets/`, and refreshes the model card (metrics table + Experiments table). Exporting anywhere
other than model/ (smoke runs) first copies the card, `model.py`, `handler.py` and
`requirements.txt` there, so the folder is a self-contained snapshot and model/ is never touched.
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

import model as M  # ../model/model.py

from . import utils

MODEL_NAME = "EfficientNet-B2, fully fine-tuned from ImageNet weights"
TASK = "multiclass-image-classification"
DATASET = "Food-101 (ethz/food101): 101 dishes, 75,750 train / 25,250 test images"
DEPLOYED_DEFAULT_NAME = "original training run"   # name of the checkpoint before `variant` was recorded
COPIED_FROM_MODEL_DIR = ("README.md", "model.py", "handler.py", "requirements.txt")
WEIGHT_FILES = (M.WEIGHTS_FILE, "config.json")
ASSETS = ("training_curves.png", "comparison.png", "confusion_matrix.png", "per_class_accuracy.png")
_EXP_START, _EXP_END = "<!-- experiments:start -->", "<!-- experiments:end -->"

# Chart colours (validated categorical slots 1-2 + a one-hue sequential ramp, see dataviz palette).
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = {"train": "#2a78d6", "test": "#eb6834"}
BEST, OTHER = "#2a78d6", "#b9b8b3"
BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f8fe", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])


@dataclass
class Variant:
    """One evaluated set of weights. Exactly one of `net` / `weights_dir` is set."""
    name: str
    scores: dict[str, float]
    y_true: np.ndarray
    y_pred: np.ndarray
    source: str
    trained_at: str | None
    device: str | None
    train_time_s: float | None
    net: object | None = None          # trained in this run -> saved with save_pretrained
    weights_dir: Path | None = None    # an existing checkpoint -> its files are copied


def new_run_variant(net, result: dict, device, train_time_s: float) -> Variant:
    from . import engine

    dev = str(getattr(device, "type", device))
    return Variant(
        name=f"retrain with training/ on {dev}, {utils.today()}",
        scores=engine.classification_metrics(result),
        y_true=result["y_true"], y_pred=result["y_pred"],
        source=f"retrained {utils.today()} with training/ ({dev})",
        trained_at=utils.today(), device=dev, train_time_s=train_time_s, net=net,
    )


def deployed_variant(folder: Path, info: dict, result: dict) -> Variant:
    """The checkpoint in model/; its provenance comes from the metrics.json saved next to it."""
    from . import engine

    return Variant(
        name=info.get("variant") or DEPLOYED_DEFAULT_NAME,
        scores=engine.classification_metrics(result),
        y_true=result["y_true"], y_pred=result["y_pred"],
        source=info.get("source", "deployed checkpoint (provenance unknown)"),
        trained_at=info.get("trained_at"), device=info.get("device"),
        train_time_s=info.get("train_time_s"), weights_dir=Path(folder),
    )


def select_best(variants: list[Variant]) -> Variant:
    """Highest (accuracy, top-5 accuracy). Ties keep the earlier variant, so list the deployed one first."""
    def key(v: Variant):
        return v.scores["accuracy"], v.scores.get("top5_accuracy", 0.0)

    best = variants[0]
    for v in variants[1:]:
        if key(v) > key(best):
            best = v
    return best


def comparison(variants: list[Variant]) -> dict[str, dict[str, float]]:
    return {v.name: dict(v.scores) for v in variants}


def save_weights(variant: Variant, target: str | Path) -> Path:
    """Put the variant's `model.safetensors` + `config.json` into `target` (no-op if already there)."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    if variant.net is not None:
        variant.net.save_pretrained(target)
    elif variant.weights_dir.resolve() != target.resolve():
        for name in WEIGHT_FILES:
            shutil.copy2(variant.weights_dir / name, target / name)
    return target


def build_metrics(best: Variant, variants: list[Variant], *, data: dict, params: int, smoke: bool,
                  **extra) -> dict:
    """metrics.json content (schema: STANDARD §4) for the deployed (= best) variant.

    `extra` keys (e.g. `evaluation`, `history`) are added as-is."""
    metrics = {
        "model": MODEL_NAME,
        "variant": best.name,
        "task": TASK,
        "dataset": DATASET,
        "split": "test",
        "primary_metric": {"name": "accuracy", "value": best.scores["accuracy"]},
        "metrics": dict(best.scores),
        "comparison": comparison(variants),
        "data": data,
        "params": int(params),
        "train_time_s": None if best.train_time_s is None else round(best.train_time_s, 1),
        "device": best.device,
        "smoke": smoke,
        "source": best.source,
        "versions": utils.lib_versions("torch", "torchvision", "huggingface_hub", "safetensors", "numpy"),
        "trained_at": best.trained_at or utils.today(),
    }
    metrics.update(extra)
    return metrics


def export(best: Variant, metrics: dict, model_dir: str | Path, assets_dir: str | Path) -> Path:
    """Write the best weights, metrics.json and the plots from `assets_dir` to `model_dir`, then
    refresh its model card (YAML + metrics table + Experiments table)."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if model_dir.resolve() != utils.MODEL_DIR.resolve():
        for name in COPIED_FROM_MODEL_DIR:
            shutil.copy2(utils.MODEL_DIR / name, model_dir / name)

    save_weights(best, model_dir)
    (model_dir / "assets").mkdir(exist_ok=True)
    for name in ASSETS:
        src, dst = Path(assets_dir) / name, model_dir / "assets" / name
        if src.is_file():
            shutil.copy2(src, dst)
        elif dst.is_file():              # a plot this run did not produce would be stale
            dst.unlink()
    utils.save_json(metrics, model_dir / "metrics.json")
    utils.update_model_card(model_dir, metrics)
    write_experiments_table(model_dir, metrics)
    print(f"exported '{best.name}' -> {model_dir}")
    return model_dir


def experiments_markdown(metrics: dict) -> str:
    """Markdown table of every variant in metrics['comparison']; the deployed one in bold."""
    rows = metrics.get("comparison") or {metrics.get("variant", "deployed"): metrics["metrics"]}
    keys = list(dict.fromkeys(k for scores in rows.values() for k in scores))
    head = "| variant | " + " | ".join(keys) + " |\n|---|" + "---|" * len(keys)
    lines = []
    for name, scores in rows.items():
        cells = [f"{scores[k]:.4f}" if isinstance(scores.get(k), (int, float)) else "n/a" for k in keys]
        if name == metrics.get("variant"):
            lines.append(f"| **{name} (deployed)** | " + " | ".join(f"**{c}**" for c in cells) + " |")
        else:
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return f"{_EXP_START}\n{head}\n" + "\n".join(lines) + f"\n{_EXP_END}"


def write_experiments_table(model_dir: str | Path, metrics: dict) -> None:
    path = Path(model_dir) / "README.md"
    text = path.read_text(encoding="utf-8")
    if _EXP_START in text and _EXP_END in text:
        table = experiments_markdown(metrics)
        text = re.sub(re.escape(_EXP_START) + r".*?" + re.escape(_EXP_END), lambda _: table, text, flags=re.S)
        path.write_text(text, encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------- plots
def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def top_confusions(y_true, y_pred, class_names: list[str], k: int = 5) -> list[tuple[str, str, int]]:
    """The k most frequent (true, predicted) mistakes."""
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    np.fill_diagonal(cm, 0)
    flat = np.argsort(cm, axis=None)[::-1][:k]
    return [(class_names[i], class_names[j], int(cm[i, j])) for i, j in zip(*np.unravel_index(flat, cm.shape))
            if cm[i, j] > 0]


def plot_confusion_matrix(y_true, y_pred, class_names: list[str], path: str | Path, title: str = "") -> Path:
    """Row-normalised confusion matrix (share of each true class). With 101 classes the cells carry
    no numbers; the most frequent mistakes are listed under the chart."""
    n = len(class_names)
    cm = np.zeros((n, n), dtype=float)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    correct, total = int(np.trace(cm)), int(cm.sum())
    rows = cm.sum(axis=1, keepdims=True)
    share = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
    fig = Figure(figsize=(7.2, 7.8), dpi=150, facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    _style(ax)
    ax.imshow(share, cmap=BLUES, vmin=0, vmax=1, interpolation="nearest")
    ticks = list(range(0, n, 10))
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_yticks(ticks, [f"{t} {class_names[t]}" for t in ticks], fontsize=7)
    ax.set_xlabel("Predicted class (index, alphabetical)", color=INK_MUTED)
    ax.set_ylabel("True class", color=INK_MUTED)
    head = f"{title}\n" if title else ""
    ax.set_title(f"{head}{correct:,}/{total:,} correct ({correct / max(total, 1):.1%}), row-normalised",
                 color=INK, fontsize=10, loc="left")
    mistakes = top_confusions(y_true, y_pred, class_names)
    if mistakes:
        text = "Most frequent mistakes (true -> predicted): " + "; ".join(f"{a} -> {b} ({c})" for a, b, c in mistakes)
        fig.text(0.01, 0.005, text, fontsize=7, color=INK_MUTED, wrap=True, va="bottom")
    return _save(fig, path)


def plot_per_class(y_true, y_pred, class_names: list[str], path: str | Path, k: int = 12, title: str = "",
                   min_support: int = 5) -> Path:
    """The k hardest and the k easiest dishes by per-class accuracy, among classes with at least
    `min_support` test images (all of them if none has that many, e.g. in a smoke run)."""
    from . import engine

    acc = engine.per_class_accuracy(np.asarray(y_true), np.asarray(y_pred), len(class_names))
    support = np.bincount(np.asarray(y_true), minlength=len(class_names))
    present = [i for i in np.argsort(acc) if not np.isnan(acc[i]) and support[i] >= min_support]
    if not present:
        present = [i for i in np.argsort(acc) if not np.isnan(acc[i])]
    k = min(k, len(present) // 2) or len(present)
    picks = present[:k] + (present[-k:] if len(present) > k else [])
    fig = Figure(figsize=(7.6, 0.28 * len(picks) + 1.4), dpi=150, facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    _style(ax)
    ys = list(range(len(picks)))
    colors = [OTHER if n < k else BEST for n in range(len(picks))]
    ax.barh(ys, [acc[i] for i in picks], color=colors, height=0.75)
    for y, i in zip(ys, picks):
        ax.text(acc[i] + 0.01, y, f"{acc[i]:.0%} (n={support[i]})", va="center", fontsize=7, color=INK)
    ax.set_yticks(ys, [class_names[i] for i in picks], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.18)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    head = f"{title}\n" if title else ""
    ax.set_title(f"{head}Per-class accuracy: {k} hardest (grey) and {k} easiest (blue) dishes",
                 color=INK, fontsize=10, loc="left")
    return _save(fig, path)


def plot_history(history: dict[str, list[float]], path: str | Path, title: str = "") -> Path:
    """Loss and accuracy per epoch, train vs test, as two small multiples (never a dual axis)."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig = Figure(figsize=(8.0, 3.4), dpi=150, facecolor=SURFACE, layout="constrained")
    if title:
        fig.suptitle(title, color=INK, fontsize=10, x=0.01, ha="left")
    for ax, metric, name in zip(fig.subplots(1, 2), ("loss", "acc"), ("Cross-entropy loss (label smoothing 0.1)", "Accuracy")):
        _style(ax)
        ax.grid(True, color=GRID, linewidth=0.6)
        for split, color in SERIES.items():
            ax.plot(epochs, history[f"{split}_{metric}"], color=color, linewidth=2, marker="o",
                    markersize=5, markeredgecolor=SURFACE, markeredgewidth=1, label=split)
        ax.set_title(name, color=INK_MUTED, fontsize=9, loc="left")
        ax.set_xlabel("epoch", color=INK_MUTED)
        ax.set_xticks(list(epochs))
        ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    return _save(fig, path)


def plot_comparison(rows: dict[str, dict[str, float]], highlight: str, path: str | Path,
                    title: str = "Test split: every evaluated variant") -> Path:
    """Horizontal bars per metric, one bar per row (variant or evaluation); `highlight` is blue."""
    keys = list(dict.fromkeys(k for scores in rows.values() for k in scores))
    names = list(rows)
    fig = Figure(figsize=(8.0, 1.4 + 0.42 * len(keys) * len(names)), dpi=150, facecolor=SURFACE,
                 layout="constrained")
    ax = fig.subplots()
    _style(ax)
    height = 0.8 / len(names)
    for n, name in enumerate(names):
        ys, vals = [], []
        for i, key in enumerate(keys):
            if isinstance(rows[name].get(key), (int, float)):
                ys.append(i + (n - (len(names) - 1) / 2) * height)
                vals.append(rows[name][key])
        ax.barh(ys, vals, height=height * 0.9, color=BEST if name == highlight else OTHER, label=name)
        for y, val in zip(ys, vals):
            ax.text(val + 0.004, y, f"{val:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(keys)), keys)
    ax.invert_yaxis()
    low = min(v for scores in rows.values() for v in scores.values() if isinstance(v, (int, float)))
    ax.set_xlim(max(0.0, low - 0.1), 1.05)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left", bbox_to_anchor=(0, 1.08), ncol=1)
    return _save(fig, path)


def _save(fig: Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path
