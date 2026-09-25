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

MODEL_NAME = "EfficientNet-B2 feature extractor (frozen ImageNet backbone + linear head)"
TASK = "multiclass-image-classification"
DATASET = ("Food-101 pizza/steak/sushi 20% subset (pizza_steak_sushi_20_percent.zip from "
           "mrdbourke/pytorch-deep-learning; images from ethz/food101)")
DEPLOYED_DEFAULT_NAME = "original bootcamp checkpoint"   # name of the checkpoint before `variant` was recorded
COPIED_FROM_MODEL_DIR = ("README.md", "model.py", "handler.py", "requirements.txt")
WEIGHT_FILES = (M.WEIGHTS_FILE, "config.json")
ASSETS = ("training_curves.png", "comparison.png", "confusion_matrix.png")
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


def new_run_variant(net, result: dict, class_names: list[str], device, train_time_s: float) -> Variant:
    from . import engine

    dev = str(getattr(device, "type", device))
    return Variant(
        name=f"retrain with training/ on {dev}, {utils.today()}",
        scores=engine.classification_metrics(result["y_true"], result["y_pred"], class_names),
        y_true=result["y_true"], y_pred=result["y_pred"],
        source=f"retrained {utils.today()} with training/ ({dev})",
        trained_at=utils.today(), device=dev, train_time_s=train_time_s, net=net,
    )


def deployed_variant(folder: Path, info: dict, result: dict, class_names: list[str]) -> Variant:
    """The checkpoint in model/; its provenance comes from the metrics.json saved next to it."""
    from . import engine

    return Variant(
        name=info.get("variant") or DEPLOYED_DEFAULT_NAME,
        scores=engine.classification_metrics(result["y_true"], result["y_pred"], class_names),
        y_true=result["y_true"], y_pred=result["y_pred"],
        source=info.get("source", "deployed checkpoint (provenance unknown)"),
        trained_at=info.get("trained_at"), device=info.get("device"),
        train_time_s=info.get("train_time_s"), weights_dir=Path(folder),
    )


def select_best(variants: list[Variant]) -> Variant:
    """Highest (accuracy, macro F1). Ties keep the earlier variant, so list the deployed one first."""
    best = variants[0]
    for v in variants[1:]:
        if (v.scores["accuracy"], v.scores["f1_macro"]) > (best.scores["accuracy"], best.scores["f1_macro"]):
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


def build_metrics(best: Variant, variants: list[Variant], *, data: dict, params: int, smoke: bool) -> dict:
    """metrics.json content (schema: STANDARD §4) for the deployed (= best) variant."""
    return {
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
    keys = list(next(iter(rows.values())).keys())
    head = "| variant | " + " | ".join(keys) + " |\n|---|" + "---|" * len(keys)
    lines = []
    for name, scores in rows.items():
        cells = [f"{scores.get(k, float('nan')):.3f}" for k in keys]
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


def plot_confusion_matrix(y_true, y_pred, class_names: list[str], path: str | Path, title: str = "") -> Path:
    """Counts per (true, predicted) class; the diagonal holds the correct predictions."""
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    fig = Figure(figsize=(4.6, 4.0), dpi=150, facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    _style(ax)
    ax.imshow(cm, cmap=BLUES, vmin=0, vmax=max(cm.max(), 1))
    for i in range(n):
        for j in range(n):
            dark_cell = cm[i, j] > cm.max() * 0.55
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12,
                    color="#ffffff" if dark_cell else INK, fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(n), class_names)
    ax.set_yticks(range(n), class_names)
    ax.set_xlabel("Predicted", color=INK_MUTED)
    ax.set_ylabel("True", color=INK_MUTED)
    accuracy = np.trace(cm) / max(cm.sum(), 1)
    head = f"{title}\n" if title else ""
    ax.set_title(f"{head}Test split: {np.trace(cm)}/{cm.sum()} correct ({accuracy:.1%})",
                 color=INK, fontsize=10, loc="left")
    return _save(fig, path)


def plot_history(history: dict[str, list[float]], path: str | Path, title: str = "") -> Path:
    """Loss and accuracy per epoch, train vs test, as two small multiples (never a dual axis)."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig = Figure(figsize=(8.0, 3.4), dpi=150, facecolor=SURFACE, layout="constrained")
    if title:
        fig.suptitle(title, color=INK, fontsize=10, x=0.01, ha="left")
    for ax, metric, name in zip(fig.subplots(1, 2), ("loss", "acc"), ("Cross-entropy loss", "Accuracy")):
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


def plot_comparison(variants: list[Variant], best: Variant, path: str | Path) -> Path:
    """Horizontal bars per metric, one bar per variant; the deployed (best) one is highlighted."""
    keys = list(best.scores.keys())
    fig = Figure(figsize=(8.0, 1.0 + 0.55 * len(keys) * len(variants) / 2 + 0.6), dpi=150,
                 facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    _style(ax)
    height = 0.8 / len(variants)
    for k, v in enumerate(variants):
        ys = [i + (k - (len(variants) - 1) / 2) * height for i in range(len(keys))]
        vals = [v.scores[m] for m in keys]
        color = BEST if v is best else OTHER
        ax.barh(ys, vals, height=height * 0.9, color=color, label=v.name + (" (deployed)" if v is best else ""))
        for y, val in zip(ys, vals):
            ax.text(val + 0.004, y, f"{val:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(keys)), keys)
    ax.invert_yaxis()
    low = min(v.scores[m] for v in variants for m in keys)
    ax.set_xlim(max(0.0, low - 0.1), 1.03)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Test split: every evaluated variant", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left", bbox_to_anchor=(0, 1.06), ncol=1)
    return _save(fig, path)


def _save(fig: Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path
