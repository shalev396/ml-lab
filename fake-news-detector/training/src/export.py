"""Write the model repo (cfg.model_dir): model.keras, vocab/stopwords/config JSON, metrics.json, the
card plots and the refreshed model card. Smoke runs write to training/outputs/smoke/model/ and never
touch model/."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from sklearn.metrics import roc_curve

import model as M  # ../model/model.py

from . import utils
from .config import Config
from .data_setup import Splits
from .engine import Evaluation, confusion
from .model_builder import describe

DATASET = "ISOT Fake and Real News (Kaggle clmentbisaillon/fake-and-real-news-dataset)"
SELECTION_RULE = "highest validation accuracy (ties: validation ROC-AUC); threshold 0.5"
VERSIONS = ("tensorflow", "keras", "numpy", "pandas", "sklearn", "nltk")
REPO_FILES = ("README.md", "model.py", "handler.py", "requirements.txt")  # copied into smoke exports


def export(cfg: Config, *, best: str, model, vocab: list[str], stop_words: set[str],
           evals: dict[str, Evaluation], splits: Splits, source: str, device: str, train_time_s: float,
           provenance: str | None = None, trained_at: str | None = None) -> dict:
    """Save the best Keras model + everything model.py needs, then metrics, plots and card.

    `provenance` / `trained_at` override the default "retrained today" note (used when converting
    weights from an earlier run). Returns the metrics dict written to metrics.json.
    """
    save_model(cfg, best=best, model=model, vocab=vocab, stop_words=stop_words, source=source,
               evals=evals, splits=splits)
    return write_report(cfg, best=best, evals=evals, splits=splits, source=source, device=device,
                        train_time_s=train_time_s, provenance=provenance, trained_at=trained_at)


def save_model(cfg: Config, *, best: str, model, vocab: list[str], stop_words: set[str], source: str,
               evals: dict[str, Evaluation], splits: Splits) -> Path:
    """Step 1: model.keras + vocab.json + stopwords.json + config.json into cfg.model_dir, then reload
    them through model.py and check they reproduce the in-memory predictions."""
    model_dir = _prepare(cfg.model_dir)
    model.save(model_dir / "model.keras")
    utils.save_json(list(vocab), model_dir / "vocab.json")
    utils.save_json(sorted(stop_words), model_dir / "stopwords.json")
    utils.save_json({
        "variant": best,
        "architecture": describe(best, cfg),
        "labels": {"0": M.LABELS[0], "1": M.LABELS[1]},
        "output": "sigmoid = P(fake); predict() returns {'fake': p, 'real': 1 - p}",
        "threshold": 0.5,
        "max_tokens": cfg.max_tokens,
        "sequence_length": cfg.sequence_length,
        "vocab_size": len(vocab),
        "model": {"variant": best, **cfg.model_kwargs()},
        "cleaning": {**M.CLEANING, "stemming": cfg.use_stemming},
        "dataset_source": source,
        "versions": utils.lib_versions(*VERSIONS),
    }, model_dir / "config.json")
    _check_roundtrip(model_dir, evals[best], splits)
    return model_dir


def write_report(cfg: Config, *, best: str, evals: dict[str, Evaluation], splits: Splits, source: str,
                 device: str, train_time_s: float, provenance: str | None = None,
                 trained_at: str | None = None) -> dict:
    """Step 2: metrics.json, the card plots in assets/ and the refreshed model card."""
    model_dir = _prepare(cfg.model_dir)
    chosen = evals[best]
    cm = confusion(splits.y_test, chosen.test_prob)
    ranked = sorted(evals.values(), key=lambda e: (e.val["accuracy"], e.val["roc_auc"]), reverse=True)
    metrics = {
        "model": best,
        "task": "binary-classification",
        "dataset": f"{DATASET}; source used: {source}",
        "split": "test",
        "primary_metric": {"name": "accuracy", "value": round(chosen.test["accuracy"], 6)},
        "metrics": {k: round(v, 6) for k, v in chosen.test.items()},
        "threshold": 0.5,
        "confusion_matrix": cm,
        "selection": {"rule": SELECTION_RULE, "val": {k: round(v, 6) for k, v in chosen.val.items()}},
        "comparison": {e.name: {"params": e.params, "epochs": e.epochs_ran,
                                "val_accuracy": round(e.val["accuracy"], 6),
                                **{k: round(v, 6) for k, v in e.test.items()}} for e in ranked},
        "data": splits.counts(),
        "params": chosen.params,
        "train_time_s": round(train_time_s, 1),
        "device": device,
        "smoke": cfg.smoke,
        "source": provenance or f"retrained {utils.today()} with training/ ({device})",
        "versions": utils.lib_versions(*VERSIONS),
        "trained_at": trained_at or utils.today(),
    }
    utils.save_json(metrics, model_dir / "metrics.json")

    assets = model_dir / "assets"
    plot_training_curves(evals, assets / "training_curves.png")
    plot_comparison(ranked, best, assets / "model_comparison.png")
    plot_confusion(cm, best, assets / "confusion_matrix.png")
    plot_roc(evals, best, splits.y_test, assets / "roc_curve.png")
    utils.update_model_card(model_dir, metrics, ml_lab={"architecture": f"{M.VARIANTS[best]}: {describe(best, cfg)}"})
    _write_comparison_table(model_dir / "README.md", metrics["comparison"], best)
    print(f"[export] {best} -> {model_dir}")
    return metrics


def _prepare(model_dir: Path) -> Path:
    """Create the export dir; outside model/ (smoke) seed it with the card + code so it is a
    complete, loadable model repo (the Space can run on it via MODEL_DIR=...)."""
    model_dir.mkdir(parents=True, exist_ok=True)
    if model_dir.resolve() != utils.MODEL_DIR.resolve():
        for name in REPO_FILES:
            shutil.copyfile(utils.MODEL_DIR / name, model_dir / name)
    return model_dir


def _check_roundtrip(model_dir: Path, chosen: Evaluation, splits: Splits, n: int = 256) -> None:
    """Reload the export through model.py (what the Space runs) and compare with the in-memory model."""
    served = M.load(model_dir, "cpu").probabilities(splits.x_test[:n])
    if not np.allclose(served, chosen.test_prob[:n], atol=1e-5):
        raise RuntimeError("exported model does not reproduce the training predictions")


_TABLE_START, _TABLE_END = "<!-- comparison:start -->", "<!-- comparison:end -->"


def _write_comparison_table(card: Path, comparison: dict, best: str) -> None:
    """Fill the card's <!-- comparison:start/end --> block from metrics.json (ranked by val accuracy)."""
    rows = ["| variant | params | epochs | val accuracy | test accuracy | test F1 | test precision | test recall "
            "| test ROC-AUC |", "|---|---|---|---|---|---|---|---|---|"]
    for name, m in comparison.items():
        cells = [f"{m['params']:,}", str(m["epochs"]), *(f"{m[k]:.4f}" for k in
                 ("val_accuracy", "accuracy", "f1", "precision", "recall", "roc_auc"))]
        label = M.VARIANTS[name]
        if name == best:
            label, cells = f"**{label}** (deployed)", [f"**{c}**" for c in cells]
        rows.append("| " + " | ".join([label, *cells]) + " |")
    table = "\n".join([_TABLE_START, *rows, _TABLE_END])
    text = card.read_text(encoding="utf-8")
    if _TABLE_START in text and _TABLE_END in text:
        text = re.sub(re.escape(_TABLE_START) + r".*?" + re.escape(_TABLE_END), lambda _: table, text, flags=re.S)
        card.write_text(text, encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------- plots
# Light chart surface, ink and the first four categorical slots of the validated default palette.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
BLUES = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")


def _color(name: str) -> str:
    """Color follows the variant (fixed order), never its rank."""
    return SERIES[list(M.VARIANTS).index(name) % len(SERIES)]


def _axes(width: float = 6.4, height: float = 4.2, ncols: int = 1):
    fig = Figure(figsize=(width, height), dpi=150, facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, ncols, squeeze=False)[0]
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side, spine in ax.spines.items():
            spine.set_visible(side in ("left", "bottom"))
            spine.set_color(AXIS)
        ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8)
        ax.xaxis.label.set_color(INK_2)
        ax.yaxis.label.set_color(INK_2)
    return fig, axes


def _save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)


def _legend(ax, **kwargs) -> None:
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, **kwargs)


def plot_training_curves(evals: dict[str, Evaluation], path: Path) -> None:
    """Validation loss and accuracy per epoch for every variant (dashed = train)."""
    fig, axes = _axes(10, 3.9, ncols=2)
    for ax, key, title in zip(axes, ("loss", "accuracy"), ("binary cross-entropy", "accuracy")):
        for e in evals.values():
            h = e.history
            if key not in h:
                continue
            epochs = np.arange(1, len(h[key]) + 1)
            ax.plot(epochs, h[key], color=_color(e.name), linewidth=1.2, linestyle="--", alpha=0.6)
            ax.plot(epochs, h[f"val_{key}"], color=_color(e.name), linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"{M.VARIANTS[e.name]} (val)")
        ax.set(xlabel="epoch")
        ax.set_xticks(range(1, max((len(e.history.get("loss", [])) for e in evals.values()), default=1) + 1))
        ax.set_title(f"{title} per epoch", fontsize=10, loc="left", color=INK)
    axes[0].plot([], [], color=MUTED, linestyle="--", linewidth=1.2, label="train (dashed)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncols=5, frameon=False,
               fontsize=8, labelcolor=INK_2)
    _save(fig, path)


def plot_comparison(ranked: list[Evaluation], best: str, path: Path) -> None:
    """Test accuracy / F1 / ROC-AUC of every variant, best on top."""
    keys, labels = ("accuracy", "f1", "roc_auc"), ("test accuracy", "test F1", "test ROC-AUC")
    fig, (ax,) = _axes(6.8, 0.75 * len(ranked) + 1.3)
    rows = np.arange(len(ranked))[::-1]
    height = 0.26
    lo = min(e.test[k] for e in ranked for k in keys)
    xmin = max(0.0, np.floor((lo - 0.02) * 50) / 50)
    for i, (key, label) in enumerate(zip(keys, labels)):
        offs = (1 - i) * height
        vals = [e.test[key] for e in ranked]
        ax.barh(rows + offs, [v - xmin for v in vals], left=xmin, height=height - 0.03, color=SERIES[i],
                edgecolor=SURFACE, linewidth=1, label=label)
        for row, v in zip(rows, vals):
            ax.text(v + 0.0015, row + offs, f"{v:.3f}", va="center", fontsize=7, color=INK_2)
    ax.set_yticks(rows, [M.VARIANTS[e.name] + (" (deployed)" if e.name == best else "") for e in ranked])
    for tick, e in zip(ax.get_yticklabels(), ranked):
        tick.set_fontweight("bold" if e.name == best else "normal")
    ax.grid(axis="y", visible=False)
    ax.set(xlim=(xmin, 1.012), xlabel="score on the test split (x-axis starts at %.2f)" % xmin)
    ax.set_title("All variants, ranked by validation accuracy", fontsize=10, loc="left", color=INK)
    _legend(ax, loc="lower left", bbox_to_anchor=(0, 1.06), ncols=3)
    _save(fig, path)


def plot_confusion(cm: dict[str, int], best: str, path: Path) -> None:
    grid = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, (ax,) = _axes(4.4, 3.8)
    ax.grid(False)
    ax.imshow(grid, cmap=LinearSegmentedColormap.from_list("blues", BLUES), vmin=0, vmax=max(grid.max(), 1))
    for (i, j), count in np.ndenumerate(grid):
        share = count / max(grid[i].sum(), 1)
        ax.text(j, i, f"{count:,}\n{share:.1%} of row", ha="center", va="center", fontsize=9,
                color="#ffffff" if count > grid.max() / 2 else INK)
    ax.set_xticks([0, 1], list(M.LABELS))
    ax.set_yticks([0, 1], list(M.LABELS))
    ax.set(xlabel="Predicted", ylabel="Actual")
    ax.set_title(f"{M.VARIANTS[best]} @ 0.5 (test split)", fontsize=10, loc="left", color=INK)
    _save(fig, path)


def plot_roc(evals: dict[str, Evaluation], best: str, y_test, path: Path) -> None:
    """Test ROC curves of every variant (best drawn thickest)."""
    fig, (ax,) = _axes(5.4, 4.4)
    for e in evals.values():
        fpr, tpr, _ = roc_curve(np.asarray(y_test).astype(int), e.test_prob)
        ax.plot(fpr, tpr, color=_color(e.name), linewidth=2.4 if e.name == best else 1.4,
                label=f"{M.VARIANTS[e.name]}  (AUC {e.test['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle="--")
    zoom = min(e.test["roc_auc"] for e in evals.values()) >= 0.97   # all near-perfect: zoom into the corner
    ax.set(xlim=(-0.002, 0.15) if zoom else (-0.01, 1), ylim=(0.5, 1.005) if zoom else (0, 1.01),
           xlabel="False positive rate (real flagged as fake)", ylabel="True positive rate (fake caught)")
    ax.set_title("ROC on the test split" + (" (zoomed: FPR <= 0.15, TPR >= 0.5)" if zoom else ""),
                 fontsize=10, loc="left", color=INK)
    _legend(ax, loc="lower right")
    _save(fig, path)
