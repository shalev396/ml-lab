"""Write the model repo (cfg.model_dir) in two steps, both called from the notebook:

    save_models(...)  weights + config.json (+ the card, model.py, handler.py, requirements.txt when the
                      target is not model/ itself) -> the folder model.load() and the endpoint handler read
    export(...)       metrics.json, experiment graphs in assets/, refreshed model card

Smoke runs write to training/outputs/smoke/model/ and never touch model/.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import joblib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.figure import Figure
from sklearn.metrics import average_precision_score, precision_recall_curve

import model as M  # ../model/model.py (put on sys.path by src/__init__.py)

from . import utils
from .config import Config
from .data_setup import Splits
from .engine import Evaluation, confusion
from .model_builder import MLP_NAME, count_params, describe, for_inference

DATASET = "ULB Credit Card Fraud (Kaggle mlg-ulb/creditcardfraud)"
SELECTION_RULE = "highest validation PR-AUC; decision threshold = max F1 on validation"
VERSIONS = ("sklearn", "imblearn", "tensorflow", "keras", "numpy", "pandas", "joblib")
CODE_FILES = ("README.md", "model.py", "handler.py", "requirements.txt", ".gitattributes", ".gitignore")


def save_models(cfg: Config, *, scaler, fitted: dict, exported: list[str], best: str,
                evals: dict[str, Evaluation]) -> Path:
    """Write scaler.joblib, model.joblib (best scikit-learn variant, resampler stripped), model.keras
    and config.json (schema, tuned thresholds, default model, versions) to cfg.model_dir."""
    model_dir = _prepare(cfg.model_dir)
    joblib.dump(scaler, model_dir / "scaler.joblib")
    models = {}
    for name in exported:
        estimator = fitted[name]
        if name == MLP_NAME:
            file = "model.keras"
            estimator.save(model_dir / file)
        else:
            file = "model.joblib"
            joblib.dump(for_inference(estimator), model_dir / file, compress=3)
        models[name] = {"file": file, "estimator": describe(name, estimator),
                        "threshold": evals[name].threshold,  # exact: rounding could move decisions
                        "val_pr_auc": round(evals[name].val["pr_auc"], 6),
                        "params": count_params(name, estimator)}
    utils.save_json({"input_columns": M.INPUT_COLUMNS, "feature_columns": M.FEATURE_COLUMNS,
                     "scaled_columns": M.SCALED_COLUMNS, "target": M.TARGET, "best": best, "models": models,
                     "selection": SELECTION_RULE, "versions": utils.lib_versions(*VERSIONS)},
                    model_dir / "config.json")
    print(f"[save] {', '.join(exported)} (default {best}) -> {model_dir}")
    return model_dir


def _prepare(model_dir: Path) -> Path:
    """Create the export dir; outside model/ (smoke) seed it with the card + code files so it is a
    complete, loadable model repo (the Space can run on it via MODEL_DIR=..., the handler too)."""
    model_dir.mkdir(parents=True, exist_ok=True)
    if model_dir.resolve() != utils.MODEL_DIR.resolve():
        for name in CODE_FILES:
            if (utils.MODEL_DIR / name).is_file():
                shutil.copyfile(utils.MODEL_DIR / name, model_dir / name)
    return model_dir


def check_roundtrip(model_dir: Path, evals: dict[str, Evaluation], splits: Splits, n: int = 512) -> None:
    """Reload the export through model.py (what the Space runs) and compare with the in-memory models."""
    predictor = M.load(model_dir, "cpu")
    for name in predictor.models:
        served = predictor.probabilities(splits.test.head(n), name)
        if not np.allclose(served, evals[name].test_prob[:n], atol=1e-5):
            raise RuntimeError(f"exported {name} does not reproduce the training predictions")
    print(f"[check] {', '.join(predictor.models)} reloaded through model.py reproduce the training predictions")


def figures(evals: dict[str, Evaluation], best: str, exported: list[str], splits: Splits,
            history: dict) -> dict[str, Figure]:
    """Every experiment graph of the model card, keyed by its assets/<name>.png file name."""
    y_test, y_val = splits.test[M.TARGET].to_numpy(), splits.val[M.TARGET].to_numpy()
    ranked = sorted(evals.values(), key=lambda e: e.val["pr_auc"], reverse=True)
    chosen = evals[best]
    return {
        "model_comparison": plot_comparison(ranked, best, set(exported)),
        "training_curves": plot_mlp_history(history),
        "pr_curve": plot_pr_curves(evals, best, [n for n in exported if n != best], y_test),
        "threshold_tuning": plot_threshold(y_val, chosen),
        "confusion_matrix": plot_confusion(confusion(y_test, chosen.test_prob, chosen.threshold), best,
                                           chosen.threshold),
    }


def export(cfg: Config, *, best: str, exported: list[str], evals: dict[str, Evaluation], splits: Splits,
           history: dict, device: str, train_time_s: float) -> dict:
    """metrics.json + assets/*.png + card refresh. Call save_models first. Returns the metrics dict."""
    model_dir = cfg.model_dir
    config = utils.load_json(model_dir / "config.json")
    if config["best"] != best or set(config["models"]) != set(exported):
        raise RuntimeError("config.json is from another run: call save_models(...) first")
    chosen = evals[best]
    y_test = splits.test[M.TARGET].to_numpy()
    ranked = sorted(evals.values(), key=lambda e: e.val["pr_auc"], reverse=True)
    counts = {f"n_{name}": len(part) for name, part in splits.items()}
    counts |= {f"n_fraud_{name}": int(part[M.TARGET].sum()) for name, part in splits.items()}
    metrics = {
        "model": best,
        "task": "binary-classification",
        "dataset": DATASET,
        "split": "test",
        "primary_metric": {"name": "pr_auc", "value": round(chosen.test["pr_auc"], 6)},
        "metrics": {k: round(v, 6) for k, v in chosen.test.items()},
        "threshold": chosen.threshold,
        "confusion_matrix": confusion(y_test, chosen.test_prob, chosen.threshold),
        "selection": {"rule": SELECTION_RULE, "val": {k: round(v, 6) for k, v in chosen.val.items()}},
        "exported": list(exported),
        "comparison": {e.name: {"val_pr_auc": round(e.val["pr_auc"], 6), "threshold": e.threshold,
                                **{k: round(v, 6) for k, v in e.test.items()}} for e in ranked},
        "data": {**counts, "n_features": len(M.FEATURE_COLUMNS), "n_classes": 2},
        "params": config["models"][best]["params"],
        "train_time_s": round(train_time_s, 1),
        "device": device,
        "smoke": cfg.smoke,
        "source": f"retrained {utils.today()} with training/notebook.ipynb ({device.upper()})",
        "versions": utils.lib_versions(*VERSIONS),
        "trained_at": utils.today(),
    }
    utils.save_json(metrics, model_dir / "metrics.json")

    for name, fig in figures(evals, best, exported, splits, history).items():
        _save(fig, model_dir / "assets" / f"{name}.png")
    architecture = "; ".join(f"{name}: {spec['estimator']}" for name, spec in config["models"].items())
    utils.update_model_card(model_dir, metrics, ml_lab={"architecture": architecture})
    _write_comparison_table(model_dir / "README.md", metrics["comparison"], set(exported), best)
    print(f"[export] metrics.json, assets/, README.md -> {model_dir}")
    return metrics


_TABLE_START, _TABLE_END = "<!-- comparison:start -->", "<!-- comparison:end -->"


def _write_comparison_table(card: Path, comparison: dict, exported: set[str], best: str) -> None:
    """Fill the card's <!-- comparison:start/end --> block from metrics.json (ranked by val PR-AUC).
    The deployed default model is in bold; the other exported model is marked."""
    rows = ["| variant | val PR-AUC | threshold | test PR-AUC | test ROC-AUC | test F1 | test precision "
            "| test recall |", "|---|---|---|---|---|---|---|---|"]
    for name, m in comparison.items():
        cells = [f"{m['val_pr_auc']:.4f}", f"{m['threshold']:.3f}", f"{m['pr_auc']:.4f}", f"{m['roc_auc']:.4f}",
                 f"{m['f1']:.4f}", f"{m['precision']:.4f}", f"{m['recall']:.4f}"]
        if name == best:
            label, cells = f"**{name}** (deployed)", [f"**{c}**" for c in cells]
        else:
            label = f"{name} (exported alternative)" if name in exported else name
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    table = "\n".join([_TABLE_START, *rows, _TABLE_END])
    text = card.read_text(encoding="utf-8")
    if _TABLE_START in text and _TABLE_END in text:
        text = re.sub(re.escape(_TABLE_START) + r".*?" + re.escape(_TABLE_END), lambda _: table, text, flags=re.S)
        card.write_text(text, encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------- plots
# Light chart surface, ink and categorical slots of the validated default palette (fixed order).
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
BLUES = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")


def _axes(width: float = 6.4, height: float = 4.2, ncols: int = 1):
    """A figure on the chart surface with recessive hairline grid and axes (backend-free)."""
    fig = Figure(figsize=(width, height), dpi=150, facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, ncols, squeeze=False)[0]
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
        ax.set_axisbelow(True)
        for side, spine in ax.spines.items():
            spine.set_visible(side in ("left", "bottom"))
            spine.set_color(AXIS)
        ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8)
        ax.xaxis.label.set_color(INK_2)
        ax.yaxis.label.set_color(INK_2)
        ax.title.set_color(INK)
    return fig, axes


def _save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)


def _legend(ax, **kwargs) -> None:
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, **kwargs)


def plot_pr_curves(evals: dict[str, Evaluation], best: str, others: list[str], y_test: np.ndarray) -> Figure:
    """Test precision-recall curves: exported models in color, every other variant in gray."""
    fig, (ax,) = _axes()
    for e in evals.values():
        if e.name not in (best, *others):
            precision, recall, _ = precision_recall_curve(y_test, e.test_prob)
            ax.plot(recall, precision, color=MUTED, linewidth=0.8, alpha=0.45, drawstyle="steps-post")
    ax.plot([], [], color=MUTED, linewidth=0.8, label="other variants")
    for color, name in zip(SERIES, (best, *others)):
        e = evals[name]
        precision, recall, _ = precision_recall_curve(y_test, e.test_prob)
        ap = average_precision_score(y_test, e.test_prob)
        ax.plot(recall, precision, color=color, linewidth=2, drawstyle="steps-post", label=f"{name}  (PR-AUC {ap:.3f})")
    e = evals[best]  # operating point of the default model at its tuned threshold
    ax.plot(e.test["recall"], e.test["precision"], "o", markersize=8, color=SERIES[0],
            markeredgecolor=SURFACE, markeredgewidth=2, label=f"{best} @ threshold {e.threshold:.2f}")
    ax.set(xlim=(0, 1.01), ylim=(0, 1.02), xlabel="Recall (frauds caught)", ylabel="Precision (flags that are fraud)")
    ax.set_title("Precision-recall on the test split", fontsize=10, loc="left")
    _legend(ax, loc="lower left")
    return fig


def plot_confusion(cm: dict[str, int], name: str, threshold: float) -> Figure:
    """2x2 counts on a log color scale (so the few fraud cells stay visible next to ~57k legit)."""
    grid = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, (ax,) = _axes(4.4, 3.8)
    ax.grid(False)
    ax.imshow(np.maximum(grid, 1), cmap=LinearSegmentedColormap.from_list("blues", BLUES),
              norm=LogNorm(1, max(grid.max(), 10)))
    for (i, j), count in np.ndenumerate(grid):
        share = count / max(grid[i].sum(), 1)
        dark = count >= np.sqrt(max(grid.max(), 10))  # upper half of the log scale -> white text
        ax.text(j, i, f"{count:,}\n{share:.2%} of row", ha="center", va="center", fontsize=9,
                color="#ffffff" if dark else INK)
    ax.set_xticks([0, 1], ["legit", "fraud"])
    ax.set_yticks([0, 1], ["legit", "fraud"])
    ax.set(xlabel="Predicted", ylabel="Actual")
    ax.set_title(f"{name} @ threshold {threshold:.2f} (test split)", fontsize=10, loc="left")
    return fig


def plot_comparison(ranked: list[Evaluation], best: str, exported: set[str]) -> Figure:
    """Validation PR-AUC (the selection metric) next to test PR-AUC for every variant."""
    fig, (ax,) = _axes(6.4, 0.42 * len(ranked) + 1.2)
    rows = np.arange(len(ranked))[::-1]  # best on top
    height = 0.36
    val = [e.val["pr_auc"] for e in ranked]
    test = [e.test["pr_auc"] for e in ranked]
    ax.barh(rows + height / 2, val, height=height, color=SERIES[0], edgecolor=SURFACE, linewidth=1,
            label="validation PR-AUC (selects the model)")
    ax.barh(rows - height / 2, test, height=height, color=SERIES[1], edgecolor=SURFACE, linewidth=1,
            label="test PR-AUC (reported once)")
    for row, value in zip(rows, val):
        ax.text(value + 0.01, row + height / 2, f"{value:.3f}", va="center", fontsize=7, color=INK_2)
    ax.set_yticks(rows, [e.name for e in ranked])
    for label, e in zip(ax.get_yticklabels(), ranked):
        label.set_fontweight("bold" if e.name == best else "normal")
        label.set_fontstyle("italic" if e.name in exported and e.name != best else "normal")
    ax.grid(axis="y", visible=False)
    ax.set(xlim=(0, 1.08), xlabel="PR-AUC (average precision)")
    ax.set_title(f"All {len(ranked)} variants by validation PR-AUC (bold = deployed, italic = alt.)", fontsize=10, loc="left")
    _legend(ax, loc="lower right")
    return fig


def plot_threshold(y_val: np.ndarray, chosen: Evaluation) -> Figure:
    """Precision / recall / F1 of the default model vs the decision threshold on validation."""
    precision, recall, thresholds = precision_recall_curve(y_val, chosen.val_prob)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    fig, (ax,) = _axes()
    for color, values, label in zip(SERIES, (precision, recall, f1), ("precision", "recall", "F1")):
        ax.plot(thresholds, values[:-1], color=color, linewidth=2, label=label)
    ax.axvline(chosen.threshold, color=INK_2, linewidth=1)
    right_edge = chosen.threshold > 0.7  # put the label on the side with room
    ax.annotate(f"tuned threshold {chosen.threshold:.2f}\n(F1 {chosen.val['f1']:.3f})", (chosen.threshold, 0.62),
                xytext=(-6 if right_edge else 6, 0), textcoords="offset points",
                ha="right" if right_edge else "left", fontsize=8, color=INK_2)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Decision threshold (flag if P(fraud) >= threshold)")
    ax.set_title(f"{chosen.name}: threshold tuning on the validation split", fontsize=10, loc="left")
    _legend(ax, loc="lower center", ncols=3)
    return fig


def plot_mlp_history(history: dict) -> Figure:
    """Keras MLP training curves: loss / PR-AUC per epoch on the fit rows and the early-stopping slice."""
    fig, axes = _axes(9, 3.4, ncols=2)
    epochs = np.arange(1, len(history["loss"]) + 1)
    for ax, key, title in zip(axes, ("loss", "pr_auc"), ("binary cross-entropy (fit rows class-weighted)", "PR-AUC")):
        ax.plot(epochs, history[key], color=SERIES[0], linewidth=2, marker="o", markersize=4, label="fit rows")
        ax.plot(epochs, history[f"val_{key}"], color=SERIES[1], linewidth=2, marker="o", markersize=4,
                label="early-stopping slice")
        ax.set(xlabel="epoch")
        ax.set_title(f"Keras MLP: {title}", fontsize=10, loc="left")
        _legend(ax)
    return fig


def plot_eda(df) -> Figure:
    """Class balance (log scale) and the Amount / Time distributions of legit vs fraud rows."""
    fig, axes = _axes(13, 3.6, ncols=3)
    counts = df[M.TARGET].value_counts().sort_index()
    axes[0].bar(["legit", "fraud"], counts.to_numpy(), color=SERIES[:2], width=0.5)
    for i, count in enumerate(counts.to_numpy()):
        axes[0].text(i, count * 1.15, f"{count:,}", ha="center", fontsize=8, color=INK_2)
    axes[0].set(yscale="log", ylim=(1, counts.max() * 4))
    axes[0].set_title("Class balance (log scale)", fontsize=10, loc="left")
    for cls, color, label in ((0, SERIES[0], "legit"), (1, SERIES[1], "fraud")):
        part = df[df[M.TARGET] == cls]
        axes[1].hist(np.log10(part["Amount"] + 1), bins=50, density=True, alpha=0.6, color=color, label=label)
        axes[2].hist(part["Time"] / 3600, bins=48, density=True, alpha=0.6, color=color, label=label)
    axes[1].set(xlabel="log10(Amount in EUR + 1)")
    axes[1].set_title("Amount by class", fontsize=10, loc="left")
    axes[2].set(xlabel="hours since the first transaction")
    axes[2].set_title("Time by class", fontsize=10, loc="left")
    for ax in axes[1:]:
        _legend(ax)
    return fig
