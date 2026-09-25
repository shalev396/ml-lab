"""Write the model repo (cfg.model_dir): weights + config.json + RFM table (`save_model`), then
metrics.json, card plots and the refreshed model card (`write_report`). Smoke runs write to
training/outputs/smoke/model/ and never touch model/."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import joblib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.figure import Figure
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

import model as M  # ../model/model.py

from . import utils
from .engine import Evaluation, confusion
from .model_builder import LOGREG, MLP, describe

DATASET = ("Kaggle benpowis/customer-propensity-to-purchase-data (sessions) + "
           "UCI Online Retail id 352 (RFM segments)")
SELECTION_RULE = "highest validation PR-AUC; decision threshold = max F1 on validation"
VERSIONS = ("torch", "sklearn", "numpy", "pandas", "joblib", "huggingface_hub", "safetensors")
FILES = {LOGREG: "logreg.joblib", MLP: M.WEIGHTS_FILE}


# --------------------------------------------------------------------------- 1. weights
def save_model(model_dir: str | Path, *, scaler, logreg, mlp, evals: dict[str, Evaluation], best: str,
               params: dict[str, int], rfm_stats: dict, splits) -> Path:
    """Weights, config.json (MLP architecture + thresholds + default model) and the RFM table.
    Reloads the folder through model.py and checks it reproduces the in-memory predictions."""
    model_dir = _prepare(Path(model_dir))
    joblib.dump(scaler, model_dir / "scaler.joblib")
    joblib.dump(logreg, model_dir / FILES[LOGREG])
    config = {
        # PropensityMLP.__init__ arguments (read back by PyTorchModelHubMixin.from_pretrained)
        "feature_names": list(mlp.feature_names),
        "hidden_units": list(mlp.hidden_units),
        "dropout": float(mlp.net[2].p) if len(mlp.net) > 2 else 0.0,
        # runtime metadata for model.Predictor
        "target": M.TARGET,
        "labels": ["buy", "no_buy"],
        "scaler": "scaler.joblib",
        "best": best,
        "selection": SELECTION_RULE,
        "models": {name: {"file": FILES[name], "estimator": describe(name, est), "params": params[name],
                          "threshold": evals[name].threshold, "val_pr_auc": round(evals[name].val["pr_auc"], 6)}
                   for name, est in ((LOGREG, logreg), (MLP, mlp))},
        "rfm": M.RFM_FILE,
        "versions": utils.lib_versions(*VERSIONS),
    }
    mlp.cpu().save_pretrained(model_dir, config=config)  # model.safetensors + config.json
    utils.save_json(config, model_dir / "config.json")   # same content, indented + trailing newline
    utils.save_json(rfm_stats, model_dir / M.RFM_FILE)
    _check_roundtrip(model_dir, evals, splits)
    print(f"[export] weights + config -> {model_dir}")
    return model_dir


def _prepare(model_dir: Path) -> Path:
    """Outside model/ (smoke) seed the folder with the card, model.py and handler so it is a complete,
    loadable model repo (`MODEL_DIR=... python app.py` serves it)."""
    model_dir.mkdir(parents=True, exist_ok=True)
    if model_dir.resolve() != utils.MODEL_DIR.resolve():
        for name in ("README.md", "model.py", "handler.py", "requirements.txt"):
            if (utils.MODEL_DIR / name).is_file():
                shutil.copyfile(utils.MODEL_DIR / name, model_dir / name)
    return model_dir


def _check_roundtrip(model_dir: Path, evals: dict[str, Evaluation], splits, n: int = 2048) -> None:
    predictor = M.load(model_dir, "cpu")
    x, _ = splits.xy("test", scaled=False)
    for name, e in evals.items():
        served = predictor.probabilities(x[:n], name)
        if not np.allclose(served, e.test_prob[:n], atol=1e-5):
            raise RuntimeError(f"exported {name} does not reproduce the training predictions")


def write_examples(splits, folder: str | Path = utils.SPACE_DIR / "examples") -> Path:
    """Real test-split sessions for the Space (full runs only: smoke never writes into space/)."""
    from .data_setup import example_sessions

    path = Path(folder) / "visitors.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    example_sessions(splits).to_csv(path, index=False, lineterminator="\n")
    return path


# --------------------------------------------------------------------------- 2. report
def build_metrics(cfg, *, evals: dict[str, Evaluation], best: str, splits, params: dict[str, int],
                  history: dict, rfm_stats: dict, device: str, train_time_s: float) -> dict:
    chosen = evals[best]
    y_test = splits.test[M.TARGET].to_numpy()
    counts = {f"n_{name}": len(part) for name, part in splits.items()}
    counts |= {f"n_orders_{name}": int(part[M.TARGET].sum()) for name, part in splits.items()}
    ranked = sorted(evals.values(), key=lambda e: e.val["pr_auc"], reverse=True)
    return {
        "model": best,
        "task": "binary-classification",
        "dataset": DATASET,
        "split": "test",
        "primary_metric": {"name": "roc_auc", "value": round(chosen.test["roc_auc"], 6)},
        "metrics": {k: round(v, 6) for k, v in chosen.test.items()},
        "threshold": chosen.threshold,
        "confusion_matrix": confusion(y_test, chosen.test_prob, chosen.threshold),
        "selection": {"rule": SELECTION_RULE, "val": {k: round(v, 6) for k, v in chosen.val.items()}},
        "comparison": {e.name: {"params": params[e.name], "val_pr_auc": round(e.val["pr_auc"], 6),
                                "threshold": e.threshold, **{k: round(v, 6) for k, v in e.test.items()},
                                "confusion_matrix": confusion(y_test, e.test_prob, e.threshold)}
                       for e in ranked},
        "mlp_training": {k: history[k] for k in ("best_epoch", "best_val_roc_auc", "seconds", "pos_weight")}
                        | {"epochs_run": len(history["train_loss"])},
        "rfm": {k: rfm_stats[k] for k in ("snapshot", "last_invoice_date", "n_customers", "n_invoice_lines", "total_revenue")},
        "data": {**counts, "n_features": len(M.FEATURES), "n_classes": 2},
        "params": params[best],
        "train_time_s": round(train_time_s, 1),
        "device": device,
        "smoke": cfg.smoke,
        "source": f"retrained {utils.today()} with training/ (local {device.upper()})",
        "versions": utils.lib_versions(*VERSIONS),
        "trained_at": utils.today(),
    }


def write_report(model_dir: str | Path, metrics: dict, *, evals: dict[str, Evaluation], splits,
                 history: dict, coefficients: dict, importance: dict, rfm_summary) -> dict:
    """metrics.json + the 6 card plots in assets/ + the refreshed model card."""
    model_dir = Path(model_dir)
    utils.save_json(metrics, model_dir / "metrics.json")
    best = metrics["model"]
    y_test = splits.test[M.TARGET].to_numpy()
    assets = model_dir / "assets"
    save(plot_training_curves(history), assets / "training_curves.png")
    save(plot_comparison(metrics["comparison"], best), assets / "model_comparison.png")
    save(plot_roc_pr(evals, best, y_test), assets / "roc_pr_curves.png")
    save(plot_confusion(metrics["confusion_matrix"], best, metrics["threshold"]), assets / "confusion_matrix.png")
    save(plot_importance(coefficients, importance), assets / "feature_importance.png")
    save(plot_rfm_segments(rfm_summary), assets / "rfm_segments.png")
    utils.update_model_card(model_dir, metrics)
    _write_comparison_table(model_dir / "README.md", metrics["comparison"], best)
    print(f"[export] metrics + card + {len(list(assets.glob('*.png')))} plots -> {model_dir}")
    return metrics


_TABLE_START, _TABLE_END = "<!-- comparison:start -->", "<!-- comparison:end -->"


def _write_comparison_table(card: Path, comparison: dict, best: str) -> None:
    rows = ["| experiment | params | val PR-AUC | threshold | test ROC-AUC | test PR-AUC | test F1 | test precision "
            "| test recall |", "|---|---|---|---|---|---|---|---|---|"]
    for name, m in comparison.items():
        label = f"**{name}** (deployed default)" if name == best else name
        rows.append(f"| {label} | {m['params']:,} | {m['val_pr_auc']:.4f} | {m['threshold']:.3f} | {m['roc_auc']:.4f} | "
                    f"{m['pr_auc']:.4f} | {m['f1']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} |")
    table = "\n".join([_TABLE_START, *rows, _TABLE_END])
    text = card.read_text(encoding="utf-8")
    if _TABLE_START in text and _TABLE_END in text:
        text = re.sub(re.escape(_TABLE_START) + r".*?" + re.escape(_TABLE_END), lambda _: table, text, flags=re.S)
        card.write_text(text, encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------- plots
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
NEGATIVE = "#c9483a"
BLUES = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
PRETTY = {LOGREG: "Logistic regression", MLP: "PyTorch MLP"}


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
        ax.title.set_color(INK)
    return fig, axes


def _legend(ax, **kwargs) -> None:
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, **kwargs)


def save(fig: Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    return path


def plot_training_curves(history: dict) -> Figure:
    """MLP: weighted BCE loss (train vs validation) and validation ROC-AUC per epoch."""
    fig, (ax1, ax2) = _axes(9, 3.4, ncols=2)
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], color=SERIES[0], linewidth=2, marker="o", markersize=4, label="train")
    ax1.plot(epochs, history["val_loss"], color=SERIES[1], linewidth=2, marker="o", markersize=4, label="validation")
    ax1.set(xlabel="epoch")
    ax1.set_title("PyTorch MLP: pos_weight BCE loss", fontsize=10, loc="left")
    _legend(ax1)
    ax2.plot(epochs, history["val_roc_auc"], color=SERIES[0], linewidth=2, marker="o", markersize=4)
    if history.get("best_epoch"):
        ax2.axvline(history["best_epoch"], color=INK_2, linewidth=1)
        ax2.annotate(f"kept: epoch {history['best_epoch']}", (history["best_epoch"], min(history["val_roc_auc"])),
                     xytext=(4, 4), textcoords="offset points", fontsize=8, color=INK_2)
    ax2.set(xlabel="epoch")
    ax2.set_title("validation ROC-AUC (early stopping)", fontsize=10, loc="left")
    return fig


def plot_comparison(comparison: dict, best: str) -> Figure:
    """Test ROC-AUC / PR-AUC / F1 / precision / recall of every experiment, side by side."""
    keys = ("roc_auc", "pr_auc", "f1", "precision", "recall")
    fig, (ax,) = _axes(7.2, 3.8)
    names = list(comparison)
    width = 0.8 / len(names)
    x = np.arange(len(keys))
    for i, (name, color) in enumerate(zip(names, SERIES)):
        values = [comparison[name][k] for k in keys]
        bars = ax.bar(x + (i - (len(names) - 1) / 2) * width, values, width=width, color=color,
                      edgecolor=SURFACE, linewidth=1,
                      label=f"{PRETTY.get(name, name)} ({comparison[name]['params']:,} params)"
                            + ("  [deployed]" if name == best else ""))
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=6.5, color=INK_2)
    ax.set_xticks(x, ["ROC-AUC", "PR-AUC", "F1", "precision", "recall"])
    ax.set(ylim=(0.5, 1.04))
    ax.grid(axis="x", visible=False)
    ax.set_title("Both experiments on the test split (F1 / precision / recall at the tuned threshold)",
                 fontsize=10, loc="left")
    _legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncols=len(names))
    return fig


def plot_roc_pr(evals: dict[str, Evaluation], best: str, y_test: np.ndarray) -> Figure:
    fig, (ax1, ax2) = _axes(9.6, 4.0, ncols=2)
    order = [best, *[n for n in evals if n != best]]
    for name, color in zip(order, SERIES):
        e = evals[name]
        fpr, tpr, _ = roc_curve(y_test, e.test_prob)
        ax1.plot(fpr, tpr, color=color, linewidth=2, label=f"{PRETTY[name]}  (AUC {e.test['roc_auc']:.4f})")
        precision, recall, _ = precision_recall_curve(y_test, e.test_prob)
        ap = average_precision_score(y_test, e.test_prob)
        ax2.plot(recall, precision, color=color, linewidth=2, drawstyle="steps-post",
                 label=f"{PRETTY[name]}  (PR-AUC {ap:.4f})")
    e = evals[best]
    ax2.plot(e.test["recall"], e.test["precision"], "o", markersize=8, color=SERIES[0], markeredgecolor=SURFACE,
             markeredgewidth=2, label=f"{PRETTY[best]} @ threshold {e.threshold:.3f}")
    ax1.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--")
    ax1.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate", ylabel="True positive rate")
    ax1.set_title("ROC (test split)", fontsize=10, loc="left")
    ax2.set(xlim=(0, 1.01), ylim=(0, 1.02), xlabel="Recall (buyers found)", ylabel="Precision (flags that buy)")
    ax2.set_title("Precision-recall (test split)", fontsize=10, loc="left")
    _legend(ax1, loc="lower right")
    _legend(ax2, loc="lower left")
    return fig


def plot_confusion(cm: dict[str, int], name: str, threshold: float) -> Figure:
    """2x2 counts on a log color scale (the ~4 % buyers stay visible next to the non-buyers)."""
    grid = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, (ax,) = _axes(4.6, 3.9)
    ax.grid(False)
    ax.imshow(np.maximum(grid, 1), cmap=LinearSegmentedColormap.from_list("blues", BLUES),
              norm=LogNorm(1, max(grid.max(), 10)))
    for (i, j), count in np.ndenumerate(grid):
        share = count / max(grid[i].sum(), 1)
        dark = count >= np.sqrt(max(grid.max(), 10))
        ax.text(j, i, f"{count:,}\n{share:.2%} of row", ha="center", va="center", fontsize=9,
                color="#ffffff" if dark else INK)
    ax.set_xticks([0, 1], ["no order", "order"])
    ax.set_yticks([0, 1], ["no order", "order"])
    ax.set(xlabel="Predicted", ylabel="Actual")
    ax.set_title(f"{PRETTY.get(name, name)} @ threshold {threshold:.3f} (test)", fontsize=10, loc="left")
    return fig


def plot_importance(coefficients: dict[str, float], importance: dict[str, float]) -> Figure:
    """Left: logistic-regression coefficients (standardized flags). Right: MLP permutation importance."""
    order = sorted(coefficients, key=lambda f: coefficients[f])
    fig, (ax1, ax2) = _axes(10, 6.2, ncols=2)
    rows = np.arange(len(order))
    coefs = [coefficients[f] for f in order]
    ax1.barh(rows, coefs, color=[SERIES[0] if c >= 0 else NEGATIVE for c in coefs], height=0.7)
    ax1.set_yticks(rows, order)
    ax1.axvline(0, color=AXIS, linewidth=1)
    ax1.set(xlabel="coefficient (log-odds per std of the flag)")
    ax1.set_title("Logistic regression coefficients", fontsize=10, loc="left")
    imp_order = sorted(importance, key=lambda f: importance[f])
    drops = [importance[f] for f in imp_order]
    ax2.barh(rows, drops, color=[SERIES[2] if d >= 0 else NEGATIVE for d in drops], height=0.7)
    ax2.set_yticks(rows, imp_order)
    ax2.axvline(0, color=AXIS, linewidth=1)
    ax2.set(xlabel="test ROC-AUC drop when the flag is shuffled")
    ax2.set_title("PyTorch MLP permutation importance", fontsize=10, loc="left")
    for ax in (ax1, ax2):
        ax.grid(axis="y", visible=False)
    return fig


def plot_rfm_segments(summary) -> Figure:
    """Customers vs revenue share per RFM segment (UCI Online Retail)."""
    fig, (ax,) = _axes(7.6, 0.42 * len(summary) + 1.3)
    rows = np.arange(len(summary))[::-1]
    h = 0.38
    ax.barh(rows + h / 2, summary["share_pct"], height=h, color=SERIES[0], label="% of customers")
    ax.barh(rows - h / 2, summary["revenue_share_pct"], height=h, color=SERIES[1], label="% of revenue")
    for row, c, r in zip(rows, summary["share_pct"], summary["revenue_share_pct"]):
        ax.text(c + 0.4, row + h / 2, f"{c:.1f}%", va="center", fontsize=7, color=INK_2)
        ax.text(r + 0.4, row - h / 2, f"{r:.1f}%", va="center", fontsize=7, color=INK_2)
    ax.set_yticks(rows, summary["segment"])
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="share (%)")
    ax.set_title(f"RFM segments: {int(summary['customers'].sum()):,} UCI Online Retail customers",
                 fontsize=10, loc="left")
    _legend(ax, loc="lower right")
    return fig
