"""Write the trained models as a Hugging Face model repo folder.

`save_models` writes the files `model.load()` needs (xgb.ubj, model.safetensors + config.json,
scaler.joblib). `export` adds metrics.json (STANDARD §4), the experiment graphs in assets/, and refreshes
the model card. Any folder other than model/ (smoke runs, the inference check) first gets a copy of the
card and code from model/, so it is a self-contained snapshot and the real model/ is never touched.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import joblib
import numpy as np
from matplotlib.figure import Figure

from . import utils

import model as M

MODEL_NAMES = {"linear_regression": "LinearRegression", "random_forest": "RandomForest",
               "xgboost": "XGBoost", "mlp": "PyTorch MLP"}
TASK = "tabular-regression"
DATASET = "California Housing (1990 US census, StatLib; sklearn fetch_california_housing)"
COPIED_FROM_MODEL_DIR = ("README.md", "model.py", "handler.py", "requirements.txt")
VERSIONS = ("xgboost", "torch", "sklearn", "numpy", "joblib", "safetensors", "huggingface_hub")

# Chart colours (same palette as the other ml-lab projects).
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, MUTED_BAR = "#2a78d6", "#eb6834", "#b9b7b1"


# --------------------------------------------------------------------------- model files
def build_config(cfg, best: str, val_metrics: dict, xgb_fit: dict, mlp_fit: dict) -> dict:
    """config.json: MLP init args (read back by PyTorchModelHubMixin) + everything model.py needs."""
    return {
        "in_features": len(M.FEATURES),
        "hidden_units": list(cfg.hidden_units),
        "dropout": cfg.dropout,
        "raw_features": M.RAW_FEATURES,
        "features": M.FEATURES,
        "engineered_features": {
            "rooms_per_household": "AveRooms",
            "bedrooms_per_room": "AveBedrms / AveRooms",
            "population_per_household": "AveOccup",
        },
        "target": M.TARGET,
        "target_unit_usd": M.TARGET_UNIT_USD,
        "output": "price_usd = max(prediction, 0) * target_unit_usd",
        "best": best,
        "models": {
            "xgboost": {"file": M.XGB_FILE, "inputs": "unscaled features",
                        "best_iteration": xgb_fit["best_iteration"], "val_rmse": val_metrics["xgboost"]["rmse"]},
            "mlp": {"file": M.MLP_FILE, "scaler": M.SCALER_FILE, "inputs": "StandardScaler(features)",
                    "best_epoch": mlp_fit["best_epoch"], "val_rmse": val_metrics["mlp"]["rmse"]},
        },
        "versions": utils.lib_versions(*VERSIONS),
    }


def prepare_dir(model_dir: str | Path) -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if model_dir.resolve() != utils.MODEL_DIR.resolve():
        for name in COPIED_FROM_MODEL_DIR:
            shutil.copy2(utils.MODEL_DIR / name, model_dir / name)
    return model_dir


def save_models(model_dir: str | Path, booster, mlp, scaler, config: dict) -> Path:
    """xgb.ubj (native XGBoost format), model.safetensors + config.json (mixin), scaler.joblib."""
    model_dir = prepare_dir(model_dir)
    booster.save_model(str(model_dir / M.XGB_FILE))
    mlp.cpu().save_pretrained(model_dir, config=config)   # README.md already exists there: kept as is
    joblib.dump(scaler, model_dir / M.SCALER_FILE)
    return model_dir


# --------------------------------------------------------------------------- metrics.json
def build_metrics(best: str, val: dict, test: dict, *, data: dict, fits: dict, train_time_s: float,
                  device: str, smoke: bool, source: str, xgb_trees: int, mlp_params: int) -> dict:
    comparison = {name: {"val_rmse": val[name]["rmse"], **{k: v for k, v in test[name].items()},
                         "exported": name in ("xgboost", "mlp"),
                         "train_time_s": fits[name]["train_time_s"]}
                  for name in test}
    return {
        "model": best,
        "task": TASK,
        "dataset": DATASET,
        "split": "test",
        "primary_metric": {"name": "r2", "value": test[best]["r2"]},
        "metrics": test[best],
        "selection": {"rule": "lowest validation RMSE among the exported models", "val": val[best]},
        "comparison": comparison,
        "data": data,
        "params": mlp_params if best == "mlp" else None,
        "model_size": {"xgboost_trees": xgb_trees, "mlp_params": mlp_params},
        "train_time_s": round(train_time_s, 1),
        "device": device,
        "smoke": smoke,
        "source": source,
        "versions": utils.lib_versions(*VERSIONS),
        "trained_at": utils.today(),
    }


# --------------------------------------------------------------------------- export
def export(model_dir: str | Path, *, booster, mlp, scaler, config: dict, metrics: dict,
           xgb_history: dict, mlp_history: dict, y_test: np.ndarray, best_pred: np.ndarray) -> dict:
    """Weights + config + metrics.json + assets/*.png, then refresh the model card."""
    model_dir = save_models(model_dir, booster, mlp, scaler, config)
    assets = model_dir / "assets"
    plot_training_curves(xgb_history, config["models"]["xgboost"]["best_iteration"], mlp_history,
                         assets / "training_curves.png")
    plot_comparison(metrics["comparison"], metrics["model"], assets / "model_comparison.png")
    plot_pred_vs_actual(y_test, best_pred, MODEL_NAMES[metrics["model"]], assets / "pred_vs_actual.png")
    plot_feature_importance(booster, assets / "feature_importance.png")
    utils.save_json(metrics, model_dir / "metrics.json")
    utils.update_model_card(model_dir, metrics)
    update_comparison_table(model_dir, metrics)
    print(f"exported -> {model_dir}")
    return metrics


def comparison_table(metrics: dict) -> str:
    rows = sorted(metrics["comparison"].items(), key=lambda kv: kv[1]["val_rmse"])
    lines = ["| model | val RMSE | test RMSE | test MAE | test R² | test RMSE ($) | exported |",
             "|---|---|---|---|---|---|---|"]
    for name, m in rows:
        label = MODEL_NAMES[name]
        cells = [f"{m['val_rmse']:.4f}", f"{m['rmse']:.4f}", f"{m['mae']:.4f}", f"{m['r2']:.4f}",
                 f"${m['rmse_usd']:,.0f}", "yes" if m["exported"] else "no"]
        if name == metrics["model"]:
            label, cells = f"**{label}** (default)", [f"**{c}**" for c in cells]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def update_comparison_table(model_dir: str | Path, metrics: dict) -> None:
    """Replace the table between <!-- comparison:start --> / <!-- comparison:end --> in README.md."""
    path = Path(model_dir) / "README.md"
    text = path.read_text(encoding="utf-8")
    start, end = "<!-- comparison:start -->", "<!-- comparison:end -->"
    if start in text and end in text:
        block = f"{start}\n{comparison_table(metrics)}\n{end}"
        text = re.sub(re.escape(start) + r".*?" + re.escape(end), lambda _: block, text, flags=re.S)
        path.write_text(text, encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------- plots
def _style(ax, grid_axis: str = "both") -> None:
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig: Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path


def plot_training_curves(xgb_history: dict, best_iteration: int, mlp_history: dict, path) -> Path:
    """XGBoost RMSE per boosting round and MLP MSE per epoch (train vs validation), side by side."""
    fig = Figure(figsize=(9.0, 3.4), dpi=150, facecolor=SURFACE, layout="constrained")
    ax_xgb, ax_mlp = fig.subplots(1, 2)
    for ax, hist, keys, xlabel, title in (
        (ax_xgb, xgb_history, ("train_rmse", "val_rmse"), "boosting round", "XGBoost: RMSE ($100k)"),
        (ax_mlp, mlp_history, ("train_loss", "val_loss"), "epoch", "PyTorch MLP: MSE ($100k²)"),
    ):
        _style(ax)
        steps = np.arange(1, len(hist[keys[0]]) + 1)
        ax.plot(steps, hist[keys[0]], color=BLUE, linewidth=1.8, label="train")
        ax.plot(steps, hist[keys[1]], color=ORANGE, linewidth=1.8, label="validation")
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xlabel(xlabel, color=INK_MUTED)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    ax_xgb.axvline(best_iteration + 1, color=INK_MUTED, linestyle="--", linewidth=1)
    ax_xgb.annotate(f"best round {best_iteration + 1}", (best_iteration + 1, ax_xgb.get_ylim()[1]),
                    xytext=(4, -12), textcoords="offset points", fontsize=8, color=INK_MUTED)
    best_epoch = int(np.argmin(mlp_history["val_loss"])) + 1
    ax_mlp.axvline(best_epoch, color=INK_MUTED, linestyle="--", linewidth=1)
    ax_mlp.annotate(f"best epoch {best_epoch}", (best_epoch, ax_mlp.get_ylim()[1]),
                    xytext=(4, -12), textcoords="offset points", fontsize=8, color=INK_MUTED)
    return _save(fig, path)


def plot_comparison(comparison: dict, best: str, path) -> Path:
    """Test RMSE ($), MAE ($) and R² of every experiment as small multiples; the default is highlighted."""
    names = sorted(comparison, key=lambda n: comparison[n]["val_rmse"], reverse=True)
    labels = [MODEL_NAMES[n] for n in names]
    colors = [BLUE if n == best else MUTED_BAR for n in names]
    fig = Figure(figsize=(10.0, 3.0), dpi=150, facecolor=SURFACE, layout="constrained")
    panels = (("rmse_usd", "Test RMSE ($, lower is better)", lambda v: f"${v / 1000:.1f}k"),
              ("mae_usd", "Test MAE ($, lower is better)", lambda v: f"${v / 1000:.1f}k"),
              ("r2", "Test R² (higher is better)", lambda v: f"{v:.3f}"))
    for i, (ax, (key, title, fmt)) in enumerate(zip(fig.subplots(1, 3), panels)):
        _style(ax, grid_axis="x")
        values = [comparison[n][key] for n in names]
        ax.barh(labels, values, color=colors, height=0.6)
        for y, v in enumerate(values):
            ax.text(v, y, " " + fmt(v), va="center", fontsize=8, color=INK)
        ax.set_xlim(0, max(values) * 1.25)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        if i:
            ax.set_yticklabels([])
    return _save(fig, path)


def plot_pred_vs_actual(y_true, y_pred, model_name: str, path) -> Path:
    """Predicted vs actual median house value on the test split (dataset values are capped at $500k)."""
    fig = Figure(figsize=(4.8, 4.6), dpi=150, facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    _style(ax)
    ax.scatter(y_true * 100, y_pred * 100, s=4, alpha=0.25, color=BLUE, linewidths=0)
    hi = max(float(np.max(y_true)), float(np.max(y_pred))) * 100
    ax.plot([0, hi], [0, hi], color=ORANGE, linewidth=1.2, linestyle="--", label="perfect prediction")
    ax.set_xlabel("actual ($k)", color=INK_MUTED)
    ax.set_ylabel("predicted ($k)", color=INK_MUTED)
    ax.set_title(f"{model_name}, test split ({len(y_true):,} districts)", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    return _save(fig, path)


def plot_feature_importance(booster, path) -> Path:
    """XGBoost total gain per feature (normalized to sum to 1)."""
    gain = booster.get_score(importance_type="total_gain")   # keys f0..f10 (trained on arrays)
    values = np.array([gain.get(f"f{i}", 0.0) for i in range(len(M.FEATURES))])
    values = values / max(values.sum(), 1e-12)
    order = np.argsort(values)
    fig = Figure(figsize=(5.4, 3.8), dpi=150, facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    _style(ax, grid_axis="x")
    ax.barh([M.FEATURES[i] for i in order], values[order], color=BLUE, height=0.6)
    for y, v in enumerate(values[order]):
        ax.text(v, y, f" {v:.1%}", va="center", fontsize=8, color=INK)
    ax.set_xlim(0, values.max() * 1.2)
    ax.set_title("XGBoost feature importance (share of total gain)", color=INK, fontsize=10, loc="left")
    return _save(fig, path)
