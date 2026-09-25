"""Write the model repo (cfg.model_dir): rf.joblib, config.json, metrics.json, card plots, refreshed card.
Smoke runs write to training/outputs/smoke/model/ and never touch model/. Also holds the notebook's plots."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from sklearn.metrics import average_precision_score, precision_recall_curve

import model as M  # ../model/model.py (put on sys.path by src/__init__.py)

from . import utils
from .config import Config
from .data_setup import SPLITS, Windows, episodes
from .engine import Evaluation, confusion
from .model_builder import RF

DATASET = "Synthetic elevator IoT month (seeded generator, 44,640 minutely rows, 11 sensors)"
SELECTION_RULE = "highest validation F1 (ties: validation PR-AUC); threshold = max F1 on validation"
VERSIONS = ("sklearn", "numpy", "pandas", "joblib", "tensorflow", "keras")
MODEL_REPO_FILES = ("README.md", "model.py", "handler.py", "requirements.txt", ".gitattributes", ".gitignore")


# --------------------------------------------------------------------------- weights + config
def save_model(cfg: Config, forest, threshold: float) -> Path:
    """rf.joblib + config.json into cfg.model_dir: everything model.load() needs (section 7 then loads it)."""
    model_dir = _prepare(cfg.model_dir)
    joblib.dump(forest, model_dir / "rf.joblib", compress=3)
    utils.save_json({
        "model": RF,
        "weights": "rf.joblib",
        "sensors": M.SENSORS,
        "window": cfg.window,
        "label_lag": cfg.label_lag,
        "fft_top_k": cfg.fft_top_k,
        "feature_names": M.feature_names(M.SENSORS, cfg.fft_top_k),
        "threshold": threshold,
        "classes": list(M.CLASSES),
        "estimator": {"type": "RandomForestClassifier", "n_estimators": cfg.rf_estimators,
                      "class_weight": "balanced", "random_state": cfg.seed},
        "data": "SYNTHETIC (seeded generator in training/src/data_setup.py)",
        "selection": SELECTION_RULE,
        "versions": utils.lib_versions("sklearn", "numpy", "pandas", "joblib"),
    }, model_dir / "config.json")
    return model_dir


def _prepare(model_dir: Path) -> Path:
    """Create the export dir; outside model/ (smoke) seed it with the repo's code + card so it is a
    complete, loadable model repo (the Space can run on it via MODEL_DIR=...)."""
    model_dir.mkdir(parents=True, exist_ok=True)
    if model_dir.resolve() != utils.MODEL_DIR.resolve():
        for name in MODEL_REPO_FILES:
            if (utils.MODEL_DIR / name).is_file():
                shutil.copyfile(utils.MODEL_DIR / name, model_dir / name)
    return model_dir


def check_roundtrip(cfg: Config, evaluation: Evaluation, windows: Windows, n: int = 32) -> None:
    """Reload the export through model.py from single windows (what the Space runs) and compare with
    the in-memory forest's test probabilities."""
    predictor = M.load(cfg.model_dir, "cpu")
    for i in np.linspace(0, len(windows.y["test"]) - 1, n).astype(int):
        frame = pd.DataFrame(windows.raw["test"][i], columns=M.SENSORS)
        frame.insert(0, "timestamp", pd.date_range(end=windows.ends["test"][i], periods=cfg.window, freq="min"))
        served = predictor.predict(frame)["failure"]
        if not np.isclose(served, evaluation.prob["test"][i], atol=1e-6):
            raise RuntimeError(f"exported model does not reproduce test window {i}: {served} vs "
                               f"{evaluation.prob['test'][i]}")


# --------------------------------------------------------------------------- metrics + card
def export(cfg: Config, *, evals: dict[str, Evaluation], best: str, windows: Windows, forest,
           histories: dict[str, dict], device: str, total_time_s: float, examples: dict[str, float]) -> dict:
    """metrics.json + assets/*.png + model card for the deployed model; returns the metrics dict."""
    if best != RF:
        raise RuntimeError(f"{best} won on validation, but only the RandomForest export is implemented; "
                           "extend export.save_model before deploying a Keras model")
    model_dir = _prepare(cfg.model_dir)
    chosen = evals[best]
    check_roundtrip(cfg, chosen, windows)
    y_test = windows.y["test"]
    counts = {f"n_{s}": int(len(windows.y[s])) for s in SPLITS}
    counts |= {f"n_failure_{s}": int(windows.y[s].sum()) for s in SPLITS}
    metrics = {
        "model": best,
        "task": "binary-classification (failure state 10 min ahead from a 60-min window)",
        "dataset": DATASET,
        "split": "test",
        "primary_metric": {"name": "f1", "value": round(chosen.test["f1"], 6)},
        "metrics": {k: round(v, 6) for k, v in chosen.test.items()},
        "threshold": chosen.threshold,
        "confusion_matrix": confusion(y_test, chosen.prob["test"], chosen.threshold),
        "selection": {"rule": SELECTION_RULE, "val": {k: round(v, 6) for k, v in chosen.val.items()}},
        "comparison": {e.name: {"val_f1": round(e.val["f1"], 6), "val_pr_auc": round(e.val["pr_auc"], 6),
                                "threshold": round(e.threshold, 6),
                                **{k: round(v, 6) for k, v in e.test.items()},
                                "params": e.params, "train_time_s": round(e.train_time_s, 1)}
                       for e in evals.values()},
        "examples": {k: round(v, 6) for k, v in examples.items()},
        "data": {**counts, "n_rows": windows.n_rows,
                 "n_features": len(M.feature_names(M.SENSORS, cfg.fft_top_k)), "n_classes": 2,
                 "window": cfg.window, "label_lag": cfg.label_lag, "stride": cfg.stride, "synthetic": True},
        "params": chosen.params,
        "params_note": "RandomForest size = total tree nodes (Keras rows: weights)",
        "train_time_s": round(total_time_s, 1),
        "device": device.lower(),
        "smoke": cfg.smoke,
        "source": f"retrained {utils.today()} with training/ (local {device.upper()}) on the synthetic month",
        "versions": utils.lib_versions(*VERSIONS),
        "trained_at": utils.today(),
    }
    utils.save_json(metrics, model_dir / "metrics.json")

    assets = model_dir / "assets"
    _save(plot_histories(histories), assets / "training_curves.png")
    _save(plot_comparison(evals, best), assets / "model_comparison.png")
    _save(plot_pr_curves(evals, best, y_test), assets / "pr_curves.png")
    _save(plot_confusion(metrics["confusion_matrix"], best, chosen.threshold), assets / "confusion_matrix.png")
    _save(plot_timeline(windows, chosen), assets / "timeline.png")
    _save(plot_importance(forest), assets / "feature_importance.png")

    utils.update_model_card(model_dir, metrics, ml_lab={"architecture": f"RandomForest ({cfg.rf_estimators} trees) "
                                                                        "on 63 window features incl. FFT"})
    _write_comparison_table(model_dir / "README.md", metrics["comparison"], best)
    print(f"[export] {best} -> {model_dir}")
    return metrics


_TABLE_START, _TABLE_END = "<!-- comparison:start -->", "<!-- comparison:end -->"


def _write_comparison_table(card: Path, comparison: dict, best: str) -> None:
    """Fill the card's <!-- comparison:start/end --> block from metrics.json."""
    rows = ["| experiment | size | val F1 | threshold | test F1 | test precision | test recall | test PR-AUC "
            "| train time |", "|---|---|---|---|---|---|---|---|---|"]
    for name, m in comparison.items():
        size = f"{m['params']:,} tree nodes" if name == RF else f"{m['params']:,} weights"
        cells = [f"{m['val_f1']:.4f}", f"{m['threshold']:.3f}", f"{m['f1']:.4f}", f"{m['precision']:.4f}",
                 f"{m['recall']:.4f}", f"{m['pr_auc']:.4f}"]
        if name == best:
            cells = [f"**{c}**" for c in cells]
        label = f"**{name}** (deployed)" if name == best else name
        rows.append(f"| {label} | {size} | {' | '.join(cells)} | {m['train_time_s']:.1f} s |")
    table = "\n".join([_TABLE_START, *rows, _TABLE_END])
    text = card.read_text(encoding="utf-8")
    if _TABLE_START in text and _TABLE_END in text:
        text = re.sub(re.escape(_TABLE_START) + r".*?" + re.escape(_TABLE_END), lambda _: table, text, flags=re.S)
        card.write_text(text, encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------- plots
# Light chart surface, ink and categorical slots of the lab's default palette (fixed order).
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
FAIL = "#d64545"
BLUES = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")


def _axes(width: float = 6.4, height: float = 4.2, nrows: int = 1, ncols: int = 1, sharex: bool = False):
    """A figure on the chart surface with a recessive hairline grid (backend-free, safe headless)."""
    fig = Figure(figsize=(width, height), dpi=150, facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(nrows, ncols, squeeze=False, sharex=sharex).ravel()
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


def _save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)


def _legend(ax, **kwargs) -> None:
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, **kwargs)


def _shade_episodes(ax, df: pd.DataFrame, label: bool = True) -> None:
    for i, (a, b) in enumerate(episodes(df)):
        ax.axvspan(a, b, color=FAIL, alpha=0.12, linewidth=0, label="Status = 1" if label and i == 0 else None)


def plot_month(df: pd.DataFrame, cfg: Config, sensors=("vibration_rms", "motor_current", "temperature", "oil_level")):
    """EDA: hourly means of key sensors over the synthetic month, failure episodes shaded, splits marked."""
    fig, axes = _axes(11, 1.7 * len(sensors) + 0.6, nrows=len(sensors), sharex=True)
    hourly = df[list(sensors)].resample("60min").mean()
    n = len(df)
    bounds = [df.index[int(n * cfg.train_frac)], df.index[int(n * (cfg.train_frac + cfg.val_frac))]]
    for i, (ax, sensor) in enumerate(zip(axes, sensors)):
        _shade_episodes(ax, df, label=i == 0)
        ax.plot(hourly.index, hourly[sensor], color=SERIES[0], linewidth=1)
        for b in bounds:
            ax.axvline(b, color=INK_2, linewidth=1, linestyle="--")
        ax.set_ylabel(sensor, fontsize=8)
    axes[0].set_title("Synthetic month (hourly means): shaded = degradation/failure, dashed = train | val | test",
                      fontsize=10, loc="left")
    _legend(axes[0], loc="upper left")
    return fig


def plot_daily_profile(df: pd.DataFrame):
    """EDA: mean motor current by hour of day, weekdays vs weekends (the usage seasonality)."""
    fig, (ax,) = _axes(6.4, 3.4)
    healthy = df[df["Status"] == 0]
    for color, (label, part) in zip(SERIES, (("weekday", healthy[healthy.index.dayofweek < 5]),
                                             ("weekend", healthy[healthy.index.dayofweek >= 5]))):
        profile = part.groupby(part.index.hour)["motor_current"].mean()
        ax.plot(profile.index, profile.to_numpy(), color=color, linewidth=2, marker="o", markersize=3, label=label)
    ax.set(xlabel="hour of day", ylabel="motor current (A)", xticks=range(0, 24, 3))
    ax.set_title("Usage seasonality (healthy minutes)", fontsize=10, loc="left")
    _legend(ax)
    return fig


def plot_lowpass(df: pd.DataFrame, start: str | None = None, hours: int = 6, cutoff: float = 0.05):
    """EDA: a Butterworth low-pass filter (scipy.signal) separates the vibration drift from minute noise."""
    from scipy.signal import butter, filtfilt

    a, b = episodes(df)[0]
    begin = pd.Timestamp(start) if start else a - pd.Timedelta(hours=1)
    part = df.loc[begin: begin + pd.Timedelta(hours=hours), "vibration_rms"]
    bb, aa = butter(4, cutoff)                     # cutoff as a fraction of Nyquist (0.5 cycles/min)
    fig, (ax,) = _axes(9, 3.2)
    _shade_episodes(ax, df.loc[part.index[0]: part.index[-1]])
    ax.plot(part.index, part.to_numpy(), color=MUTED, linewidth=0.8, label="raw (1/min)")
    ax.plot(part.index, filtfilt(bb, aa, part.to_numpy()), color=SERIES[0], linewidth=2,
            label=f"Butterworth low-pass (order 4, {cutoff:g} x Nyquist)")
    ax.set(ylabel="vibration_rms")
    ax.set_title("Vibration around the start of the first degradation episode", fontsize=10, loc="left")
    _legend(ax, loc="upper left")
    return fig


def plot_window(frame: pd.DataFrame, title: str = "Last 60 minutes, per sensor"):
    """Sparkline grid of one 60-minute window (used by the notebook and the Space)."""
    fig, axes = _axes(12, 5.6, nrows=3, ncols=4)
    for ax, sensor in zip(axes, M.SENSORS):
        ax.plot(np.arange(len(frame)), frame[sensor].to_numpy(), color=SERIES[0], linewidth=1.2)
        ax.set_title(sensor, fontsize=8, loc="left")
    for ax in axes[len(M.SENSORS):]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=10, color=INK)
    return fig


def plot_histories(histories: dict[str, dict]):
    """Keras loss and validation PR-AUC per epoch for the two neural experiments."""
    fig, axes = _axes(10, 3.4, ncols=2)
    for color, (name, h) in zip(SERIES[1:], histories.items()):
        epochs = np.arange(1, len(h["loss"]) + 1)
        axes[0].plot(epochs, h["loss"], color=color, linewidth=2, marker="o", markersize=3, label=f"{name} train")
        axes[0].plot(epochs, h["val_loss"], color=color, linewidth=1.4, linestyle="--", label=f"{name} val")
        axes[1].plot(epochs, h["pr_auc"], color=color, linewidth=2, marker="o", markersize=3, label=f"{name} train")
        axes[1].plot(epochs, h["val_pr_auc"], color=color, linewidth=1.4, linestyle="--", label=f"{name} val")
    axes[0].set(xlabel="epoch", title="binary cross-entropy (class-weighted train)")
    axes[1].set(xlabel="epoch", ylim=(0, 1.02), title="PR-AUC (early stopping monitors val)")
    for ax in axes:
        ax.title.set_fontsize(10)
        _legend(ax)
    return fig


def plot_comparison(evals: dict[str, Evaluation], best: str):
    """Test F1 / precision / recall / PR-AUC of every experiment (bold = deployed)."""
    keys = ("f1", "precision", "recall", "pr_auc")
    fig, (ax,) = _axes(7.2, 3.8)
    names = list(evals)
    x = np.arange(len(names))
    width = 0.2
    colors = (*SERIES, "#8a5cd1")
    for j, (key, color) in enumerate(zip(keys, colors)):
        values = [evals[n].test[key] for n in names]
        ax.bar(x + (j - 1.5) * width, values, width=width, color=color, edgecolor=SURFACE, label=f"test {key}")
    lo = min(min(e.test[k] for k in keys) for e in evals.values())
    ax.set_xticks(x, names)
    for label in ax.get_xticklabels():
        label.set_fontweight("bold" if label.get_text() == best else "normal")
    ax.set(ylim=(max(0.0, lo - 0.1), 1.005))
    ax.set_title("Test metrics of every experiment (bold = deployed, chosen on validation F1)", fontsize=10, loc="left")
    _legend(ax, loc="lower right", ncols=2)
    return fig


def plot_pr_curves(evals: dict[str, Evaluation], best: str, y_test: np.ndarray):
    fig, (ax,) = _axes()
    for color, e in zip(SERIES, evals.values()):
        precision, recall, _ = precision_recall_curve(y_test, e.prob["test"])
        ap = average_precision_score(y_test, e.prob["test"])
        ax.plot(recall, precision, color=color, linewidth=2.4 if e.name == best else 1.4, drawstyle="steps-post",
                label=f"{e.name}  (PR-AUC {ap:.3f})")
    e = evals[best]
    ax.plot(e.test["recall"], e.test["precision"], "o", markersize=8, color=SERIES[0], markeredgecolor=SURFACE,
            markeredgewidth=2, label=f"{best} @ threshold {e.threshold:.2f}")
    ax.axhline(float(np.mean(y_test)), color=MUTED, linewidth=0.8, linestyle="--", label="chance")
    ax.set(xlim=(0, 1.01), ylim=(0, 1.02), xlabel="Recall (failure windows caught)",
           ylabel="Precision (alarms that are real)")
    ax.set_title("Precision-recall on the test segment", fontsize=10, loc="left")
    _legend(ax, loc="lower left")
    return fig


def plot_confusion(cm: dict[str, int], name: str, threshold: float):
    grid = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, (ax,) = _axes(4.4, 3.8)
    ax.grid(False)
    ax.imshow(grid / np.maximum(grid.sum(axis=1, keepdims=True), 1),
              cmap=LinearSegmentedColormap.from_list("blues", BLUES), vmin=0, vmax=1)
    for (i, j), count in np.ndenumerate(grid):
        share = count / max(grid[i].sum(), 1)
        ax.text(j, i, f"{count:,}\n{share:.1%} of row", ha="center", va="center", fontsize=9,
                color="#ffffff" if share > 0.5 else INK)
    ax.set_xticks([0, 1], ["healthy", "failure"])
    ax.set_yticks([0, 1], ["healthy", "failure"])
    ax.set(xlabel="Predicted", ylabel="Actual")
    ax.set_title(f"{name} @ threshold {threshold:.2f} (test windows)", fontsize=10, loc="left")
    return fig


def plot_timeline(windows: Windows, e: Evaluation):
    """P(failure) of the deployed model for every window of the month vs the true label."""
    fig, (ax,) = _axes(11, 3.4)
    for split, color in zip(SPLITS, SERIES):
        times, y = windows.label_times[split], windows.y[split]
        ax.fill_between(times, 0, y, step="mid", color=FAIL, alpha=0.15, linewidth=0,
                        label="true failure window" if split == "train" else None)
        ax.plot(times, e.prob[split], color=color, linewidth=0.9, label=f"{split}: P(failure)")
    ax.axhline(e.threshold, color=INK_2, linewidth=0.9, linestyle="--", label=f"threshold {e.threshold:.2f}")
    ax.set(ylim=(-0.02, 1.05), ylabel="P(failure in 10 min)")
    ax.set_title(f"{e.name}: predicted failure probability vs truth over the whole month", fontsize=10, loc="left")
    _legend(ax, loc="upper left", ncols=5)
    return fig


def plot_importance(forest, top: int = 15):
    """Mean decrease in impurity of the forest's top features."""
    names = np.array(M.feature_names())
    imp = forest.feature_importances_
    order = np.argsort(imp)[::-1][:top][::-1]
    fig, (ax,) = _axes(6.4, 0.28 * top + 1.0)
    ax.barh(names[order], imp[order], color=SERIES[0])
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="mean decrease in impurity")
    ax.set_title(f"RandomForest: top {top} of 63 features", fontsize=10, loc="left")
    return fig
