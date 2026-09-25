"""Write the model repo (cfg.model_dir): weights, config.json, frozen prices, metrics.json, card plots and
the refreshed model card. Smoke runs write to training/outputs/smoke/model/ and never touch model/."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import model as M  # ../model/model.py (put on sys.path by src/__init__.py)

from . import utils
from .config import Config
from .data_setup import Track
from .model_builder import NAIVE

VERSIONS = ("tensorflow", "keras", "numpy", "pandas", "sklearn")
SELECTION_RULE = "lowest validation RMSE among the univariate models the Space can run recursively"
SEED_FILES = ("README.md", "model.py", "handler.py", "requirements.txt")


# --------------------------------------------------------------------------- 1. deployable files
def save_model(cfg: Config, net, track: Track, prices: pd.DataFrame, name: str, source: str) -> Path:
    """model.keras + config.json + prices.csv: everything `model.load()` needs. Returns the model dir."""
    model_dir = _prepare(cfg.model_dir)
    net.save(model_dir / "model.keras")
    frozen = prices[["Close"]].round(6)
    frozen.index = frozen.index.strftime("%Y-%m-%d")
    frozen.to_csv(model_dir / "prices.csv", index_label="Date")
    utils.save_json({
        "model_name": f"LSTM({cfg.lstm_units}) on log-returns",
        "variant": name,
        "architecture": f"Input({cfg.lookback}, 1) -> LSTM({cfg.lstm_units}) -> Dropout({cfg.dropout}) -> Dense(1)",
        "target": "next-day log-return, standardized with the train-split mean/std",
        "lookback": cfg.lookback,
        "units": cfg.lstm_units,
        "dropout": cfg.dropout,
        "ret_mu": track.ret_mu,
        "ret_sigma": track.ret_sigma,
        "ticker": cfg.ticker,
        "prices_file": "prices.csv",
        "price_column": "Close (split-adjusted, not dividend-adjusted)",
        "prices_source": source,
        "date_range": {"first": frozen.index[0], "last": frozen.index[-1], "rows": int(len(frozen))},
        "default_horizon": M.DEFAULT_HORIZON,
        "max_horizon": M.MAX_HORIZON,
        "history_days": M.HISTORY_DAYS,
        "dtype": "float32",
        "versions": utils.lib_versions(*VERSIONS),
    }, model_dir / "config.json")
    _check_roundtrip(model_dir, net, track)
    return model_dir


def _prepare(model_dir: Path) -> Path:
    """Create the export dir; outside model/ (smoke) seed it with the card + code so it is a complete,
    loadable model repo (the Space can run on it via MODEL_DIR=...)."""
    model_dir.mkdir(parents=True, exist_ok=True)
    if model_dir.resolve() != utils.MODEL_DIR.resolve():
        for name in SEED_FILES:
            shutil.copyfile(utils.MODEL_DIR / name, model_dir / name)
    return model_dir


def _check_roundtrip(model_dir: Path, net, track: Track) -> None:
    """Reload through model.py (what the Space runs) and compare a 5-day forecast with the in-memory net."""
    predictor = M.load(model_dir, "cpu")
    o = len(track.close) - 1
    window = M.standardize(M.log_returns(track.close[o - predictor.lookback:o + 1]), track.ret_mu, track.ret_sigma)
    expected = M.to_prices(track.close[o], M.recursive_forecast(net, window[None], 5)[0], track.ret_mu,
                           track.ret_sigma)
    served = predictor.forecast(predictor.resolve_cutoff(track.dates[o]), 5)
    if not np.allclose(served, expected, rtol=1e-4):
        raise RuntimeError(f"exported model does not reproduce the training forecast: {served} vs {expected}")


# --------------------------------------------------------------------------- 2. metrics + card
def export_report(cfg: Config, *, best: str, evals: dict, histories: dict, params: dict, horizon: dict,
                  tracks: dict[str, Track], dataset: str, device: str, train_time_s: float) -> dict:
    """metrics.json + assets/*.png + refreshed README.md in cfg.model_dir. Returns the metrics dict."""
    model_dir = cfg.model_dir
    chosen, naive = evals[best], evals[NAIVE]
    track = tracks["returns"]
    ranked = sorted(evals, key=lambda n: evals[n]["val"]["rmse"])
    metrics = {
        "model": best,
        "task": "time-series-forecasting (next-day close, applied recursively for multi-day forecasts)",
        "dataset": dataset,
        "split": "test",
        "primary_metric": {"name": "rmse", "value": round(chosen["test"]["rmse"], 4)},
        "metrics": {k: round(v, 4) for k, v in chosen["test"].items()},
        "baseline": {"model": NAIVE, **{k: round(v, 4) for k, v in naive["test"].items()}},
        "beats_naive": bool(chosen["test"]["rmse"] < naive["test"]["rmse"]),
        "multi_step": {k: (round(v, 4) if isinstance(v, float) else
                           [round(x, 4) for x in v] if isinstance(v, list) else v) for k, v in horizon.items()},
        "selection": {"rule": SELECTION_RULE, "val": {k: round(v, 4) for k, v in chosen["val"].items()}},
        "comparison": {n: {"val_rmse": round(evals[n]["val"]["rmse"], 4),
                           **{f"test_{k}": round(v, 4) for k, v in evals[n]["test"].items()},
                           "params": params.get(n)} for n in ranked},
        "data": {"n_days": int(len(track.close)), "first_day": str(track.dates[0].date()),
                 "last_day": str(track.dates[-1].date()), "lookback": cfg.lookback,
                 **{f"n_{s}": int(len(v)) for s, v in track.idx.items()},
                 **{f"{s}_span": f"{track.dates[v[0]].date()} to {track.dates[v[-1]].date()}"
                    for s, v in track.idx.items()}},
        "params": params[best],
        "train_time_s": round(train_time_s, 1),
        "device": device,
        "smoke": cfg.smoke,
        "source": f"retrained {utils.today()} with training/ ({device.upper()})",
        "versions": utils.lib_versions(*VERSIONS),
        "trained_at": utils.today(),
    }
    utils.save_json(metrics, model_dir / "metrics.json")

    assets = model_dir / "assets"
    plot_training_curves(histories, assets / "training_curves.png")
    plot_comparison(evals, ranked, best, assets / "model_comparison.png")
    plot_test_forecast(track, evals, best, assets / "test_forecast.png")
    plot_horizon(horizon, assets / "multi_step_mae.png")
    utils.update_model_card(model_dir, metrics, ml_lab={
        "architecture": f"LSTM({cfg.lstm_units}) on standardized daily log-returns, {cfg.lookback}-day window"})
    _write_comparison_table(model_dir / "README.md", metrics["comparison"], best)
    print(f"[export] {best} -> {model_dir}")
    return metrics


_TABLE_START, _TABLE_END = "<!-- comparison:start -->", "<!-- comparison:end -->"


def _write_comparison_table(card: Path, comparison: dict, best: str) -> None:
    """Fill the card's <!-- comparison:start/end --> block from metrics.json (ranked by validation RMSE)."""
    rows = ["| variant | params | val RMSE ($) | test RMSE ($) | test MAE ($) | test MAPE (%) |",
            "|---|---|---|---|---|---|"]
    for name, m in comparison.items():
        label = f"**{name}** (deployed)" if name == best else name
        params = f"{m['params']:,}" if m.get("params") else "-"
        rows.append(f"| {label} | {params} | {m['val_rmse']:.3f} | {m['test_rmse']:.3f} | {m['test_mae']:.3f} "
                    f"| {m['test_mape']:.3f} |")
    table = "\n".join([_TABLE_START, *rows, _TABLE_END])
    text = card.read_text(encoding="utf-8")
    if _TABLE_START in text and _TABLE_END in text:
        text = re.sub(re.escape(_TABLE_START) + r".*?" + re.escape(_TABLE_END), lambda _: table, text, flags=re.S)
        card.write_text(text, encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------- plots
# Light chart surface, ink and categorical slots of the validated default palette (fixed order).
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#8e5cd9")


def _axes(width: float = 6.4, height: float = 4.0, ncols: int = 1):
    """A figure on the chart surface with a recessive hairline grid (backend-free)."""
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


def _save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)


def _legend(ax, **kwargs) -> None:
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, **kwargs)


def plot_training_curves(histories: dict, path: Path) -> None:
    """Train / validation MSE (scaled target) per epoch for every trained variant."""
    fig, axes = _axes(3.2 * len(histories), 3.0, ncols=len(histories))
    for ax, (name, hist) in zip(axes, histories.items()):
        epochs = np.arange(1, len(hist["loss"]) + 1)
        ax.plot(epochs, hist["loss"], color=SERIES[0], linewidth=1.8, label="train")
        ax.plot(epochs, hist["val_loss"], color=SERIES[1], linewidth=1.8, label="validation")
        ax.axvline(hist["best_epoch"], color=MUTED, linewidth=0.8, linestyle="--")
        ax.set_yscale("log")
        ax.set(xlabel="epoch")
        ax.set_title(name, fontsize=9, loc="left")
    axes[0].set_ylabel("MSE on the scaled target (log)")
    _legend(axes[0])
    _save(fig, path)


def plot_comparison(evals: dict, ranked: list[str], best: str, path: Path) -> None:
    """Validation and test RMSE ($) of every variant and the naive baseline."""
    fig, (ax,) = _axes(6.4, 0.5 * len(ranked) + 1.3)
    rows = np.arange(len(ranked))[::-1]
    h = 0.36
    val = [evals[n]["val"]["rmse"] for n in ranked]
    test = [evals[n]["test"]["rmse"] for n in ranked]
    ax.barh(rows + h / 2, val, height=h, color=SERIES[0], label="validation RMSE (selects the model)")
    ax.barh(rows - h / 2, test, height=h, color=SERIES[1], label="test RMSE (reported once)")
    for row, v in zip(rows, test):
        ax.text(v, row - h / 2, f" {v:.2f}", va="center", fontsize=7, color=INK_2)
    ax.set_yticks(rows, ranked)
    for label, name in zip(ax.get_yticklabels(), ranked):
        label.set_fontweight("bold" if name == best else "normal")
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="RMSE of next-day close ($, lower is better)", xlim=(0, max(val + test) * 1.15))
    ax.set_title("All variants, ranked by validation RMSE (bold = deployed)", fontsize=10, loc="left")
    _legend(ax, loc="upper right")
    _save(fig, path)


def plot_test_forecast(track: Track, evals: dict, best: str, path: Path, zoom: int = 90) -> None:
    """Last `zoom` test days: actual close vs the deployed model's and naive one-step predictions."""
    fig, (ax,) = _axes(7.2, 3.8)
    idx = track.idx["test"][-zoom:]
    dates = track.dates[idx]
    ax.plot(dates, track.close[idx], color=INK, linewidth=1.6, label="actual close")
    ax.plot(dates, evals[best]["test_pred"][-zoom:], color=SERIES[0], linewidth=1.4, label=f"{best} (one-step)")
    ax.plot(dates, evals[NAIVE]["test_pred"][-zoom:], color=SERIES[1], linewidth=1.0, linestyle="--",
            label="naive (yesterday's close)")
    ax.set(ylabel="close ($)")
    ax.set_title(f"Test split, last {len(idx)} trading days: next-day predictions", fontsize=10, loc="left")
    fig.autofmt_xdate()
    _legend(ax, loc="upper left")
    _save(fig, path)


def plot_horizon(horizon: dict, path: Path) -> None:
    """MAE by forecast step of the recursive forecast vs naive persistence, over many test origins."""
    fig, (ax,) = _axes(6.4, 3.6)
    steps = np.arange(1, horizon["horizon"] + 1)
    ax.plot(steps, horizon["mae_by_step"], color=SERIES[0], linewidth=2, marker="o", markersize=3,
            label=f"LSTM recursive (mean {horizon['mae_mean']:.2f})")
    ax.plot(steps, horizon["naive_mae_by_step"], color=SERIES[1], linewidth=2, linestyle="--",
            label=f"naive: last close (mean {horizon['naive_mae_mean']:.2f})")
    ax.set(xlabel="business days after the cutoff", ylabel="MAE ($)")
    ax.set_title(f"Multi-step forecasts from {horizon['n_origins']} test cutoffs", fontsize=10, loc="left")
    _legend(ax, loc="upper left")
    _save(fig, path)
