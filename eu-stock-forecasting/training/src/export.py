"""Write the model repo (cfg.model_dir): config.json + eustocks.csv (what the Predictor serves), metrics.json,
card plots and the refreshed model card. Smoke runs write to training/outputs/smoke/model/ and never touch model/."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import model as M  # ../model/model.py (put on sys.path by src/__init__.py)

from . import utils
from .config import Config
from .data_setup import Splits
from .engine import NAIVE, method_of

DATASET = "EuStockMarkets (R datasets: DAX, SMI, CAC, FTSE daily closes 1991-1998)"
SELECTION_RULE = ("per index, validation window only: among ARIMA, Holt-Winters and VAR, the models whose one-step "
                  "validation MAPE is within 2% (relative) of the best; deploy the one with the fewest fitted "
                  "parameters (ties: lower validation MAPE); test metrics are never used")
VERSIONS = ("statsmodels", "pandas", "numpy", "scipy", "sklearn", "tensorflow", "keras")
REPO_FILES = ("model.py", "handler.py", "requirements.txt", "README.md", ".gitattributes", ".gitignore")


# --------------------------------------------------------------------------- the deployable model
def save_model(cfg: Config, frame: pd.DataFrame, specs: dict[str, dict]) -> Path:
    """config.json (method + orders per index) and eustocks.csv (the full data) -> cfg.model_dir.
    Outside model/ (smoke) the code files and card are copied first, so the folder is a complete repo."""
    model_dir = cfg.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    if model_dir.resolve() != utils.MODEL_DIR.resolve():
        for name in REPO_FILES:
            if (utils.MODEL_DIR / name).is_file():
                shutil.copyfile(utils.MODEL_DIR / name, model_dir / name)
    frame.to_csv(model_dir / M.DATA_FILE, float_format="%.2f")
    utils.save_json({
        "framework": M.FRAMEWORK,
        "data_file": M.DATA_FILE,
        "indices": list(M.INDICES),
        "index_start": M.INDEX_START,
        "freq": M.FREQ,
        "first_date": f"{frame.index[0]:%Y-%m-%d}",
        "last_date": f"{frame.index[-1]:%Y-%m-%d}",
        "history_days": cfg.history_days,
        "max_horizon": cfg.max_horizon,
        "min_history": cfg.min_history,
        "selection": SELECTION_RULE,
        "models": specs,
        "versions": utils.lib_versions("statsmodels", "pandas", "numpy"),
    }, model_dir / "config.json")
    return model_dir


# --------------------------------------------------------------------------- metrics + card
def _row(stats: pd.Series) -> dict:
    return {k.lower(): round(float(v), 6) for k, v in stats.items()}


def export(cfg: Config, *, splits: Splits, tables: dict[str, pd.DataFrame], deployed: dict[str, str],
           specs: dict[str, dict], backtests: dict[str, dict], params: dict[str, int], preds: dict,
           histories: dict, device: str, train_time_s: float) -> dict:
    """metrics.json + assets/*.png + card for the run. Returns the metrics dict."""
    model_dir, target = cfg.model_dir, cfg.target
    best = deployed[target]
    table = tables[target]
    bt = backtests[target]["table"]
    best_bt = bt.loc[M.describe(specs[target]["deployed"], specs[target])]
    metrics = {
        "model": M.describe(specs[target]["deployed"], specs[target]),
        "task": "time-series-forecasting",
        "dataset": DATASET,
        "split": "test",
        "target": target,
        "primary_metric": {"name": "mape", "value": round(float(table.loc[best, "MAPE"]), 6)},
        "metrics": {**{k: v for k, v in _row(table.loc[best]).items() if k != "val_mape"},
                    f"backtest_mae_h{cfg.backtest_horizon}": round(float(best_bt["MAE"]), 6),
                    f"backtest_mape_h{cfg.backtest_horizon}": round(float(best_bt["MAPE"]), 6)},
        "evaluation": (f"one-step-ahead walk-forward on the last {splits.n_test} business days (test); "
                       f"validation = the {splits.n_val} days before; backtest = {cfg.backtest_horizon}-day "
                       f"forecasts from {backtests[target]['origins']} origins in the test window"),
        "selection": SELECTION_RULE,
        "deployed": {ix: M.describe(specs[ix]["deployed"], specs[ix]) for ix in specs},
        "comparison": {name: _row(stats) for name, stats in table.iterrows()},
        "comparison_by_index": {ix: {name: _row(stats) for name, stats in t.iterrows()} for ix, t in tables.items()},
        "backtest": {ix: {name: _row(stats) for name, stats in b["table"].iterrows()} for ix, b in backtests.items()},
        "data": {"n_rows": len(splits.frame), "n_fit": splits.n_fit, "n_val": splits.n_val, "n_test": splits.n_test,
                 "n_series": len(M.INDICES), "first_date": f"{splits.frame.index[0]:%Y-%m-%d}",
                 "last_date": f"{splits.frame.index[-1]:%Y-%m-%d}"},
        "params": int(params[best]),
        "params_by_variant": {k: int(v) for k, v in params.items()},
        "train_time_s": round(train_time_s, 1),
        "device": device,
        "smoke": cfg.smoke,
        "source": f"retrained {utils.today()} with training/ ({device.upper()})",
        "versions": utils.lib_versions(*VERSIONS),
        "trained_at": utils.today(),
    }
    utils.save_json(metrics, model_dir / "metrics.json")

    assets = model_dir / "assets"
    plot_series(splits, assets / "series.png")
    plot_comparison(tables, deployed, assets / "model_comparison.png")
    plot_overlay(splits, target, preds[target], best, assets / "holdout_overlay.png")
    plot_training_curves(histories[target], target, assets / "training_curves.png")
    plot_backtest(splits, target, backtests[target], specs[target], assets / "forecast_backtest.png")

    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    for ix, fit_tables in tables.items():  # full tables for reference (not part of the model repo)
        fit_tables.round(4).to_csv(cfg.outputs_dir / f"comparison_{ix}.csv")

    architecture = "; ".join(f"{ix}: {name}" for ix, name in metrics["deployed"].items())
    utils.update_model_card(model_dir, metrics, ml_lab={"architecture": f"per-index classical forecaster ({architecture})"})
    _fill(model_dir / "README.md", "comparison", _comparison_md(table, best))
    _fill(model_dir / "README.md", "by-index", _by_index_md(tables, deployed))
    _fill(model_dir / "README.md", "backtest", _backtest_md(backtests, specs, cfg.backtest_horizon))
    print(f"[export] {metrics['model']} on {target}: test MAPE {metrics['primary_metric']['value']:.3f}% -> {model_dir}")
    return metrics


def _fill(card: Path, key: str, body: str) -> None:
    """Replace the block between <!-- key:start --> and <!-- key:end --> in the card."""
    start, end = f"<!-- {key}:start -->", f"<!-- {key}:end -->"
    text = card.read_text(encoding="utf-8")
    if start in text and end in text:
        text = re.sub(re.escape(start) + r".*?" + re.escape(end), lambda _: f"{start}\n{body}\n{end}", text, flags=re.S)
        card.write_text(text, encoding="utf-8", newline="\n")


def _comparison_md(table: pd.DataFrame, best: str) -> str:
    rows = ["| variant | val MAPE % | test ME | test RMSE | test MAE | test MPE % | test MAPE % | test MASE |",
            "|---|---|---|---|---|---|---|---|"]
    for name, m in table.iterrows():
        label = f"**{name}** (deployed)" if name == best else name
        rows.append(f"| {label} | {m['val_MAPE']:.3f} | {m['ME']:.2f} | {m['RMSE']:.2f} | {m['MAE']:.2f} | "
                    f"{m['MPE']:.3f} | {m['MAPE']:.3f} | {m['MASE']:.2f} |")
    return "\n".join(rows)


def _by_index_md(tables: dict[str, pd.DataFrame], deployed: dict[str, str]) -> str:
    kinds = ["ARIMA", "Holt-Winters", "VAR", "MLP", "LSTM", NAIVE]
    rows = ["| index | " + " | ".join(kinds) + " | deployed |", "|---|" + "---|" * (len(kinds) + 1)]
    for ix, t in tables.items():
        cells = []
        for kind in kinds:
            name = next((n for n in t.index if n.startswith(kind)), None)
            if name is None:  # smoke runs train the nets on the target index only
                cells.append("n/a")
                continue
            cell = f"{t.loc[name, 'val_MAPE']:.3f} / {t.loc[name, 'MAPE']:.3f}"
            cells.append(f"**{cell}**" if name == deployed[ix] else cell)
        rows.append(f"| {ix} | " + " | ".join(cells) + f" | {deployed[ix]} |")
    return "\n".join(rows)


def _backtest_md(backtests: dict[str, dict], specs: dict[str, dict], horizon: int) -> str:
    names = list(next(iter(backtests.values()))["table"].index)
    kinds = [re.sub(r"\(.*\)", "", n).strip() for n in names]
    rows = ["| index | " + " | ".join(f"{k} MAE" for k in kinds) + " | origins |", "|---|" + "---|" * (len(kinds) + 1)]
    for ix, b in backtests.items():
        best = M.describe(specs[ix]["deployed"], specs[ix])
        cells = [f"**{m:.1f}**" if n == best else f"{m:.1f}" for n, m in b["table"]["MAE"].items()]
        rows.append(f"| {ix} | " + " | ".join(cells) + f" | {b['origins']} x {horizon} days |")
    return "\n".join(rows)


# --------------------------------------------------------------------------- plots
# Light chart surface, ink and categorical slots of the validated default palette (fixed order).
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1", "#d1a21b")


def _axes(width: float = 7.0, height: float = 4.0, nrows: int = 1, ncols: int = 1, **kwargs):
    """A figure on the chart surface with a recessive hairline grid and axes (backend-free)."""
    fig = Figure(figsize=(width, height), dpi=150, facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(nrows, ncols, squeeze=False, **kwargs).ravel()
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side, spine in ax.spines.items():
            spine.set_visible(side in ("left", "bottom"))
            spine.set_color(AXIS)
        ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=7)
        ax.title.set_color(INK)
    return fig, axes


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    return path


def _legend(ax, **kwargs) -> None:
    ax.legend(frameon=False, fontsize=7, labelcolor=INK_2, **kwargs)


def plot_series(splits: Splits, path: Path) -> Path:
    """The four indices with the validation and test windows shaded."""
    fig, axes = _axes(9, 4.6, 2, 2, sharex=True)
    val, test = splits.val.index, splits.test.index
    for ax, (color, ix) in zip(axes, zip(SERIES, M.INDICES)):
        ax.plot(splits.frame.index, splits.frame[ix], color=color, linewidth=1)
        ax.axvspan(val[0], val[-1], color=SERIES[4], alpha=0.12, linewidth=0)
        ax.axvspan(test[0], test[-1], color=SERIES[1], alpha=0.12, linewidth=0)
        ax.set_title(ix, fontsize=9, loc="left")
    axes[0].text(val[0], axes[0].get_ylim()[1], " val", fontsize=7, color=INK_2, va="top")
    axes[0].text(test[0], axes[0].get_ylim()[1], " test", fontsize=7, color=INK_2, va="top")
    fig.suptitle("EuStockMarkets closes (synthetic business-day dates); shaded: validation, test",
                 fontsize=9, color=INK, x=0.01, ha="left")
    return _save(fig, path)


def plot_comparison(tables: dict[str, pd.DataFrame], deployed: dict[str, str], path: Path) -> Path:
    """Test MAPE of every variant per index; the deployed method in blue, naive persistence as a line."""
    fig, axes = _axes(9, 4.2, 1, len(tables))
    for ax, (ix, table) in zip(axes, tables.items()):
        names = [n for n in table.index if n != NAIVE]
        rows = np.arange(len(names))[::-1]
        colors = [SERIES[0] if n == deployed[ix] else (AXIS if method_of(n) is None else MUTED) for n in names]
        ax.barh(rows, table.loc[names, "MAPE"], color=colors, height=0.6)
        ax.axvline(table.loc[NAIVE, "MAPE"], color=SERIES[1], linewidth=1.2, linestyle="--")
        for row, value in zip(rows, table.loc[names, "MAPE"]):
            ax.text(value, row, f" {value:.2f}", va="center", fontsize=6.5, color=INK_2)
        ax.set_yticks(rows, names)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, table["MAPE"].max() * 1.3)
        ax.set_title(ix, fontsize=9, loc="left")
        ax.set_xlabel("test MAPE (%)", fontsize=7, color=INK_2)
    handles = [Line2D([], [], color=SERIES[1], linestyle="--", label="naive persistence"),
               Patch(color=SERIES[0], label="deployed (fewest params within 2% of best val MAPE)"),
               Patch(color=MUTED, label="other classical"), Patch(color=AXIS, label="neural nets (not deployed)")]
    fig.legend(handles=handles, frameon=False, fontsize=7, labelcolor=INK_2, loc="outside lower center", ncols=4)
    fig.suptitle("One-step-ahead test MAPE, all variants (lower is better)", fontsize=9, color=INK, x=0.01, ha="left")
    return _save(fig, path)


def plot_overlay(splits: Splits, index: str, preds: dict[str, np.ndarray], best: str, path: Path) -> Path:
    """Test window: actual closes and the one-step predictions of every variant."""
    fig, (ax,) = _axes(8, 4)
    dates, n_val = splits.test.index, splits.n_val
    ax.plot(dates, splits.test[index], color=INK, linewidth=1.8, label="actual")
    others = [n for n in preds if n not in (best, NAIVE)]
    for color, name in zip(SERIES[1:], others):
        ax.plot(dates, preds[name][n_val:], color=color, linewidth=0.9, alpha=0.8, label=name)
    ax.plot(dates, preds[best][n_val:], color=SERIES[0], linewidth=1.6, label=f"{best} (deployed)")
    ax.set_ylabel("index level", fontsize=7, color=INK_2)
    ax.set_title(f"{index}: one-step-ahead predictions on the {len(dates)}-day test window", fontsize=9, loc="left")
    _legend(ax, loc="upper left", ncols=2)
    return _save(fig, path)


def plot_training_curves(histories: dict[str, dict], index: str, path: Path) -> Path:
    """MLP and LSTM loss per epoch (scaled MSE) on the fit and validation windows."""
    fig, axes = _axes(8, 3.2, 1, len(histories))
    for ax, (name, hist) in zip(axes, histories.items()):
        epochs = np.arange(1, len(hist["loss"]) + 1)
        ax.plot(epochs, hist["loss"], color=SERIES[0], linewidth=1.6, label="fit window")
        ax.plot(epochs, hist["val_loss"], color=SERIES[1], linewidth=1.6, label="validation window")
        best = int(np.argmin(hist["val_loss"])) + 1
        ax.axvline(best, color=MUTED, linewidth=0.8, linestyle=":")
        ax.set_yscale("log")
        ax.set_xlabel("epoch", fontsize=7, color=INK_2)
        ax.set_title(f"{index} {name}: MSE (scaled), best epoch {best}", fontsize=9, loc="left")
        _legend(ax)
    return _save(fig, path)


def plot_backtest(splits: Splits, index: str, bt: dict, spec: dict, path: Path) -> Path:
    """What the Space serves: multi-step forecasts from every backtest origin vs actual and naive."""
    method = spec["deployed"]
    fig, (ax,) = _axes(8, 4)
    context = splits.frame[index].iloc[splits.n_fit + splits.n_val - 40:]
    ax.plot(context.index, context, color=INK, linewidth=1.4, label="actual")
    for i, p in enumerate(bt["paths"]):
        ax.plot(p["dates"], p["naive"], color=SERIES[1], linewidth=0.9, linestyle="--",
                label="naive (last close)" if i == 0 else None)
        ax.plot(p["dates"], p[method], color=SERIES[0], linewidth=1.6,
                label=f"{M.describe(method, spec)} forecast" if i == 0 else None)
        ax.plot([p["origin"]], [splits.frame[index].loc[p["origin"]]], "o", color=SERIES[0], markersize=3)
    mae = bt["table"].loc[M.describe(method, spec), "MAE"]
    naive = bt["table"].loc[NAIVE, "MAE"]
    ax.set_title(f"{index}: {bt['horizon']}-day forecasts from {bt['origins']} origins in the test window "
                 f"(MAE {mae:.1f} vs naive {naive:.1f})", fontsize=9, loc="left")
    ax.set_ylabel("index level", fontsize=7, color=INK_2)
    _legend(ax, loc="upper left")
    return _save(fig, path)
