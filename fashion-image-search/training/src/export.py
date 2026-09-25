"""Write the model repo (cfg.model_dir) and its card graphs.

`save_model` writes the servable files (backbone.keras, embeddings.npy, catalog.parquet,
projection.npz, config.json) and checks that model.py reproduces the in-memory search.
`export` adds metrics.json, the assets/ graphs and refreshes the model card.
Smoke runs write to training/outputs/smoke/model/ and never touch model/.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from PIL import Image

import model as M  # ../model/model.py

from . import utils
from .config import Config
from .data_setup import LABELS, Splits
from .engine import Variant, per_category
from .model_builder import CNN, describe

DATASET = "Fashion Product Images (Small), Kaggle paramaggarwal/fashion-product-images-small"
SELECTION_RULE = "highest validation Precision@10 (article type); test reported once"
VERSIONS = ("tensorflow", "keras", "numpy", "pandas", "pyarrow", "PIL", "sklearn")
REPO_FILES = ("README.md", "model.py", "handler.py", "requirements.txt", ".gitattributes", ".gitignore")
CATALOG_COLUMNS = ["id", "name", "gender", "master_category", "sub_category", "article_type", "base_colour"]


# --------------------------------------------------------------------------- servable model
def save_model(cfg: Config, embedder, variant: Variant, splits: Splits) -> Path:
    """backbone.keras + float16 catalog embeddings + catalog.parquet (with JPEG thumbnails) + config.json."""
    model_dir = _prepare(cfg.model_dir)
    embedder.save(model_dir / M.BACKBONE_FILE)
    np.save(model_dir / M.EMBEDDINGS_FILE, variant.index.astype(np.float16))

    catalog = splits.index[CATALOG_COLUMNS].copy()
    catalog["id"] = catalog["id"].astype("int64")
    catalog["thumbnail"] = [M.thumbnail_bytes(p, cfg.thumbnail_px) for p in splits.index["image_path"]]
    catalog.to_parquet(model_dir / M.CATALOG_FILE, index=False, compression="zstd")

    projection = None
    if variant.projection is not None:
        projection = "projection.npz"
        np.savez(model_dir / projection, **variant.projection)
    elif (model_dir / "projection.npz").exists():
        (model_dir / "projection.npz").unlink()   # stale file from an earlier PCA export

    utils.save_json({
        "name": utils.SLUG,
        "variant": variant.name,
        "description": describe(variant.name),
        "backbone": M.BACKBONE_FILE,
        "embeddings": M.EMBEDDINGS_FILE,
        "catalog": M.CATALOG_FILE,
        "projection": projection,
        "image_size": cfg.image_size,
        "batch_size": cfg.batch_size,
        "backbone_dim": M.EMBEDDING_DIM,
        "embedding_dim": variant.dim,
        "n_items": len(catalog),
        "similarity": "cosine similarity, exact brute-force k-NN over L2-normalized embeddings",
        "preprocessing": f"RGB -> bilinear resize {cfg.image_size}x{cfg.image_size} -> x/127.5 - 1 (inside backbone.keras)",
        "thumbnail_px": cfg.thumbnail_px,
        "labels": {"master_categories": sorted(catalog["master_category"].unique().tolist()),
                   "n_article_types": int(catalog["article_type"].nunique())},
        "versions": utils.lib_versions(*VERSIONS),
    }, model_dir / M.CONFIG_FILE)
    _check_roundtrip(model_dir, variant, splits)
    size = sum(p.stat().st_size for p in model_dir.iterdir() if p.is_file()) / 1e6
    print(f"[export] {variant.name}: {len(catalog):,} products -> {model_dir}  ({size:.1f} MB)")
    return model_dir


def _prepare(model_dir: Path) -> Path:
    """Create the export dir; outside model/ (smoke) seed it with the card + code so it is a complete,
    loadable model repo (the Space can run on it via MODEL_DIR=...)."""
    model_dir.mkdir(parents=True, exist_ok=True)
    if model_dir.resolve() != utils.MODEL_DIR.resolve():
        for name in REPO_FILES:
            if (utils.MODEL_DIR / name).is_file():
                shutil.copyfile(utils.MODEL_DIR / name, model_dir / name)
    return model_dir


def _check_roundtrip(model_dir: Path, variant: Variant, splits: Splits, n: int = 16) -> None:
    """Reload through model.py (what the Space runs): same query embeddings, same neighbours."""
    predictor = M.load(model_dir, "cpu")
    served = predictor.embed(list(splits.test["image_path"].head(n)))
    if not np.allclose(served, variant.test[:n], atol=1e-4):
        raise RuntimeError("exported embedder does not reproduce the training embeddings")
    top_served, _ = M.top_k(predictor.index, served, 5)
    top_train, _ = M.top_k(variant.index, variant.test[:n], 5)
    if (top_served[:, 0] != top_train[:, 0]).mean() > 0.1:
        raise RuntimeError("exported index does not reproduce the training neighbours")


# --------------------------------------------------------------------------- metrics + card
def export(cfg: Config, *, results: dict, best: str, variants: dict[str, Variant], splits: Splits,
           val_curves: dict, random_baseline: dict, params: int, device: str, embed_time_s: float,
           train_time_s: float) -> dict:
    """metrics.json + assets/*.png + refreshed model card. Returns the metrics dict."""
    model_dir = cfg.model_dir
    ranked = sorted(results, key=lambda n: results[n]["val"][cfg.selection_metric], reverse=True)
    chosen = results[best]
    metrics = {
        "model": best,
        "task": "image-retrieval",
        "dataset": DATASET if splits.source.startswith("kaggle") else splits.source,
        "split": "test",
        "primary_metric": {"name": "p_at_5_article_type", "value": round(chosen["test"]["p_at_5_article_type"], 6)},
        "metrics": {k: round(v, 6) for k, v in chosen["test"].items()},
        "selection": {"rule": SELECTION_RULE, "val": {k: round(v, 6) for k, v in chosen["val"].items()}},
        "comparison": {name: {"dim": variants[name].dim,
                              f"val_{cfg.selection_metric}": round(results[name]["val"][cfg.selection_metric], 6),
                              **{k: round(v, 6) for k, v in results[name]["test"].items()}} for name in ranked},
        "random_baseline": {f"p_at_k_{label}": round(v, 6) for label, v in random_baseline.items()},
        "data": {"n_index": len(splits.index), "n_val": len(splits.val), "n_test": len(splits.test),
                 "n_master_categories": int(splits.index["master_category"].nunique()),
                 "n_article_types": int(splits.index["article_type"].nunique()),
                 "source": splits.source},
        "params": int(params),
        "embedding_dim": variants[best].dim,
        "embed_time_s": round(embed_time_s, 1),
        "train_time_s": round(train_time_s, 1),
        "device": device,
        "smoke": cfg.smoke,
        "source": f"rebuilt {utils.today()} with training/ ({device.upper()})",
        "versions": utils.lib_versions(*VERSIONS),
        "trained_at": utils.today(),
    }
    table = per_category(variants[best], splits, cfg, "test", k=5)
    metrics["per_category"] = {"split": "test", "k": 5, "model": best,
                               "by_master_category": {cat: {"queries": int(row["queries"]),
                                                            **{c: round(float(row[c]), 6) for c in table.columns if c != "queries"}}
                                                      for cat, row in table.iterrows()}}
    utils.save_json(metrics, model_dir / "metrics.json")

    assets = model_dir / "assets"
    plot_precision_curves(val_curves, best, random_baseline, assets / "precision_curves.png")
    plot_comparison(results, ranked, best, cfg, assets / "variant_comparison.png")
    plot_tsne(variants[best], splits, cfg, assets / "tsne_embeddings.png")
    plot_queries(variants[best], splits, assets / "query_examples.png")
    architecture = f"{describe(best)} over {len(splits.index):,} catalog products"
    utils.update_model_card(model_dir, metrics, ml_lab={"architecture": architecture})
    _write_comparison_table(model_dir / "README.md", metrics["comparison"], best, cfg)
    cards = [model_dir / "README.md"]
    if not cfg.smoke and model_dir.resolve() == utils.MODEL_DIR.resolve():
        cards.append(utils.MODEL_DIR.parent / "README.md")   # project README quotes the same line
    for card in cards:
        _write_per_category(card, metrics["per_category"])
    print(f"[export] metrics.json + {len(list(assets.glob('*.png')))} graphs + card -> {model_dir}")
    return metrics


_TABLE_START, _TABLE_END = "<!-- comparison:start -->", "<!-- comparison:end -->"


def _write_comparison_table(card: Path, comparison: dict, best: str, cfg: Config) -> None:
    """Fill the card's <!-- comparison:start/end --> block from metrics.json (ranked by validation)."""
    sel = f"val_{cfg.selection_metric}"
    rows = ["| variant | dim | val P@10 (article type) | test P@5 (article type) | test P@10 (article type) "
            "| test P@5 (master cat.) | test P@10 (master cat.) |", "|---|---|---|---|---|---|---|"]
    for name, m in comparison.items():
        label = f"**{name}** (deployed)" if name == best else name
        rows.append(f"| {label} | {m['dim']:,} | {m[sel]:.4f} | {m['p_at_5_article_type']:.4f} | "
                    f"{m['p_at_10_article_type']:.4f} | {m['p_at_5_master_category']:.4f} | "
                    f"{m['p_at_10_master_category']:.4f} |")
    table = "\n".join([_TABLE_START, *rows, _TABLE_END])
    text = card.read_text(encoding="utf-8")
    if _TABLE_START in text and _TABLE_END in text:
        text = re.sub(re.escape(_TABLE_START) + r".*?" + re.escape(_TABLE_END), lambda _: table, text, flags=re.S)
        card.write_text(text, encoding="utf-8", newline="\n")


_PERCAT_START, _PERCAT_END = "<!-- per_category:start -->", "<!-- per_category:end -->"


def _write_per_category(card: Path, per_cat: dict) -> None:
    """Fill a <!-- per_category:start/end --> block with test P@5 (article type) per master category,
    straight from metrics.json['per_category']."""
    k = per_cat["k"]
    parts = [f"{cat} {m[f'p_at_{k}_article_type']:.2f} (n={m['queries']})" for cat, m in per_cat["by_master_category"].items()]
    line = f"Per master category (test P@{k}, article type; `metrics.json` -> `per_category`, n = test queries): {', '.join(parts)}."
    text = card.read_text(encoding="utf-8")
    if _PERCAT_START in text and _PERCAT_END in text:
        block = "\n".join([_PERCAT_START, line, _PERCAT_END])
        text = re.sub(re.escape(_PERCAT_START) + r".*?" + re.escape(_PERCAT_END), lambda _: block, text, flags=re.S)
        card.write_text(text, encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------- plots
# Light chart surface, ink and categorical slots of the validated default palette (fixed order).
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1", "#d1a31b", "#d6457a", "#4bb3c9")
PRETTY = {"master_category": "master category", "article_type": "article type"}


def _axes(width: float = 6.4, height: float = 4.2, ncols: int = 1):
    """A figure on the chart surface with recessive hairline grid and axes (backend-free)."""
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


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    return path


def _legend(ax, **kwargs) -> None:
    ax.legend(frameon=False, fontsize=7, labelcolor=INK_2, **kwargs)


def _colors(names: list[str], best: str) -> dict[str, str]:
    """Deployed variant always gets the first slot; the others follow in a fixed order."""
    others = [n for n in names if n != best]
    return {best: SERIES[0], **{n: SERIES[1 + i % (len(SERIES) - 1)] for i, n in enumerate(others)}}


def plot_samples(splits: Splits, path: Path, n: int = 12) -> Path:
    """A grid of catalog photos with their labels (EDA)."""
    rows = splits.index.groupby("master_category", group_keys=False).head(2).head(n)
    fig = Figure(figsize=(1.5 * len(rows), 2.6), dpi=120, facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, len(rows), squeeze=False)[0]
    for ax, (_, row) in zip(axes, rows.iterrows()):
        with Image.open(row["image_path"]) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(f"{row['master_category']}\n{row['article_type']}", fontsize=7, color=INK_2)
        ax.axis("off")
    return _save(fig, path)


def plot_precision_curves(val_curves: dict, best: str | None, random_baseline: dict, path: Path) -> Path:
    """Precision@K for K = 1..max on the validation queries, every variant, both label granularities.
    `best=None` (before selection) draws every variant alike, with no "(deployed)" label."""
    fig, axes = _axes(10, 3.8, ncols=2)
    names = list(val_curves)
    colors = _colors(names, best) if best in val_curves else dict(zip(names, [SERIES[i % len(SERIES)] for i in range(len(names))]))
    for ax, label in zip(axes, LABELS):
        for name, c in val_curves.items():
            k = np.arange(1, len(c[label]) + 1)
            ax.plot(k, c[label], color=colors[name], linewidth=2.4 if name == best else 1.4,
                    label=f"{name} (deployed)" if name == best else name)
        ax.axhline(random_baseline[label], color=MUTED, linewidth=1, linestyle="--", label="random ranking")
        max_k = len(c[label])
        ax.set(xlabel="K (neighbours returned)", ylabel="Precision@K", ylim=(0, 1.02), xlim=(1, max_k))
        ax.set_xticks([1, *range(5, max_k + 1, 5)])
        ax.set_title(f"Validation queries: same {PRETTY[label]}", fontsize=10, loc="left", color=INK)
    _legend(axes[1], loc="lower left")
    return _save(fig, path)


def plot_comparison(results: dict, ranked: list[str], best: str, cfg: Config, path: Path) -> Path:
    """All variants: validation P@10 (selects the model) next to test P@10, both granularities."""
    fig, axes = _axes(10, 0.45 * len(ranked) + 1.4, ncols=2)
    rows = np.arange(len(ranked))[::-1]
    h = 0.36
    for ax, label in zip(axes, LABELS):
        key = f"p_at_10_{label}"
        val = [results[n]["val"][key] for n in ranked]
        test = [results[n]["test"][key] for n in ranked]
        ax.barh(rows + h / 2, val, height=h, color=SERIES[0], edgecolor=SURFACE, label="validation")
        ax.barh(rows - h / 2, test, height=h, color=SERIES[1], edgecolor=SURFACE, label="test (reported)")
        for r, v in zip(rows, test):
            ax.text(v + 0.01, r - h / 2, f"{v:.3f}", va="center", fontsize=7, color=INK_2)
        ax.set_yticks(rows, ranked)
        for tick, name in zip(ax.get_yticklabels(), ranked):
            tick.set_fontweight("bold" if name == best else "normal")
        ax.grid(axis="y", visible=False)
        ax.set(xlim=(0, 1.12), xlabel=f"Precision@10 (same {PRETTY[label]})")
        ax.set_title(f"Same {PRETTY[label]} (bold = deployed)", fontsize=10, loc="left", color=INK)
    _legend(axes[0], loc="lower right")
    return _save(fig, path)


def plot_tsne(variant: Variant, splits: Splits, cfg: Config, path: Path) -> Path:
    """2-D t-SNE of a catalog sample, coloured by master category."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    rng = np.random.default_rng(cfg.seed)
    n = min(cfg.tsne_sample, len(variant.index))
    sel = rng.choice(len(variant.index), size=n, replace=False)
    x = variant.index[sel]
    x = PCA(n_components=min(50, x.shape[1], n), random_state=cfg.seed).fit_transform(x)
    xy = TSNE(n_components=2, init="pca", perplexity=min(30, (n - 1) // 3), random_state=cfg.seed).fit_transform(x)
    cats = splits.index["master_category"].to_numpy()[sel]
    fig, (ax,) = _axes(6.4, 5.0)
    ax.grid(False)
    for color, cat in zip(SERIES, pd.Series(cats).value_counts().index):
        m = cats == cat
        ax.scatter(xy[m, 0], xy[m, 1], s=6, color=to_rgba(color, 0.7), linewidths=0, label=f"{cat} ({m.sum()})")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"t-SNE of {n:,} catalog embeddings ({variant.name})", fontsize=10, loc="left", color=INK)
    _legend(ax, loc="best", markerscale=2.5)
    return _save(fig, path)


def plot_queries(variant: Variant, splits: Splits, path: Path, k: int = 5) -> Path:
    """One held-out test query per master category -> its top-k catalog matches (score + article type)."""
    test = splits.test.reset_index(drop=True)
    picks = [test.index[test["master_category"] == c][0] for c in test["master_category"].value_counts().index[:4]]
    idx, sims = M.top_k(variant.index, variant.test[picks], k)
    fig = Figure(figsize=(1.45 * (k + 1), 2.1 * len(picks)), dpi=130, facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(len(picks), k + 1, squeeze=False)
    for r, q in enumerate(picks):
        cells = [(test.loc[q, "image_path"], f"QUERY\n{test.loc[q, 'article_type']}", INK)]
        for j, s in zip(idx[r], sims[r]):
            item = splits.index.iloc[j]
            same = item["article_type"] == test.loc[q, "article_type"]
            cells.append((item["image_path"], f"{s:.2f}\n{item['article_type']}", SERIES[2] if same else INK_2))
        for ax, (img_path, title, color) in zip(axes[r], cells):
            with Image.open(img_path) as img:
                ax.imshow(img.convert("RGB"))
            ax.set_title(title, fontsize=7, color=color)
            ax.axis("off")
    fig.suptitle(f"Held-out queries -> top-{k} matches (green = same article type)", fontsize=9, color=INK)
    return _save(fig, path)
