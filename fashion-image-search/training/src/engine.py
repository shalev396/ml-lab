"""Engine: embed the photos, build every variant's index, and score retrieval with Precision@K.

There is no gradient training in this project: the "fit" is embedding the catalog (and fitting
PCA for the reduced variants). Evaluation asks, for each held-out query photo, how many of its
top-K catalog neighbours share its label (master category and the finer article type).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image

import model as M  # ../model/model.py

from . import model_builder as mb
from .config import Config
from .data_setup import LABELS, Splits

SPLITS = ("index", "val", "test")


@dataclass
class Variant:
    name: str
    index: np.ndarray        # (n_catalog, dim) L2-normalized, float16 round-tripped exactly as served
    val: np.ndarray          # (n_val, dim) query embeddings
    test: np.ndarray         # (n_test, dim)
    projection: dict | None = None   # PCA applied after the backbone (served variants only)

    @property
    def dim(self) -> int:
        return int(self.index.shape[1])

    def queries(self, split: str) -> np.ndarray:
        return getattr(self, split)


# --------------------------------------------------------------------------- "training"
def embed_splits(cfg: Config, embedder, images: dict[str, list[Image.Image]]) -> tuple[dict[str, np.ndarray], float]:
    """Raw MobileNetV2 features for every split with the SAME code path the Predictor uses
    (`model.backbone_features`). Returns ({split: (n, 1280)}, seconds spent embedding)."""
    raw, start = {}, time.perf_counter()
    for split in SPLITS:
        t0 = time.perf_counter()
        raw[split] = M.backbone_features(embedder, images[split], cfg.batch_size, cfg.image_size)
        rate = len(images[split]) / max(time.perf_counter() - t0, 1e-9)
        print(f"[embed] {split:5s} {len(images[split]):5,} photos -> {raw[split].shape}  ({rate:.0f} img/s)")
    return raw, time.perf_counter() - start


def build_variants(cfg: Config, raw: dict[str, np.ndarray], images: dict[str, list[Image.Image]]) -> dict[str, Variant]:
    """Every variant's catalog index + query embeddings (catalog stored as float16, like the export)."""
    variants = {}

    def add(name, feats: dict[str, np.ndarray], projection=None):
        variants[name] = Variant(name, M.as_stored(feats["index"]), feats["val"], feats["test"], projection)

    add(f"pixels_{cfg.pixel_size}", {s: mb.pixel_features(images[s], cfg.pixel_size) for s in SPLITS})
    add(f"color_hist_{cfg.hist_bins}", {s: mb.color_histogram(images[s], cfg.hist_bins) for s in SPLITS})
    add(mb.CNN, {s: M.project(raw[s], None) for s in SPLITS})
    for dims in cfg.pca_dims:
        pca = mb.fit_pca(raw["index"], dims)
        add(f"{mb.CNN}_pca{dims}", {s: M.project(raw[s], pca) for s in SPLITS}, pca)
    for v in variants.values():
        print(f"[index] {v.name:18s} {v.dim:5,}-d  catalog {len(v.index):,}")
    return variants


# --------------------------------------------------------------------------- evaluation
def neighbour_labels(variant: Variant, splits: Splits, split: str, label: str, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(query labels (n,), labels of each query's top-k catalog items (n, k))."""
    idx, _ = M.top_k(variant.index, variant.queries(split), k)
    return getattr(splits, split)[label].to_numpy(), splits.index[label].to_numpy()[idx]


def precision_curve(variant: Variant, splits: Splits, split: str, label: str, max_k: int) -> np.ndarray:
    """Mean Precision@K for K = 1..max_k: share of the top-K neighbours with the query's label."""
    truth, neighbours = neighbour_labels(variant, splits, split, label, max_k)
    hits = neighbours == truth[:, None]
    return np.cumsum(hits, axis=1).mean(axis=0) / np.arange(1, hits.shape[1] + 1)


def evaluate(variant: Variant, splits: Splits, split: str, cfg: Config) -> dict[str, float]:
    """{"p_at_{k}_{label}": value} for k in cfg.ks and both label granularities."""
    out = {}
    for label in LABELS:
        curve = precision_curve(variant, splits, split, label, max(cfg.ks))
        out |= {f"p_at_{k}_{label}": float(curve[k - 1]) for k in cfg.ks}
    return out


def evaluate_all(variants: dict[str, Variant], splits: Splits, cfg: Config) -> dict[str, dict[str, dict[str, float]]]:
    """{variant: {"val": metrics, "test": metrics}}."""
    return {name: {split: evaluate(v, splits, split, cfg) for split in ("val", "test")} for name, v in variants.items()}


def curves(variants: dict[str, Variant], splits: Splits, split: str, cfg: Config) -> dict[str, dict[str, np.ndarray]]:
    """{variant: {label: Precision@1..curve_max_k}} on one query split."""
    return {name: {label: precision_curve(v, splits, split, label, cfg.curve_max_k) for label in LABELS}
            for name, v in variants.items()}


def comparison_table(results: dict, cfg: Config) -> pd.DataFrame:
    """One row per variant, ranked by the validation selection metric."""
    rows = {name: {f"{split}_{k}": v for split, m in r.items() for k, v in m.items()} for name, r in results.items()}
    table = pd.DataFrame(rows).T
    return table.sort_values(f"val_{cfg.selection_metric}", ascending=False)


def select_best(results: dict, cfg: Config) -> str:
    """Highest validation score on cfg.selection_metric (test is never looked at)."""
    return max(results, key=lambda name: results[name]["val"][cfg.selection_metric])


def random_baseline(splits: Splits, split: str = "test") -> dict[str, float]:
    """Expected Precision@K of a random ranking = chance that a random catalog item shares the label."""
    out = {}
    for label in LABELS:
        share = splits.index[label].value_counts(normalize=True)
        out[label] = float(getattr(splits, split)[label].map(share).fillna(0).mean())
    return out


def check_against_sklearn(variant: Variant, split: str = "test", k: int = 10) -> float:
    """Cross-check `model.top_k` against sklearn's brute-force cosine NearestNeighbors.
    Returns the max similarity difference over the top-k (should be ~1e-6)."""
    from sklearn.neighbors import NearestNeighbors

    queries = variant.queries(split)
    dist, _ = NearestNeighbors(metric="cosine", algorithm="brute").fit(variant.index).kneighbors(queries, k)
    _, sims = M.top_k(variant.index, queries, k)
    diff = float(np.abs((1.0 - dist) - sims).max())
    if diff > 1e-4:
        raise RuntimeError(f"model.top_k disagrees with sklearn NearestNeighbors (max diff {diff:.2e})")
    return diff


def per_category(variant: Variant, splits: Splits, cfg: Config, split: str = "test", k: int = 5) -> pd.DataFrame:
    """Precision@k per master category (both label granularities) for one variant."""
    frame = getattr(splits, split)
    out = {}
    for label in LABELS:
        truth, neighbours = neighbour_labels(variant, splits, split, label, k)
        out[f"p_at_{k}_{label}"] = pd.Series((neighbours == truth[:, None]).mean(axis=1)).groupby(
            frame["master_category"].to_numpy()).mean()
    table = pd.DataFrame(out)
    table.insert(0, "queries", frame["master_category"].value_counts())
    return table.sort_values("queries", ascending=False)
