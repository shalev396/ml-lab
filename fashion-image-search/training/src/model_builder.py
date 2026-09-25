"""Models: the MobileNetV2 embedder from model/model.py plus the variants it is compared with.

Every variant maps a product photo to an L2-normalized vector; search is the same exact cosine
k-NN (`model.top_k`) for all of them, so the comparison isolates the representation:

    pixels_32         raw 32x32 RGB pixels (3,072-d)                      baseline, no learning at all
    color_hist_8      8x8x8 RGB colour histogram, sqrt-scaled (512-d)     baseline, colour only
    mobilenetv2       frozen ImageNet MobileNetV2, global avg pool (1,280-d)
    mobilenetv2_pca*  the same embedding reduced with PCA fit on the catalog (unsupervised)
"""
from __future__ import annotations

import numpy as np
from PIL import Image

import model as M  # ../model/model.py (put on sys.path by src/__init__.py)

from .config import Config

CNN = "mobilenetv2"


def build_embedder(cfg: Config):
    """model.build_embedder with ImageNet weights (downloaded once, ~9 MB). There is no silent
    fallback to random weights: if the download fails this raises."""
    return M.build_embedder(weights="imagenet", image_size=cfg.image_size)


def count_params(embedder) -> int:
    return int(embedder.count_params())


def variant_names(cfg: Config) -> list[str]:
    return [f"pixels_{cfg.pixel_size}", f"color_hist_{cfg.hist_bins}", CNN, *(f"{CNN}_pca{d}" for d in cfg.pca_dims)]


def describe(name: str) -> str:
    if name.startswith("pixels_"):
        side = int(name.split("_")[1])
        return f"raw {side}x{side} RGB pixels ({3 * side * side:,}-d), cosine k-NN"
    if name.startswith("color_hist_"):
        bins = int(name.split("_")[2])
        return f"{bins}x{bins}x{bins} RGB colour histogram, sqrt-scaled ({bins ** 3}-d), cosine k-NN"
    if name == CNN:
        return f"MobileNetV2 (ImageNet, frozen) global-avg-pool {M.EMBEDDING_DIM}-d, cosine k-NN"
    if name.startswith(f"{CNN}_pca"):
        return f"MobileNetV2 {M.EMBEDDING_DIM}-d -> PCA {name.split('pca')[1]}-d (fit on the catalog), cosine k-NN"
    raise KeyError(name)


# --------------------------------------------------------------------------- baseline features
def pixel_features(images: list[Image.Image], side: int = 32) -> np.ndarray:
    """Mean-centred raw pixels: two photos are close when the same pixels have the same colours."""
    x = np.stack([np.asarray(img.resize((side, side), Image.BILINEAR), dtype=np.float32).ravel() / 255.0
                  for img in images])
    return M.l2_normalize(x - x.mean(axis=1, keepdims=True))


def color_histogram(images: list[Image.Image], bins: int = 8) -> np.ndarray:
    """Joint RGB histogram (Hellinger / sqrt scaling): position-free colour distribution."""
    feats = []
    for img in images:
        q = (np.asarray(img, dtype=np.uint16) * bins // 256).reshape(-1, 3)
        codes = (q[:, 0] * bins + q[:, 1]) * bins + q[:, 2]
        hist = np.bincount(codes, minlength=bins ** 3).astype(np.float32)
        feats.append(np.sqrt(hist / hist.sum()))
    return M.l2_normalize(np.stack(feats))


def fit_pca(features: np.ndarray, dims: int) -> dict:
    """PCA projection {mean, components (dims, D)} fit on the catalog embeddings (SVD, no labels)."""
    features = np.asarray(features, dtype=np.float32)
    dims = int(min(dims, *features.shape))
    mean = features.mean(axis=0)
    _, _, vt = np.linalg.svd(features - mean, full_matrices=False)
    return {"mean": mean.astype(np.float32), "components": vt[:dims].astype(np.float32)}
