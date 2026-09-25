"""Every tunable of the fashion-image-search run in one dataclass."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from . import utils


@dataclass
class Config:
    # --- data: stratified (by master category) catalog / validation / test split ---------------
    seed: int = 42
    kaggle_dataset: str = "paramaggarwal/fashion-product-images-small"
    min_category_count: int = 20      # master categories with fewer products are dropped
    index_size: int = 8_000           # products embedded into the searchable catalog
    n_val: int = 500                  # held-out query photos that pick the variant
    n_test: int = 500                 # held-out query photos reported once
    fallback_images: int = 10_000     # CIFAR-10 images written to disk if the Kaggle download fails

    # --- embedder + variants -----------------------------------------------------------------
    image_size: int = 224             # MobileNetV2 input resolution
    batch_size: int = 64
    pca_dims: tuple[int, ...] = (256, 128)   # PCA variants of the MobileNetV2 embedding
    pixel_size: int = 32              # raw-pixel baseline: RGB thumbnail side
    hist_bins: int = 8                # colour-histogram baseline: bins per RGB channel

    # --- evaluation ---------------------------------------------------------------------------
    ks: tuple[int, ...] = (1, 5, 10)  # Precision@K reported in metrics.json
    curve_max_k: int = 20             # Precision@K curves go from K=1 to this
    selection_metric: str = "p_at_10_article_type"   # on the validation queries
    tsne_sample: int = 2_000

    # --- export -------------------------------------------------------------------------------
    thumbnail_px: int = 160           # catalog thumbnails: longest side (never upscaled)
    n_examples: int = 4               # Space example photos (one per master category, from test)

    # --- run ----------------------------------------------------------------------------------
    smoke: bool = field(default_factory=lambda: os.getenv("SMOKE_TEST", "0") == "1")

    def __post_init__(self):
        self.pca_dims = tuple(self.pca_dims)
        self.ks = tuple(self.ks)
        if self.smoke:  # tiny, fast end-to-end run: same code path, a few hundred images
            self.index_size = min(self.index_size, 300)
            self.n_val = min(self.n_val, 40)
            self.n_test = min(self.n_test, 40)
            self.fallback_images = min(self.fallback_images, 600)
            self.pca_dims = tuple(d for d in self.pca_dims if d <= 64) or (32,)
            self.tsne_sample = min(self.tsne_sample, 150)

    @property
    def model_dir(self) -> Path:
        """Where export writes the model repo; smoke runs never touch the real model/."""
        return utils.OUTPUTS_DIR / "smoke" / "model" if self.smoke else utils.MODEL_DIR

    @property
    def outputs_dir(self) -> Path:
        """Run artifacts that are not part of the model repo."""
        return utils.OUTPUTS_DIR / "smoke" if self.smoke else utils.OUTPUTS_DIR

    @property
    def examples_dir(self) -> Path:
        """Space example photos (held-out test products); smoke runs write to outputs/ instead."""
        return utils.OUTPUTS_DIR / "smoke" / "examples" if self.smoke else utils.SPACE_DIR / "examples"
