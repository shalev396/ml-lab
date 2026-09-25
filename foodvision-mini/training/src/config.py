"""Every tunable of the FoodVision Mini training run, in one dataclass.

Defaults reproduce the original bootcamp recipe (PyTorch Deep Learning bootcamp, notebook 09):
frozen ImageNet EfficientNet-B2 backbone, new head trained with Adam(lr=1e-3), plain
cross-entropy, batch 32, 10 epochs, seed 42, no data augmentation.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

from . import utils

DATA_URL = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip"
# Same file served by GitHub's raw CDN directly (used when the redirecting URL above fails).
DATA_FALLBACK_URL = ("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/"
                     "data/pizza_steak_sushi_20_percent.zip")


@dataclass
class Config:
    # --- data
    data_url: str = DATA_URL
    data_fallback_url: str = DATA_FALLBACK_URL
    dataset_dir_name: str = "pizza_steak_sushi_20_percent"  # folder under training/data/
    subset_per_class: int = 0          # 0 = every image; smoke keeps the first N per class and split
    # --- model
    dropout: float = 0.3
    # --- optimisation
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    seed: int = 42
    # --- run mode (SMOKE_TEST=1): tiny subset + 1 epoch, exports to outputs/smoke/model
    smoke: bool = field(default_factory=lambda: os.getenv("SMOKE_TEST", "0") == "1")

    def __post_init__(self):
        if self.smoke:
            self.epochs = 1
            self.subset_per_class = self.subset_per_class or 8   # 24 train / 24 test images

    @property
    def data_dir(self) -> Path:
        return utils.DATA_DIR / self.dataset_dir_name

    @property
    def run_dir(self) -> Path:
        """Scratch folder of this run (plots, the staged best model). Gitignored."""
        return utils.OUTPUTS_DIR / ("smoke" if self.smoke else "run")

    @property
    def model_dir(self) -> Path:
        """Where export() writes weights + card. Smoke runs never touch the real model/ repo."""
        return utils.OUTPUTS_DIR / "smoke" / "model" if self.smoke else utils.MODEL_DIR
