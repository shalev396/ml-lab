"""Every tunable of the FoodVision Big training run, in one dataclass.

Defaults reproduce the original run (PyTorch Deep Learning bootcamp, `foodvision_big/train.py`):
ImageNet EfficientNet-B2 fully fine-tuned on all of Food-101 (75,750 train / 25,250 test),
`Adam(lr=1e-4)`, cross-entropy with label smoothing 0.1, `TrivialAugmentWide` on the training
images, batch 32, 5 epochs, seed 42, AMP on CUDA, and the epoch with the lowest test loss kept.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

from . import utils


@dataclass
class Config:
    # --- data
    dataset_id: str = "ethz/food101"   # Hub dataset; its "validation" split is Food-101's official test split
    source: str = "hub"                # "hub" = full download (~5 GB, cached) · "slice" = read only a few parquet row groups
    train_images: int = 0              # 0 = every image; >0 = a seeded random subset ("hub") or slice size ("slice")
    test_images: int = 0
    slice_chunks: int = 8              # "slice": number of 100-image row groups spread evenly over the split
    # --- model
    dropout: float = 0.3
    freeze_backbone: bool = False      # the original run fine-tunes the whole network
    use_compile: bool = False          # torch.compile (the original used it on CUDA; same maths, only faster)
    # --- optimisation
    epochs: int = 5
    batch_size: int = 32
    lr: float = 1e-4
    label_smoothing: float = 0.1
    augment: bool = True               # TrivialAugmentWide in front of the EfficientNet-B2 preprocessing (train only)
    keep: str = "best_test_loss"       # which epoch to keep: "best_test_loss" (original) or "last"
    seed: int = 42
    log_every: int = 200               # print a progress line every N training batches
    # --- run mode (SMOKE_TEST=1): a tiny streamed slice + 1 epoch, exports to outputs/smoke/model
    smoke: bool = field(default_factory=lambda: os.getenv("SMOKE_TEST", "0") == "1")

    def __post_init__(self):
        if self.smoke:
            self.epochs = 1
            self.source = "slice"
            self.train_images = self.train_images or 64
            self.test_images = self.test_images or 64
            self.slice_chunks = min(self.slice_chunks, 4)
            self.log_every = 1

    @property
    def data_dir(self) -> Path:
        return utils.DATA_DIR

    @property
    def run_dir(self) -> Path:
        """Scratch folder of this run (plots, the staged best model). Gitignored."""
        return utils.OUTPUTS_DIR / ("smoke" if self.smoke else "run")

    @property
    def model_dir(self) -> Path:
        """Where export() writes weights + card. Smoke runs never touch the real model/ repo."""
        return utils.OUTPUTS_DIR / "smoke" / "model" if self.smoke else utils.MODEL_DIR
