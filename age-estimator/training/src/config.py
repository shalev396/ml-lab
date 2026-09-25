"""Every tunable of the Age & Gender Estimator training run, in one dataclass.

Defaults reproduce the run that produced the deployed weights: ImageNet EfficientNet-B2 trunk
fine-tuned end to end with two cross-entropy heads (age bins + gender), Adam(lr=1e-3), batch 64,
10 epochs, 90/10 random train/validation split of UTKFace (seed 42), best epoch by validation age MAE.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

from . import utils

HF_DATASET_ID = "nu-delta/utkface"


@dataclass
class Config:
    # --- data
    dataset_id: str = HF_DATASET_ID
    val_frac: float = 0.1              # held-out validation fraction (the reported metrics are on it)
    subset: int = 0                    # 0 = all 23,705 faces; smoke keeps a small random slice
    # --- model
    dropout: float = 0.3
    # --- optimisation
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-3
    freeze_epochs: int = 0             # epochs that train only the heads before unfreezing the trunk
    gender_loss_weight: float = 1.0    # loss = CE(age) + w * CE(gender)
    augment: bool = True               # horizontal flip + mild colour jitter on the training faces
    amp: bool = True                   # mixed precision on CUDA (no-op on CPU/MPS)
    seed: int = 42
    # --- run mode (SMOKE_TEST=1): tiny slice + 1 epoch, writes only to outputs/smoke/
    smoke: bool = field(default_factory=lambda: os.getenv("SMOKE_TEST", "0") == "1")

    def __post_init__(self):
        if self.smoke:
            self.epochs = 1
            self.subset = self.subset or 96      # ~86 train / 10 validation faces
            self.batch_size = min(self.batch_size, 16)

    @property
    def data_dir(self) -> Path:
        return utils.DATA_DIR / "utkface"

    @property
    def run_dir(self) -> Path:
        """Scratch folder of this run (best checkpoint + a loadable copy of the trained model)."""
        return utils.OUTPUTS_DIR / ("smoke" if self.smoke else "run")

    @property
    def model_dir(self) -> Path:
        """Where export() writes weights + card. Smoke runs never touch the real model/ repo."""
        return utils.OUTPUTS_DIR / "smoke" / "model" if self.smoke else utils.MODEL_DIR
