"""Every tunable of the Email Spam Classifier training run, in one dataclass.

Defaults reproduce the run that produced the deployed weights: DistilBERT (uncased) with the
last 2 of its 6 transformer blocks unfrozen, a new `Dropout(0.3) -> Linear(768, 1)` head,
AdamW with differential learning rates (head 5e-4, encoder 2e-5), linear warmup (10%) + decay,
BCE-with-logits with `pos_weight = n_ham / n_spam`, batch 32, max 256 tokens, up to 5 epochs
with early stopping (patience 2) on validation loss, seed 42.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

from . import utils

BASE_MODEL = "distilbert/distilbert-base-uncased"


@dataclass
class Config:
    # --- data
    dataset: str = "enron"             # "enron" (SetFit/enron_spam) | "sms" (ucirvine/sms_spam)
    val_size: float = 0.15             # stratified 70 / 15 / 15 split of the cleaned dataset
    test_size: float = 0.15
    train_subset: int = 0              # 0 = all rows; smoke keeps a small stratified sample
    eval_subset: int = 0               # same for val / test
    # --- model
    base_model: str = BASE_MODEL
    max_len: int = 256
    dropout: float = 0.3
    unfreeze_last_n: int = 2           # transformer blocks that are fine-tuned (0 = frozen encoder)
    threshold: float = 0.5
    # --- optimisation
    epochs: int = 5
    batch_size: int = 32
    eval_batch_size: int = 64
    lr_head: float = 5e-4
    lr_backbone: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    patience: int = 2                  # early stopping on validation loss
    seed: int = 42
    # --- baseline experiment (TF-IDF + logistic regression)
    baseline_max_features: int = 50_000
    baseline_ngram_max: int = 2
    baseline_C: float = 10.0
    # --- run mode (SMOKE_TEST=1): small sample, short sequences, 1 epoch, exports to outputs/smoke/model
    smoke: bool = field(default_factory=lambda: os.getenv("SMOKE_TEST", "0") == "1")

    def __post_init__(self):
        if self.smoke:
            self.epochs = 1
            self.train_subset = self.train_subset or 128
            self.eval_subset = self.eval_subset or 64
            self.max_len = min(self.max_len, 64)
            self.baseline_max_features = min(self.baseline_max_features, 5_000)

    @property
    def run_dir(self) -> Path:
        """Scratch folder of this run (best checkpoint, plots). Never the real model/ repo."""
        return utils.OUTPUTS_DIR / ("smoke" if self.smoke else "run")

    @property
    def checkpoint_dir(self) -> Path:
        """Best checkpoint as a loadable model folder (weights + config + tokenizer)."""
        return self.run_dir / "checkpoint"

    @property
    def model_dir(self) -> Path:
        """Where export() writes weights + card. Smoke runs never touch the real model/ repo."""
        return utils.OUTPUTS_DIR / "smoke" / "model" if self.smoke else utils.MODEL_DIR
