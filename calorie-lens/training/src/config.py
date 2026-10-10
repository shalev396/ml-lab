"""Every tunable of the calorie-lens run, in one dataclass.

The run trains, in order:
  (0) NutriText (stage 2b): food name -> nutrients per 100 g, from scratch on USDA FoodData Central,
      compared with a TF-IDF nearest-neighbour baseline and the mean;
  (1) IngredientNet + the frozen NutritionLayer (stages 1 + 2a) end to end, once per backbone in
      `experiments` (torchvision ConvNeXt-B vs DINOv2-B with registers); the winner on validation is
      then retrained with the auxiliary depth head (E3). Validation picks every checkpoint, the mass
      blend and the display threshold; the test split is looked at once, at the end.

Data: shalev396/calorie-lens-data (built by dataset_build.py from Nutrition5k + USDA). Images are fitted
into a 378 x 504 canvas (4:3; 27 x 36 DINOv2 patches). Augmentation runs batched on the GPU.
"""
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from . import utils


@dataclass
class Config:
    # --- data
    dataset_id: str = "shalev396/calorie-lens-data"
    build_data: bool = field(default_factory=lambda: os.getenv("BUILD_DATA", "0") == "1")
    image_size: tuple = (378, 504)          # (H, W)
    overhead_share: float = 0.5             # of training samples; the rest are side-angle frames
    samples_per_epoch: int = 12_000
    side_loss_weight: float = 0.5           # side frames may not show the whole plate: mass losses count half
    workers: int = field(default_factory=lambda: int(os.getenv("DATA_WORKERS", utils.num_workers())))
    seed: int = 42
    # --- model (stage 1)
    backbone: str = "convnext_base"
    d_model: int = 256
    decoder_layers: int = 2
    heads: int = 8
    drop_path: float = 0.1
    depth_aux: bool = False
    pretrained: bool = True
    # --- optimisation
    epochs: int = 30                        # upper bound: early stopping on the validation score
    patience: int = 6
    batch_size: int = 64                    # effective; micro-batches of 32 x 2 on GPUs under 70 GB
    micro_batch: int | None = field(default_factory=lambda: int(os.getenv("MICRO_BATCH")) if os.getenv("MICRO_BATCH") else None)
    lr_backbone: float = 1e-4               # top backbone stage; lower stages x layer_decay per stage
    layer_decay: float = 0.8
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    use_compile: bool = True                # CUDA only; falls back to eager if compiling fails
    channels_last: bool = True              # ConvNeXt only
    loss_weights: dict = field(default_factory=lambda: {
        "presence": 1.0, "grams": 0.5, "soft": 1.0, "totals_head": 1.0, "nutrition": 2.0, "depth": 0.5})
    resume: bool = True
    log_every: int = 50
    experiments: tuple = (
        {"name": "E1 ConvNeXt-B", "backbone": "convnext_base"},
        {"name": "E2 DINOv2-B", "backbone": "dinov2", "lr_backbone": 6e-5, "layer_decay": 0.75},
    )
    # --- NutriText (stage 2b)
    nt_vocab: int = 8000
    nt_d: int = 256
    nt_layers: int = 4
    nt_heads: int = 4
    nt_max_len: int = 48
    nt_epochs: int = 20
    nt_batch: int = 512
    nt_lr: float = 1e-3
    nt_branded_weight: float = 0.3          # label values of packaged products are noisier than lab data
    nt_patience: int = 4
    # --- smoke run (SMOKE_TEST=1): tiny random ConvNeXt on the mini build -> outputs/smoke/model
    smoke: bool = field(default_factory=lambda: os.getenv("SMOKE_TEST", "0") == "1")
    smoke_applied: bool = False             # set once, so for_experiment() copies never re-shrink over overrides

    def __post_init__(self):
        if self.smoke and not self.smoke_applied:
            self.smoke_applied = True
            self.backbone, self.pretrained, self.image_size = "convnext_tiny", False, (96, 128)
            self.d_model, self.epochs, self.patience, self.samples_per_epoch = 32, 2, 2, 48
            self.batch_size, self.micro_batch, self.workers, self.use_compile = 8, 4, 0, False
            self.nt_vocab, self.nt_d, self.nt_layers, self.nt_epochs, self.nt_batch = 500, 32, 1, 1, 64
            self.experiments = ({"name": "smoke convnext_tiny", "backbone": "convnext_tiny"},)
            self.log_every = 2

    def for_experiment(self, spec: dict) -> "Config":
        """A copy with one experiment's overrides applied (its "name" is metadata only)."""
        return replace(self, **{k: v for k, v in spec.items() if k != "name"})

    @property
    def data_dir(self) -> Path:
        """Downloaded dataset (or, for smoke runs, the mini build made by dataset_build)."""
        return utils.OUTPUTS_DIR / "smoke" / "data" if self.smoke else utils.DATA_DIR / "calorie-lens-data"

    @property
    def raw_dir(self) -> Path:
        """Nutrition5k metadata + USDA zips, only needed to (re)build the dataset."""
        return utils.DATA_DIR

    @property
    def run_dir(self) -> Path:
        return utils.OUTPUTS_DIR / ("smoke" if self.smoke else "run")

    @property
    def checkpoint_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def model_dir(self) -> Path:
        """Where export() writes weights + card. Smoke runs never touch the real model/ repo."""
        return utils.OUTPUTS_DIR / "smoke" / "model" if self.smoke else utils.MODEL_DIR

    def to_dict(self) -> dict:
        return asdict(self)
