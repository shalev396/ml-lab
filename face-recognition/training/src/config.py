"""Every tunable of the face-recognition run, in one dataclass.

Three stages, all trained here (no pretrained face network is used):
  (0) the face detector (model.FaceDetector): finds faces and their 5 landmarks, trained on Open
      Images face boxes + LFW, with the landmark labels in training/labels/landmark_labels.json.gz;
  (1) the face embedders: CNNs trained with a CosFace margin loss on every LFW identity except the
      held-out ones -> 512-d embeddings each. The model is an ensemble of `embedders`: an EfficientNet-B2
      that starts from torchvision's ImageNet weights and a ResNet-18 trained from scratch;
  (2) the identity head: an MLP (embedding -> 256 -> 42) on the frozen, concatenated embeddings of
      the 42 people with at least 25 photos.
The 42 people keep the original 75/25 split (seed 42); 15% of their training photos are the
validation split used for every choice (checkpoint, head epoch, verification threshold). The test
split is looked at once, at the end. 10% of the other people with >= 2 photos are never trained
on ("unseen"): they measure how well the embedder generalises to new faces.
"""
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import utils


@dataclass
class Config:
    # --- data: Labeled Faces in the Wild (funneled), every photo of every person
    min_faces_per_person: int = 25     # the 42 people the head names
    test_size: float = 0.25            # stratified per identity (the original split)
    val_size: float = 0.15             # of the 42 people's training photos
    unseen_fraction: float = 0.10      # of the other people with >= 2 photos, never trained on
    seed: int = 42
    # --- stage 0: face detector (see src/detector.py)
    det_epochs: int = 30
    det_size: int = 640                # training crops are det_size x det_size
    det_batch_size: int = 16
    det_lr: float = 2e-3
    det_landmark_weight: float = 0.5
    det_mosaic: float = 0.3            # share of 2x2 mosaics (many small faces)
    det_lfw_photos: int = 5000         # LFW photos mixed into the Open Images photos
    det_val_images: int = 600          # held-out Open Images photos (checkpoint selection)
    det_val_every: int = 3
    det_workers: int = field(default_factory=lambda: int(os.getenv("DET_WORKERS", utils.num_workers())))
    det_smoke_images: int = 60
    eval_wider: bool = True            # AP on WIDER FACE val (evaluation only; CC BY-NC-ND, downloaded once)
    wider_images: int | None = None    # None = all 3,226 val photos
    # --- face alignment: detector landmarks -> 5-point template
    crop_size: int = 128
    # --- stage 1: embedders (each: any torchvision ResNet / EfficientNet / ConvNeXt; `pretrained`
    # starts the backbone from torchvision's ImageNet weights, else from random weights)
    embedders: tuple = (
        {"arch": "efficientnet_b2", "pretrained": True, "input_size": 128, "epochs": 25},
        {"arch": "resnet18", "pretrained": False, "input_size": 128, "epochs": 80},
    )
    embedding_dim: int = 512           # per embedder
    batch_size: int = 128
    lr: float = 1e-3                   # neck + CosFace weights; a pretrained backbone gets lr * backbone_lr_mult
    backbone_lr_mult: float = 0.3
    weight_decay: float = 5e-4
    warmup_epochs: int = 2
    cosface_scale: float = 30.0
    cosface_margin: float = 0.35
    # --- stage 2: identity head (MLP on the frozen embeddings)
    hidden_units: int = 256
    dropout: float = 0.3
    head_epochs: int = 200             # the epoch with the lowest validation loss is kept
    head_batch_size: int = 64
    head_lr: float = 1e-3
    head_weight_decay: float = 1e-4
    # --- verification + plots
    verify_grid: tuple = tuple(round(0.2 + 0.025 * i, 3) for i in range(41))   # 0.2 .. 1.2
    tsne_max_points: int = 1500
    # --- smoke run (SMOKE_TEST=1): 12 people, 2 epochs -> outputs/smoke/model
    smoke: bool = field(default_factory=lambda: os.getenv("SMOKE_TEST", "0") == "1")

    def __post_init__(self):
        if self.smoke:
            self.embedders = ({"arch": "resnet18", "pretrained": False, "input_size": 64, "epochs": 2},
                              {"arch": "resnet18", "pretrained": False, "input_size": 64, "epochs": 1})
            self.head_epochs = 20
            self.det_epochs, self.det_size, self.det_batch_size, self.det_lfw_photos = 1, 320, 8, 60
            self.det_val_images, self.det_val_every, self.det_workers, self.eval_wider = 12, 1, 0, False

    @property
    def lfw_home(self) -> Path:
        return utils.DATA_DIR / "sklearn_lfw"

    @property
    def landmark_labels(self) -> Path:
        return utils.TRAINING_DIR / "labels" / "landmark_labels.json.gz"

    @property
    def open_images_dir(self) -> Path:
        return utils.DATA_DIR / "openimages"

    @property
    def wider_dir(self) -> Path:
        return utils.DATA_DIR / "wider_face"

    @property
    def crops_cache(self) -> Path:
        return utils.DATA_DIR / f"lfw_aligned_{self.crop_size}{'_smoke' if self.smoke else ''}.npz"

    @property
    def model_dir(self) -> Path:
        """Where export() writes weights + card. Smoke runs never touch the real model/ repo."""
        return utils.OUTPUTS_DIR / "smoke" / "model" if self.smoke else utils.MODEL_DIR

    def to_dict(self) -> dict:
        return asdict(self)


# AP (IoU 0.5) of the reference detector the landmark labels came from (MTCNN, facenet-pytorch), on
# the same photos with the same evaluation code (detector.evaluate_detections), measured 2026-09-26.
REFERENCE_DETECTOR_AP: dict[str, dict] = {"held-out Open Images": {"large": 0.7382, "medium": 0.7197, "small": 0.7048}, "WIDER FACE val": {"large": 0.8126, "medium": 0.7631, "small": 0.6396}}

# Every embedder that was tried, all with this code's data, alignment, splits and seeds: an MLP head
# (early-stopped on validation) on the frozen embeddings. Chosen by validation accuracy, then
# verification AUC on the unseen people, then size; the test column was not used for any choice.
# Ensembles concatenate the members' embeddings. Recorded 2026-09-26 on an RTX 2080 Ti.
EXPERIMENTS: dict[str, dict] = {
    "ResNet-18 scratch, 42 people only, unaligned": {"train_accuracy": 1.0, "val_accuracy": 0.9281, "test_accuracy": 0.9243, "unseen_auc": 0.8381, "params": 11440192},
    "ResNet-18 scratch, all LFW, unaligned": {"train_accuracy": 1.0, "val_accuracy": 0.976, "test_accuracy": 0.9567, "unseen_auc": 0.9399, "params": 11440192},
    "ResNet-50 ImageNet init, unaligned": {"train_accuracy": 1.0, "val_accuracy": 0.9863, "test_accuracy": 0.9815, "unseen_auc": 0.9404, "params": 24558144},
    "ResNet-18 scratch, aligned 112 px": {"train_accuracy": 1.0, "val_accuracy": 0.9932, "test_accuracy": 0.9753, "unseen_auc": 0.9475, "params": 11440192},
    "ResNet-18 scratch, aligned 128 px": {"train_accuracy": 1.0, "val_accuracy": 0.9932, "test_accuracy": 0.9768, "unseen_auc": 0.9614, "params": 11440192},
    "ResNet-50 ImageNet init, aligned": {"train_accuracy": 1.0, "val_accuracy": 0.9897, "test_accuracy": 0.9892, "unseen_auc": 0.9442, "params": 24558144},
    "ResNet-50 ImageNet init, aligned, 15 epochs": {"train_accuracy": 1.0, "val_accuracy": 0.9863, "test_accuracy": 0.9861, "unseen_auc": 0.9406, "params": 24558144},
    "EfficientNet-B2 ImageNet init, aligned": {"train_accuracy": 1.0, "val_accuracy": 0.9897, "test_accuracy": 0.9845, "unseen_auc": 0.9519, "params": 8423426},
    "Ensemble: ResNet-50 ImageNet init + ResNet-18 scratch": {"train_accuracy": 1.0, "val_accuracy": 0.9966, "test_accuracy": 0.9938, "unseen_auc": 0.9753, "params": 35998336},
    "Ensemble: ResNet-50 ImageNet init + EfficientNet-B2 ImageNet init": {"train_accuracy": 1.0, "val_accuracy": 0.9966, "test_accuracy": 0.9938, "unseen_auc": 0.9631, "params": 32981570},
    "Ensemble: EfficientNet-B2 ImageNet init + ResNet-18 scratch": {"train_accuracy": 1.0, "val_accuracy": 0.9966, "test_accuracy": 0.9954, "unseen_auc": 0.9772, "params": 19863618},
}
