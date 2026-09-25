"""Builds networks FROM model/model.py (the same class the Space deploys).

- `build_model`: a fresh FoodVisionNet with the ImageNet backbone frozen, ready to train.
- `load_deployed`: the checkpoint currently in model/ (or on the Hub), evaluated as a baseline
  variant so a new run only replaces it when it is actually better.
"""
from pathlib import Path

import torch

import model as M  # ../model/model.py, put on sys.path by src/__init__.py

from . import utils
from .config import Config

LFS_POINTER_MAX_BYTES = 1024   # a git-lfs pointer file (GIT_LFS_SKIP_SMUDGE=1 checkout) is ~130 bytes


def build_model(cfg: Config, class_names: list[str]) -> M.FoodVisionNet:
    """EfficientNet-B2 feature extractor: ImageNet backbone frozen, only the new head trains.

    Seeding first makes the random head initialisation reproducible.
    """
    utils.set_seeds(cfg.seed, framework="torch")
    net = M.FoodVisionNet(class_names=class_names, dropout=cfg.dropout, pretrained=True)
    for param in net.backbone.features.parameters():
        param.requires_grad = False
    return net


def trainable_params(net) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def deployed_dir() -> Path | None:
    """Folder holding the real deployed weights: ../model, or a Hub download when model/ only has
    code (a fresh Colab clone fetches the submodule with GIT_LFS_SKIP_SMUDGE=1). None if neither."""
    weights = utils.MODEL_DIR / M.WEIGHTS_FILE
    if weights.is_file() and weights.stat().st_size > LFS_POINTER_MAX_BYTES:
        return utils.MODEL_DIR
    try:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(M.REPO_ID, allow_patterns=[M.WEIGHTS_FILE, "config.json", "metrics.json"]))
        print(f"model/ has no weights locally -> downloaded the deployed checkpoint from {M.REPO_ID}")
        return path
    except Exception as err:  # offline / repo not published yet
        print(f"no deployed checkpoint to compare against ({type(err).__name__}: {err})")
        return None


def load_deployed(device) -> tuple[M.FoodVisionNet | None, Path | None, dict]:
    """(net, folder, its metrics.json) of the deployed checkpoint, or (None, None, {}) if there is none."""
    folder = deployed_dir()
    if folder is None:
        return None, None, {}
    device = torch.device(device)
    net = M.FoodVisionNet.from_pretrained(folder, pretrained=False, map_location=device.type, strict=True)
    info = utils.load_json(folder / "metrics.json") if (folder / "metrics.json").is_file() else {}
    return net.to(device).eval(), folder, info
