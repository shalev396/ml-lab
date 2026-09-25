"""Builds the trainable network from model/model.py (the class the Space deploys) + a trivial baseline."""
import numpy as np

import model as M  # ../model/model.py, put on sys.path by src/__init__.py

from . import utils
from .config import Config


def build_model(cfg: Config, age_representatives: list[float]) -> M.AgeGenderNet:
    """EfficientNet-B2 trunk initialised from ImageNet + fresh age/gender heads.
    With `cfg.freeze_epochs > 0` the trunk starts frozen (see `engine.set_trunk_trainable`)."""
    utils.set_seeds(cfg.seed, framework="torch")
    net = M.AgeGenderNet(
        age_bin_edges=M.DEFAULT_AGE_BIN_EDGES,
        age_labels=M.age_labels_for(M.DEFAULT_AGE_BIN_EDGES),
        age_representatives=age_representatives,
        gender_labels=M.DEFAULT_GENDER_LABELS,
        dropout=cfg.dropout,
        pretrained=True,
    )
    if cfg.freeze_epochs > 0:
        for p in net.features.parameters():
            p.requires_grad = False
    return net


def baseline_predictions(train_ages: np.ndarray, train_genders: np.ndarray, n: int) -> dict:
    """No-learning reference: every face gets the median training age and the majority gender."""
    majority = int(np.bincount(train_genders).argmax())
    return {"age_pred": np.full(n, float(np.median(train_ages))), "gender_pred": np.full(n, majority)}
