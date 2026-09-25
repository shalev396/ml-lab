"""Builds the networks: the deployed DistilBERT classifier (from model/model.py) and the baseline."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from transformers import AutoConfig, AutoTokenizer

import model as M  # ../model/model.py, put on sys.path by src/__init__.py

from . import utils
from .config import Config


def load_tokenizer(cfg: Config):
    """The base model's WordPiece tokenizer (saved next to the weights at export)."""
    return AutoTokenizer.from_pretrained(cfg.base_model)


def build_model(cfg: Config, pretrained: bool = True) -> M.SpamClassifier:
    """DistilBERT + new head with every encoder weight frozen except the last `unfreeze_last_n` blocks.

    `backbone_config` is stored in config.json so inference rebuilds the encoder offline.
    Seeding first makes the head initialisation reproducible.
    """
    utils.set_seeds(cfg.seed, framework="torch")
    backbone_config = AutoConfig.from_pretrained(cfg.base_model).to_dict()
    net = M.SpamClassifier(base_model=cfg.base_model, backbone_config=backbone_config, dropout=cfg.dropout,
                           max_len=cfg.max_len, threshold=cfg.threshold, labels=M.LABELS, pretrained=pretrained)
    for param in net.backbone.parameters():
        param.requires_grad = False
    if cfg.unfreeze_last_n > 0:
        for block in net.encoder_layers()[-cfg.unfreeze_last_n:]:
            for param in block.parameters():
                param.requires_grad = True
    return net


def param_groups(net: M.SpamClassifier, cfg: Config) -> list[dict]:
    """AdamW groups with differential learning rates: new head (fast) vs unfrozen encoder blocks (slow)."""
    head = [p for p in net.classifier.parameters() if p.requires_grad]
    backbone = [p for p in net.backbone.parameters() if p.requires_grad]
    groups = [{"params": head, "lr": cfg.lr_head}]
    if backbone:
        groups.append({"params": backbone, "lr": cfg.lr_backbone})
    return groups


def trainable_params(net) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def build_baseline(cfg: Config) -> Pipeline:
    """Baseline experiment: word 1-2-gram TF-IDF + logistic regression (class-balanced)."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, cfg.baseline_ngram_max), max_features=cfg.baseline_max_features,
                                  sublinear_tf=True, min_df=2 if not cfg.smoke else 1)),
        ("clf", LogisticRegression(C=cfg.baseline_C, max_iter=2000, class_weight="balanced",
                                   random_state=cfg.seed)),
    ])
