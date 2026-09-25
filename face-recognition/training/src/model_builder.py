"""Builds every model of the project FROM ../model/model.py.

* `FaceDetector`: our face + 5-landmark detector, trained from random weights (src/detector.py).
* `FaceAligner`: the detector + the 5-point alignment.
* `FaceRecognizer`: the ensemble of embedders + the identity head; each embedder's backbone starts
  from torchvision's ImageNet weights (`pretrained`) or from random weights.
"""
from __future__ import annotations

import hashlib

import torch

import model as M

from . import utils
from .config import Config


# --------------------------------------------------------------------------- detector
def build_detector(device) -> M.FaceDetector:
    return M.FaceDetector().to(device)


def detector_id(det: M.FaceDetector) -> str:
    """Short fingerprint of the detector weights (keys the alignment cache)."""
    h = hashlib.sha1()
    for v in det.state_dict().values():
        h.update(v.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()[:12]


def build_aligner(cfg: Config, det: M.FaceDetector, selection: str = "center") -> M.FaceAligner:
    """Detector + 5-point alignment. LFW is labelled by the centered person -> "center" selection."""
    return M.FaceAligner(None, crop_size=cfg.crop_size, selection=selection, detector=det)


# --------------------------------------------------------------------------- embedders + head
def build_recognizer(cfg: Config, class_names: list[str], verify_threshold: float = 0.5) -> M.FaceRecognizer:
    """model.FaceRecognizer with one member per cfg.embedders entry; pretrained members start from
    torchvision's ImageNet weights."""
    members = [{"arch": e["arch"], "input_size": e["input_size"]} for e in cfg.embedders]
    net = M.FaceRecognizer(class_names=class_names, members=members, embedding_dim=cfg.embedding_dim,
                           hidden_units=cfg.hidden_units, dropout=cfg.dropout, crop_size=cfg.crop_size,
                           verify_threshold=verify_threshold)
    for member, spec in zip(net.members, cfg.embedders):
        if spec["pretrained"]:
            imagenet, _ = M.build_backbone(spec["arch"], pretrained=True)
            member.backbone.load_state_dict(imagenet.state_dict())
    return net


def member_name(spec: dict) -> str:
    return f"{spec['arch']} ({'ImageNet init' if spec['pretrained'] else 'from scratch'}, {spec['input_size']} px)"


def count_params(module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def pipeline_params(det: M.FaceDetector, net: M.FaceRecognizer) -> dict[str, int]:
    """Parameter counts of the detector, each embedder, the identity head and the whole pipeline."""
    counts = {"detector": count_params(det)}
    for i, member in enumerate(net.members):
        counts[f"embedder_{i + 1}_{member.arch}"] = count_params(member)
    counts["embedders"] = sum(count_params(m) for m in net.members)
    counts["head"] = count_params(net.head)
    counts["total"] = counts["detector"] + counts["embedders"] + counts["head"]
    return counts
