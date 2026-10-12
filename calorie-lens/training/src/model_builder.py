"""Build the networks: the deployed CalorieLens (model/model.py) and what only training needs.

  build_model()     CalorieLens with the dataset's ingredients + per-gram table, the pretrained
                    backbone, and output biases set from the training data (presence frequency, median
                    grams, mean totals), so the first steps already predict a plausible plate
  DepthHead         auxiliary metric-depth decoder on the image tokens (experiment E3; overhead
                    scans only). Lives in TrainNet, not in CalorieLens, so the exported model and
                    `from_pretrained(strict=True)` never see it
  TrainNet          CalorieLens + optional DepthHead
  param_groups()    AdamW groups: layer-wise learning-rate decay over the backbone stages, the head
                    rate for everything new, no weight decay on norms / biases / queries
  pick_micro_batch  64 on >= 70 GB GPUs, else 32 (with gradient accumulation to the same batch)
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

import model as M


def nutritext_kwargs(cfg, vocab_size: int) -> dict:
    return {"vocab_size": int(vocab_size), "d": cfg.nt_d, "layers": cfg.nt_layers, "heads": cfg.nt_heads,
            "max_len": cfg.nt_max_len}


def build_model(cfg, labels, pretrained: bool | None = None, nutritext: dict | None = None) -> M.CalorieLens:
    """A fresh CalorieLens for `cfg.backbone` (+ an untrained NutriText of the given shape)."""
    pretrained = cfg.pretrained if pretrained is None else pretrained
    ing = labels.ingredients
    train = labels.split == "train"
    dens = labels.densities
    mean_density = (labels.grams[train].sum(0) @ dens) / max(labels.grams[train].sum(), 1.0)
    backbone_config = None
    if cfg.backbone == "dinov2":
        from transformers import AutoConfig

        backbone_config = AutoConfig.from_pretrained(cfg.backbone_weights or M.DINO_BASE, drop_path_rate=cfg.drop_path).to_dict()
    net = M.CalorieLens(
        ingredients=list(ing["name"]), ingredient_ids=[int(i) for i in ing["ingr_id"]], densities=dens.tolist(),
        density_sources=list(ing["source"]), mean_density=[float(v) for v in mean_density],
        backbone=cfg.backbone, backbone_config=backbone_config, image_size=list(cfg.image_size),
        d_model=cfg.d_model, decoder_layers=cfg.decoder_layers, heads=cfg.heads, drop_path=cfg.drop_path,
        backbone_weights=cfg.backbone_weights, multi_pool=cfg.multi_pool, nutritext=nutritext, pretrained=pretrained)
    init_output_biases(net, labels)
    return net


@torch.no_grad()
def init_output_biases(net: M.CalorieLens, labels) -> None:
    """Presence bias = logit(train frequency), grams bias = log(median grams when present), totals head
    = log1p(train mean); the last totals layer starts at zero so it outputs exactly that mean."""
    train = labels.split == "train"
    g = labels.grams[train]
    freq = np.clip((g > 0).mean(0), 1e-3, 0.5)
    med = np.array([np.median(c[c > 0]) if (c > 0).any() else 10.0 for c in g.T])
    net.decoder.presence_b.copy_(torch.from_numpy(np.log(freq / (1 - freq))).float())
    net.decoder.grams_b.copy_(torch.from_numpy(np.log(np.clip(med, 0.5, None))).float())
    last = net.totals_head[-1]
    nn.init.zeros_(last.weight)
    last.bias.copy_(torch.from_numpy(np.log1p(labels.totals[train].mean(0))).float())


class DepthHead(nn.Module):
    """Image tokens (N, L, d) on an (h, w) grid -> metric depth at 4x the token resolution."""

    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(d, 128, 3, padding=1), nn.GELU(), nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 64, 3, padding=1), nn.GELU(), nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(64, 1, 3, padding=1))
        nn.init.constant_(self.net[-1].bias, float(np.log(0.36)))     # ~ camera-to-plate distance

    def forward(self, tokens: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        n, _, d = tokens.shape
        fmap = tokens.transpose(1, 2).reshape(n, d, *grid)
        with torch.autocast(tokens.device.type, enabled=False):
            return torch.exp(self.net(fmap.float())[:, 0])           # metres, always positive (fp32)


class TrainNet(nn.Module):
    """The deployed net + training-only heads. forward(x) -> CalorieLens outputs (+ "depth")."""

    def __init__(self, net: M.CalorieLens, depth_aux: bool, d_model: int):
        super().__init__()
        self.net = net
        self.depth_head = DepthHead(d_model) if depth_aux else None

    def forward(self, x: torch.Tensor) -> dict:
        out = self.net(x, return_features=self.depth_head is not None)
        if self.depth_head is not None:
            out["depth"] = self.depth_head(out.pop("tokens"), out["grid"])
        return out


def _no_decay(name: str, p: nn.Parameter) -> bool:
    return p.ndim <= 1 or name.endswith("queries") or "norm" in name or "embeddings" in name.split(".")[-2:]


def param_groups(tnet: TrainNet, cfg) -> list[dict]:
    """Backbone stage i of n gets lr_backbone * layer_decay ** (n - 1 - i); the rest gets lr_head.
    NutriText is not trained here (it has its own stage) and is left out."""
    stages = tnet.net.backbone.stages()
    owner = {}
    for i, stage in enumerate(stages):
        for p in stage.parameters():
            owner[id(p)] = i
    groups: dict[tuple, dict] = {}
    for name, p in tnet.named_parameters():
        if not p.requires_grad or name.startswith("net.nutritext."):
            continue
        i = owner.get(id(p))
        lr = cfg.lr_head if i is None else cfg.lr_backbone * cfg.layer_decay ** (len(stages) - 1 - i)
        wd = 0.0 if _no_decay(name, p) else cfg.weight_decay
        groups.setdefault((lr, wd), {"params": [], "lr": lr, "weight_decay": wd})["params"].append(p)
    return list(groups.values())


def pick_micro_batch(cfg, device) -> int:
    """Per-step batch: explicit cfg.micro_batch, else 64 on a >= 70 GB GPU, 32 below (A100 40 GB, L4)."""
    if cfg.micro_batch:
        return int(cfg.micro_batch)
    if device.type != "cuda":
        return min(cfg.batch_size, 8)
    vram = torch.cuda.get_device_properties(device).total_memory / 2 ** 30
    return min(cfg.batch_size, 64 if vram >= 70 else 32)


def maybe_compile(module: nn.Module, cfg, device) -> nn.Module:
    """torch.compile on CUDA when cfg.use_compile; the caller falls back to eager if the first step fails."""
    if device.type != "cuda" or not cfg.use_compile:
        return module
    return torch.compile(module)
