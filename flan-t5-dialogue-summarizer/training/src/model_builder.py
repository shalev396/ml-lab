"""Build the trainable model: FLAN-T5 base weights (frozen) + LoRA adapters on the attention q/v projections."""
from __future__ import annotations

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .config import Config


def load_tokenizer(cfg: Config):
    return AutoTokenizer.from_pretrained(cfg.base_model)


def load_base(cfg: Config, device) -> torch.nn.Module:
    """The pretrained FLAN-T5 in float32 (T5 overflows in fp16; bf16 autocast is applied only while training)."""
    return AutoModelForSeq2SeqLM.from_pretrained(cfg.base_model, dtype=torch.float32).to(device)


def lora_config(cfg: Config) -> LoraConfig:
    return LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      target_modules=list(cfg.lora_target_modules), bias="none",
                      task_type=TaskType.SEQ_2_SEQ_LM)


def add_lora(model: torch.nn.Module, cfg: Config):
    """Freeze the base weights and insert rank-r adapters; only the adapters are trained and saved."""
    return get_peft_model(model, lora_config(cfg))


def count_params(model: torch.nn.Module) -> dict[str, int]:
    """Trainable (LoRA) vs total parameters. Merging the adapter later adds no parameters to the base."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    lora = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    return {"trainable": trainable, "total": total, "base": total - lora}
