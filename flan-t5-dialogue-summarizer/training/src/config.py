"""Every tunable of the flan-t5-dialogue-summarizer run in one dataclass."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import utils


@dataclass
class Config:
    # --- models ---------------------------------------------------------------------------------
    base_model: str = "google/flan-t5-base"        # 247.6M parameters, the deployed base
    smoke_base_model: str = "google/flan-t5-small"  # 77M parameters, smoke runs only
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- data: knkarthick/dialogsum (train 12,460 / validation 500 / test 500 dialogues) -----------
    seed: int = 42
    train_samples: int = 12460        # the whole train split (a smaller number = seeded subset)
    val_samples: int = 500            # the whole validation split: loss after every epoch, picks the checkpoint
    test_samples: int = 500           # all unique test dialogues (3 human summaries each): ROUGE, reported once
    max_input_length: int = 512       # prompt tokens (2 % of DialogSum train prompts are longer)
    max_target_length: int = 128      # summary tokens (99th percentile ~96)

    # --- LoRA (PEFT) ----------------------------------------------------------------------------
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q", "v")   # T5 attention query / value projections

    # --- optimisation ---------------------------------------------------------------------------
    epochs: int = 6                   # upper bound: early stopping usually ends the run sooner
    patience: int = 2                 # stop after this many epochs without a new best validation loss
    lr: float = 1e-3                  # LoRA tunes 0.7 % of the weights, so a high lr works
    batch_size: int = 8
    grad_accum_steps: int = 1
    weight_decay: float = 0.01
    warmup_ratio: float = 0.0         # linear decay from lr to 0
    max_grad_norm: float = 1.0
    log_every: int = 25               # steps between training-loss log points
    resume: bool = True               # continue from the last epoch checkpoint of an identical run

    # --- evaluation / generation ----------------------------------------------------------------
    eval_batch_size: int = 16
    max_new_tokens: int = 96
    num_beams: int = 1                # greedy: fast and identical for every variant
    few_shot_k: int = 2               # solved examples in the few-shot prompt
    shot_max_chars: int = 450         # in-context examples are short train dialogues
    shot_max_input_length: int = 1024  # one/few-shot prompts are longer than the zero-shot one
    rag_top_k: int = 3
    rag_max_new_tokens: int = 64

    # --- run ------------------------------------------------------------------------------------
    smoke: bool = field(default_factory=lambda: os.getenv("SMOKE_TEST", "0") == "1")

    def __post_init__(self):
        self.lora_target_modules = tuple(self.lora_target_modules)
        if self.smoke:  # same code path, tiny model + data: finishes in a few minutes on a CPU
            self.base_model = self.smoke_base_model
            self.train_samples = min(self.train_samples, 64)
            self.val_samples = min(self.val_samples, 16)
            self.test_samples = min(self.test_samples, 12)
            self.max_input_length = min(self.max_input_length, 256)
            self.max_target_length = min(self.max_target_length, 64)
            self.shot_max_input_length = min(self.shot_max_input_length, 512)
            self.epochs = 1
            self.log_every = 2
            self.max_new_tokens = min(self.max_new_tokens, 48)

    @property
    def model_dir(self) -> Path:
        """Where export writes the model repo; smoke runs never touch the real model/."""
        return utils.OUTPUTS_DIR / "smoke" / "model" if self.smoke else utils.MODEL_DIR

    @property
    def outputs_dir(self) -> Path:
        return utils.OUTPUTS_DIR / "smoke" if self.smoke else utils.OUTPUTS_DIR

    @property
    def checkpoint_dir(self) -> Path:
        return self.outputs_dir / "checkpoints"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["lora_target_modules"] = list(self.lora_target_modules)
        return d
