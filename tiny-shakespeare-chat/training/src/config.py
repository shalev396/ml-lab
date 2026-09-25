"""Every tunable of the Tiny Shakespeare Chat training run, in one dataclass.

Defaults reproduce the run that produced the deployed weights: a 6-layer / 6-head / 384-d
character GPT (block 256, dropout 0.2) pretrained for 5,000 iterations on the raw play, then
chat-tuned for 2,000 iterations on consecutive dialogue-line pairs.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

from . import utils

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
# The TensorFlow text-generation tutorial serves the byte-identical file (1,115,394 chars).
DATA_FALLBACK_URL = "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"


@dataclass
class Config:
    # --- data
    data_url: str = DATA_URL
    data_fallback_url: str = DATA_FALLBACK_URL
    corpus_file: str = "tinyshakespeare.txt"     # cached under training/data/
    val_frac: float = 0.10                       # stage A: last 10 % of the play
    chat_val_frac: float = 0.05                  # stage B: last 5 % of the chat-formatted stream
    # --- model (6L / 6H / 384d -> 10,751,232 parameters with the 68-token vocabulary)
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256
    dropout: float = 0.2
    # --- stage A: pretrain on the raw play (next-character prediction)
    batch_size: int = 64
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 100
    max_iters: int = 5000
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eval_interval: int = 250                     # iterations between loss estimates
    eval_iters: int = 40                         # random batches per loss estimate
    # --- stage B: chat-tune on "<|user|> line <|end|> <|bot|> next line <|end|>" pairs
    chat_lr: float = 1e-4
    chat_min_lr: float = 1e-5
    chat_warmup_iters: int = 50
    chat_iters: int = 2000
    # --- evaluation: deterministic pass over the val splits (0 = every token)
    eval_max_tokens: int = 0
    # --- generation (defaults of model.py; used for the notebook samples)
    temperature: float = 0.8
    top_k: int = 40
    max_new_tokens: int = 200
    seed: int = 42
    # --- run mode (SMOKE_TEST=1): tiny model, a few hundred steps, exports to outputs/smoke/model
    smoke: bool = field(default_factory=lambda: os.getenv("SMOKE_TEST", "0") == "1")

    def __post_init__(self):
        if self.smoke:
            self.n_layer, self.n_head, self.n_embd, self.block_size, self.dropout = 2, 2, 64, 64, 0.1
            self.batch_size = 32
            self.max_iters, self.warmup_iters = 200, 10
            self.chat_iters, self.chat_warmup_iters = 60, 5
            self.eval_interval, self.eval_iters = 50, 8
            self.eval_max_tokens = self.eval_max_tokens or 8192
            self.max_new_tokens = 120

    @property
    def corpus_path(self) -> Path:
        return utils.DATA_DIR / self.corpus_file

    @property
    def run_dir(self) -> Path:
        """Scratch folder for this run (staged weights for the inference check, plots)."""
        return utils.OUTPUTS_DIR / ("smoke" if self.smoke else "run")

    @property
    def model_dir(self) -> Path:
        """Where export() writes weights + card. Smoke runs never touch the real model/ repo."""
        return utils.OUTPUTS_DIR / "smoke" / "model" if self.smoke else utils.MODEL_DIR
