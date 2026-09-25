"""Builds the networks FROM model/model.py (the class the Space deploys) plus a bigram baseline."""
import copy

import torch

import model as M  # ../model/model.py, put on sys.path by src/__init__.py

from . import utils
from .config import Config


def build_gpt(cfg: Config, tok: M.CharTokenizer) -> M.GPT:
    """Fresh, seeded GPT whose vocabulary (written to config.json on export) is `tok`'s."""
    utils.set_seeds(cfg.seed, framework="torch")
    return M.GPT(chars=tok.chars, specials=tok.specials, n_layer=cfg.n_layer, n_head=cfg.n_head,
                 n_embd=cfg.n_embd, block_size=cfg.block_size, dropout=cfg.dropout)


def load_deployed(device) -> tuple[M.GPT | None, dict]:
    """The model deployed right now in ../model (for comparison only), with its metrics.json.

    Returns (None, {}) with a message when the weights are not checked out (e.g. a clone made
    with GIT_LFS_SKIP_SMUDGE=1 holds only an LFS pointer file)."""
    weights = utils.MODEL_DIR / M.WEIGHTS_FILE
    if not weights.is_file() or weights.stat().st_size < 1_000_000:
        print(f"no deployed weights in {utils.MODEL_DIR} (not downloaded): skipping that comparison row")
        return None, {}
    net = M.GPT.from_pretrained(utils.MODEL_DIR, map_location="cpu", strict=True).to(device).eval()
    info = utils.load_json(utils.MODEL_DIR / "metrics.json") if (utils.MODEL_DIR / "metrics.json").is_file() else {}
    return net, info


def snapshot(net: M.GPT) -> M.GPT:
    """Frozen CPU copy (e.g. the stage-A model, kept for the comparison after chat-tuning)."""
    return copy.deepcopy(net).cpu().eval()


class BigramLM:
    """Count-based bigram baseline with add-one smoothing: P(next char | current char)."""

    def __init__(self, vocab_size: int):
        self.counts = torch.ones(vocab_size, vocab_size, dtype=torch.float64)

    def fit(self, ids: torch.Tensor) -> "BigramLM":
        self.counts.index_put_((ids[:-1], ids[1:]), torch.ones(len(ids) - 1, dtype=torch.float64),
                               accumulate=True)
        return self

    def loss(self, ids: torch.Tensor) -> float:
        """Mean cross-entropy (nats/char) of predicting ids[1:] from ids[:-1]."""
        log_p = (self.counts / self.counts.sum(1, keepdim=True)).log()
        return float(-log_p[ids[:-1], ids[1:]].mean())
