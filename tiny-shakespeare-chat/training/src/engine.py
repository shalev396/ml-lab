"""Training loop (AdamW, warmup + cosine LR, grad clip, AMP on CUDA), deterministic evaluation
and chat sampling. Device-agnostic: everything runs on whatever `device` is passed in."""
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

import model as M  # ../model/model.py, put on sys.path by src/__init__.py

from . import utils
from .data_setup import get_batch


def make_optimizer(net: torch.nn.Module, lr: float, weight_decay: float,
                   betas: tuple[float, float]) -> torch.optim.AdamW:
    """AdamW with weight decay on matrices/embeddings only (not on biases or LayerNorm gains)."""
    params = [p for p in net.parameters() if p.requires_grad]
    groups = [{"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
              {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


def get_lr(it: int, max_lr: float, min_lr: float, warmup_iters: int, max_iters: int) -> float:
    """Linear warmup, then cosine decay to `min_lr`."""
    if it < warmup_iters:
        return max_lr * (it + 1) / max(1, warmup_iters)
    if it >= max_iters:
        return min_lr
    ratio = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (max_lr - min_lr)


@torch.inference_mode()
def estimate_loss(net, splits: dict[str, torch.Tensor], eval_iters: int, block_size: int,
                  batch_size: int, device) -> dict[str, float]:
    """Mean loss over `eval_iters` random batches per split (the cheap in-training estimate)."""
    net.eval()
    out = {}
    for name, data in splits.items():
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            with utils.autocast(device):
                _, loss = net(x, y)
            losses.append(loss.item())
        out[name] = float(np.mean(losses))
    net.train()
    return out


def train_stage(net, train_ids: torch.Tensor, val_ids: torch.Tensor, *, iters: int, lr: float,
                min_lr: float, warmup_iters: int, cfg, device, stage: str) -> dict:
    """Train one stage (pretrain or chat-tune) in place. Returns its history:
    {"iter", "train_loss", "val_loss", "lr", "time_s"}; losses are estimated every eval_interval."""
    optimizer = make_optimizer(net, lr, cfg.weight_decay, (cfg.beta1, cfg.beta2))
    scaler = utils.grad_scaler(device)
    history = {"iter": [], "train_loss": [], "val_loss": [], "lr": []}
    splits = {"train": train_ids, "val": val_ids}
    net.train()
    start = time.perf_counter()

    def log(it: int, lr_now: float) -> None:
        losses = estimate_loss(net, splits, cfg.eval_iters, cfg.block_size, cfg.batch_size, device)
        for key, value in (("iter", it), ("train_loss", losses["train"]), ("val_loss", losses["val"]),
                           ("lr", lr_now)):
            history[key].append(value)
        print(f"[{stage}] iter {it:5d}/{iters} | train {losses['train']:.4f} | val {losses['val']:.4f} "
              f"| lr {lr_now:.2e} | {time.perf_counter() - start:.0f}s")

    for it in range(iters):
        lr_now = get_lr(it, lr, min_lr, warmup_iters, iters)
        for group in optimizer.param_groups:
            group["lr"] = lr_now
        if it % cfg.eval_interval == 0:
            log(it, lr_now)
        x, y = get_batch(train_ids, cfg.block_size, cfg.batch_size, device)
        with utils.autocast(device):
            _, loss = net(x, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
    log(iters, get_lr(iters, lr, min_lr, warmup_iters, iters))
    history["time_s"] = round(time.perf_counter() - start, 1)
    net.eval()
    return history


@torch.inference_mode()
def evaluate(net, ids: torch.Tensor, device, max_tokens: int = 0, batch_size: int = 32) -> dict:
    """Deterministic loss over a split: non-overlapping windows of `block_size` chars, every
    target scored once. Returns {"loss" (nats/char), "bpc" (bits/char), "ppl", "n_tokens",
    "loss_by_position" (mean loss at each context position 0..block_size-1)}."""
    net.eval()
    T = net.block_size
    if max_tokens:
        ids = ids[:max_tokens + 1]
    n_windows = (len(ids) - 1) // T
    if n_windows < 1:
        raise ValueError(f"split too small ({len(ids)} tokens) for block_size={T}")
    x_all = ids[:n_windows * T].view(n_windows, T)
    y_all = ids[1:n_windows * T + 1].view(n_windows, T)
    per_token = []
    for i in range(0, n_windows, batch_size):
        x, y = x_all[i:i + batch_size].to(device), y_all[i:i + batch_size].to(device)
        with utils.autocast(device):
            logits, _ = net(x)
        per_token.append(F.cross_entropy(logits.float().transpose(1, 2), y, reduction="none").cpu())
    losses = torch.cat(per_token)                                   # (n_windows, T)
    loss = float(losses.mean())
    return {"loss": loss, "bpc": loss / math.log(2), "ppl": math.exp(loss), "n_tokens": int(losses.numel()),
            "loss_by_position": losses.mean(0).tolist()}


def reply(net: M.GPT, message: str, history=None, *, max_new_tokens: int = M.MAX_NEW_TOKENS,
          temperature: float = M.TEMPERATURE, top_k: int = M.TOP_K, seed: int | None = None) -> str:
    """Chat reply from an in-memory network, with exactly the prompt format + sampler of model.py."""
    tok = net.tokenizer()
    ids = tok.encode(M.build_prompt(message, history))[-(net.block_size - M.REPLY_ROOM):]
    device = next(net.parameters()).device
    gen = None if seed is None else torch.Generator(device=device.type).manual_seed(seed)
    out = M.sample(net.eval(), ids, max_new_tokens=max_new_tokens, temperature=temperature,
                   top_k=top_k, stop_id=tok.end_id, generator=gen)
    return tok.decode(out).strip()


def continue_text(net: M.GPT, prompt: str, *, max_new_tokens: int = 200, temperature: float = 0.8,
                  top_k: int = M.TOP_K, seed: int | None = None) -> str:
    """Raw continuation (no chat format, no stop token): shows what stage A learned."""
    tok = net.tokenizer()
    device = next(net.parameters()).device
    gen = None if seed is None else torch.Generator(device=device.type).manual_seed(seed)
    out = M.sample(net.eval(), tok.encode(prompt)[-net.block_size:], max_new_tokens=max_new_tokens,
                   temperature=temperature, top_k=top_k, generator=gen)
    return prompt + tok.decode(out)
