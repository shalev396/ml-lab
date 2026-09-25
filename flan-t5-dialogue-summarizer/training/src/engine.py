"""Training loop (LoRA, AdamW, linear decay, resumable per epoch) and evaluation (ROUGE, prompting, RAG)."""
from __future__ import annotations

import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from rouge_score import rouge_scorer
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

import model as M  # ../model/model.py

from . import utils
from .config import Config
from .data_setup import Splits, few_shot_prompt

ROUGE_KEYS = ("rouge1", "rouge2", "rougeL")


# --------------------------------------------------------------------------- precision
def train_autocast(device):
    """bf16 autocast on GPUs that support it, full fp32 everywhere else.

    T5 overflows in fp16 (the loss is NaN from the first step), so fp16 is never used, even on GPUs
    where utils.amp_dtype would pick it (e.g. a Colab T4, which trains in fp32)."""
    return utils.autocast(device) if utils.amp_dtype(device) == torch.bfloat16 else nullcontext()


def precision_name(device) -> str:
    return "bf16 autocast" if utils.amp_dtype(device) == torch.bfloat16 else "fp32"


# --------------------------------------------------------------------------- training
def _run_key(cfg: Config) -> dict:
    """Settings that must match for a checkpoint to be resumed."""
    keys = ("base_model", "seed", "train_samples", "max_input_length", "max_target_length", "lora_r",
            "lora_alpha", "lora_dropout", "epochs", "lr", "batch_size", "grad_accum_steps", "weight_decay")
    d = cfg.to_dict()
    return {k: d[k] for k in keys} | {"lora_target_modules": list(cfg.lora_target_modules)}


@torch.no_grad()
def evaluate_loss(model, loader, device) -> float:
    """Mean token-level cross-entropy on a loader (teacher forcing)."""
    model.eval()
    total, tokens = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with train_autocast(device):
            loss = model(**batch).loss
        n = int((batch["labels"] != -100).sum())
        total += float(loss) * n
        tokens += n
    return total / max(tokens, 1)


def train(model, train_loader, val_loader, cfg: Config, device) -> dict:
    """Fine-tune the LoRA adapters. Returns the history: training loss every `log_every` steps and the
    validation loss after every epoch (epoch 0 = before training). After each epoch the adapter and the
    optimizer state are saved to cfg.checkpoint_dir, so an interrupted run continues where it stopped.

    Checkpoint selection: the adapter of the epoch with the lowest validation loss is kept (history
    "best_epoch"), and training stops early after `cfg.patience` epochs without a new best. The model
    returned holds that best adapter, not the last one, so a late overfitting epoch is never exported."""
    utils.set_seeds(cfg.seed, "torch")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum_steps)
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg.warmup_ratio * total_steps), total_steps)
    history = {"step": [], "train_loss": [], "epoch": [], "val_loss": [], "epoch_time_s": [],
               "steps_per_epoch": steps_per_epoch, "precision": precision_name(device), "best_epoch": 0}
    start_epoch = 0
    best_adapter = None               # lowest-validation-loss adapter so far (None = untrained model)

    ckpt = cfg.checkpoint_dir / "last.pt"
    if cfg.resume and ckpt.is_file():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if state["run"] == _run_key(cfg):
            set_peft_model_state_dict(model, state["adapter"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            history, start_epoch = state["history"], state["epoch"]
            best_adapter = state.get("best_adapter")
            torch.set_rng_state(state["rng"])
            print(f"[train] resumed after epoch {start_epoch} from {ckpt}")
        else:
            print(f"[train] ignoring {ckpt}: it belongs to a run with different settings")

    if start_epoch == 0:
        history["epoch"].append(0)
        history["val_loss"].append(evaluate_loss(model, val_loader, device))
        print(f"[train] epoch 0 (before training): val loss {history['val_loss'][-1]:.4f}")

    step = start_epoch * steps_per_epoch
    for epoch in range(start_epoch + 1, cfg.epochs + 1):
        if epoch - history["best_epoch"] > cfg.patience:
            print(f"[train] early stop: no new best validation loss for {cfg.patience} epochs")
            break
        model.train()
        t0, running, count = time.perf_counter(), 0.0, 0
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}", leave=False)
        for i, batch in enumerate(bar, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with train_autocast(device):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step + 1}: {float(loss)}")
            (loss / cfg.grad_accum_steps).backward()
            running, count = running + loss.item(), count + 1
            if i % cfg.grad_accum_steps == 0 or i == len(train_loader):
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % cfg.log_every == 0 or i == len(train_loader):
                    history["step"].append(step)
                    history["train_loss"].append(running / count)
                    bar.set_postfix(loss=f"{running / count:.3f}")
                    running, count = 0.0, 0
        epoch_time = time.perf_counter() - t0
        val_loss = evaluate_loss(model, val_loader, device)
        history["epoch"].append(epoch)
        history["val_loss"].append(val_loss)
        history["epoch_time_s"].append(epoch_time)
        if val_loss < min(history["val_loss"][:-1]):
            history["best_epoch"] = epoch
            best_adapter = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
        print(f"[train] epoch {epoch}: train loss {history['train_loss'][-1]:.4f} · val loss {val_loss:.4f} · "
              f"{epoch_time:.0f} s ({epoch_time / steps_per_epoch:.2f} s/step)"
              + (" · new best" if history["best_epoch"] == epoch else f" · best is epoch {history['best_epoch']}"))
        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"run": _run_key(cfg), "epoch": epoch, "adapter": get_peft_model_state_dict(model),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "history": history, "rng": torch.get_rng_state(), "best_adapter": best_adapter}, ckpt)
        if epoch - history["best_epoch"] >= cfg.patience and epoch < cfg.epochs:
            print(f"[train] early stop: no new best validation loss for {cfg.patience} epochs")
            break

    if best_adapter is None:
        raise RuntimeError("validation loss never improved on the untrained model; check the data and lr")
    set_peft_model_state_dict(model, best_adapter)
    print(f"[train] kept epoch {history['best_epoch']} (lowest val loss "
          f"{history['val_loss'][history['epoch'].index(history['best_epoch'])]:.4f})")

    history["train_time_s"] = sum(history["epoch_time_s"])   # includes epochs of a resumed run
    history["seconds_per_step"] = history["train_time_s"] / max(len(history["epoch_time_s"]) * steps_per_epoch, 1)
    model.eval()
    return history


# --------------------------------------------------------------------------- ROUGE
_SCORER = rouge_scorer.RougeScorer(list(ROUGE_KEYS), use_stemmer=True)


def rouge_per_example(predictions: list[str], references: list[list[str]]) -> dict[str, np.ndarray]:
    """F-measure of every prediction, averaged over its human references (DialogSum test: 3 each)."""
    out = {k: np.zeros(len(predictions)) for k in ROUGE_KEYS}
    for i, (pred, refs) in enumerate(zip(predictions, references)):
        scores = [_SCORER.score(ref, pred) for ref in refs]
        for k in ROUGE_KEYS:
            out[k][i] = np.mean([s[k].fmeasure for s in scores])
    return out


def rouge(predictions: list[str], references: list[list[str]]) -> dict[str, float]:
    per = rouge_per_example(predictions, references)
    return {k: round(float(v.mean()), 6) for k, v in per.items()} | {
        "gen_len_words": round(float(np.mean([len(p.split()) for p in predictions])), 2)}


# --------------------------------------------------------------------------- variants
def summarize(model, tokenizer, prompts: list[str], cfg: Config, max_input_length: int | None = None) -> list[str]:
    """The same batched generation the Predictor runs (model.generate), with this run's settings."""
    return M.generate(model, tokenizer, prompts, max_input_length=max_input_length or cfg.max_input_length,
                      max_new_tokens=cfg.max_new_tokens, num_beams=cfg.num_beams, batch_size=cfg.eval_batch_size)


def evaluate_variants(peft_model, tokenizer, splits: Splits, shots: list[tuple[str, str]], cfg: Config) -> dict:
    """ROUGE on the test subset for every variant. The base model is the same network with the adapter
    switched off (peft `disable_adapter`), so no second copy is loaded.

    Returns {variant: {"rouge1", "rouge2", "rougeL", "gen_len_words", "predictions", "per_example_rougeL",
    "seconds"}}."""
    dialogues, refs = list(splits.test["dialogue"]), list(splits.test["references"])
    zero = [M.build_prompt(d) for d in dialogues]
    one = [few_shot_prompt(d, shots[:1]) for d in dialogues]
    few = [few_shot_prompt(d, shots) for d in dialogues]
    plan = [
        ("base zero-shot", False, zero, cfg.max_input_length),
        ("base one-shot", False, one, cfg.shot_max_input_length),
        (f"base few-shot (k={len(shots)})", False, few, cfg.shot_max_input_length),
        ("LoRA fine-tuned", True, zero, cfg.max_input_length),
    ]
    results = {}
    for name, use_adapter, prompts, max_len in plan:
        t0 = time.perf_counter()
        ctx = nullcontext() if use_adapter else peft_model.disable_adapter()
        with ctx:
            preds = summarize(peft_model, tokenizer, prompts, cfg, max_len)
        scores = rouge(preds, refs)
        results[name] = {**scores, "predictions": preds,
                         "per_example_rougeL": rouge_per_example(preds, refs)["rougeL"].tolist(),
                         "seconds": round(time.perf_counter() - t0, 1)}
        print(f"[eval] {name:<24} ROUGE-1 {scores['rouge1']:.4f} · ROUGE-2 {scores['rouge2']:.4f} · "
              f"ROUGE-L {scores['rougeL']:.4f} · {results[name]['seconds']} s")
    return results


# --------------------------------------------------------------------------- RAG
def evaluate_retrieval(retriever, questions: list[tuple[str, str]], k: int = 3) -> dict:
    """Hit rate of the retriever on held-out questions: is the answering doc ranked 1st / in the top k?"""
    ranks = []
    for question, doc_id in questions:
        ids = [s["id"] for s in retriever.search(question, k=len(retriever.docs))]
        ranks.append(ids.index(doc_id) + 1)
    ranks = np.array(ranks)
    return {"hit_at_1": round(float((ranks == 1).mean()), 4), f"hit_at_{k}": round(float((ranks <= k).mean()), 4),
            "mrr": round(float((1 / ranks).mean()), 4), "n_questions": len(questions)}


def rag_answers(model, tokenizer, retriever, questions: list[str], cfg: Config) -> list[dict]:
    """Grounded answers (retrieved context in the prompt) next to ungrounded ones (question only)."""
    rows = []
    for q in questions:
        sources = retriever.search(q, k=cfg.rag_top_k)
        grounded, bare = M.generate(model, tokenizer, [M.build_rag_prompt(q, sources), q],
                                    max_input_length=cfg.max_input_length, max_new_tokens=cfg.rag_max_new_tokens)
        rows.append({"question": q, "top_source": sources[0]["title"], "grounded": grounded, "ungrounded": bare})
    return rows


def save_history(history: dict, path: Path) -> None:
    utils.save_json(history, path)
