"""Builds the pipeline variants and the judge FROM space/pipeline.py (the code the Space serves)."""
from __future__ import annotations

import gc

import pipeline as P

from . import utils


def build_variant(cfg, name: str, device: str) -> P.QuizPipeline:
    """One pipeline variant from cfg.variants[name] = [ASR model, quiz LLM] on `device`."""
    asr_model, llm_model = cfg.variants[name]
    return P.load(device, asr_model=asr_model, llm_model=llm_model)


def matches(pipe: P.QuizPipeline | None, cfg, name: str) -> bool:
    """True when `pipe` already is variant `name` (so it can be reused instead of reloaded)."""
    return pipe is not None and [pipe.asr_model_id, pipe.llm_model_id] == list(cfg.variants[name])


def describe(pipe: P.QuizPipeline) -> list[dict]:
    """Parameter count of each pretrained model inside a pipeline."""
    return [{"stage": "ASR", "model": pipe.asr_model_id, "params": pipe.asr.n_params,
             "params_human": utils.human_params(pipe.asr.n_params)},
            {"stage": "quiz LLM", "model": pipe.llm_model_id, "params": pipe.llm.n_params,
             "params_human": utils.human_params(pipe.llm.n_params)}]


def build_judge(cfg, device: str, reuse: P.QuizPipeline | None = None) -> P.ChatLLM:
    """The G-Eval-lite judge LLM; reuses a loaded pipeline's LLM when it is the same checkpoint."""
    if reuse is not None and reuse.llm_model_id == cfg.judge_model:
        return reuse.llm
    return P.ChatLLM(cfg.judge_model, device)


def release() -> None:
    """Free memory after the caller dropped its references (`del pipe`)."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
