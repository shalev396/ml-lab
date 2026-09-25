"""Builds the pipeline FROM space/pipeline.py (the file the Space runs). Nothing is trained:
every component is a pretrained Hub checkpoint."""
from __future__ import annotations

import pipeline as P

from . import utils


def build_pipeline(cfg, device: str, variant: str | None = None) -> P.Pipeline:
    """The pipeline for `variant` (a key of cfg.variants), or the device default when None."""
    asr, llm = cfg.variants[variant] if variant else P.default_models(device)
    return P.load(device, asr_model=asr, llm_model=llm, max_seconds=cfg.max_seconds,
                  max_new_tokens=cfg.max_new_tokens, n_key_points=cfg.n_key_points)


def describe(pipe: P.Pipeline) -> list[dict]:
    """Rows: component, Hub id, parameter count."""
    counts = pipe.param_counts()
    return [{"component": "speech-to-text", "hub_id": pipe.asr_id, "params": counts["asr"],
             "params_human": utils.human_params(counts["asr"])},
            {"component": "summarizer LLM", "hub_id": pipe.llm_id, "params": counts["llm"],
             "params_human": utils.human_params(counts["llm"])}]
