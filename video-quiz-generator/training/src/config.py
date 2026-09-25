"""Every setting of the quiz-generator evaluation run, in one dataclass.

There is nothing to train: the pipeline is built from pretrained checkpoints. The run
generates quizzes over a small grid (samples x difficulties) with every pipeline variant,
checks them against the JSON schema and scores them with an LLM judge (G-Eval-lite).
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

import pipeline as P

from . import utils


def _variants() -> dict:
    """name -> [ASR model, quiz LLM]. Exactly the two configurations the Space runs."""
    return {"cpu": [P.ASR_MODELS["cpu"], P.LLM_MODELS["cpu"]],      # CPU hardware
            "gpu": [P.ASR_MODELS["cuda"], P.LLM_MODELS["cuda"]]}    # ZeroGPU (deployed)


@dataclass
class Config:
    # --- what is evaluated
    variants: dict = field(default_factory=_variants)
    deployed_variant: str = "gpu"             # the Space runs on ZeroGPU
    samples: list = field(default_factory=lambda: ["jfk", "gettysburg"])   # see data_setup.SAMPLES
    difficulties: list = field(default_factory=lambda: ["easy", "medium", "hard"])
    n_questions: int = 4
    repair: bool = True                       # the retry policy of pipeline.generate_quiz (False: first reply only)
    # --- G-Eval-lite judge (evaluation only, never used by the Space)
    judge: bool = True
    judge_model: str = P.LLM_MODELS["cuda"]   # Qwen2.5-1.5B-Instruct
    judge_max_new_tokens: int = 60
    seed: int = 42
    # --- run mode (SMOKE_TEST=1): one variant, one sample, one quiz of 2 questions, small judge
    smoke: bool = field(default_factory=lambda: os.getenv("SMOKE_TEST", "0") == "1")

    def __post_init__(self):
        if self.smoke:
            self.variants = {"cpu": self.variants["cpu"]}
            self.deployed_variant = "cpu"
            self.samples = ["jfk"]
            self.difficulties = ["mixed"]
            self.n_questions = 2
            self.judge_model = P.LLM_MODELS["cpu"]

    @property
    def out_dir(self) -> Path:
        """Where export() writes results. Smoke runs go to outputs/smoke and never touch ../model."""
        return utils.OUTPUTS_DIR / ("smoke" if self.smoke else "results")

    @property
    def model_card(self) -> Path:
        """README whose results table export() refreshes (a copy under outputs/smoke for smoke runs)."""
        return self.out_dir / "model_README.md" if self.smoke else utils.MODEL_DIR / "README.md"
