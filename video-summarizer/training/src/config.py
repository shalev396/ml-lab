"""Every knob of the video-summarizer evaluation run, in one dataclass.

Nothing is trained here: the notebook runs the pretrained pipeline (`space/pipeline.py`) on the
sample media, measures it and exports the Space examples + a results.json.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

import pipeline as P

from . import utils

# The words spoken in the JFK clip (inaugural address, 20 Jan 1961) -- the WER reference.
JFK_REFERENCE = ("And so, my fellow Americans, ask not what your country can do for you, "
                 "ask what you can do for your country.")


@dataclass
class Config:
    # --- data (cached in training/data/)
    audio_sample: str = "jfk.wav"             # 11 s, 16 kHz mono speech clip
    video_sample: str = "sample.mp4"          # the same audio inside an H.264/AAC video
    transcript_sample: str = "lecture_transcript.txt"   # pasted-transcript input (skips ASR)
    reference: str = JFK_REFERENCE
    # --- pipeline
    max_seconds: float = P.MAX_SECONDS
    max_new_tokens: int = P.MAX_NEW_TOKENS
    n_key_points: int = P.N_KEY_POINTS
    # --- variants to evaluate: name -> (ASR id, LLM id). "cpu" = what CPU hardware runs,
    #     "gpu" = what the ZeroGPU Space runs (timed here on whatever device this machine has).
    variants: dict = field(default_factory=lambda: {
        "cpu": (P.ASR_CPU, P.LLM_CPU),
        "gpu": (P.ASR_GPU, P.LLM_GPU),
    })
    seed: int = 42
    # --- run mode (SMOKE_TEST=1): only the small pair, outputs go to outputs/smoke/
    smoke: bool = field(default_factory=lambda: os.getenv("SMOKE_TEST", "0") == "1")

    def __post_init__(self):
        if self.smoke:
            self.variants = {"cpu": self.variants["cpu"]}
            self.max_new_tokens = min(self.max_new_tokens, 256)

    @property
    def data_dir(self) -> Path:
        return utils.DATA_DIR

    @property
    def outputs_dir(self) -> Path:
        """results.json + plots. Smoke runs use outputs/smoke/."""
        return utils.OUTPUTS_DIR / "smoke" if self.smoke else utils.OUTPUTS_DIR

    @property
    def examples_dir(self) -> Path:
        """Where export writes the Space examples. Smoke runs never touch space/."""
        return self.outputs_dir / "examples" if self.smoke else utils.SPACE_DIR / "examples"
