"""Training package for video-quiz-generator. Makes ../space importable as the top-level module `pipeline`.

This is a pipeline-only project (no trained weights): the code under test is the Space's own
`space/pipeline.py`, so the notebook evaluates exactly what the Space serves.
"""
import sys
from pathlib import Path

_SPACE_DIR = Path(__file__).resolve().parents[2] / "space"
if str(_SPACE_DIR) not in sys.path:
    sys.path.insert(0, str(_SPACE_DIR))
