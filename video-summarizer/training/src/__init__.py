"""Training package for video-summarizer. Makes ../space importable, so `import pipeline as P`
is the exact pipeline the Space runs (a pipeline-only project: no model/model.py)."""
import sys
from pathlib import Path

_SPACE_DIR = Path(__file__).resolve().parents[2] / "space"
if str(_SPACE_DIR) not in sys.path:
    sys.path.insert(0, str(_SPACE_DIR))
