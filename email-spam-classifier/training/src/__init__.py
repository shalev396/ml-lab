"""Training package for email-spam-classifier. Makes ../model importable as the top-level module `model`."""
import sys
from pathlib import Path

_MODEL_DIR = Path(__file__).resolve().parents[2] / "model"
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))
