"""Shared training helpers — the SAME file is copied into every <slug>/training/src/utils.py.

Device-agnostic by design: code asks "is there a GPU?" (CUDA → Apple MPS → CPU) and
never asks "am I on Colab?". Frameworks are imported lazily so a PyTorch project never
pays the TensorFlow import cost (and vice versa).
"""
from __future__ import annotations

import json
import os
import platform
import random
import re
from contextlib import nullcontext
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

TRAINING_DIR = Path(__file__).resolve().parents[1]   # <slug>/training
PROJECT_DIR = TRAINING_DIR.parent                     # <slug>
SLUG = PROJECT_DIR.name
HF_USER = "shalev396"
REPO_ID = f"{HF_USER}/{SLUG}"
MODEL_DIR = PROJECT_DIR / "model"                     # HF model repo (git submodule)
SPACE_DIR = PROJECT_DIR / "space"                     # HF Space repo (git submodule)
DATA_DIR = TRAINING_DIR / "data"                      # gitignored download cache
OUTPUTS_DIR = TRAINING_DIR / "outputs"                # gitignored run outputs


# --------------------------------------------------------------------------- devices
def get_device():
    """PyTorch device: CUDA if present, else Apple MPS, else CPU."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_dtype(device):
    """Mixed-precision dtype for this device, or None (= full fp32).

    bf16 on Ampere+ (compute capability >= 8.0), fp16 on older GPUs (e.g. Colab T4), none on CPU/MPS.
    """
    import torch

    if getattr(device, "type", str(device)) != "cuda":
        return None
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def autocast(device, enabled: bool = True):
    """`with autocast(device):` — AMP on CUDA, a no-op everywhere else."""
    import torch

    dtype = amp_dtype(device) if enabled else None
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def grad_scaler(device, enabled: bool = True):
    """GradScaler that is only active when training in fp16 (bf16/fp32 need no scaling)."""
    import torch

    active = enabled and amp_dtype(device) == torch.float16
    return torch.amp.GradScaler("cuda", enabled=active)


def num_workers() -> int:
    """DataLoader workers: 0 on Windows (spawn is slow inside notebooks), else up to 4."""
    return 0 if platform.system() == "Windows" else min(4, os.cpu_count() or 1)


def setup_tf() -> str:
    """TensorFlow: enable GPU memory growth when a GPU exists. Float32 everywhere."""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except (RuntimeError, ValueError):
            pass
    return "gpu" if gpus else "cpu"


def device_name(framework: str = "torch") -> str:
    """Human-readable device string for logs and metrics.json ("cuda", "mps", "cpu", "gpu")."""
    if framework == "torch":
        return get_device().type
    if framework == "tensorflow":
        return setup_tf()
    return "cpu"


# --------------------------------------------------------------------------- reproducibility
def set_seeds(seed: int = 42, framework: str | None = None) -> None:
    """Seed python, numpy and (optionally) torch or tensorflow."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if framework == "torch":
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    elif framework == "tensorflow":
        import tensorflow as tf

        tf.random.set_seed(seed)


# --------------------------------------------------------------------------- io
def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return path


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lib_versions(*names: str) -> dict[str, str]:
    """{"torch": "2.11.0", ...} for the given importable packages (missing ones are skipped)."""
    import importlib

    out = {}
    for name in names:
        try:
            out[name] = importlib.import_module(name).__version__
        except Exception:
            continue
    return out


def count_params(model) -> int:
    """Total parameter count for a torch nn.Module or a Keras model."""
    if hasattr(model, "count_params"):
        return int(model.count_params())
    return int(sum(p.numel() for p in model.parameters()))


def human_params(n: int) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.2f}".rstrip("0").rstrip(".") + unit
    return str(n)


def today() -> str:
    return date.today().isoformat()


# --------------------------------------------------------------------------- Hugging Face
_TABLE_START, _TABLE_END = "<!-- metrics:start -->", "<!-- metrics:end -->"


def update_model_card(model_dir: str | Path, metrics: dict, ml_lab: dict | None = None) -> Path:
    """Refresh model/README.md from metrics.json: YAML `model-index`, `ml_lab` block and the
    markdown metrics table between the <!-- metrics:start/end --> markers. Prose is untouched."""
    import yaml

    path = Path(model_dir) / "README.md"
    raw = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    match = re.match(r"^---\n(.*?)\n---\n?(.*)$", raw, flags=re.S)
    if not match:
        raise ValueError(f"{path} has no YAML front matter")
    data, text = yaml.safe_load(match.group(1)) or {}, match.group(2)

    primary = metrics["primary_metric"]
    block = dict(data.get("ml_lab") or {})
    block.update(ml_lab or {})
    block["metric"] = {"name": primary["name"], "value": primary["value"], "split": metrics.get("split", "test")}
    if metrics.get("params"):
        block["params"] = int(metrics["params"])
        block["params_human"] = human_params(int(metrics["params"]))
    block["trained_at"] = metrics.get("trained_at", today())
    data["ml_lab"] = block

    numeric = {k: v for k, v in metrics.get("metrics", {}).items() if isinstance(v, (int, float))}
    # model-index dataset.type must be a Hub-style id: the Hub dataset if there is one, else a slug of the name
    dataset_id = (data.get("datasets") or [re.sub(r"[^\w.-]+", "-", str(block.get("dataset", "custom"))).strip("-").lower()])[0]
    data["model-index"] = [{
        "name": SLUG,
        "results": [{
            "task": {"type": data.get("pipeline_tag", "other")},
            "dataset": {"name": block.get("dataset", dataset_id), "type": dataset_id,
                        "split": metrics.get("split", "test")},
            "metrics": [{"type": k, "value": round(float(v), 6), "name": k} for k, v in numeric.items()],
        }],
    }]

    rows = "\n".join(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |"
                     for k, v in numeric.items())
    table = f"{_TABLE_START}\n| metric ({metrics.get('split', 'test')}) | value |\n|---|---|\n{rows}\n{_TABLE_END}"
    if _TABLE_START in text and _TABLE_END in text:
        text = re.sub(re.escape(_TABLE_START) + r".*?" + re.escape(_TABLE_END), lambda _: table, text, flags=re.S)

    front = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=1000)
    path.write_text(f"---\n{front}---\n{text}", encoding="utf-8", newline="\n")
    return path


def upload_model(model_dir: str | Path = MODEL_DIR, repo_id: str = REPO_ID,
                 message: str = "Update model from training notebook") -> str:
    """Upload model/ to the Hub (creates the repo if needed). Needs a token: `hf auth login`
    locally, or an HF_TOKEN secret/env var (huggingface_hub picks it up automatically)."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    info = api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(model_dir),
                             commit_message=message,
                             ignore_patterns=[".git", ".git/*", "__pycache__/*", "*.pyc"])
    return str(info)
