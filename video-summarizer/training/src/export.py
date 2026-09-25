"""Export for a pipeline-only project: there are no weights. It writes the Space examples
(space/examples, or outputs/smoke/examples for smoke runs) and training/outputs/results.json + plots."""
from __future__ import annotations

import platform
import shutil
from pathlib import Path

from . import utils

LIBS = ("torch", "transformers", "av", "numpy", "huggingface_hub")


def write_examples(samples: dict[str, Path], examples_dir: Path) -> list[Path]:
    """Copy the sample media + transcript into the Space's examples/ folder."""
    examples_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for path in samples.values():
        target = examples_dir / path.name
        if not target.is_file() or target.read_bytes() != path.read_bytes():
            shutil.copyfile(path, target)
        out.append(target)
    return out


def plot_timings(results: dict[str, list[dict]], path: Path) -> Path:
    """Stacked bars: decode / ASR / LLM seconds for every variant x sample."""
    import matplotlib.pyplot as plt

    labels, stages = [], {"decode_s": [], "asr_s": [], "llm_s": []}
    for variant, rows in results.items():
        for row in rows:
            labels.append(f"{variant}\n{row['sample']}")
            for key in stages:
                stages[key].append(row[key])
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(labels)), 4))
    bottom = [0.0] * len(labels)
    for (key, values), color in zip(stages.items(), ["#9aa5b1", "#2a78d6", "#eb6834"]):
        ax.bar(labels, values, bottom=bottom, label=key.removesuffix("_s"), color=color)
        bottom = [b + v for b, v in zip(bottom, values)]
    ax.set_ylabel("seconds")
    ax.set_title(f"Pipeline time per stage ({utils.device_name('torch')})")
    ax.legend()
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def build_results(cfg, results: dict[str, list[dict]], table: dict[str, dict],
                  params: dict[str, list[dict]], device: str) -> dict:
    """The results.json payload (no absolute paths)."""
    return {
        "task": "video/audio -> transcript -> summary + key points (pretrained pipeline, nothing trained)",
        "dataset": "JFK inaugural-address clip (11 s) as WAV + MP4, and an original lecture transcript",
        "primary_metric": {"name": "wer", "variant": next(iter(table)), "value": table[next(iter(table))]["wer"]},
        "variants": table,
        "params": params,
        "runs": results,
        "config": {"max_seconds": cfg.max_seconds, "max_new_tokens": cfg.max_new_tokens,
                   "n_key_points": cfg.n_key_points, "reference": cfg.reference},
        "device": device,
        "machine": f"{platform.system()} {platform.machine()}",
        "smoke": cfg.smoke,
        "versions": utils.lib_versions(*LIBS),
        "evaluated_at": utils.today(),
    }


def save_results(payload: dict, outputs_dir: Path) -> Path:
    return utils.save_json(payload, outputs_dir / "results.json")
