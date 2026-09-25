"""Write the evaluation results: metrics.json, every quiz + judgment, plots, and the results table.

Pipeline-only project: there are no weights. Results go to training/outputs/results (full run)
or training/outputs/smoke (smoke run). A full run also refreshes the table between
<!-- metrics:start --> / <!-- metrics:end --> in ../model/README.md; a smoke run refreshes a copy.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import utils  # noqa: E402

_START, _END = "<!-- metrics:start -->", "<!-- metrics:end -->"
TABLE_METRICS = ["schema_valid_rate", "first_try_valid_rate", "strict_json_first_try_rate", "question_yield",
                 "answer_agreement", "judge_relevance", "judge_clarity", "asr_wer", "mean_attempts",
                 "repair_rate", "regenerate_rate", "mean_quiz_seconds"]
RATE_METRICS = ["schema_valid_rate", "first_try_valid_rate", "question_yield", "answer_agreement"]


def build_metrics(cfg, summary: dict, runs: dict, params: dict, seconds: float, device: str) -> dict:
    """metrics.json in the ml-lab schema. The deployed variant's numbers are the headline."""
    name = cfg.deployed_variant if cfg.deployed_variant in summary else next(iter(summary))
    asr, llm = cfg.variants[name]
    n_quizzes = sum(len(r["quizzes"]) for r in runs.values())
    return {
        "model": f"{asr} + {llm} ({name} variant)",
        "task": "video/transcript -> multiple-choice quiz (strict JSON)",
        "dataset": "JFK inaugural clip (11 s audio) + Gettysburg Address (text)",
        "split": "eval",
        "primary_metric": {"name": "schema_valid_rate", "value": summary[name]["schema_valid_rate"]},
        "metrics": {k: v for k, v in summary[name].items() if isinstance(v, (int, float))},
        "comparison": summary,
        "data": {"samples": list(cfg.samples), "difficulties": list(cfg.difficulties),
                 "n_questions_per_quiz": cfg.n_questions, "n_quizzes": n_quizzes},
        "params": params.get(name),
        "judge_model": cfg.judge_model if cfg.judge else None,
        "train_time_s": round(seconds, 1),     # wall time of the evaluation run (nothing is trained)
        "device": device,
        "smoke": cfg.smoke,
        "source": f"pipeline evaluation {utils.today()} with training/ ({device})",
        "versions": utils.lib_versions("torch", "transformers", "av", "numpy"),
        "trained_at": utils.today(),
    }


def plot_comparison(summary: dict, path: Path) -> Path:
    names = list(summary)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    width = 0.8 / max(1, len(names))
    for k, name in enumerate(names):
        vals = [summary[name][m] or 0 for m in RATE_METRICS]
        bars = ax.bar([i + k * width for i in range(len(RATE_METRICS))], vals, width, label=name)
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks([i + width * (len(names) - 1) / 2 for i in range(len(RATE_METRICS))],
                  [m.replace("_", "\n") for m in RATE_METRICS], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_title("Pipeline variants: validity and answer agreement")
    ax.legend(title="variant")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_judge(judgments: list[dict], path: Path) -> Path | None:
    import pandas as pd

    df = pd.DataFrame(judgments)
    if df.empty:
        return None
    table = df.groupby(["variant", "difficulty"])[["relevance", "clarity"]].mean().unstack("variant")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for ax, metric in zip(axes, ["relevance", "clarity"]):
        table[metric].plot.bar(ax=ax, rot=0)
        ax.set_title(f"judge {metric} (1-5)")
        ax.set_ylim(0, 5.5)
        ax.set_xlabel("difficulty")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def update_results_table(readme: Path, metrics: dict) -> Path:
    """Replace the table between the markers with every variant's metrics (no YAML: not a Hub repo)."""
    comp = metrics["comparison"]
    names = list(comp)
    head = "| metric | " + " | ".join(names) + " |\n|---|" + "---|" * len(names)
    rows = []
    for m in TABLE_METRICS:
        cells = []
        for n in names:
            v = comp[n].get(m)
            cells.append("n/a" if v is None else (f"{v:.1f}" if m == "mean_quiz_seconds"
                                                  else f"{v:.2f}" if m == "mean_attempts" else f"{v:.3f}"))
        rows.append(f"| {m} | " + " | ".join(cells) + " |")
    note = (f"\n\n_{metrics['data']['n_quizzes']} quizzes of {metrics['data']['n_questions_per_quiz']} questions "
            f"({', '.join(metrics['data']['samples'])} x {', '.join(metrics['data']['difficulties'])}); judge "
            f"{metrics.get('judge_model') or 'off'}; run {metrics['trained_at']} on {metrics['device']}"
            f"{' (SMOKE)' if metrics['smoke'] else ''}._")
    table = f"{_START}\n{head}\n" + "\n".join(rows) + note + f"\n{_END}"
    text = readme.read_text(encoding="utf-8").replace("\r\n", "\n")
    if _START in text and _END in text:
        text = re.sub(re.escape(_START) + r".*?" + re.escape(_END), lambda _: table, text, flags=re.S)
        readme.write_text(text, encoding="utf-8", newline="\n")
    return readme


def export(cfg, runs: dict, judgments: list[dict], metrics: dict) -> Path:
    """Write everything to cfg.out_dir and refresh the results table. Returns cfg.out_dir."""
    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    utils.save_json(metrics, out / "metrics.json")
    utils.save_json({n: r["quizzes"] for n, r in runs.items()}, out / "quizzes.json")
    utils.save_json({n: r["asr"] for n, r in runs.items()}, out / "asr.json")
    utils.save_json(judgments, out / "judgments.json")
    plot_comparison(metrics["comparison"], out / "assets" / "variant_comparison.png")
    plot_judge(judgments, out / "assets" / "judge_scores.png")
    if cfg.smoke:                      # exercise the card update on a copy; never touch ../model
        shutil.copyfile(utils.MODEL_DIR / "README.md", cfg.model_card)
    update_results_table(cfg.model_card, metrics)
    return out
