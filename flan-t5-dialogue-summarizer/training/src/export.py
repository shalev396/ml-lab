"""Write the model repo (cfg.model_dir): LoRA adapter, tokenizer, config.json, metrics.json, card plots and
the refreshed model card. Smoke runs write to training/outputs/smoke/model/ and never touch model/."""
from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

import model as M  # ../model/model.py

from . import utils
from .config import Config
from .data_setup import Splits
from .engine import ROUGE_KEYS

DATASET = "DialogSum (knkarthick/dialogsum)"
DEPLOYED = "LoRA fine-tuned"
VERSIONS = ("torch", "transformers", "peft", "sentence_transformers", "datasets", "rouge_score", "numpy")
REPO_FILES = ("README.md", "model.py", "handler.py", "requirements.txt", M.KB_FILE)


def save_model(cfg: Config, peft_model, tokenizer) -> Path:
    """Write the servable model: LoRA adapter + tokenizer + config.json (runtime settings) into cfg.model_dir."""
    model_dir = _prepare(cfg.model_dir)
    with tempfile.TemporaryDirectory() as tmp:  # peft also writes a README.md; keep only the weights
        peft_model.save_pretrained(tmp)
        for name in M.ADAPTER_FILES:
            shutil.copyfile(Path(tmp) / name, model_dir / name)
    tokenizer.save_pretrained(model_dir)
    utils.save_json({
        "base_model": cfg.base_model, "embed_model": cfg.embed_model,
        "max_input_length": cfg.max_input_length, "max_new_tokens": cfg.max_new_tokens,
        "num_beams": cfg.num_beams, "rag_top_k": cfg.rag_top_k, "rag_max_new_tokens": cfg.rag_max_new_tokens,
        "prompt_template": M.PROMPT_TEMPLATE, "rag_prompt_template": M.RAG_PROMPT_TEMPLATE,
        "lora": {"r": cfg.lora_r, "alpha": cfg.lora_alpha, "dropout": cfg.lora_dropout,
                 "target_modules": list(cfg.lora_target_modules)},
        "versions": utils.lib_versions(*VERSIONS),
    }, model_dir / "config.json")
    print(f"[export] adapter + tokenizer + config.json -> {model_dir}")
    return model_dir


def export(cfg: Config, *, history: dict, results: dict, retrieval: dict, splits: Splits, params: dict,
           device: str) -> dict:
    """After save_model: check the round trip, then write metrics.json, the card plots and the model card.
    Returns the metrics dict written to metrics.json."""
    model_dir = cfg.model_dir
    _check_roundtrip(model_dir, splits, results[DEPLOYED]["predictions"])

    # metrics.json ---------------------------------------------------------------------------
    tuned, base = results[DEPLOYED], results["base zero-shot"]
    scores = {k: tuned[k] for k in ROUGE_KEYS}
    metrics = {
        "model": f"{cfg.base_model} + LoRA (r={cfg.lora_r}, alpha={cfg.lora_alpha}, {'/'.join(cfg.lora_target_modules)})",
        "task": "dialogue-summarization",
        "dataset": DATASET,
        "split": "test",
        "primary_metric": {"name": "rougeL", "value": scores["rougeL"]},
        "metrics": scores,
        "improvement_over_base": {k: round(tuned[k] - base[k], 6) for k in ROUGE_KEYS},
        "comparison": {name: {k: r[k] for k in (*ROUGE_KEYS, "gen_len_words")} for name, r in results.items()},
        "rag_retrieval": retrieval,
        "data": {**splits.sizes(), "test_references_per_dialogue": 3, "unit": "dialogues"},
        "params": params["base"],
        "trainable_params": params["trainable"],
        "train_time_s": round(history["train_time_s"], 1),
        "seconds_per_step": round(history["seconds_per_step"], 3),
        "best_epoch": history.get("best_epoch"),
        "epochs_run": len(history["epoch_time_s"]),
        "eval_time_s": round(sum(r["seconds"] for r in results.values()), 1),
        "device": device,
        "precision": history["precision"],
        "smoke": cfg.smoke,
        "source": f"trained {utils.today()} with training/ ({device}{', smoke run' if cfg.smoke else ''})",
        "hyperparameters": {k: v for k, v in cfg.to_dict().items() if k not in ("smoke", "smoke_base_model")},
        "samples": [{"dialogue": splits.test[i]["dialogue"], "reference": splits.test[i]["references"][0],
                     **{name: results[name]["predictions"][i] for name in ("base zero-shot", DEPLOYED)}}
                    for i in range(min(3, len(splits.test)))],
        "versions": utils.lib_versions(*VERSIONS),
        "trained_at": utils.today(),
    }
    utils.save_json(metrics, model_dir / "metrics.json")

    # 3. card plots + card text -------------------------------------------------------------------
    assets = model_dir / "assets"
    plot_loss(history, assets / "loss_curve.png")
    plot_comparison(metrics["comparison"], assets / "rouge_comparison.png")
    plot_per_dialogue(np.array(base["per_example_rougeL"]), np.array(tuned["per_example_rougeL"]),
                      assets / "rouge_per_dialogue.png")
    architecture = (f"{cfg.base_model} (frozen) + LoRA r={cfg.lora_r} on attention "
                    f"{'/'.join(cfg.lora_target_modules)}, merged at load time")
    utils.update_model_card(model_dir, metrics, ml_lab={"architecture": architecture})
    _fill_results(model_dir / "README.md", metrics)
    if not cfg.smoke:
        _fill_project_readme(metrics)
    print(f"[export] metrics.json + assets + card -> {model_dir}")
    return metrics


def _prepare(model_dir: Path) -> Path:
    """Create the export dir; outside model/ (smoke) seed it with the card, code and knowledge base so it
    is a complete, loadable model repo (the Space can serve it via MODEL_DIR=...)."""
    model_dir.mkdir(parents=True, exist_ok=True)
    if model_dir.resolve() != utils.MODEL_DIR.resolve():
        for name in REPO_FILES:
            (model_dir / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(utils.MODEL_DIR / name, model_dir / name)
    return model_dir


def _check_roundtrip(model_dir: Path, splits: Splits, expected: list[str], n: int = 3) -> None:
    """Reload the export through model.py (base + merged adapter, what the Space runs) and compare its
    greedy summaries with the in-memory adapter's. Merging changes float rounding, so a rare one-token
    difference is reported, not raised; an empty or wholly different result is an error."""
    predictor = M.load(model_dir, "cpu")
    same = 0
    for dialogue, want in zip(splits.test["dialogue"][:n], expected[:n]):
        got = predictor.predict(dialogue)
        if not got:
            raise RuntimeError("the exported model returned an empty summary")
        same += got == want
        if got != want:
            print(f"[export] round-trip differs:\n  in-memory: {want}\n  exported:  {got}")
    if same == 0:
        raise RuntimeError("the exported adapter does not reproduce any in-memory summary")
    print(f"[export] round trip through model.py: {same}/{min(n, len(expected))} summaries identical")


# --------------------------------------------------------------------------- card text
_RES_START, _RES_END = "<!-- results:start -->", "<!-- results:end -->"


def _replace_block(text: str, start: str, end: str, body: str) -> str:
    return re.sub(re.escape(start) + r".*?" + re.escape(end), lambda _: f"{start}{body}{end}", text, flags=re.S)


def results_markdown(metrics: dict) -> str:
    """Comparison table (ROUGE x100, deployed variant in bold) + retrieval + sample summaries."""
    n = metrics["data"]["n_test"]
    rows = ["| variant | ROUGE-1 | ROUGE-2 | ROUGE-L | avg. words |", "|---|---|---|---|---|"]
    for name, m in metrics["comparison"].items():
        label = f"**{name}** (deployed)" if name == DEPLOYED else name
        rows.append(f"| {label} | {100 * m['rouge1']:.2f} | {100 * m['rouge2']:.2f} | {100 * m['rougeL']:.2f} "
                    f"| {m['gen_len_words']:.1f} |")
    gain = metrics["improvement_over_base"]["rougeL"]
    r = metrics["rag_retrieval"]
    top_k = next(k for k in r if k.startswith("hit_at_") and k != "hit_at_1")
    lines = [
        "", f"ROUGE F-measure x100 on {n} DialogSum test dialogues, each scored against its 3 human summaries "
        f"(mean). Greedy decoding for every variant. The adapter adds **{100 * gain:+.2f} ROUGE-L** over "
        f"the zero-shot base model.", "", *rows, "",
        f"RAG retriever (MiniLM, {r['n_questions']} held-out store questions): hit@1 {r['hit_at_1']:.2f} · "
        f"{top_k.replace('_at_', '@')} {r[top_k]:.2f} · MRR {r['mrr']:.2f}.", "",
        f"Run: {metrics['source']} · {metrics['precision']} · train {metrics['train_time_s']:.0f} s "
        f"({metrics['seconds_per_step']:.2f} s/step) · eval {metrics['eval_time_s']:.0f} s.", "",
        "![Loss curve](assets/loss_curve.png)", "![ROUGE comparison](assets/rouge_comparison.png)",
        "![ROUGE-L per test dialogue](assets/rouge_per_dialogue.png)", "",
    ]
    if metrics["smoke"]:
        lines.insert(1, "> Smoke run (tiny model and data): these numbers only prove the pipeline works.\n")
    return "\n".join(lines)


def _fill_results(card: Path, metrics: dict) -> None:
    text = card.read_text(encoding="utf-8")
    if _RES_START in text and _RES_END in text:
        card.write_text(_replace_block(text, _RES_START, _RES_END, results_markdown(metrics)),
                        encoding="utf-8", newline="\n")


def _fill_project_readme(metrics: dict) -> None:
    """The one-line result in <slug>/README.md (between <!-- result:start/end -->)."""
    path = utils.PROJECT_DIR / "README.md"
    if not path.is_file():
        return
    m, gain = metrics["metrics"], metrics["improvement_over_base"]["rougeL"]
    line = (f"ROUGE-L = {100 * m['rougeL']:.2f} ({metrics['data']['n_test']} test dialogues), "
            f"{100 * gain:+.2f} over zero-shot flan-t5-base")
    text = path.read_text(encoding="utf-8")
    path.write_text(_replace_block(text, "<!-- result:start -->", "<!-- result:end -->", line),
                    encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------- plots
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1")


def _axes(width: float = 6.4, height: float = 4.0):
    fig = Figure(figsize=(width, height), dpi=150, facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    return fig, ax


def _save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)


def plot_loss(history: dict, path: Path) -> None:
    """Training loss (every log_every steps) and validation loss (after each epoch) on one step axis."""
    fig, ax = _axes()
    ax.plot(history["step"], history["train_loss"], color=SERIES[0], linewidth=1.6, label="train (running mean)")
    val_steps = [e * history["steps_per_epoch"] for e in history["epoch"]]
    ax.plot(val_steps, history["val_loss"], color=SERIES[1], linewidth=2, marker="o", markersize=5,
            label="validation (end of epoch)")
    if history.get("best_epoch"):
        i = history["epoch"].index(history["best_epoch"])
        ax.plot(val_steps[i], history["val_loss"][i], "o", markersize=11, markerfacecolor="none",
                markeredgecolor=INK, markeredgewidth=1.6, label=f"kept: epoch {history['best_epoch']} (lowest val loss)")
    ax.set(xlabel="optimizer step", ylabel="cross-entropy per summary token")
    ax.set_title("LoRA fine-tuning loss", fontsize=10, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    _save(fig, path)


def plot_comparison(comparison: dict, path: Path) -> None:
    """ROUGE-1/2/L of every variant (x100), grouped by metric."""
    fig, ax = _axes(7.2, 4.0)
    names = list(comparison)
    x = np.arange(len(ROUGE_KEYS))
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        vals = [100 * comparison[name][k] for k in ROUGE_KEYS]
        bars = ax.bar(x + (i - (len(names) - 1) / 2) * width, vals, width, color=SERIES[i % len(SERIES)],
                      edgecolor=SURFACE, linewidth=1, label=name)
        ax.bar_label(bars, fmt="%.1f", fontsize=7, color=INK_2, padding=2)
    ax.set_ylim(0, 1.3 * max(100 * comparison[n][k] for n in names for k in ROUGE_KEYS))  # room for the legend
    ax.set_xticks(x, ["ROUGE-1", "ROUGE-2", "ROUGE-L"])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("F-measure x100 (test)")
    ax.set_title("Prompting the base model vs the LoRA fine-tune", fontsize=10, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, ncols=2)
    _save(fig, path)


def plot_per_dialogue(base: np.ndarray, tuned: np.ndarray, path: Path) -> None:
    """ROUGE-L of every test dialogue: zero-shot base (x) vs fine-tuned (y). Above the diagonal = better."""
    fig, ax = _axes(4.8, 4.6)
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
    ax.scatter(100 * base, 100 * tuned, s=14, color=SERIES[0], alpha=0.7, edgecolors="none")
    better = float((tuned > base).mean())
    ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="base zero-shot ROUGE-L", ylabel="LoRA fine-tuned ROUGE-L")
    ax.set_title(f"Per test dialogue: fine-tune better on {better:.0%}", fontsize=10, loc="left", color=INK)
    _save(fig, path)

