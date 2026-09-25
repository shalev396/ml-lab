"""Export the chat-tuned GPT as a Hugging Face model repo folder.

Writes `model.safetensors` + `config.json` (PyTorchModelHubMixin; the tied embedding / LM-head
weight is stored once), `metrics.json` (STANDARD §4), card plots in `assets/`, then refreshes the
metrics in `README.md`. Exporting anywhere other than model/ (smoke runs) first copies the card,
`model.py`, `handler.py` and `requirements.txt` there, so the folder is a self-contained snapshot
and the real model/ is never touched.
"""
import shutil
from pathlib import Path

from matplotlib.figure import Figure

from . import utils

TASK = "text-generation (character-level chat)"
DATASET = "Tiny Shakespeare (karpathy/char-rnn tinyshakespeare/input.txt, 1,115,394 characters)"
COPIED_FROM_MODEL_DIR = ("README.md", "model.py", "handler.py", "requirements.txt")

# Chart colours (validated categorical slots + neutrals, same as the other ml-lab projects).
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = {"train": "#2a78d6", "val": "#eb6834"}
SPLIT_COLORS = {"chat_val_loss": "#2a78d6", "text_val_loss": "#eb6834"}
SPLIT_LABELS = {"chat_val_loss": "chat val (dialogue pairs)", "text_val_loss": "text val (raw play)"}


def model_name(net) -> str:
    return (f"char-GPT {len(net.blocks)}L/{net.blocks[0].attn.n_head}H/{net.wte.embedding_dim}d, "
            f"block {net.block_size}, pretrained + chat-tuned")


def scores(chat_eval: dict, text_eval: dict) -> dict[str, float]:
    """Performance metrics of one model from its two `engine.evaluate` results (lower = better)."""
    return {"chat_val_loss": chat_eval["loss"], "chat_val_bpc": chat_eval["bpc"], "chat_val_ppl": chat_eval["ppl"],
            "text_val_loss": text_eval["loss"], "text_val_bpc": text_eval["bpc"]}


def build_metrics(metrics: dict[str, float], *, model: str, data: dict, params: int,
                  train_time_s: float | None, device: str, smoke: bool, source: str,
                  comparison: dict | None = None, training: dict | None = None,
                  trained_at: str | None = None) -> dict:
    """metrics.json content (schema: STANDARD §4). `metrics` holds the deployed model's val scores."""
    out = {
        "model": model,
        "task": TASK,
        "dataset": DATASET,
        "split": "val",
        "primary_metric": {"name": "chat_val_loss", "value": metrics["chat_val_loss"],
                           "unit": "nats per character, lower is better"},
        "metrics": metrics,
        "data": data,
        "params": int(params),
        "train_time_s": None if train_time_s is None else round(train_time_s, 1),
        "device": device,
        "smoke": smoke,
        "source": source,
        "versions": utils.lib_versions("torch", "huggingface_hub", "safetensors", "numpy"),
        "trained_at": trained_at or utils.today(),
    }
    if comparison:
        out["comparison"] = comparison
    if training:
        out["training"] = training
    return out


def save_weights(net, folder: str | Path) -> Path:
    """model.safetensors + config.json only (PyTorchModelHubMixin): a folder `model.load()` can serve."""
    folder = Path(folder)
    net.cpu().save_pretrained(folder)
    return folder


def export(net, model_dir: str | Path, metrics: dict, *, histories: dict | None = None,
           loss_by_position: dict | None = None) -> dict:
    """Write weights, config, metrics and plots to `model_dir`, then update its model card.

    histories: {"pretrain": engine history, "chat": engine history} -> assets/training_curves.png
    loss_by_position: {"chat val": [...], "text val": [...]} -> assets/loss_by_position.png
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if model_dir.resolve() != utils.MODEL_DIR.resolve():
        for name in COPIED_FROM_MODEL_DIR:
            shutil.copy2(utils.MODEL_DIR / name, model_dir / name)

    net.cpu().save_pretrained(model_dir)     # model.safetensors + config.json (README.md is kept)
    assets = model_dir / "assets"
    if histories:
        plot_history(histories, assets / "training_curves.png")
    if metrics.get("comparison"):
        plot_comparison(metrics["comparison"], metrics["model"], assets / "comparison.png")
    if loss_by_position:
        plot_loss_by_position(loss_by_position, assets / "loss_by_position.png")
    utils.save_json(metrics, model_dir / "metrics.json")
    utils.update_model_card(model_dir, metrics)
    print(f"exported -> {model_dir}")
    return metrics


# --------------------------------------------------------------------------- plots
def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def plot_history(histories: dict[str, dict], path: str | Path) -> Path:
    """Estimated train / val loss per stage, as small multiples (stage A, stage B)."""
    titles = {"pretrain": "Stage A: pretrain on the raw play", "chat": "Stage B: chat-tune on dialogue pairs"}
    stages = [s for s in ("pretrain", "chat") if s in histories]
    fig = Figure(figsize=(4.2 * len(stages), 3.3), dpi=150, facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, len(stages), squeeze=False)[0]
    for ax, stage in zip(axes, stages):
        _style(ax)
        ax.grid(True, color=GRID, linewidth=0.6)
        h = histories[stage]
        for split, color in SERIES.items():
            ax.plot(h["iter"], h[f"{split}_loss"], color=color, linewidth=2, marker="o", markersize=3.5,
                    markeredgecolor=SURFACE, markeredgewidth=0.8, label=split)
        ax.annotate(f"{h['val_loss'][-1]:.3f}", (h["iter"][-1], h["val_loss"][-1]), xytext=(4, 6),
                    textcoords="offset points", fontsize=8, color=SERIES["val"])
        ax.set_title(titles[stage], color=INK, fontsize=10, loc="left")
        ax.set_xlabel("iteration", color=INK_MUTED)
        ax.set_ylabel("cross-entropy (nats/char)", color=INK_MUTED)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    return _save(fig, path)


def plot_comparison(comparison: dict[str, dict], deployed: str, path: str | Path) -> Path:
    """Val loss of every variant on both splits (horizontal bars, lower is better)."""
    names = list(comparison)
    keys = [k for k in SPLIT_COLORS if any(k in comparison[n] for n in names)]
    fig = Figure(figsize=(8.0, 0.55 * len(names) * len(keys) + 1.2), dpi=150, facecolor=SURFACE,
                 layout="constrained")
    ax = fig.subplots()
    _style(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    height = 0.8 / len(keys)
    for j, key in enumerate(keys):
        ys = [i + (j - (len(keys) - 1) / 2) * height for i in range(len(names))]
        vals = [comparison[n].get(key, float("nan")) for n in names]
        ax.barh(ys, vals, height=height * 0.92, color=SPLIT_COLORS[key], label=SPLIT_LABELS[key])
        for y, v in zip(ys, vals):
            if v == v:
                ax.text(v, y, f" {v:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(names)), [("* " if deployed in n or "deployed" in n else "") + n for n in names],
                  fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("validation cross-entropy (nats/char), lower is better", color=INK_MUTED)
    ax.set_xlim(0, max(v for n in names for v in comparison[n].values() if v == v) * 1.18)
    ax.set_title("Every variant on both validation splits (* = deployed)", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    return _save(fig, path)


def plot_loss_by_position(curves: dict[str, list[float]], path: str | Path) -> Path:
    """Mean loss at each position of the context window: how much the model uses its context."""
    colors = list(SPLIT_COLORS.values())
    fig = Figure(figsize=(6.4, 3.2), dpi=150, facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    _style(ax)
    ax.grid(True, color=GRID, linewidth=0.6)
    for (label, ys), color in zip(curves.items(), colors):
        ax.plot(range(1, len(ys) + 1), ys, color=color, linewidth=1.6, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("characters of context seen (log scale)", color=INK_MUTED)
    ax.set_ylabel("cross-entropy (nats/char)", color=INK_MUTED)
    ax.set_title("Deployed model: loss vs. context length", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    return _save(fig, path)


def _save(fig: Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path
