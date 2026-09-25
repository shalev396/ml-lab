"""Export the trained FaceDetector + FaceRecognizer as a Hugging Face model folder.

Writes `face_detector.safetensors`, `model.safetensors` + `config.json` (PyTorchModelHubMixin),
`metrics.json` (STANDARD §4) and the card plots in `assets/`, then refreshes the metrics in
`README.md`. Exporting anywhere other than model/ (smoke runs) first copies the card and the code
there, so the folder is a self-contained snapshot and the real model/ is never touched.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from safetensors.torch import save_file

import model as M

from . import utils

TASK = "face-identification (42 people) + face-verification"
DATASET = ("LFW (Labeled Faces in the Wild, funneled): the 42 people with >= 25 photos are named; "
           "every other person trains the embedder")
COPIED_FILES = ("README.md", "model.py", "handler.py", "requirements.txt", "LICENSE")

# Chart colours (validated categorical slots + a one-hue sequential ramp, see the dataviz palette).
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = {"train": "#2a78d6", "val": "#eb6834", "unseen": "#1f9e89"}
PAIRS = {"same person": "#2a78d6", "different people": "#eb6834"}
BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f8fe", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])


def model_name(cfg) -> str:
    members = " + ".join(f"{e['arch']} ({'ImageNet init' if e['pretrained'] else 'from scratch'})" for e in cfg.embedders)
    return f"FaceDetector + 5-point alignment + face embedders trained with CosFace [{members}] + MLP head"


def build_metrics(cfg, scores: dict[str, float], *, data: dict, params: dict[str, int], train_time_s: float | None,
                  device: str, source: str, experiments: dict | None = None, extra: dict | None = None) -> dict:
    """metrics.json content (schema: STANDARD §4). `scores` = the deployed pipeline's test metrics."""
    metrics = {
        "model": model_name(cfg),
        "task": TASK,
        "dataset": DATASET,
        "split": "test",
        "primary_metric": {"name": "accuracy", "value": scores["accuracy"]},
        "metrics": scores,
        "data": data,
        "params": int(params["total"]),
        "params_breakdown": {k: int(v) for k, v in params.items()},
        "train_time_s": None if train_time_s is None else round(train_time_s, 1),
        "device": device,
        "smoke": cfg.smoke,
        "source": source,
        "versions": utils.lib_versions("torch", "torchvision", "sklearn", "huggingface_hub", "safetensors", "numpy"),
        "trained_at": utils.today(),
    }
    if experiments:
        metrics["experiments"] = experiments
    metrics.update(extra or {})
    return metrics


def prepare_model_dir(model_dir: str | Path) -> Path:
    """Create `model_dir`; when it is not ../model, copy the code and the card into it."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if model_dir.resolve() != utils.MODEL_DIR.resolve():
        for name in COPIED_FILES:
            shutil.copy2(utils.MODEL_DIR / name, model_dir / name)
    return model_dir


def save_model(det, net, model_dir: str | Path) -> Path:
    """face_detector.safetensors + model.safetensors / config.json into `model_dir`."""
    model_dir = prepare_model_dir(model_dir)
    save_file({k: v.detach().cpu().contiguous() for k, v in det.state_dict().items()}, model_dir / M.DETECTOR_WEIGHTS)
    net.save_pretrained(model_dir)
    return model_dir


def export(det, net, model_dir: str | Path, metrics: dict, *, class_names: list[str], y_true, y_pred,
           history: dict, head_history: dict, detector_history: dict | None = None, detector_ap: dict | None = None,
           experiments: dict | None = None, tsne_embeddings=None, tsne_labels=None, test_embeddings=None,
           test_labels=None, verify_threshold: float = 0.5, tsne_max_points: int = 1500, seed: int = 42) -> dict:
    """Write weights, config, metrics and plots to `model_dir`, then update its model card."""
    model_dir = save_model(det, net, model_dir)   # README.md exists and is kept
    assets = model_dir / "assets"
    if detector_history:
        plot_detector_history(detector_history, assets / "detector_training.png")
    if detector_ap:
        plot_detector_ap(detector_ap, assets / "detector_ap.png")
    plot_confusion_matrix(y_true, y_pred, class_names, assets / "confusion_matrix.png")
    plot_history(history, head_history, assets / "training_curves.png")
    if experiments:
        plot_experiments(experiments, assets / "model_comparison.png")
    if test_embeddings is not None:
        plot_verification(test_embeddings, test_labels, verify_threshold, assets / "verification_distances.png")
    if tsne_embeddings is not None:
        plot_tsne(tsne_embeddings, tsne_labels, assets / "tsne_embeddings.png", tsne_max_points, seed)
    utils.save_json(metrics, model_dir / "metrics.json")
    utils.update_model_card(model_dir, metrics)
    print(f"exported -> {model_dir}")
    return metrics


def plot_history(histories: dict[str, dict], head_history: dict, path) -> Path:
    """One row per embedder: CosFace loss, then nearest-centroid accuracy on the 42 people (train vs
    validation) and verification AUC on the unseen people; last panel: the identity head, train vs
    validation accuracy. A growing train/validation gap = overfitting. Dashed line = kept epoch."""
    rows = len(histories)
    fig = Figure(figsize=(10.0, 2.8 * rows), dpi=150, facecolor=SURFACE, layout="constrained")
    axes = np.atleast_2d(fig.subplots(rows, 3))
    for r, (name, history) in enumerate(histories.items()):
        panels = [({"train": history["train_loss"]}, f"{name}: CosFace loss"),
                  ({"train": history["train_acc"], "val": history["val_acc"], "unseen": history["unseen_auc"]},
                   "accuracy (42 people) / AUC (unseen people)")]
        if r == 0:
            panels.append(({"train": head_history["train_acc"], "val": head_history["val_acc"]}, "Identity head: accuracy"))
        for c, (series, title) in enumerate(panels):
            ax = axes[r, c]
            _style(ax)
            ax.grid(True, color=GRID, linewidth=0.6)
            for split, values in series.items():
                ax.plot(range(1, len(values) + 1), values, color=SERIES[split], linewidth=2, label=split,
                        marker="o" if len(values) < 5 else None)
            best = (head_history if c == 2 else history).get("best_epoch")
            if best and c > 0:
                ax.axvline(best, color=INK_MUTED, linewidth=1, linestyle="--")
            ax.set_title(title, color=INK, fontsize=8, loc="left")
            ax.set_xlabel("epoch", color=INK_MUTED)
            ax.legend(frameon=False, fontsize=7, labelcolor=INK)
        for c in range(len(panels), 3):
            axes[r, c].set_visible(False)
    return _save(fig, path)


def plot_detector_history(history: dict, path) -> Path:
    """Detector: training losses per epoch, and AP on the held-out Open Images photos per size band."""
    fig = Figure(figsize=(9.0, 3.0), dpi=150, facecolor=SURFACE, layout="constrained")
    ax1, ax2 = fig.subplots(1, 2)
    colors = ["#2a78d6", "#eb6834", "#1f9e89"]
    for ax in (ax1, ax2):
        _style(ax)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_xlabel("epoch", color=INK_MUTED)
    epochs = range(1, len(history["loss_face"]) + 1)
    for key, color in zip(("loss_face", "loss_box", "loss_landmarks"), colors):
        ax1.plot(epochs, history[key], color=color, linewidth=2, label=key.replace("loss_", ""),
                 marker="o" if len(history[key]) < 5 else None)
    ax1.set_title("Face detector: training losses", color=INK, fontsize=9, loc="left")
    for band, color in zip(("large", "medium", "small"), colors):
        ax2.plot(history["val_epoch"], history[f"val_{band}"], color=color, linewidth=2, marker="o", label=f"{band} faces")
    if history.get("best_epoch"):
        ax2.axvline(history["best_epoch"], color=INK_MUTED, linewidth=1, linestyle="--")
    ax2.set_title("Held-out Open Images photos: AP (IoU 0.5)", color=INK, fontsize=9, loc="left")
    for ax in (ax1, ax2):
        ax.legend(frameon=False, fontsize=7, labelcolor=INK)
    return _save(fig, path)


def plot_detector_ap(scores: dict[str, dict[str, float]], path) -> Path:
    """Grouped bars: detector x size band (AP at IoU 0.5), e.g. ours vs the reference detector on WIDER FACE."""
    names, bands = list(scores), ("large", "medium", "small")
    colors = ["#2a78d6", "#eb6834", "#1f9e89", "#8a5cd6"]
    fig = Figure(figsize=(7.0, 3.2), dpi=150, facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    _style(ax)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    width = 0.8 / len(names)
    for n_i, (name, color) in enumerate(zip(names, colors)):
        xs = np.arange(len(bands)) + (n_i - (len(names) - 1) / 2) * width
        ys = [scores[name][b] for b in bands]
        ax.bar(xs, ys, width=width * 0.92, color=color, label=name)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.01, f"{y:.3f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(range(len(bands)), [f"{b} faces\n(>= {h} px)" for b, h in zip(bands, (64, 32, 16))], fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_title("Face detection AP (IoU 0.5), WIDER FACE val", color=INK, fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    return _save(fig, path)


def plot_experiments(experiments: dict[str, dict], path, keys=("val_accuracy", "test_accuracy", "unseen_auc")) -> Path | None:
    """Horizontal grouped bars: every experiment x (validation accuracy, test accuracy, unseen-people AUC).
    Returns None (no plot) when there are no experiments."""
    names = list(experiments)
    if not names:
        return None
    colors = ["#2a78d6", "#eb6834", "#1f9e89"]
    fig = Figure(figsize=(8.0, 0.55 * len(names) + 1.2), dpi=150, facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    _style(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    height = 0.8 / len(keys)
    values = [experiments[n][k] for n in names for k in keys if experiments[n].get(k) is not None]
    low = max(0.0, min(values) - 0.02)
    for k_i, (key, color) in enumerate(zip(keys, colors)):
        for n_i, name in enumerate(names):
            v = experiments[name].get(key)
            if v is None:
                continue
            y = n_i + (k_i - (len(keys) - 1) / 2) * height
            ax.barh(y, v - low, height=height * 0.92, left=low, color=color,
                    label=key.replace("_", " ") if n_i == 0 else None)
            ax.text(v + 0.001, y, f"{v:.3f}", va="center", ha="left", fontsize=6, color=INK)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(low, 1.0 + (1.0 - low) * 0.08)
    ax.set_title("Embedder experiments (chosen on validation; test looked at once)", color=INK, fontsize=9,
                 loc="left", pad=22)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    return _save(fig, path)


# --------------------------------------------------------------------------- plots
def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _save(fig: Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path


def plot_confusion_matrix(y_true, y_pred, class_names: list[str], path) -> Path:
    """Counts per (true, predicted) person; off-diagonal cells are the mistakes (annotated)."""
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    size = max(4.6, min(11.0, 0.22 * n + 2.5))
    fig = Figure(figsize=(size, size * 0.92), dpi=150, facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    _style(ax)
    ax.imshow(cm, cmap=BLUES, vmin=0, vmax=max(cm.max(), 1))
    for i, j in zip(*np.nonzero(cm)):
        if i != j or n <= 12:
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="#b3261e" if i != j else INK, fontweight="bold")
    font = 6 if n > 20 else 8
    ax.set_xticks(range(n), class_names, rotation=90, fontsize=font)
    ax.set_yticks(range(n), class_names, fontsize=font)
    ax.set_xlabel("Predicted", color=INK_MUTED)
    ax.set_ylabel("True", color=INK_MUTED)
    correct, total = int(np.trace(cm)), int(cm.sum())
    ax.set_title(f"MLP head, test split: {correct}/{total} correct ({correct / max(total, 1):.2%}); "
                 "red = mistakes", color=INK, fontsize=9, loc="left")
    return _save(fig, path)


def plot_verification(embeddings, labels, threshold: float, path) -> Path:
    """Cosine-distance histograms of same-person vs different-people test pairs + the threshold."""
    from .engine import pair_distances

    dist, same = pair_distances(np.asarray(embeddings), np.asarray(labels))
    fig = Figure(figsize=(7.2, 3.2), dpi=150, facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    _style(ax)
    bins = np.linspace(0, max(1.6, float(dist.max())), 60)
    for (name, color), mask in zip(PAIRS.items(), (same, ~same)):
        ax.hist(dist[mask], bins=bins, density=True, alpha=0.75, color=color,
                label=f"{name} ({int(mask.sum()):,} pairs)")
    ax.axvline(threshold, color=INK, linewidth=1.2, linestyle="--")
    ax.text(threshold, ax.get_ylim()[1] * 0.95, f"  threshold {threshold}", color=INK, fontsize=8, va="top")
    ax.set_xlabel("cosine distance between embeddings", color=INK_MUTED)
    ax.set_ylabel("density", color=INK_MUTED)
    ax.set_title("Verification: all pairs of test faces (threshold picked on validation pairs)", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    return _save(fig, path)


def plot_tsne(embeddings, labels, path, max_points: int = 1500, seed: int = 42) -> Path:
    """2-D t-SNE of the face embeddings, one colour per person (legend omitted: 42 people)."""
    from sklearn.manifold import TSNE

    embeddings, labels = np.asarray(embeddings), np.asarray(labels)
    if len(embeddings) > max_points:
        keep = np.random.default_rng(seed).permutation(len(embeddings))[:max_points]
        embeddings, labels = embeddings[keep], labels[keep]
    perplexity = float(min(30, max(2, len(embeddings) // 5)))
    pts = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto",
               random_state=seed).fit_transform(embeddings)
    fig = Figure(figsize=(6.4, 5.2), dpi=150, facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    _style(ax)
    import matplotlib

    cmap = matplotlib.colormaps["tab20"]
    for label in np.unique(labels):
        mask = labels == label
        ax.scatter(pts[mask, 0], pts[mask, 1], s=8, alpha=0.8, color=cmap(int(label) % 20), linewidths=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"t-SNE of the embeddings: {len(np.unique(labels))} people, {len(pts)} faces "
                 "(each tight cluster = one person)", color=INK, fontsize=9, loc="left")
    return _save(fig, path)
