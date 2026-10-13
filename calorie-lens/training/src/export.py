"""Assemble the deployed model, write it with metrics + graphs into model/, refresh the model card.

  assemble()   the winning stage-1 network + NutriText -> one CalorieLens whose config carries the
               readout fitted on validation (mass blend, kcal blend, kcal scale), the test-time views and the
               display threshold (built from the winner's own saved config, so nothing is downloaded and the
               DINOv2 / ConvNeXt shape is exact)
  export()     model.safetensors + config.json + nutritext_tokenizer.json, metrics.json, assets/*.png,
               NOTICE (+ the Apache-2.0 text) when the shipped backbone is DINOv2, then the model card's
               YAML, metrics table and Experiments table. Exporting anywhere but model/ (smoke runs)
               first copies the card and code there, so model/ is never touched by a smoke run.
"""
from __future__ import annotations

import re
import shutil
import urllib.request
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

import model as M

from . import utils

TASK = "image -> ingredient grams -> nutrition (multi-label + regression)"
DATASET = ("Nutrition5k (CC BY 4.0) cleaned + FlavoriaFoodWeight1700 (CC BY 4.0) + USDA FoodData Central (CC0): "
           "shalev396/calorie-lens-data")
COPIED_FROM_MODEL_DIR = ("README.md", "model.py", "handler.py", "requirements.txt")
WEIGHT_FILES = (M.WEIGHTS_FILE, "config.json", M.TOKENIZER_FILE)
NOTICE_FILES = ("NOTICE", "LICENSE-APACHE-2.0.txt")
APACHE_URL = "https://www.apache.org/licenses/LICENSE-2.0.txt"
_EXP_START, _EXP_END = "<!-- experiments:start -->", "<!-- experiments:end -->"

# Chart colours: the validated reference palette (categorical slots 1-4 in fixed order; slots 3-4 are
# under 3:1 on the light surface, so every multi-series chart carries a legend and value labels).
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
BEST, OTHER = "#2a78d6", "#b9b8b3"


# --------------------------------------------------------------------------- model files
STAGE1_PARTS = ("backbone.", "decoder.", "totals_head.")
READOUT_DEFAULTS = {"mass_blend": 1.0, "kcal_blend": 0.0, "kcal_scale": 0.0}


def assemble(stage1: M.CalorieLens, nutritext: dict | None, readout: dict, presence_threshold: float,
             tta_views: int = 1) -> M.CalorieLens:
    """One deployable CalorieLens: the stage-1 weights + NutriText ({"net", "kwargs"}) + the readout fitted on
    validation ({"mass_blend", "kcal_blend", "kcal_scale"}), test-time views and display threshold."""
    config = dict(stage1._hub_mixin_config)
    config.update({k: float(readout.get(k, d)) for k, d in READOUT_DEFAULTS.items()})
    config.update(tta_views=int(tta_views), presence_threshold=float(presence_threshold), pretrained=False,
                  nutritext=nutritext["kwargs"] if nutritext else None)
    final = M.CalorieLens(**config)
    state = {k: v for k, v in stage1.state_dict().items() if k.startswith(STAGE1_PARTS)}
    missing, unexpected = final.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith("nutritext.") for k in missing):
        raise RuntimeError(f"stage-1 weights do not fit: missing {missing[:5]}, unexpected {unexpected[:5]}")
    if nutritext:
        final.nutritext.load_state_dict(nutritext["net"].state_dict())
    return final.eval()


def save_model(net: M.CalorieLens, tokenizer, target: str | Path) -> Path:
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    net.save_pretrained(target)
    if tokenizer is not None:
        tokenizer.save(str(target / M.TOKENIZER_FILE))
    return target


def apache_sources(net: M.CalorieLens) -> list[str]:
    """The model's pretrained weights if they are distributed under Apache-2.0: DINOv2 (any size) or a timm
    ConvNeXt checkpoint (Hub cards: apache-2.0). torchvision's ImageNet-1k ConvNeXt is BSD: nothing to add."""
    src = (net.backbone_weights or M.DINO_BASE) if net.backbone_name == "dinov2" else net.backbone_weights
    return [src] if src and (net.backbone_name == "dinov2" or src.startswith("timm/")) else []


def write_notice(model_dir: Path, sources: list[str]) -> None:
    """Apache-2.0 pretrained weights (`sources`): ship NOTICE + the licence text with the fine-tuned weights."""
    if not sources:
        for name in NOTICE_FILES:
            (model_dir / name).unlink(missing_ok=True)
        return
    origin = {True: ("DINOv2 (https://github.com/facebookresearch/dinov2), Copyright (c) Meta Platforms, Inc. "
                     "and affiliates"),
              False: ("ConvNeXt (https://github.com/facebookresearch/ConvNeXt, Copyright (c) Meta Platforms, Inc. "
                      "and affiliates), as published by timm (https://github.com/huggingface/pytorch-image-models)")}
    lines = [f"- {s}: {origin['dino' in s.lower()]}" for s in sources]
    (model_dir / "NOTICE").write_text(
        "Calorie Lens\nCopyright (c) 2026 Shalev Ben Moshe (MIT, see LICENSE)\n\n"
        "The image backbone in model.safetensors was fine-tuned from these pretrained weights, licensed under the\n"
        "Apache License 2.0 (LICENSE-APACHE-2.0.txt):\n" + "\n".join(lines) + "\n"
        "The fine-tuned weights were modified by training on Nutrition5k and FlavoriaFoodWeight1700.\n", encoding="utf-8")
    with urllib.request.urlopen(APACHE_URL, timeout=60) as r:
        (model_dir / "LICENSE-APACHE-2.0.txt").write_bytes(r.read())


def build_metrics(*, headline: dict, test: dict, test_clean: dict, test_original: dict, ci: tuple, experiments: dict,
                  nutritext: dict, ablation: dict, mass_blend: dict, threshold: float, winner: str, data: dict,
                  params: dict, train_time_s: float, device: str, smoke: bool, **extra) -> dict:
    """metrics.json: headline numbers flat in `metrics` (model card / model-index), the rest nested."""
    o, sd = test.get("overhead", {}), test.get("side_dish", {})
    flat = {"kcal_mae": o.get("kcal_mae"), "kcal_pmae": o.get("kcal_pmae"), "mass_pmae": o.get("mass_pmae"),
            "fat_pmae": o.get("fat_pmae"), "carb_pmae": o.get("carb_pmae"), "protein_pmae": o.get("protein_pmae"),
            "side_dish_kcal_pmae": sd.get("kcal_pmae"), "test_clean_kcal_pmae": test_clean.get("overhead", {}).get("kcal_pmae"),
            "presence_map": o.get("presence_map"), "ingredient_grams_mae": o.get("ingredient_grams_mae"),
            "flavoria_mass_pmae": test.get("flavoria", {}).get("mass_pmae"),
            "nutritext_kcal_mae_per_100g": nutritext.get("test", {}).get("generic", {}).get("nutritext", {}).get("kcal_mae")}
    return {
        "model": f"Calorie Lens ({winner})", "variant": winner, "task": TASK, "dataset": DATASET, "split": "test",
        "primary_metric": {"name": "kcal_pmae", "value": headline["kcal_pmae"]},
        "metrics": {k: round(float(v), 4) for k, v in flat.items() if v is not None},
        "test": test, "test_clean": test_clean, "test_original_labels": test_original,
        "kcal_pmae_ci95_overhead": list(ci), "experiments": experiments, "nutritext": nutritext,
        "ablation": ablation, "mass_blend": mass_blend, "presence_threshold": threshold, "data": data,
        "params": int(params["total"]), "params_breakdown": params, "train_time_s": round(train_time_s, 1),
        "device": device, "smoke": smoke, "source": "training/notebook.ipynb",
        "versions": utils.lib_versions("torch", "torchvision", "transformers", "tokenizers", "huggingface_hub",
                                       "safetensors", "numpy"),
        "trained_at": utils.today(), **extra}


def experiments_markdown(metrics: dict) -> str:
    """Every candidate: validation score (picks the winner) and test calorie / mass errors."""
    rows = metrics.get("experiments", {})
    head = ("| candidate | val score | test kcal PMAE (overhead) | test mass PMAE | side-dish kcal PMAE | Flavoria weight PMAE "
            "| views | params |\n|---|---|---|---|---|---|---|---|")
    lines = []
    for name, r in rows.items():
        cells = [f"{r.get('val_score', float('nan')):.2f}", f"{r.get('test_kcal_pmae', float('nan')):.2f}%",
                 f"{r.get('test_mass_pmae', float('nan')):.2f}%", f"{r.get('test_side_dish_kcal_pmae', float('nan')):.2f}%",
                 f"{r['flavoria_mass_pmae']:.2f}%" if r.get("flavoria_mass_pmae") is not None else "-",
                 str(r.get("views", 1)), utils.human_params(int(r.get("params", 0)))]
        if name == metrics.get("variant"):
            lines.append(f"| **{name} (deployed)** | " + " | ".join(f"**{c}**" for c in cells) + " |")
        else:
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return f"{_EXP_START}\n{head}\n" + "\n".join(lines) + f"\n{_EXP_END}"


def write_experiments_table(model_dir: Path, metrics: dict) -> None:
    path = model_dir / "README.md"
    text = path.read_text(encoding="utf-8")
    if _EXP_START in text and _EXP_END in text:
        table = experiments_markdown(metrics)
        text = re.sub(re.escape(_EXP_START) + r".*?" + re.escape(_EXP_END), lambda _: table, text, flags=re.S)
        path.write_text(text, encoding="utf-8", newline="\n")


def export(net: M.CalorieLens, tokenizer, metrics: dict, model_dir: str | Path, assets_dir: str | Path) -> Path:
    """Write weights + tokenizer, NOTICE, metrics.json and graphs to model_dir; refresh its card."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if model_dir.resolve() != utils.MODEL_DIR.resolve():
        for name in COPIED_FROM_MODEL_DIR:
            shutil.copy2(utils.MODEL_DIR / name, model_dir / name)
    save_model(net, tokenizer, model_dir)
    write_notice(model_dir, apache_sources(net))
    (model_dir / "assets").mkdir(exist_ok=True)
    for old in (model_dir / "assets").glob("*.png"):
        old.unlink()                                   # a graph this run did not make would be stale
    for src in Path(assets_dir).glob("*.png"):
        shutil.copy2(src, model_dir / "assets" / src.name)
    utils.save_json(metrics, model_dir / "metrics.json")
    utils.update_model_card(model_dir, metrics)
    write_experiments_table(model_dir, metrics)
    print(f"exported '{metrics['variant']}' -> {model_dir}")
    return model_dir


def params_breakdown(net: M.CalorieLens) -> dict:
    count = lambda m: sum(p.numel() for p in m.parameters()) if m is not None else 0
    parts = {"backbone": count(net.backbone), "query_decoder": count(net.decoder), "totals_head": count(net.totals_head),
             "nutritext": count(net.nutritext), "nutrition_layer_frozen_weights": int(net.nutrition.weight.numel())}
    parts["total"] = count(net)
    return parts


# --------------------------------------------------------------------------- plots
def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig: Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path


def plot_histories(histories: dict[str, list[dict]], path, title: str = "Training: every experiment") -> Path:
    """Two small multiples, one line per experiment (fixed colour order, direct end labels):
    training loss, and the validation score that picks the epoch."""
    fig = Figure(figsize=(8.0, 3.4), dpi=150, facecolor=SURFACE, layout="constrained")
    fig.suptitle(title, color=INK, fontsize=10, x=0.01, ha="left")
    for ax, key, label in zip(fig.subplots(1, 2), ("train", "val_score"),
                              ("Training loss (weighted sum)", "Validation score: mean kcal % error (lower is better)")):
        _style(ax)
        for (name, hist), color in zip(histories.items(), SLOTS):
            ep = [h["epoch"] for h in hist]
            y = [h["train"]["total"] if key == "train" else h["val_score"] for h in hist]
            ax.plot(ep, y, color=color, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
                    markeredgewidth=1, label=name)
            ax.annotate(name.split()[0], (ep[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=8, color=INK, va="center")
        ax.set_title(label, color=INK_MUTED, fontsize=9, loc="left")
        ax.set_xlabel("epoch", color=INK_MUTED)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    return _save(fig, path)


def plot_bars(rows: dict[str, float], highlight: str | None, path, title: str, xlabel: str, fmt: str = "{:.1f}") -> Path:
    """One horizontal bar per row (value printed at the end); `highlight` in blue, the rest grey."""
    names = list(rows)
    fig = Figure(figsize=(8.0, 1.1 + 0.42 * len(names)), dpi=150, facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    _style(ax)
    ax.grid(False, axis="y")
    vals = [rows[n] for n in names]
    ax.barh(range(len(names)), vals, height=0.62, color=[BEST if n == highlight else OTHER for n in names])
    for i, v in enumerate(vals):
        ax.text(v, i, " " + fmt.format(v), va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.18 if vals else 1)
    ax.set_xlabel(xlabel, color=INK_MUTED)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    return _save(fig, path)


def plot_scatter(pred: np.ndarray, true: np.ndarray, path, title: str) -> Path:
    """Predicted vs true kcal per dish (overhead test), with the y = x line."""
    fig = Figure(figsize=(5.2, 5.0), dpi=150, facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    _style(ax)
    hi = float(max(pred.max(), true.max()) * 1.05)
    ax.plot([0, hi], [0, hi], color=INK_MUTED, linewidth=1, linestyle="--")
    ax.scatter(true, pred, s=14, color=SLOTS[0], alpha=0.55, edgecolors=SURFACE, linewidths=0.5)
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_xlabel("true kcal", color=INK_MUTED); ax.set_ylabel("predicted kcal", color=INK_MUTED)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    return _save(fig, path)


def plot_error_by_bin(pred: np.ndarray, true: np.ndarray, path, edges=(0, 100, 200, 300, 400, 600, 10_000)) -> Path:
    """Mean absolute calorie error per true-calorie bin (dish counts in the labels)."""
    rows = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (true >= lo) & (true < hi)
        if m.any():
            rows[f"{lo}-{hi if hi < 10_000 else '+'} kcal (n={int(m.sum())})"] = float(np.abs(pred[m] - true[m]).mean())
    return plot_bars(rows, None, path, "Test (overhead): calorie error by plate size", "mean absolute error (kcal)")


def plot_nutritext(evals: dict, path) -> Path:
    """R^2 per nutrient on held-out generic USDA foods: NutriText vs TF-IDF kNN vs the mean (grouped bars)."""
    part = evals["generic"]
    methods = [m for m in ("nutritext", "tfidf_knn", "mean") if m in part]
    nutrients = [k[:-3] for k in part[methods[0]] if k.endswith("_r2")]
    fig = Figure(figsize=(8.0, 3.6), dpi=150, facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    _style(ax)
    ax.grid(False, axis="x")
    width = 0.8 / len(methods)
    for j, (meth, color) in enumerate(zip(methods, SLOTS)):
        vals = [max(part[meth].get(f"{n}_r2", 0.0), 0.0) for n in nutrients]
        xs = np.arange(len(nutrients)) + (j - (len(methods) - 1) / 2) * width
        ax.bar(xs, vals, width=width * 0.9, color=color, label=meth)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(range(len(nutrients)), nutrients)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("R² (held-out generic foods)", color=INK_MUTED)
    ax.set_title(f"NutriText (stage 2b) vs baselines, {part['n']} unseen USDA foods", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, ncol=3, loc="upper right")
    return _save(fig, path)


def save_heatmap_example(pred: M.Predictor, image, path) -> Path:
    """The Space's view of one test photo: the photo, then where the top ingredient queries looked."""
    result, maps = pred.analyze(image, top_k=4)
    tiles = [M.to_pil(image)] + list(maps.values())
    names = ["photo"] + [f"{r['name']} {r['grams']:.0f} g" for r in result["ingredients"]]
    fig = Figure(figsize=(2.4 * len(tiles), 2.4), dpi=150, facecolor=SURFACE, layout="constrained")
    for ax, tile, name in zip(np.atleast_1d(fig.subplots(1, len(tiles))), tiles, names):
        ax.imshow(tile)
        ax.set_axis_off()
        ax.set_title(name, fontsize=8, color=INK)
    return _save(fig, path)
