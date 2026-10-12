"""Training and evaluation of the fused model (stages 1 + 2a).

Losses (every regression target divided by its train mean, so they weigh alike):
  presence      asymmetric loss (ASL) on "is ingredient k on the plate" per positive label; most of
                the 253 are absent
  grams         |log1p(grams) - log1p(true)| of the ingredients that are present (their own scale)
  soft          sum_k |sigmoid(logit_k) x grams_k - true_k| / mean plate mass: the K-neuron layer itself
  totals_head   |log1p(direct totals) - log1p(true)| for kcal, mass, fat, carb, protein
  totals_abs    |direct totals - true| / train mean: the same head on the scale of the metric (percent MAE
                is dominated by big plates; the log loss alone weighs a 50 and an 800 kcal plate alike)
  nutrition     |NutritionLayer(reconciled grams) - NutritionLayer(true grams)| over all 8 nutrients:
                end to end through the frozen stage-2a table
  depth         (E3) scale-invariant log error + L1 of the auxiliary depth map, overhead scans only
Side-angle frames count side_loss_weight on the mass / nutrient losses (part of the plate can be
out of view) and fully on presence. Each loss only uses the labels a sample has (Flavoria plates: the weight
and the mapped ingredients' grams).

Reading the model (fitted on validation, never on test): the mass blend, the kcal blend (the grams tilted
toward the totals head's kcal per gram, model.CalorieLens.tilt), the kcal scale and the number of test-time
views. Every combination is computed from per-view raw outputs (`Preds`), exactly as
model.CalorieLens.forward combines them.

The loop: AdamW (fused), warmup + cosine per step, bf16 autocast, gradient accumulation to the
effective batch, clipping, an EMA of the weights (evaluated and kept), torch.compile with an eager
fallback, a last.pt checkpoint every epoch (resumed when the run key matches, as in flan-t5) and
early stopping on the validation score. If a step runs out of GPU memory, batches are split into
more chunks instead of crashing.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

import model as M

from . import utils
from .data_setup import gpu_augment
from .model_builder import maybe_compile, param_groups

TARGETS = M.TOTALS                       # kcal, mass, fat, carb, protein: the paper's five


# --------------------------------------------------------------------------- losses
def make_norms(labels, device) -> dict:
    train = labels.split == "train"
    nutr = labels.grams[train] @ labels.densities
    return {"mass": float(labels.totals[train, 1].mean()),
            "nutr": torch.tensor(np.maximum(nutr.mean(0), 1e-3), device=device, dtype=torch.float32),
            "tot": torch.tensor(np.maximum(labels.totals[train].mean(0), 1e-3), device=device, dtype=torch.float32),
            "dens": torch.tensor(labels.densities, device=device, dtype=torch.float32)}


def asl_loss(logits: torch.Tensor, y: torch.Tensor, gamma_neg: float = 4.0, gamma_pos: float = 0.0,
             clip: float = 0.05, known: torch.Tensor | None = None) -> torch.Tensor:
    """Asymmetric loss (Ridnik et al., 2021): down-weights the many easy negatives of multi-label data.
    Summed over the K classes and divided by the number of positive labels in the batch (as focal
    losses normalise), so it stays comparable to the regression losses whatever K is. known: (N, K) mask of
    the labels that exist (a Flavoria plate only knows its mapped ingredients are present)."""
    p = torch.sigmoid(logits)
    p_neg = (p - clip).clamp(min=0) if clip else p
    loss_pos = y * torch.log(p.clamp(min=1e-8)) * (1 - p) ** gamma_pos
    loss_neg = (1 - y) * torch.log((1 - p_neg).clamp(min=1e-8)) * p_neg ** gamma_neg
    if known is not None:
        y, loss_pos, loss_neg = y * known, loss_pos * known, loss_neg * known
    return -(loss_pos + loss_neg).sum() / y.sum().clamp_min(1.0)


def _wmean(v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (v * w).sum() / w.sum().clamp_min(1e-6)


def depth_loss(pred: torch.Tensor, gt: torch.Tensor, overhead: torch.Tensor) -> torch.Tensor:
    """Scale-invariant log error (Eigen et al.) + L1 in cm, on valid pixels of overhead scans."""
    gt = F.interpolate(gt[:, None], size=pred.shape[-2:], mode="nearest")[:, 0]
    valid = (gt > 0) & overhead[:, None, None]
    if not valid.any():
        return pred.sum() * 0.0
    d = (torch.log(pred.clamp_min(1e-3)) - torch.log(gt.clamp_min(1e-3))) * valid
    n = valid.sum((1, 2)).clamp_min(1)
    si = (d ** 2).sum((1, 2)) / n - 0.5 * (d.sum((1, 2)) / n) ** 2
    l1 = ((pred - gt).abs() * valid).sum((1, 2)) / n * 100
    keep = valid.flatten(1).any(1)
    return (si[keep] + 0.1 * l1[keep]).mean()


def compute_losses(out: dict, batch: dict, depth: torch.Tensor | None, norms: dict, cfg) -> dict:
    """Every loss over the labels a sample has: Nutrition5k dishes know everything; a Flavoria plate knows its
    weight and its mapped ingredients' grams, so it trains presence (positives), grams and the mass of the totals
    head, never the losses that need the whole composition (soft grams, NutritionLayer)."""
    g, tot = batch["grams"].float(), batch["totals"].float()
    kg = batch.get("known_grams", torch.ones_like(g, dtype=torch.bool)).bool()
    kt = batch.get("known_totals", torch.ones_like(tot, dtype=torch.bool)).float()
    full = kg.all(1)                                       # the whole composition is known
    overhead = batch["overhead"].bool()
    w = torch.where(overhead, 1.0, cfg.side_loss_weight)
    present = (g > 0) & kg
    losses = {"presence": asl_loss(out["logits"].float(), present.float(), known=kg.float())}
    lg = torch.log1p(torch.exp(out["log_grams"].float()))
    per = ((lg - torch.log1p(g)).abs() * present).sum(1) / present.sum(1).clamp_min(1)
    losses["grams"] = _wmean(per, w * present.any(1))
    losses["soft"] = _wmean((out["soft"] - g).abs().sum(1) / norms["mass"], w * full)
    known_mean = lambda e: (e * kt).sum(1) / kt.sum(1).clamp_min(1.0)
    losses["totals_head"] = _wmean(known_mean((out["log_totals"].float() - torch.log1p(tot)).abs()), w)
    if cfg.loss_weights.get("totals_abs"):
        direct = torch.expm1(out["log_totals"].float().clamp(0.0, M.MAX_LOG_GRAMS))
        losses["totals_abs"] = _wmean(known_mean((direct - tot).abs() / norms["tot"]), w)
    target = g @ norms["dens"]
    losses["nutrition"] = _wmean(((out["totals"].float() - target).abs() / norms["nutr"]).mean(1), w * full)
    if "depth" in out and depth is not None:
        losses["depth"] = depth_loss(out["depth"], depth, overhead)
    losses["total"] = sum(cfg.loss_weights[k] * v for k, v in losses.items())
    return losses


# --------------------------------------------------------------------------- evaluation
@dataclass
class Preds:
    """Per-image raw outputs, enough to recompute any readout / threshold / aggregation / view average offline."""
    dish: np.ndarray            # (N,) dish row
    overhead: np.ndarray        # (N,) bool
    soft: np.ndarray            # (N, K) sigmoid(logit) x grams
    prob: np.ndarray            # (N, K)
    log_totals: np.ndarray      # (N, 5) direct totals head in log1p space (TARGETS), as the model outputs it

    def subset(self, mask: np.ndarray) -> "Preds":
        return Preds(*(getattr(self, f)[mask] for f in self.__dataclass_fields__))

    @property
    def direct(self) -> np.ndarray:
        """(N, 5) the totals head in grams / kcal."""
        return np.expm1(np.clip(self.log_totals, 0.0, M.MAX_LOG_GRAMS))

    @property
    def mass_direct(self) -> np.ndarray:
        return self.direct[:, 1]


def _to(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def evaluate_views(net: M.CalorieLens, loader, device, views: int = 1) -> list[Preds]:
    """Run one (uncompiled, EMA) network over a loader; one Preds per test-time view, in model.tta_views
    order (the image, its mirror, upside down, rotated 180 degrees)."""
    assert net.tta_views == 1, "evaluate one view at a time; views are combined here (combine)"
    net.eval()
    parts = [{k: [] for k in Preds.__dataclass_fields__} for _ in range(views)]
    for batch in loader:
        batch = _to(batch, device)
        x, _ = gpu_augment(batch, train=False)
        for v, xv in enumerate(M.tta_views(x, views).chunk(views)):
            with utils.autocast(device):
                out = net(xv)
            parts[v]["dish"].append(batch["dish"].cpu().numpy())
            parts[v]["overhead"].append(batch["overhead"].cpu().numpy())
            parts[v]["soft"].append(out["soft"].float().cpu().numpy())
            parts[v]["prob"].append(torch.sigmoid(out["logits"].float()).cpu().numpy())
            parts[v]["log_totals"].append(out["log_totals"].float().cpu().numpy())
    return [Preds(*(np.concatenate(p[k]) for k in Preds.__dataclass_fields__)) for p in parts]


def evaluate(net: M.CalorieLens, loader, device) -> Preds:
    return evaluate_views(net, loader, device, 1)[0]


def combine(preds: list[Preds]) -> Preds:
    """Mean over networks / views of the same images, as model.CalorieLens.forward averages them:
    presence probabilities, soft grams and log totals."""
    first = preds[0]
    assert all(np.array_equal(p.dish, first.dish) for p in preds), "every view must cover the same images"
    mean = lambda f: np.mean([getattr(p, f) for p in preds], axis=0).astype(np.float32)
    return Preds(first.dish, first.overhead, mean("soft"), mean("prob"), mean("log_totals"))


def dish_mean(p: Preds) -> Preds:
    """Average all frames of each dish (multi-view inference: 4 cameras x 5 frames); totals in grams / kcal."""
    dishes, inv = np.unique(p.dish, return_inverse=True)
    n = np.bincount(inv).astype(np.float32)[:, None]

    def avg(a):
        out = np.zeros((len(dishes), *a.shape[1:]), np.float32)
        np.add.at(out, inv, a)
        return out / n.reshape(-1, *([1] * (a.ndim - 1)))
    return Preds(dishes, np.zeros(len(dishes), bool), avg(p.soft), avg(p.prob), np.log1p(avg(p.direct)))


PLAIN = {"mass_blend": 1.0, "kcal_blend": 0.0, "kcal_scale": 0.0}   # readout during training: mass head only


def _ro(ro: dict | None) -> dict:
    """A readout: {"mass_blend", "kcal_blend", "kcal_scale"} (missing keys = PLAIN; extra keys ignored)."""
    return {k: float((ro or {}).get(k, v)) for k, v in PLAIN.items()}


def readout(p: Preds, densities: np.ndarray, ro: dict | None = None) -> np.ndarray:
    """(N, K) grams exactly as the deployed model computes them (model.CalorieLens.read_grams)."""
    r = _ro(ro)
    kpg = torch.from_numpy(np.ascontiguousarray(densities[:, 0]))
    grams, _ = M.CalorieLens.read_grams(torch.from_numpy(p.soft), torch.from_numpy(p.log_totals), kpg,
                                        r["mass_blend"], r["kcal_blend"], r["kcal_scale"])
    return grams.numpy()


def predicted_targets(p: Preds, densities: np.ndarray, how: str = "reconciled", ro: dict | None = None) -> np.ndarray:
    """(N, 5) predictions of TARGETS. how: "reconciled" (the K neurons read with `ro`: rescaled to the blended
    mass, tilted toward the blended kcal per gram), "items" (sum of the soft grams, no rescaling) or "direct"
    (the totals head alone)."""
    if how == "direct":
        return p.direct
    grams = p.soft if how == "items" else readout(p, densities, ro)
    nutr = grams @ densities
    return np.stack([nutr[:, 0], grams.sum(1), nutr[:, 1], nutr[:, 2], nutr[:, 3]], 1)


def regression_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """MAE and percent MAE (MAE / mean of the truth, the Nutrition5k convention) per target."""
    out = {}
    for i, t in enumerate(TARGETS):
        mae = float(np.abs(pred[:, i] - true[:, i]).mean())
        out[f"{t}_mae"], out[f"{t}_pmae"] = mae, 100 * mae / float(max(true[:, i].mean(), 1e-6))
    return out


def presence_metrics(prob: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict:
    from sklearn.metrics import average_precision_score

    cols = y.any(0)
    ap = average_precision_score(y[:, cols], prob[:, cols], average="macro")
    pred = prob >= threshold
    tp, fp, fn = (pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum()
    return {"presence_map": float(ap), "presence_f1": float(2 * tp / max(2 * tp + fp + fn, 1))}


def gram_metrics(grams: np.ndarray, true: np.ndarray) -> dict:
    """Per-ingredient grams error on present ingredients + the 1st/2nd/3rd largest ingredient (Food-R1)."""
    present = true > 0
    out = {"ingredient_grams_mae": float(np.abs(grams - true)[present].mean())}
    order = np.argsort(-true, axis=1)[:, :3]
    for r in range(3):
        k = order[:, r]
        has = true[np.arange(len(k)), k] > 0
        out[f"top{r + 1}_grams_mae"] = float(np.abs(grams[np.arange(len(k)), k] - true[np.arange(len(k)), k])[has].mean())
    return out


def summarize(p: Preds, labels, ro: dict | None = None, threshold: float = 0.5, original: bool = False,
              details: bool = True) -> dict:
    """Metrics of one split read with `ro`: overhead scans, side frames (per frame) and side dishes (all frames
    averaged). details=False: only the overhead + side-dish regression metrics (fast, for fitting readouts)."""
    dens = labels.densities
    truth = labels.orig_totals if original else labels.totals
    out = {}
    views = {"overhead": p.subset(p.overhead), "side_frame": p.subset(~p.overhead)}
    views["side_dish"] = dish_mean(views["side_frame"]) if len(views["side_frame"].dish) else views["side_frame"]
    if not details:
        views.pop("side_frame")
    for view, q in views.items():
        if not len(q.dish):
            continue
        grams = readout(q, dens, ro)
        nutr = grams @ dens
        pred = np.stack([nutr[:, 0], grams.sum(1), nutr[:, 1], nutr[:, 2], nutr[:, 3]], 1)
        out[view] = {"n": int(len(q.dish)), **regression_metrics(pred, truth[q.dish])}
        if view != "side_frame" and details:
            g_true = labels.grams[q.dish]
            out[view].update(presence_metrics(q.prob, g_true > 0, threshold))
            out[view].update(gram_metrics(grams, g_true))
            for how in ("direct", "items"):
                out[view][f"{how}_kcal_pmae"] = regression_metrics(predicted_targets(q, dens, how), truth[q.dish])["kcal_pmae"]
    return out


def flavoria_metrics(p: Preds, labels, ro: dict | None = None) -> dict:
    """Flavoria test plates (a second cafeteria and camera; weight is their only label): plate-weight error of the
    readout and of the totals head alone, the mean signed error, and the error on plates over 600 g."""
    if not len(p.dish):
        return {}
    true = labels.totals[p.dish, M.TOTALS.index("mass")]
    out = {"n": int(len(p.dish))}
    for name, pred in (("mass", readout(p, labels.densities, ro).sum(1)), ("direct_mass", p.direct[:, 1])):
        err = pred - true
        out[f"{name}_mae"] = float(np.abs(err).mean())
        out[f"{name}_pmae"] = 100 * out[f"{name}_mae"] / float(true.mean())
        out[f"{name}_bias"] = float(err.mean())
        if (true > 600).any():
            out[f"{name}_mae_over_600g"] = float(np.abs(err[true > 600]).mean())
    return out


def val_score(metrics: dict) -> float:
    """Model selection: mean of the overhead and the side-dish calorie percent errors (lower is better)."""
    vals = [metrics[v]["kcal_pmae"] for v in ("overhead", "side_dish") if v in metrics]
    return float(np.mean(vals)) if vals else float("inf")


READOUT_GRIDS = {"mass_blend": np.linspace(0, 1, 21), "kcal_blend": np.linspace(0, 1, 11),
                 "kcal_scale": np.linspace(0, 1, 5)}


def fit_readout(p: Preds, labels, rounds: int = 2) -> dict:
    """Mass blend, kcal blend and kcal scale that minimise the validation score: coordinate descent over
    READOUT_GRIDS (each knob on its grid, the others held), `rounds` passes. kcal_scale only acts with a
    kcal blend. -> {"mass_blend", "kcal_blend", "kcal_scale", "val_score", "scores": {knob: {value: score}}}."""
    cur, scores = dict(PLAIN), {}
    best = val_score(summarize(p, labels, cur, details=False))
    for _ in range(rounds):
        for knob, grid in READOUT_GRIDS.items():
            if knob == "kcal_scale" and cur["kcal_blend"] == 0:
                continue
            scores[knob] = {round(float(v), 2): val_score(summarize(p, labels, {**cur, knob: float(v)}, details=False))
                            for v in grid}
            v = min(scores[knob], key=scores[knob].get)
            if scores[knob][v] < best - 1e-9:
                cur[knob], best = v, scores[knob][v]
    return {**cur, "val_score": best, "scores": scores}


# --------------------------------------------------------------------------- candidates (network x views)
def candidate_preds(view_preds: dict[str, list[Preds]], network: str, views: int) -> Preds:
    """One network read with its first `views` test-time views, averaged as model.CalorieLens.forward does."""
    return combine(view_preds[network][:views])


def fit_candidates(view_preds: dict[str, list[Preds]], labels, view_options=(1, 2, 4), log=print) -> dict[str, dict]:
    """Fit the readout of every network with each number of test-time views, on validation.
    -> network: {"network", "views", "mass_blend", "kcal_blend", "kcal_scale", "val_score", ...}; fewer views win ties."""
    out = {}
    for name, views in view_preds.items():
        best = None
        for v in view_options:
            if v > len(views):
                continue
            fit = fit_readout(candidate_preds(view_preds, name, v), labels)
            if best is None or fit["val_score"] < best["val_score"] - 1e-9:
                best = {"network": name, "views": v, **fit}
        out[name] = best
        log(f"candidate {name}: val score {best['val_score']:.2f} (views {best['views']}, mass blend "
            f"{best['mass_blend']}, kcal blend {best['kcal_blend']}, kcal scale {best['kcal_scale']})")
    return out


def fit_presence_threshold(p: Preds, labels, grid=np.linspace(0.1, 0.9, 17)) -> float:
    """Display threshold: best micro-F1 of the ingredient list on validation overhead + side dishes."""
    q = dish_mean(p.subset(~p.overhead)) if (~p.overhead).any() else p
    probs = np.concatenate([p.subset(p.overhead).prob, q.prob])
    y = labels.grams[np.concatenate([p.dish[p.overhead], q.dish])] > 0
    f1 = {float(t): presence_metrics(probs, y, t)["presence_f1"] for t in grid}
    return round(max(f1, key=f1.get), 2)


def bootstrap_ci(p: Preds, labels, ro: dict | None = None, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95 % CI of the overhead calorie percent MAE, resampling whole plate groups."""
    q = p.subset(p.overhead)
    if not len(q.dish):
        return float("nan"), float("nan")
    pred = predicted_targets(q, labels.densities, "reconciled", ro)[:, 0]
    true = labels.totals[q.dish, 0]
    groups = labels.group[q.dish]
    uniq = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        rows = np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq))])
        stats.append(100 * np.abs(pred[rows] - true[rows]).mean() / true[rows].mean())
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


# --------------------------------------------------------------------------- training
def file_log(path):
    """print() that also appends each line, timestamped, to `path`: the training log survives cleared cell
    outputs, a closed browser tab and kernel restarts. Inside a notebook it also appends the traceback of
    any cell that fails."""
    import traceback

    path.parent.mkdir(parents=True, exist_ok=True)

    def append(msg: str) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")

    def log(msg: str) -> None:
        print(msg)
        append(msg)

    try:
        from IPython import get_ipython
        shell = get_ipython()
    except ImportError:
        shell = None
    if shell is not None:
        def on_error(shell, etype, value, tb, tb_offset=None):
            append("ERROR\n" + "".join(traceback.format_exception(etype, value, tb)).rstrip())
            shell.showtraceback((etype, value, tb), tb_offset=tb_offset)
        shell.set_custom_exc((Exception,), on_error)
    return log


def run_key(cfg, name: str) -> str:
    """Checkpoints only resume into the same run (same experiment + every setting that changes it)."""
    # micro_batch is left out on purpose: a resume may land on another GPU type (same effective batch)
    keys = ("backbone", "image_size", "d_model", "decoder_layers", "heads", "drop_path", "depth_aux", "batch_size",
            "lr_backbone", "layer_decay", "lr_head", "weight_decay", "warmup_epochs", "ema_decay", "grad_clip",
            "epochs", "samples_per_epoch", "overhead_share", "side_loss_weight", "flavoria_share", "side_zoom", "loss_weights",
            "backbone_weights", "multi_pool", "seed", "smoke")
    blob = json.dumps({"name": name, **{k: getattr(cfg, k) for k in keys}}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def train(tnet, loaders, cfg, device, name: str, log=print) -> dict:
    """Train one experiment. Returns {"net": best EMA CalorieLens, "history", "best_epoch", "val",
    "val_score", "seconds", "micro_batch", "chunks", "compiled"}."""
    raw = tnet.to(device)
    convnext = cfg.backbone != "dinov2"
    if cfg.channels_last and convnext and device.type == "cuda":
        raw = raw.to(memory_format=torch.channels_last)
    micro = loaders.train.batch_size
    accum = max(1, cfg.batch_size // micro)
    steps = max(1, len(loaders.train) // accum)
    opt = torch.optim.AdamW(param_groups(raw, cfg), betas=(0.9, 0.999), fused=device.type == "cuda")
    warm, total = int(cfg.warmup_epochs * steps), steps * cfg.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(1, warm) if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    ema = AveragedModel(raw.net, multi_avg_fn=get_ema_multi_avg_fn(cfg.ema_decay), use_buffers=True)
    norms = make_norms(loaders.labels, device)
    fwd, compiled, chunks = maybe_compile(raw, cfg, device), cfg.use_compile and device.type == "cuda", 1
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    scaler = utils.grad_scaler(device)                 # only active for fp16 (pre-Ampere GPUs such as a T4)
    key = run_key(cfg, name)
    ckpt = cfg.checkpoint_dir / f"{_slug(name)}-{key}.pt"
    history, best, start, t0 = [], {"score": float("inf"), "epoch": 0, "state": None, "val": {}}, 1, time.time()

    if cfg.resume and ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if state.get("key") == key:
            raw.load_state_dict(state["model"]); ema.load_state_dict(state["ema"])
            opt.load_state_dict(state["opt"]); sched.load_state_dict(state["sched"]); scaler.load_state_dict(state["scaler"])
            history, best, start, chunks = state["history"], state["best"], state["epoch"] + 1, state.get("chunks", 1)
            gen.set_state(state["gen"])
            log(f"[{name}] resumed after epoch {state['epoch']} (best {best['score']:.2f} @ {best['epoch']})")
        else:
            log(f"[{name}] checkpoint {ckpt.name} is from another run - starting fresh")

    for epoch in range(start, cfg.epochs + 1):
        if best["epoch"] and epoch - best["epoch"] > cfg.patience:
            break
        loaders.sampler.set_epoch(epoch)
        raw.train()
        sums, n_steps, skipped, te = {}, 0, 0, time.time()
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(loaders.train):
            batch = _to(batch, device)
            x_all, depth_all = gpu_augment(batch, True, gen, cfg.side_zoom)
            if cfg.channels_last and convnext and device.type == "cuda":
                x_all = x_all.contiguous(memory_format=torch.channels_last)
            while True:                                   # split the batch further on out-of-memory
                snapshot, out, losses = dict(sums), None, None
                try:
                    for c in range(chunks):
                        sl = slice(c * len(x_all) // chunks, (c + 1) * len(x_all) // chunks)
                        sub = {k: v[sl] if torch.is_tensor(v) else v for k, v in batch.items()}
                        with utils.autocast(device):
                            try:
                                out = fwd(x_all[sl])
                            except torch.OutOfMemoryError:
                                raise
                            except Exception as e:     # torch.compile failure (also on a recompile) -> eager
                                if fwd is raw:
                                    raise
                                log(f"[{name}] torch.compile failed ({type(e).__name__}: {str(e)[:120]}); eager mode")
                                fwd, compiled = raw, False
                                out = fwd(x_all[sl])
                        losses = compute_losses(out, sub, None if depth_all is None else depth_all[sl], norms, cfg)
                        if not torch.isfinite(losses["total"]):
                            skipped += 1
                            continue
                        scaler.scale(losses["total"] / (accum * chunks)).backward()
                        for k, v in losses.items():
                            sums[k] = sums.get(k, 0.0) + float(v.detach()) / chunks
                    break
                except torch.OutOfMemoryError:
                    if chunks >= len(x_all):
                        raise
                    out = losses = None                   # drop the graph before retrying
                    sums = snapshot
                    opt.zero_grad(set_to_none=True)       # this accumulation window restarts (rare)
                    torch.cuda.empty_cache()
                    chunks *= 2
                    log(f"[{name}] out of GPU memory: splitting each batch of {len(x_all)} into {chunks} chunks")
            out = losses = None
            if (i + 1) % accum == 0:
                scaler.unscale_(opt)
                norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), cfg.grad_clip)
                if torch.isfinite(norm):
                    scaler.step(opt)
                    ema.update_parameters(raw.net)
                else:                                     # never write NaN / inf into the weights
                    skipped += 1
                    if scaler.is_enabled():
                        scaler.step(opt)                  # the scaler sees the inf, skips the step, lowers its scale
                scaler.update()
                opt.zero_grad(set_to_none=True); sched.step()
                n_steps += 1
                if n_steps % cfg.log_every == 0:
                    log(f"[{name}] epoch {epoch} step {n_steps}/{steps}  loss {sums.get('total', float('nan')) / (i + 1):.4f}  "
                        f"{(i + 1) * micro / (time.time() - te):.0f} img/s")
        train_loss = {k: v / max(1, len(loaders.train)) for k, v in sums.items()}
        val = summarize(evaluate(ema.module, loaders.val, device), loaders.labels)
        score = val_score(val)
        history.append({"epoch": epoch, "train": train_loss, "val_score": score, "skipped": skipped,
                        "val": {v: {k: m[k] for k in ("kcal_pmae", "mass_pmae", "kcal_mae")} for v, m in val.items()},
                        "seconds": round(time.time() - te, 1), "lr_head": opt.param_groups[-1]["lr"]})
        log(f"[{name}] epoch {epoch}: train {train_loss.get('total', float('nan')):.4f}  val score {score:.2f} "
            f"(overhead kcal {val.get('overhead', {}).get('kcal_pmae', float('nan')):.2f}%, "
            f"side dish kcal {val.get('side_dish', {}).get('kcal_pmae', float('nan')):.2f}%)  {time.time() - te:.0f}s"
            + (f"  peak GPU {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB" if device.type == "cuda" else ""))
        if score < best["score"]:
            best = {"score": score, "epoch": epoch, "val": val,
                    "state": {k: v.detach().cpu().clone() for k, v in ema.module.state_dict().items()}}
        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"key": key, "epoch": epoch, "model": raw.state_dict(), "ema": ema.state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "history": history, "best": best, "chunks": chunks, "gen": gen.get_state()}, ckpt)
    if best["state"] is None:
        raise RuntimeError(f"[{name}] no finished epoch: nothing to keep")
    net = copy.deepcopy(ema.module).cpu()
    net.load_state_dict(best["state"])
    net = net.to(memory_format=torch.contiguous_format)
    return {"net": net, "history": history, "best_epoch": best["epoch"], "val": best["val"],
            "val_score": best["score"], "seconds": round(time.time() - t0), "micro_batch": micro,
            "chunks": chunks, "compiled": compiled}


@torch.no_grad()
def throughput(tnet, loaders, cfg, device, steps: int = 20) -> dict:
    """Images/s of the data loader alone and of a full training step (forward + backward) at the
    current micro-batch. The loader should deliver >= 1.5x what the model consumes."""
    t, n = time.time(), 0
    for i, batch in enumerate(loaders.train):
        n += len(batch["image"])
        if i + 1 >= steps:
            break
    loader_ips = n / (time.time() - t)
    raw = tnet.to(device).train()
    fwd = maybe_compile(raw, cfg, device)
    batch = _to(next(iter(loaders.train)), device)
    x, depth = gpu_augment(batch, True, side_zoom=cfg.side_zoom)
    norms = make_norms(loaders.labels, device)
    with torch.enable_grad():
        for i in range(steps + 3):
            if i == 3:                                   # warm-up (compile, cudnn autotune)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t = time.time()
            with utils.autocast(device):
                out = fwd(x)
            compute_losses(out, batch, depth, norms, cfg)["total"].backward()
            raw.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    model_ips = steps * len(x) / (time.time() - t)
    mem = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0.0
    return {"loader_img_s": round(loader_ips, 1), "train_img_s": round(model_ips, 1),
            "loader_ok": loader_ips >= 1.5 * model_ips, "peak_gb": round(mem, 1), "micro_batch": len(x)}


# --------------------------------------------------------------------------- model selection
def _view_errors(p: Preds, labels, ro: dict | None = None) -> dict:
    """Per-dish |predicted - true| kcal and the truth, for the overhead and side-dish views."""
    out = {}
    views = {"overhead": p.subset(p.overhead)}
    if (~p.overhead).any():
        views["side_dish"] = dish_mean(p.subset(~p.overhead))
    for view, q in views.items():
        if len(q.dish):
            pred = predicted_targets(q, labels.densities, "reconciled", ro)[:, 0]
            out[view] = (q.dish, np.abs(pred - labels.totals[q.dish, 0]), labels.totals[q.dish, 0])
    return out


def paired_score_ci(pa: Preds, pb: Preds, labels, ro_a: dict | None = None, ro_b: dict | None = None, n: int = 1000,
                    seed: int = 0) -> tuple[float, float, float]:
    """Validation score of model a minus model b (each with its readout), with a 95 % CI from resampling whole
    plate groups (both models see the same resample). The CI containing 0 means the two are not distinguishable."""
    ea, eb = _view_errors(pa, labels, ro_a), _view_errors(pb, labels, ro_b)
    groups = np.unique(np.concatenate([labels.group[d] for d, _, _ in ea.values()]))
    rng = np.random.default_rng(seed)

    def score(errs, weights):
        vals = [100 * (w * e).sum() / (w * t).sum() for (d, e, t), w in zip(errs.values(), weights)]
        return float(np.mean(vals))

    gidx = {v: {g: i for i, g in enumerate(groups)} for v in ea}
    pos = {v: np.array([gidx[v][g] for g in labels.group[ea[v][0]]]) for v in ea}
    diff = score(ea, [np.ones(len(e[0])) for e in ea.values()]) - score(eb, [np.ones(len(e[0])) for e in eb.values()])
    stats = []
    for _ in range(n):
        counts = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
        w = [counts[pos[v]] for v in ea]
        stats.append(score(ea, w) - score(eb, w))
    return diff, float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def pick_candidate(fits: dict[str, dict], view_preds: dict[str, list[Preds]], labels) -> tuple[str, str]:
    """Lowest validation score wins (the test split is never consulted). -> (label, reason with the paired
    bootstrap CI against the runner-up)."""
    order = sorted(fits, key=lambda k: fits[k]["val_score"])
    best = order[0]
    if len(order) == 1:
        return best, "only candidate"
    a, b = fits[best], fits[order[1]]
    diff, lo, hi = paired_score_ci(candidate_preds(view_preds, a["network"], a["views"]),
                                   candidate_preds(view_preds, b["network"], b["views"]), labels, a, b)
    return best, f"lowest validation score; vs {order[1]}: {diff:+.2f} (95% CI [{lo:+.2f}, {hi:+.2f}])"


def ablation(base: Preds, final: Preds, labels, fit: dict) -> dict:
    """Overhead test calorie % error of each way to read the model, from the network on the photo alone (`base`)
    to the deployed candidate (`final`: with its test-time views), with the deployed readout."""
    r = _ro(fit)
    pmae = lambda p, how, ro=None: regression_metrics(
        predicted_targets(p.subset(p.overhead), labels.densities, how, ro), labels.totals[p.dish[p.overhead]])["kcal_pmae"]
    rows = {"direct totals head": pmae(base, "direct"), "sum of ingredient grams (no rescale)": pmae(base, "items"),
            f"grams rescaled to the plate mass (w={r['mass_blend']:g})": pmae(base, "reconciled", {"mass_blend": r["mass_blend"]})}
    if r["kcal_blend"] > 0:
        rows[f"+ toward the kcal per gram (b={r['kcal_blend']:g}, s={r['kcal_scale']:g})"] = pmae(base, "reconciled", r)
    if fit["views"] > 1:
        rows[f"+ average of {fit['views']} views (deployed)"] = pmae(final, "reconciled", r)
    return rows


def experiment_row(result: dict, test: dict, params: int) -> dict:
    """One trained network: its own validation score (photo alone, plain mass readout) and test numbers."""
    o, sd = test.get("overhead", {}), test.get("side_dish", {})
    return {"val_score": result["val_score"], "best_epoch": result["best_epoch"], "train_seconds": result["seconds"],
            "test_kcal_pmae": o.get("kcal_pmae"), "test_mass_pmae": o.get("mass_pmae"),
            "test_side_dish_kcal_pmae": sd.get("kcal_pmae"), "params": int(params),
            "micro_batch": result.get("micro_batch"), "compiled": result.get("compiled")}


def candidate_row(fit: dict, test: dict, params: int) -> dict:
    """One line of the Experiments table: a candidate (network x views + its fitted readout), selected on
    validation, test reported alongside."""
    o, sd = test.get("overhead", {}), test.get("side_dish", {})
    return {"network": fit["network"], "views": fit["views"], **_ro(fit), "val_score": fit["val_score"],
            "test_kcal_pmae": o.get("kcal_pmae"), "test_mass_pmae": o.get("mass_pmae"),
            "test_side_dish_kcal_pmae": sd.get("kcal_pmae"), "params": int(params)}
