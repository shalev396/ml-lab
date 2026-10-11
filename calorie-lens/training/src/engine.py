"""Training and evaluation of the fused model (stages 1 + 2a).

Losses (every regression target divided by its train mean, so they weigh alike):
  presence      asymmetric loss (ASL) on "is ingredient k on the plate" per positive label; most of
                the 253 are absent
  grams         |log1p(grams) - log1p(true)| of the ingredients that are present (their own scale)
  soft          sum_k |sigmoid(logit_k) x grams_k - true_k| / mean plate mass: the K-neuron layer itself
  totals_head   |log1p(direct totals) - log1p(true)| for kcal, mass, fat, carb, protein
  nutrition     |NutritionLayer(reconciled grams) - NutritionLayer(true grams)| over all 8 nutrients:
                end to end through the frozen stage-2a table
  depth         (E3) scale-invariant log error + L1 of the auxiliary depth map, overhead scans only
Side-angle frames count side_loss_weight on the mass / nutrient losses (part of the plate can be
out of view) and fully on presence.

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
            "dens": torch.tensor(labels.densities, device=device, dtype=torch.float32)}


def asl_loss(logits: torch.Tensor, y: torch.Tensor, gamma_neg: float = 4.0, gamma_pos: float = 0.0,
             clip: float = 0.05) -> torch.Tensor:
    """Asymmetric loss (Ridnik et al., 2021): down-weights the many easy negatives of multi-label data.
    Summed over the K classes and divided by the number of positive labels in the batch (as focal
    losses normalise), so it stays comparable to the regression losses whatever K is."""
    p = torch.sigmoid(logits)
    p_neg = (p - clip).clamp(min=0) if clip else p
    loss_pos = y * torch.log(p.clamp(min=1e-8)) * (1 - p) ** gamma_pos
    loss_neg = (1 - y) * torch.log((1 - p_neg).clamp(min=1e-8)) * p_neg ** gamma_neg
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
    g, tot = batch["grams"].float(), batch["totals"].float()
    overhead = batch["overhead"].bool()
    w = torch.where(overhead, 1.0, cfg.side_loss_weight)
    present = g > 0
    losses = {"presence": asl_loss(out["logits"].float(), present.float())}
    lg = torch.log1p(torch.exp(out["log_grams"].float()))
    per = ((lg - torch.log1p(g)).abs() * present).sum(1) / present.sum(1).clamp_min(1)
    losses["grams"] = _wmean(per, w * present.any(1))
    losses["soft"] = _wmean((out["soft"] - g).abs().sum(1) / norms["mass"], w)
    losses["totals_head"] = _wmean((out["log_totals"].float() - torch.log1p(tot)).abs().mean(1), w)
    target = g @ norms["dens"]
    losses["nutrition"] = _wmean(((out["totals"].float() - target).abs() / norms["nutr"]).mean(1), w)
    if "depth" in out and depth is not None:
        losses["depth"] = depth_loss(out["depth"], depth, overhead)
    losses["total"] = sum(cfg.loss_weights[k] * v for k, v in losses.items())
    return losses


# --------------------------------------------------------------------------- evaluation
@dataclass
class Preds:
    """Per-image raw outputs, enough to recompute any blend / threshold / aggregation offline."""
    dish: np.ndarray            # (N,) dish row
    overhead: np.ndarray        # (N,) bool
    soft: np.ndarray            # (N, K) sigmoid(logit) x grams
    prob: np.ndarray            # (N, K)
    mass_direct: np.ndarray     # (N,)
    direct: np.ndarray          # (N, 5) direct totals head (TARGETS)

    def subset(self, mask: np.ndarray) -> "Preds":
        return Preds(*(getattr(self, f)[mask] for f in self.__dataclass_fields__))


def _to(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def evaluate(net: M.CalorieLens, loader, device, tta: bool = False) -> Preds:
    """Run the (uncompiled, EMA) net over a loader. tta: also average the horizontally flipped image."""
    net.eval()
    parts = {k: [] for k in Preds.__dataclass_fields__}
    for batch in loader:
        batch = _to(batch, device)
        x, _ = gpu_augment(batch, train=False)
        with utils.autocast(device):
            out = net(x)
            if tta:
                flip = net(x.flip(-1))
                out = {k: (out[k] + flip[k]) / 2 if k in ("soft", "mass_direct", "log_totals", "logits") else out[k]
                       for k in out}
        parts["dish"].append(batch["dish"].cpu().numpy())
        parts["overhead"].append(batch["overhead"].cpu().numpy())
        parts["soft"].append(out["soft"].float().cpu().numpy())
        parts["prob"].append(torch.sigmoid(out["logits"].float()).cpu().numpy())
        parts["mass_direct"].append(out["mass_direct"].float().cpu().numpy())
        parts["direct"].append(torch.expm1(out["log_totals"].float().clamp(0, M.MAX_LOG_GRAMS)).cpu().numpy())
    return Preds(*(np.concatenate(parts[k]) for k in Preds.__dataclass_fields__))


def dish_mean(p: Preds) -> Preds:
    """Average all frames of each dish (multi-view inference: 4 cameras x 5 frames)."""
    dishes, inv = np.unique(p.dish, return_inverse=True)
    n = np.bincount(inv).astype(np.float32)[:, None]

    def avg(a):
        out = np.zeros((len(dishes), *a.shape[1:]), np.float32)
        np.add.at(out, inv, a)
        return out / n.reshape(-1, *([1] * (a.ndim - 1)))
    return Preds(dishes, np.zeros(len(dishes), bool), avg(p.soft), avg(p.prob), avg(p.mass_direct), avg(p.direct))


def reconcile_np(soft: np.ndarray, mass_direct: np.ndarray, blend: float) -> np.ndarray:
    return M.CalorieLens.reconcile(torch.from_numpy(soft), torch.from_numpy(mass_direct), blend).numpy()


def predicted_targets(p: Preds, densities: np.ndarray, how: str = "reconciled", blend: float = 1.0) -> np.ndarray:
    """(N, 5) predictions of TARGETS. how: "reconciled" (the K neurons, rescaled to the blended mass),
    "items" (sum of the soft grams, no rescaling) or "direct" (the totals head alone)."""
    if how == "direct":
        return p.direct
    grams = p.soft if how == "items" else reconcile_np(p.soft, p.mass_direct, blend)
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


def summarize(p: Preds, labels, blend: float = 1.0, threshold: float = 0.5, original: bool = False) -> dict:
    """Metrics of one split: overhead scans, side frames (per frame) and side dishes (all frames averaged)."""
    dens = labels.densities
    truth = labels.orig_totals if original else labels.totals
    out = {}
    views = {"overhead": p.subset(p.overhead), "side_frame": p.subset(~p.overhead)}
    views["side_dish"] = dish_mean(views["side_frame"]) if len(views["side_frame"].dish) else views["side_frame"]
    for view, q in views.items():
        if not len(q.dish):
            continue
        pred = predicted_targets(q, dens, "reconciled", blend)
        out[view] = {"n": int(len(q.dish)), **regression_metrics(pred, truth[q.dish])}
        if view != "side_frame":
            g_true = labels.grams[q.dish]
            out[view].update(presence_metrics(q.prob, g_true > 0, threshold))
            out[view].update(gram_metrics(reconcile_np(q.soft, q.mass_direct, blend), g_true))
            for how in ("direct", "items"):
                out[view][f"{how}_kcal_pmae"] = regression_metrics(predicted_targets(q, dens, how), truth[q.dish])["kcal_pmae"]
    return out


def val_score(metrics: dict) -> float:
    """Model selection: mean of the overhead and the side-dish calorie percent errors (lower is better)."""
    vals = [metrics[v]["kcal_pmae"] for v in ("overhead", "side_dish") if v in metrics]
    return float(np.mean(vals)) if vals else float("inf")


def fit_mass_blend(p: Preds, labels, grid=np.linspace(0, 1, 21)) -> tuple[float, dict]:
    """Weight of the direct mass head vs the summed ingredient grams that minimises the validation score."""
    scores = {round(float(w), 2): val_score(summarize(p, labels, blend=float(w))) for w in grid}
    best = min(scores, key=scores.get)
    return best, scores


def fit_presence_threshold(p: Preds, labels, grid=np.linspace(0.1, 0.9, 17)) -> float:
    """Display threshold: best micro-F1 of the ingredient list on validation overhead + side dishes."""
    q = dish_mean(p.subset(~p.overhead)) if (~p.overhead).any() else p
    probs = np.concatenate([p.subset(p.overhead).prob, q.prob])
    y = labels.grams[np.concatenate([p.dish[p.overhead], q.dish])] > 0
    f1 = {float(t): presence_metrics(probs, y, t)["presence_f1"] for t in grid}
    return round(max(f1, key=f1.get), 2)


def bootstrap_ci(p: Preds, labels, blend: float = 1.0, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95 % CI of the overhead calorie percent MAE, resampling whole plate groups."""
    q = p.subset(p.overhead)
    if not len(q.dish):
        return float("nan"), float("nan")
    pred = predicted_targets(q, labels.densities, "reconciled", blend)[:, 0]
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
            "epochs", "samples_per_epoch", "overhead_share", "side_loss_weight", "loss_weights", "seed", "smoke")
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
            x_all, depth_all = gpu_augment(batch, True, gen)
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
    x, depth = gpu_augment(batch, True)
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
def _view_errors(p: Preds, labels, blend: float) -> dict:
    """Per-dish |predicted - true| kcal and the truth, for the overhead and side-dish views."""
    out = {}
    views = {"overhead": p.subset(p.overhead)}
    if (~p.overhead).any():
        views["side_dish"] = dish_mean(p.subset(~p.overhead))
    for view, q in views.items():
        if len(q.dish):
            pred = predicted_targets(q, labels.densities, "reconciled", blend)[:, 0]
            out[view] = (q.dish, np.abs(pred - labels.totals[q.dish, 0]), labels.totals[q.dish, 0])
    return out


def paired_score_ci(pa: Preds, pb: Preds, labels, blend_a: float = 1.0, blend_b: float = 1.0, n: int = 1000,
                    seed: int = 0) -> tuple[float, float, float]:
    """Validation score of model a minus model b, with a 95 % CI from resampling whole plate groups
    (both models see the same resample). The CI containing 0 means the two are not distinguishable."""
    ea, eb = _view_errors(pa, labels, blend_a), _view_errors(pb, labels, blend_b)
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


def pick_winner(results: dict, val_preds: dict, labels, prefer: str = "convnext") -> tuple[str, str]:
    """Lowest validation score wins; if a `prefer` model (cheaper on the Space's CPU fallback, no extra
    licence) is within the paired bootstrap CI of the winner, it is shipped instead. -> (name, reason)."""
    best = min(results, key=lambda k: results[k]["val_score"])
    if prefer in best.lower().replace("-", "").replace(" ", ""):
        return best, "lowest validation score"
    for name in results:
        if prefer in name.lower().replace("-", "").replace(" ", ""):
            diff, lo, hi = paired_score_ci(val_preds[name], val_preds[best], labels)
            if lo <= 0.0:
                return name, f"within the CI of {best} (score diff {diff:+.2f}, 95% CI [{lo:+.2f}, {hi:+.2f}])"
            return best, f"beats {name} (score diff {-diff:+.2f}, {name} 95% CI [{lo:+.2f}, {hi:+.2f}])"
    return best, "lowest validation score"


def ablation(p: Preds, labels, blend: float) -> dict:
    """Overhead test calorie % error of each way to read the model: the totals head alone, the K gram
    neurons without rescaling, rescaled to the direct mass, and rescaled to the fitted blend."""
    over = p.subset(p.overhead)
    pmae = lambda how, w=1.0: regression_metrics(predicted_targets(over, labels.densities, how, w),
                                                 labels.totals[over.dish])["kcal_pmae"]
    return {"direct totals head": pmae("direct"), "sum of ingredient grams (no rescale)": pmae("items"),
            "grams rescaled to direct mass (w=1)": pmae("reconciled", 1.0),
            f"grams rescaled to fitted blend (w={blend:g})": pmae("reconciled", blend)}


def experiment_row(result: dict, test: dict, params: int) -> dict:
    """One line of the Experiments table (selection on validation, test reported alongside)."""
    o, sd = test.get("overhead", {}), test.get("side_dish", {})
    return {"val_score": result["val_score"], "best_epoch": result["best_epoch"], "train_seconds": result["seconds"],
            "test_kcal_pmae": o.get("kcal_pmae"), "test_mass_pmae": o.get("mass_pmae"),
            "test_side_dish_kcal_pmae": sd.get("kcal_pmae"), "params": int(params),
            "micro_batch": result["micro_batch"], "compiled": result["compiled"]}
