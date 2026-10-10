"""Stage 2b: NutriText, food name -> nutrients per 100 g, trained from scratch on USDA FoodData Central.

  prepare()          the USDA table -> one row per name (generic foods + their alternative names,
                     branded products down-weighted), split by name group so near-identical names
                     ("Chicken, broilers or fryers, breast, ..." variants) never straddle train / test
  train_tokenizer()  a BPE tokenizer of our own on the training names ([PAD] = 0, [UNK] = 1)
  train_nutritext()  AdamW + warmup/cosine + bf16, best epoch on the validation loss, early stopping
  tfidf_knn()        the baseline it has to beat: mean of the 5 nearest training names (char n-grams)
  zero_shot_n5k()    the external test: the 253 Nutrition5k ingredient names (never seen as such)
                     against their exact per-gram table
"""
from __future__ import annotations

import copy
import math
import time
import zlib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import model as M

from . import utils

TARGETS = ("kcal", "protein", "fat", "carb", "fiber", "sugars", "sat_fat", "sodium", "water", "alcohol")
# loss divisors (per 100 g) so every nutrient weighs about the same; sodium is compared in log1p space
SCALES = {"kcal": 100.0, "protein": 10.0, "fat": 10.0, "carb": 10.0, "fiber": 3.0, "sugars": 10.0,
          "sat_fat": 3.0, "water": 20.0, "alcohol": 5.0}
EVAL = ("kcal", "protein", "fat", "carb", "fiber", "sugars", "sat_fat", "sodium")
GENERIC = ("foundation", "sr_legacy", "fndds")


def _group(name: str, branded: bool) -> str:
    words = name.lower().replace(",", " ").split() if branded else [w.strip() for w in name.lower().split(",")]
    return " ".join(words[:3] if branded else words[:2])


def prepare(usda: pd.DataFrame, branded_weight: float = 0.3, val_frac: float = 0.05, test_frac: float = 0.05,
            max_rows: int | None = None, seed: int = 42) -> pd.DataFrame:
    """name, source, weight, group, split, TARGETS (per 100 g; NaN = not reported)."""
    gen = usda[usda["source"].isin(GENERIC)]
    rows = gen.assign(main=True)
    alts = gen[gen["alt_names"].notna()].explode("alt_names").dropna(subset=["alt_names"])
    alts = alts.assign(name=alts["alt_names"], main=False)
    br = usda[usda["source"] == "branded"].assign(main=True)
    df = pd.concat([rows, alts, br], ignore_index=True)
    df["name"] = df["name"].astype(str).str.strip()
    df["branded"] = df["source"] == "branded"
    df["weight"] = np.where(df["branded"], branded_weight, 1.0)
    # alternative names share the group of their food's main name
    main_name = df["fdc_id"].map(gen.drop_duplicates("fdc_id").set_index("fdc_id")["name"]).fillna(df["name"])
    df["group"] = [_group(n, b) for n, b in zip(main_name, df["branded"])]
    h = np.array([zlib.crc32(f"{seed}:{g}".encode()) / 2 ** 32 for g in df["group"]])
    df["split"] = np.where(h < test_frac, "test", np.where(h < test_frac + val_frac, "val", "train"))
    df = df[["name", "source", "main", "weight", "group", "split", *TARGETS]]
    if max_rows:
        df = df.sample(min(max_rows, len(df)), random_state=seed)
    return df.reset_index(drop=True)


def train_tokenizer(names, vocab_size: int):
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.normalizer = normalizers.Sequence([normalizers.NFKD(), normalizers.StripAccents(), normalizers.Lowercase()])
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Whitespace(), pre_tokenizers.Digits(individual_digits=True)])
    tok.train_from_iterator(names, trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
                                                       special_tokens=["[PAD]", "[UNK]"], show_progress=False))
    assert tok.token_to_id("[PAD]") == 0
    return tok


class Names(Dataset):
    def __init__(self, ids: list[list[int]], y: np.ndarray, w: np.ndarray):
        self.ids, self.y, self.w = ids, torch.from_numpy(y.astype(np.float32)), torch.from_numpy(w.astype(np.float32))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i], self.y[i], self.w[i]


def _collate(batch):
    ids, y, w = zip(*batch)
    T = max(len(i) for i in ids)
    out = torch.zeros(len(ids), T, dtype=torch.long)
    for r, i in enumerate(ids):
        out[r, :len(i)] = torch.tensor(i)
    mask = (out != 0) | (torch.arange(T)[None] == 0)
    return out, mask, torch.stack(y), torch.stack(w)


def encode(tok, names, max_len: int) -> list[list[int]]:
    return [e.ids[:max_len] or [1] for e in tok.encode_batch(list(names))]


def nt_loss(pred: dict, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Weighted mean over samples of the mean scaled L1 over the nutrients the sample reports."""
    terms, masks = [], []
    for j, t in enumerate(TARGETS):
        target = y[:, j]
        ok = torch.isfinite(target)
        t_ = torch.nan_to_num(target)
        if t == "sodium":
            err = (torch.log1p(pred[t].float()) - torch.log1p(t_.clamp_min(0))).abs()
        else:
            err = (pred[t].float() - t_).abs() / SCALES[t]
        terms.append(err * ok)
        masks.append(ok)
    err, ok = torch.stack(terms, 1), torch.stack(masks, 1).float()
    per = err.sum(1) / ok.sum(1).clamp_min(1)
    return (per * w).sum() / w.sum().clamp_min(1e-6)


def train_nutritext(cfg, tok, data: pd.DataFrame, device, log=print) -> dict:
    """Train NutriText on data[split == train]; the best epoch on data[split == val] is kept."""
    kw = {"vocab_size": tok.get_vocab_size(), "d": cfg.nt_d, "layers": cfg.nt_layers, "heads": cfg.nt_heads,
          "max_len": cfg.nt_max_len}
    net = M.NutriText(**kw).to(device)
    loaders = {}
    for split in ("train", "val"):
        part = data[data["split"] == split]
        ds = Names(encode(tok, part["name"], cfg.nt_max_len), part[list(TARGETS)].to_numpy(), part["weight"].to_numpy())
        loaders[split] = DataLoader(ds, batch_size=cfg.nt_batch, shuffle=split == "train", collate_fn=_collate,
                                    num_workers=0, drop_last=split == "train" and len(ds) > cfg.nt_batch)
    steps = len(loaders["train"]) * cfg.nt_epochs
    warm = len(loaders["train"])
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.nt_lr, weight_decay=0.01, fused=device.type == "cuda")
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    best, history, t0 = {"loss": float("inf"), "epoch": 0, "state": None}, [], time.time()
    for epoch in range(1, cfg.nt_epochs + 1):
        net.train()
        total, n = 0.0, 0
        for ids, mask, y, w in loaders["train"]:
            ids, mask, y, w = ids.to(device), mask.to(device), y.to(device), w.to(device)
            with utils.autocast(device):
                pred = net(ids, mask)
            loss = nt_loss(pred, y, w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            total, n = total + loss.item() * len(ids), n + len(ids)
        val = _val_loss(net, loaders["val"], device)
        history.append({"epoch": epoch, "train_loss": total / max(n, 1), "val_loss": val})
        log(f"[NutriText] epoch {epoch}: train {total / max(n, 1):.4f}  val {val:.4f}")
        if val < best["loss"]:
            best = {"loss": val, "epoch": epoch, "state": copy.deepcopy(net.state_dict())}
        elif epoch - best["epoch"] >= cfg.nt_patience:
            break
    net.load_state_dict(best["state"])
    return {"net": net.cpu().eval(), "kwargs": kw, "history": history, "best_epoch": best["epoch"],
            "seconds": round(time.time() - t0)}


@torch.no_grad()
def _val_loss(net, loader, device) -> float:
    net.eval()
    total, n = 0.0, 0
    for ids, mask, y, w in loader:
        ids, mask, y, w = ids.to(device), mask.to(device), y.to(device), w.to(device)
        with utils.autocast(device):
            pred = net(ids, mask)
        total, n = total + float(nt_loss(pred, y, w)) * float(w.sum()), n + float(w.sum())
    return total / max(n, 1e-6)


@torch.no_grad()
def predict(net, tok, names, device, batch: int = 2048) -> np.ndarray:
    """(N, len(EVAL)) per-100 g predictions."""
    net = net.to(device).eval()
    ids = encode(tok, names, net.max_len)
    out = []
    for i in range(0, len(ids), batch):
        x, mask, _, _ = _collate([(e, torch.zeros(1), torch.zeros(())) for e in ids[i:i + batch]])
        p = net(x.to(device), mask.to(device))
        out.append(torch.stack([p[t].float() for t in EVAL], 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(EVAL)))


def tfidf_knn(train: pd.DataFrame, names, k: int = 5, chunk: int = 2000) -> np.ndarray:
    """Baseline: similarity-weighted mean of the k nearest training names (char 2-4-gram TF-IDF)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=2, lowercase=True)
    A = vec.fit_transform(train["name"])
    Y = train[list(EVAL)].to_numpy(np.float64)
    out = []
    for i in range(0, len(names), chunk):
        S = (vec.transform(list(names[i:i + chunk])) @ A.T).toarray()
        top = np.argpartition(-S, k, axis=1)[:, :k]
        s = np.take_along_axis(S, top, 1) + 1e-6
        vals = Y[top]                                           # (n, k, nutrients), NaN where unreported
        wts = np.where(np.isfinite(vals), s[..., None], 0.0)
        out.append(np.nansum(np.nan_to_num(vals) * wts, 1) / np.clip(wts.sum(1), 1e-9, None))
    return np.concatenate(out)


def metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    """Per-nutrient MAE (per 100 g; sodium in mg) and R^2 on the rows that report it."""
    out = {}
    for j, t in enumerate(EVAL):
        ok = np.isfinite(y[:, j])
        if not ok.any():
            continue
        err = pred[ok, j] - y[ok, j]
        var = np.var(y[ok, j])
        out[f"{t}_mae"] = float(np.abs(err).mean())
        out[f"{t}_r2"] = float(1 - np.mean(err ** 2) / var) if var > 0 else float("nan")
    return out


def evaluate_all(net, tok, data: pd.DataFrame, device) -> dict:
    """NutriText vs TF-IDF kNN vs the training mean, on the generic and the branded test names."""
    train = data[data["split"] == "train"]
    mean = np.average(np.nan_to_num(train[list(EVAL)].to_numpy()), axis=0, weights=train["weight"])
    out = {}
    for name, part in (("generic", data[(data["split"] == "test") & data["source"].isin(GENERIC) & data["main"]]),
                       ("branded", data[(data["split"] == "test") & (data["source"] == "branded")])):
        if not len(part):
            continue
        y = part[list(EVAL)].to_numpy(np.float64)
        out[name] = {"n": int(len(part)),
                     "nutritext": metrics(predict(net, tok, part["name"].tolist(), device), y),
                     "tfidf_knn": metrics(tfidf_knn(train, part["name"].tolist()), y),
                     "mean": metrics(np.broadcast_to(mean, y.shape), y)}
    return out


def zero_shot_n5k(net, tok, ingredients: pd.DataFrame, data: pd.DataFrame, device) -> dict:
    """The 253 Nutrition5k names -> per-100 g kcal/fat/carb/protein vs their exact table (x 100)."""
    names = ingredients["name"].tolist()
    truth = ingredients[["kcal", "protein", "fat", "carb"]].to_numpy(np.float64) * 100
    cols = [EVAL.index(c) for c in ("kcal", "protein", "fat", "carb")]
    train = data[data["split"] == "train"]
    res = {}
    for how, pred in (("nutritext", predict(net, tok, names, device)), ("tfidf_knn", tfidf_knn(train, names))):
        err = np.abs(pred[:, cols] - truth)
        res[how] = {f"{c}_mae": float(err[:, i].mean()) for i, c in enumerate(("kcal", "protein", "fat", "carb"))}
        res[how]["kcal_median_ae"] = float(np.median(err[:, 0]))
    return res
