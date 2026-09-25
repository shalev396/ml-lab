"""Data: download -> clean (model.clean_text) -> stratified 70/15/15 split -> cache in training/data.

Primary dataset: Enron-Spam (Metsis et al., 2006) as published on the Hub by SetFit
(`SetFit/enron_spam`, 33,716 emails, `text` = subject + body). After cleaning and dropping empty
texts: 33,665 emails -> 23,565 train / 5,050 val / 5,050 test (about 51% spam).
Alternative (`Config.dataset = "sms"`): the SMS Spam Collection (`ucirvine/sms_spam`, 5,574 SMS).

The cleaned splits are cached as `training/data/<dataset>_{train,val,test}.csv`; later runs
read the CSVs and never touch the network.
"""
from __future__ import annotations

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import model as M  # ../model/model.py (clean_text, encode)

from . import utils
from .config import Config

SPLITS = ("train", "val", "test")
ENRON_JSONL = "https://huggingface.co/datasets/SetFit/enron_spam/resolve/main/{split}.jsonl"


def load_enron_spam() -> pd.DataFrame:
    """SetFit/enron_spam (all splits) -> DataFrame[text, label]. Falls back to the raw JSONL files."""
    try:
        from datasets import load_dataset

        ds = load_dataset("SetFit/enron_spam", cache_dir=str(utils.DATA_DIR / "hf"))
        frames = [ds[split].to_pandas()[["text", "label"]] for split in ds.keys()]
    except Exception as err:   # Hub / loader hiccup: read the same files directly
        print(f"datasets loader failed ({err}); reading the JSONL files directly")
        frames = [pd.read_json(ENRON_JSONL.format(split=s), lines=True)[["text", "label"]]
                  for s in ("train", "test")]
    return pd.concat(frames, ignore_index=True)


def load_sms_spam() -> pd.DataFrame:
    """ucirvine/sms_spam -> DataFrame[text, label] (0 = ham, 1 = spam)."""
    from datasets import load_dataset

    ds = load_dataset("ucirvine/sms_spam", split="train", cache_dir=str(utils.DATA_DIR / "hf"))
    return ds.to_pandas().rename(columns={"sms": "text"})[["text", "label"]]


def split_paths(cfg: Config) -> dict:
    return {name: utils.DATA_DIR / f"{cfg.dataset}_{name}.csv" for name in SPLITS}


def load_splits(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(train, val, test) DataFrames with a cleaned `text` and an int `label` (1 = spam).

    Built once (download -> clean -> drop empty -> stratified split) and cached as CSV.
    Smoke runs take a small stratified sample of the cached splits.
    """
    paths = split_paths(cfg)
    if all(p.is_file() for p in paths.values()):
        train, val, test = (pd.read_csv(paths[n], keep_default_na=False) for n in SPLITS)
    else:
        df = load_enron_spam() if cfg.dataset == "enron" else load_sms_spam()
        df["text"] = df["text"].map(M.clean_text)
        df = df[df["text"].str.len() > 0].reset_index(drop=True)
        df["label"] = df["label"].astype(int)
        holdout = cfg.val_size + cfg.test_size
        train, rest = train_test_split(df, test_size=holdout, stratify=df["label"], random_state=cfg.seed)
        val, test = train_test_split(rest, test_size=cfg.test_size / holdout, stratify=rest["label"],
                                     random_state=cfg.seed)
        utils.DATA_DIR.mkdir(parents=True, exist_ok=True)
        for name, part in zip(SPLITS, (train, val, test)):
            part.to_csv(paths[name], index=False)
    train, val, test = (d[["text", "label"]].reset_index(drop=True) for d in (train, val, test))
    if cfg.train_subset:
        train = stratified_sample(train, cfg.train_subset, cfg.seed)
    if cfg.eval_subset:
        val, test = (stratified_sample(d, cfg.eval_subset, cfg.seed) for d in (val, test))
    return train, val, test


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """`n` rows with the same spam ratio as `df` (deterministic)."""
    if n >= len(df):
        return df
    sample, _ = train_test_split(df, train_size=n, stratify=df["label"], random_state=seed)
    return sample.reset_index(drop=True)


def summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Rows, ham / spam counts, spam share and median words per split."""
    rows = {}
    for name, df in splits.items():
        spam = int(df["label"].sum())
        rows[name] = {"emails": len(df), "ham": len(df) - spam, "spam": spam,
                      "spam_share": round(spam / max(len(df), 1), 3),
                      "median_words": int(df["text"].str.split().str.len().median())}
    return pd.DataFrame(rows).T


def compute_pos_weight(train: pd.DataFrame) -> float:
    """n_ham / n_spam on the train split: BCE `pos_weight` that balances the two classes."""
    spam = int(train["label"].sum())
    return (len(train) - spam) / max(spam, 1)


class EmailDataset(Dataset):
    """(cleaned text, label) pairs; tokenization happens per batch in `collate`."""

    def __init__(self, df: pd.DataFrame):
        self.texts = df["text"].tolist()
        self.labels = df["label"].astype("float32").tolist()

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        return self.texts[idx], self.labels[idx]


def make_collate(tokenizer, max_len: int):
    """Batch of (text, label) -> {"input_ids", "attention_mask", "labels"}, padded to the longest."""
    def collate(batch):
        texts, labels = zip(*batch)
        enc = M.encode(tokenizer, [M.prepare_text(t) for t in texts], max_len)
        enc["labels"] = torch.tensor(labels, dtype=torch.float32)
        return enc
    return collate


def create_dataloaders(cfg: Config, splits: dict[str, pd.DataFrame], tokenizer, device) -> dict[str, DataLoader]:
    """{"train", "val", "test"} DataLoaders. Train is shuffled with a seeded generator."""
    collate = make_collate(tokenizer, cfg.max_len)
    common = dict(num_workers=utils.num_workers(), pin_memory=(device.type == "cuda"), collate_fn=collate)
    loaders = {}
    for name, df in splits.items():
        train = name == "train"
        loaders[name] = DataLoader(
            EmailDataset(df), batch_size=cfg.batch_size if train else cfg.eval_batch_size, shuffle=train,
            generator=torch.Generator().manual_seed(cfg.seed) if train else None, **common)
    return loaders
