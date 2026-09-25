"""Download -> cache in training/data -> clean -> balanced subsample -> stratified splits -> vectorizer.

Primary source: Kaggle `clmentbisaillon/fake-and-real-news-dataset` (ISOT): `Fake.csv` (~23.5k
articles) + `True.csv` (~21.4k articles), fetched anonymously with kagglehub.
Fallback:       Hugging Face mirror `GonzaloA/fake_news` (its label 1 = true, remapped to ours).
Label convention everywhere: 0 = real, 1 = fake.

Leakage handling: only `title + text` are used (`subject` and `date` separate the classes on their
own and are dropped); the "CITY (Reuters) -" dateline and every "reuters" token are removed by
`model.combine` / `model.clean_text`, the same functions the Space runs.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import model as M  # ../model/model.py (put on sys.path by src/__init__.py)

from . import utils
from .config import Config

NLTK_DIR = utils.DATA_DIR / "nltk_data"


# --------------------------------------------------------------------------- download
def download(cfg: Config) -> tuple[Path, Path]:
    """Fake.csv / True.csv in training/data (downloaded once with kagglehub, then cached)."""
    fake_csv, true_csv = utils.DATA_DIR / "Fake.csv", utils.DATA_DIR / "True.csv"
    if fake_csv.is_file() and true_csv.is_file():
        return fake_csv, true_csv
    import kagglehub

    root = Path(kagglehub.dataset_download(cfg.kaggle_dataset))
    utils.DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(next(root.rglob("Fake.csv")), fake_csv)
    shutil.copyfile(next(root.rglob("True.csv")), true_csv)
    return fake_csv, true_csv


def _load_hf_fallback(cfg: Config) -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset(cfg.hf_fallback, cache_dir=str(utils.DATA_DIR / "hf"))
    df = pd.concat([ds[split].to_pandas() for split in ds], ignore_index=True)
    df = df[["title", "text", "label"]].dropna()
    df["label"] = 1 - df["label"].astype(int)   # mirror: 1 = true -> ours: 1 = fake
    return df


def load_raw(cfg: Config) -> tuple[pd.DataFrame, str]:
    """Raw `title, text, label` rows + a source tag (Kaggle, else the HF mirror)."""
    try:
        fake_csv, true_csv = download(cfg)
        fake = pd.read_csv(fake_csv, encoding="utf-8")
        true = pd.read_csv(true_csv, encoding="utf-8")
        fake["label"], true["label"] = 1, 0
        df, source = pd.concat([fake, true], ignore_index=True), f"kaggle:{cfg.kaggle_dataset}"
    except Exception as err:  # network / Kaggle outage -> documented mirror
        print(f"[data] Kaggle source failed ({type(err).__name__}: {err}); using hf:{cfg.hf_fallback}")
        df, source = _load_hf_fallback(cfg), f"hf:{cfg.hf_fallback}"
    return df[["title", "text", "label"]].fillna(""), source


# --------------------------------------------------------------------------- cleaning
def get_stopwords() -> set[str]:
    """NLTK English stopwords, cached in training/data/nltk_data."""
    import nltk

    NLTK_DIR.mkdir(parents=True, exist_ok=True)
    if str(NLTK_DIR) not in nltk.data.path:
        nltk.data.path.insert(0, str(NLTK_DIR))
    try:
        from nltk.corpus import stopwords

        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", download_dir=str(NLTK_DIR), quiet=True)
        from nltk.corpus import stopwords

        return set(stopwords.words("english"))


def get_stemmer(cfg: Config):
    if not cfg.use_stemming:
        return None
    from nltk.stem import PorterStemmer

    return PorterStemmer()


def prepare(cfg: Config, raw: pd.DataFrame, stop_words: set[str]) -> pd.DataFrame:
    """Deduplicate -> balanced subsample -> model.combine + model.clean_text -> drop near-empty rows.

    Returns columns `text_clean` (what the network reads) and `label`.
    """
    df = raw.drop_duplicates(subset=["title", "text"]).reset_index(drop=True)
    n_target = cfg.subsample or len(df)
    per_class = min(n_target // 2, int(df["label"].value_counts().min()))
    df = pd.concat([df[df["label"] == c].sample(n=per_class, random_state=cfg.seed)
                    for c in (0, 1)]).reset_index(drop=True)
    stemmer = get_stemmer(cfg)
    df["text_clean"] = [M.clean_text(M.combine(t, b), stop_words, stemmer)
                        for t, b in zip(df["title"].astype(str), df["text"].astype(str))]
    df = df[df["text_clean"].str.split().str.len() >= M.MIN_TOKENS].reset_index(drop=True)
    return df[["title", "text", "text_clean", "label"]]


# --------------------------------------------------------------------------- splits + vectorizer
@dataclass
class Splits:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    test_frame: pd.DataFrame   # raw title/text of the test rows (for example inputs)

    def counts(self) -> dict[str, int]:
        return {"n_train": len(self.y_train), "n_val": len(self.y_val), "n_test": len(self.y_test),
                "n_fake_test": int(self.y_test.sum()), "n_classes": 2}


def split(cfg: Config, df: pd.DataFrame) -> Splits:
    """Stratified train / val / test split (reproducible with cfg.seed)."""
    idx = np.arange(len(df))
    y = df["label"].to_numpy().astype(np.float32)
    idx_tmp, idx_test = train_test_split(idx, test_size=cfg.test_fraction, stratify=y, random_state=cfg.seed)
    idx_train, idx_val = train_test_split(idx_tmp, test_size=cfg.val_fraction, stratify=y[idx_tmp],
                                          random_state=cfg.seed)
    x = df["text_clean"].to_numpy()
    return Splits(x[idx_train], y[idx_train], x[idx_val], y[idx_val], x[idx_test], y[idx_test],
                  df.iloc[idx_test][["title", "text", "label"]].reset_index(drop=True))


def build_vectorizer(cfg: Config, train_texts):
    """model.make_vectorizer adapted on the TRAIN split only."""
    import tensorflow as tf

    vec = M.make_vectorizer(cfg.max_tokens, cfg.sequence_length)
    vec.adapt(tf.data.Dataset.from_tensor_slices(list(train_texts)).batch(256))
    return vec
