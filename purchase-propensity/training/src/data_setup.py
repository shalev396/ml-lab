"""Data: download -> cache in training/data -> preprocess -> splits, plus the RFM segmentation.

Data A (propensity): Kaggle `benpowis/customer-propensity-to-purchase-data` (`training_sample.csv`,
    455,401 one-day website sessions, 23 binary flags + `ordered`), downloaded anonymously with
    kagglehub and cached as data/propensity_training_sample.csv.
    Fallback: download `training_sample.csv` by hand from the Kaggle page and save it under that name.
Data B (RFM): UCI Online Retail (541,909 invoice lines, Dec 2010 - Dec 2011). The xlsx comes from
    the UCI archive (two alternate URLs), is cleaned once and cached as parquet.
"""
from __future__ import annotations

import io
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import model as M  # ../model/model.py (put on sys.path by src/__init__.py)

from . import utils

KAGGLE_ID = "benpowis/customer-propensity-to-purchase-data"
KAGGLE_FILE = "training_sample.csv"
PROPENSITY_CSV = "propensity_training_sample.csv"
RETAIL_URLS = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx",
    "https://archive.ics.uci.edu/static/public/352/online+retail.zip",
)
RETAIL_XLSX = "online_retail.xlsx"
RETAIL_CACHE = "online_retail_clean.parquet"

# Standard RFM segment grid on the (R score, F score) pair, first match wins.
SEGMENT_MAP = [
    ["[1-2][1-2]", "Hibernating"],
    ["[1-2][3-4]", "At Risk"],
    ["[1-2]5", "Cannot Lose Them"],
    ["3[1-2]", "About to Sleep"],
    ["33", "Need Attention"],
    ["[3-4][4-5]", "Loyal Customers"],
    ["41", "Promising"],
    ["51", "New Customers"],
    ["[4-5][2-3]", "Potential Loyalists"],
    ["5[4-5]", "Champions"],
]


# --------------------------------------------------------------------------- Data A: sessions
def download_propensity(data_dir: Path = utils.DATA_DIR) -> Path:
    """Kaggle sessions CSV -> data/propensity_training_sample.csv (skipped when cached)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / PROPENSITY_CSV
    if path.is_file():
        print(f"[data] cached: {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
        return path
    try:
        import kagglehub

        print(f"[data] downloading {KAGGLE_ID} with kagglehub ...")
        folder = Path(kagglehub.dataset_download(KAGGLE_ID))
        shutil.copyfile(next(folder.rglob(KAGGLE_FILE)), path)
    except Exception as err:
        raise RuntimeError(
            f"could not download {KAGGLE_ID} ({err!r}). Fallback: download {KAGGLE_FILE} from "
            f"https://www.kaggle.com/datasets/{KAGGLE_ID} and save it as {path}") from err
    print(f"[data] saved: {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def load_sessions(cfg, data_dir: Path = utils.DATA_DIR) -> pd.DataFrame:
    """All sessions (smoke: a stratified `cfg.smoke_rows` subsample) with the 23 flags + target."""
    df = pd.read_csv(download_propensity(data_dir))
    missing = [c for c in (*M.FEATURES, M.TARGET) if c not in df.columns]
    if missing:
        raise ValueError(f"unexpected CSV schema, missing {missing}")
    df = df[["UserID", *M.FEATURES, M.TARGET]].copy()
    df[M.FEATURES] = df[M.FEATURES].astype(np.int8)
    df[M.TARGET] = df[M.TARGET].astype(np.int8)
    if cfg.smoke and len(df) > cfg.smoke_rows:
        df, _ = train_test_split(df, train_size=cfg.smoke_rows, stratify=df[M.TARGET], random_state=cfg.seed)
        df = df.reset_index(drop=True)
    return df


@dataclass
class Splits:
    """Raw 0/1 frames per split plus the StandardScaler fit on the train split."""
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    scaler: StandardScaler

    def items(self):
        return (("train", self.train), ("val", self.val), ("test", self.test))

    def xy(self, name: str, scaled: bool = True) -> tuple[np.ndarray, np.ndarray]:
        part = getattr(self, name)
        x = part[M.FEATURES].to_numpy(dtype=np.float64)
        return (self.scaler.transform(x) if scaled else x), part[M.TARGET].to_numpy(dtype=np.int64)


def split(df: pd.DataFrame, cfg) -> Splits:
    """Stratified train / val / test (68/12/20 by default) + a StandardScaler fit on train only."""
    rest, test = train_test_split(df, test_size=cfg.test_size, stratify=df[M.TARGET], random_state=cfg.seed)
    train, val = train_test_split(rest, test_size=cfg.val_size, stratify=rest[M.TARGET], random_state=cfg.seed)
    scaler = make_scaler().fit(train[M.FEATURES].to_numpy(dtype=np.float64))
    return Splits(train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True), scaler)


def make_scaler() -> StandardScaler:
    """Both models see standardized flags (mean 0 / std 1 per feature on the train split)."""
    return StandardScaler()


def split_summary(splits: Splits) -> pd.DataFrame:
    return pd.DataFrame({name: {"sessions": len(part), "orders": int(part[M.TARGET].sum()),
                                "order_rate": round(float(part[M.TARGET].mean()), 4)}
                         for name, part in splits.items()}).T


def order_rate_by_flag(df: pd.DataFrame) -> pd.DataFrame:
    """For each flag: share of sessions with it ON and the order rate with it ON vs OFF."""
    rows = []
    for name in M.FEATURES:
        on = df[name] == 1
        rows.append({"feature": name, "share_on": on.mean(),
                     "order_rate_on": df.loc[on, M.TARGET].mean() if on.any() else np.nan,
                     "order_rate_off": df.loc[~on, M.TARGET].mean()})
    return pd.DataFrame(rows).set_index("feature").sort_values("order_rate_on", ascending=False)


def make_loaders(splits: Splits, batch_size: int, device) -> tuple[DataLoader, DataLoader]:
    """Train (shuffled) and validation loaders of standardized float32 tensors."""
    def dataset(name):
        x, y = splits.xy(name)
        return TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1))

    pin = getattr(device, "type", str(device)) == "cuda"
    train = DataLoader(dataset("train"), batch_size=batch_size, shuffle=True,
                       num_workers=utils.num_workers(), pin_memory=pin)
    val = DataLoader(dataset("val"), batch_size=batch_size, shuffle=False,
                     num_workers=utils.num_workers(), pin_memory=pin)
    return train, val


def example_sessions(splits: Splits, n_pos: int = 3, n_neg: int = 5, seed: int = 42) -> pd.DataFrame:
    """A few real test-split sessions (both classes) for the Space examples."""
    test = splits.test
    pos = test[test[M.TARGET] == 1].sample(min(n_pos, int(test[M.TARGET].sum())), random_state=seed)
    neg = test[test[M.TARGET] == 0].sample(n_neg, random_state=seed)
    rows = pd.concat([pos, neg]).sample(frac=1, random_state=seed)
    return rows[[*M.FEATURES, M.TARGET]].reset_index(drop=True)


# --------------------------------------------------------------------------- Data B: retail
def load_retail(data_dir: Path = utils.DATA_DIR) -> pd.DataFrame:
    """UCI Online Retail -> cleaned invoice lines (cached as parquet: the xlsx parse takes ~1 min)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = data_dir / RETAIL_CACHE
    if cache.is_file():
        print(f"[data] cached: {cache.name} ({cache.stat().st_size / 1e6:.1f} MB)")
        return pd.read_parquet(cache)
    xlsx = data_dir / RETAIL_XLSX
    if not xlsx.is_file():
        _download_retail(xlsx)
    print("[data] parsing the Online Retail xlsx (about a minute, cached afterwards) ...")
    df = clean_retail(pd.read_excel(xlsx, engine="openpyxl", dtype={"InvoiceNo": str, "StockCode": str}))
    df.to_parquet(cache, index=False)
    print(f"[data] cleaned + cached: {cache.name} ({len(df):,} rows)")
    return df


def _download_retail(xlsx: Path) -> None:
    import requests

    errors = []
    for url in RETAIL_URLS:
        try:
            print(f"[data] downloading {url} ...")
            r = requests.get(url, timeout=300, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            if url.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                    xlsx.write_bytes(zf.read(next(n for n in zf.namelist() if n.lower().endswith(".xlsx"))))
            else:
                xlsx.write_bytes(r.content)
            return
        except Exception as err:  # try the next mirror
            errors.append(f"{url}: {err!r}")
    raise RuntimeError("could not download UCI Online Retail:\n" + "\n".join(errors))


def clean_retail(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows without CustomerID, cancellations (InvoiceNo 'C...') and non-positive quantity/price."""
    out = df.dropna(subset=["CustomerID"]).copy()
    out = out[~out["InvoiceNo"].astype(str).str.startswith("C")]
    out = out[(out["Quantity"] > 0) & (out["UnitPrice"] > 0)]
    out["CustomerID"] = out["CustomerID"].astype(int)
    out["InvoiceDate"] = pd.to_datetime(out["InvoiceDate"])
    out["TotalPrice"] = out["Quantity"] * out["UnitPrice"]
    return out.reset_index(drop=True)


def _quintile(s: pd.Series, ascending: bool) -> pd.Series:
    """Rank-based (tie-safe) quintile scores 1..5; ascending=False gives 5 to the smallest values."""
    return pd.qcut(s.rank(method="first", ascending=ascending), 5, labels=[1, 2, 3, 4, 5]).astype(int)


def rfm_snapshot(retail: pd.DataFrame) -> pd.Timestamp:
    """Recency reference date: the day after the last invoice (so a same-day buyer has recency 0)."""
    return retail["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)


def compute_rfm(retail: pd.DataFrame, snapshot: pd.Timestamp | None = None) -> pd.DataFrame:
    """Recency (days since last order) / Frequency (invoices) / Monetary (GBP) per customer + segment."""
    snapshot = snapshot or rfm_snapshot(retail)
    rfm = retail.groupby("CustomerID").agg(
        recency=("InvoiceDate", lambda s: (snapshot - s.max()).days),
        frequency=("InvoiceNo", "nunique"),
        monetary=("TotalPrice", "sum"),
    )
    rfm["r_score"] = _quintile(rfm["recency"], ascending=False)
    rfm["f_score"] = _quintile(rfm["frequency"], ascending=True)
    rfm["m_score"] = _quintile(rfm["monetary"], ascending=True)
    rfm["rfm_score"] = rfm["r_score"].astype(str) + rfm["f_score"].astype(str) + rfm["m_score"].astype(str)
    rfm["segment"] = [M.rfm_segment(r, f, SEGMENT_MAP) for r, f in zip(rfm["r_score"], rfm["f_score"])]
    return rfm.reset_index()


def summarize_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    summary = (rfm.groupby("segment")
               .agg(customers=("CustomerID", "count"), avg_recency_days=("recency", "mean"),
                    avg_frequency=("frequency", "mean"), avg_monetary=("monetary", "mean"),
                    total_revenue=("monetary", "sum"))
               .round(2).sort_values("customers", ascending=False).reset_index())
    summary["share_pct"] = (summary["customers"] / summary["customers"].sum() * 100).round(1)
    summary["revenue_share_pct"] = (summary["total_revenue"] / summary["total_revenue"].sum() * 100).round(1)
    return summary


def rfm_artifact(retail: pd.DataFrame, rfm: pd.DataFrame, summary: pd.DataFrame) -> dict:
    """What the Space needs for the RFM explorer: the segment table, the grid and the quintile edges.

    Edges are the upper bound of scores 1..4 (recency: of scores 5..2), so `model.rfm_scores` places a
    new customer; ties at a boundary go to the lower score (approximate for the tied rank-based scores).
    """
    def upper(col, score_col, order):
        return [float(rfm.loc[rfm[score_col] == s, col].max()) for s in order]

    return {
        "source": "UCI Online Retail (id 352), cleaned invoice lines",
        "snapshot": str(rfm_snapshot(retail).date()),  # recency reference date (last invoice + 1 day)
        "last_invoice_date": str(retail["InvoiceDate"].max().date()),
        "n_customers": int(len(rfm)),
        "n_invoice_lines": int(len(retail)),
        "total_revenue": round(float(rfm["monetary"].sum()), 2),
        "segments": summary.to_dict(orient="records"),
        "segment_map": SEGMENT_MAP,
        "edges": {"recency": upper("recency", "r_score", (5, 4, 3, 2)),
                  "frequency": upper("frequency", "f_score", (1, 2, 3, 4)),
                  "monetary": upper("monetary", "m_score", (1, 2, 3, 4))},
    }
