"""USDA FoodData Central (public domain, CC0) -> one table of foods with nutrients per 100 g.

It feeds two things:
  * stage 2b, NutriText (food name -> nutrients): every generic food (Foundation, SR Legacy, FNDDS
    survey foods + their additional descriptions) and the branded foods (label values, noisier);
  * the extended per-gram nutrients (fiber, sugars, saturated fat, sodium) of the fused model's
    NutritionLayer, through training/labels/ingredient_fdc_map.csv (one reviewed USDA food per
    Nutrition5k ingredient; suggest_matches() proposes the candidates).
"""
from __future__ import annotations

import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

FDC_URL = "https://fdc.nal.usda.gov/fdc-datasets"
RELEASES = {
    "foundation": "FoodData_Central_foundation_food_csv_2026-04-30.zip",
    "sr_legacy": "FoodData_Central_sr_legacy_food_csv_2018-04.zip",
    "fndds": "FoodData_Central_survey_food_csv_2024-10-31.zip",
    "branded": "FoodData_Central_branded_food_csv_2026-04-30.zip",
}
GENERIC = ("foundation", "sr_legacy", "fndds")
# per 100 g (sodium in mg); several FDC nutrient ids, in order of preference
NUTRIENT_IDS = {
    "kcal": (1008, 2048, 2047), "protein": (1003,), "fat": (1004,), "carb": (1005,),
    "fiber": (1079,), "sugars": (2000, 1063), "sat_fat": (1258,), "sodium": (1093,),
    "water": (1051,), "alcohol": (1018,),
}
COLUMNS = tuple(NUTRIENT_IDS)
_ALL_IDS = sorted({i for ids in NUTRIENT_IDS.values() for i in ids})
_ATTR_COMMON_NAME, _ATTR_ADDITIONAL = 1000, 1001


def download(dest: Path, sources=tuple(RELEASES)) -> dict[str, Path]:
    """Fetch the FDC CSV zips (branded ~450 MB, the rest ~14 MB) into dest; skips files present."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    paths = {}
    for src in sources:
        path = dest / RELEASES[src]
        if not path.exists():
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(f"{FDC_URL}/{RELEASES[src]}", tmp)
            tmp.replace(path)
        paths[src] = path
    return paths


def _read(zip_path: Path, name: str, **kw) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        member = next(n for n in zf.namelist() if n.endswith("/" + name))
        with zf.open(member) as f:
            return pd.read_csv(f, low_memory=False, **kw)


def _nutrients(zip_path: Path, keep: pd.Index | None = None) -> pd.DataFrame:
    """food_nutrient.csv -> one row per fdc_id with COLUMNS (first available id per nutrient).

    The survey (FNDDS) release stores the legacy nutrient *numbers* (208 = energy) in nutrient_id;
    real ids start at 1001, so values below 1000 are mapped through nutrient.csv."""
    nut = _read(zip_path, "nutrient.csv", usecols=["id", "nutrient_nbr"]).dropna()
    nut = nut[nut["nutrient_nbr"] == nut["nutrient_nbr"].round()]     # 205.2 (carb by summation) != 205
    nbr_to_id = dict(zip(nut["nutrient_nbr"].astype(int), nut["id"].astype(int)))
    parts = []
    with zipfile.ZipFile(zip_path) as zf:
        member = next(n for n in zf.namelist() if n.endswith("/food_nutrient.csv"))
        with zf.open(member) as f:
            for chunk in pd.read_csv(f, usecols=["fdc_id", "nutrient_id", "amount"], chunksize=2_000_000,
                                     dtype={"fdc_id": np.int64, "nutrient_id": np.int32, "amount": np.float32}):
                legacy = chunk["nutrient_id"] < 1000
                if legacy.any():
                    chunk.loc[legacy, "nutrient_id"] = chunk.loc[legacy, "nutrient_id"].map(nbr_to_id).fillna(-1).astype(np.int32)
                chunk = chunk[chunk["nutrient_id"].isin(_ALL_IDS)]
                if keep is not None:
                    chunk = chunk[chunk["fdc_id"].isin(keep)]
                parts.append(chunk)
    long = pd.concat(parts, ignore_index=True)
    wide = long.pivot_table(index="fdc_id", columns="nutrient_id", values="amount", aggfunc="first")
    out = pd.DataFrame(index=wide.index)
    for col, ids in NUTRIENT_IDS.items():
        out[col] = np.nan
        for i in reversed(ids):                 # the preferred id is applied last and wins
            if i in wide.columns:
                out[col] = wide[i].where(wide[i].notna(), out[col])
    return out


def _alt_names(zip_path: Path, foods: pd.DataFrame) -> pd.Series:
    """SR "common names" and FNDDS "additional descriptions" as extra names for the same nutrients."""
    try:
        attr = _read(zip_path, "food_attribute.csv", usecols=["fdc_id", "food_attribute_type_id", "value"])
    except (StopIteration, ValueError):
        return pd.Series(dtype=object)
    attr = attr[attr["food_attribute_type_id"].isin([_ATTR_COMMON_NAME, _ATTR_ADDITIONAL]) & attr["value"].notna()]
    desc = foods.set_index("fdc_id")["name"]
    attr = attr[attr["fdc_id"].isin(desc.index)]
    names = attr["value"].astype(str).str.strip()
    full = np.where(attr["food_attribute_type_id"] == _ATTR_ADDITIONAL,      # "Milk, whole, leche fresca"
                    attr["fdc_id"].map(desc) + ", " + names, names)
    return pd.Series(full, index=attr["fdc_id"]).groupby(level=0).agg(list)


def load_source(source: str, zip_path: Path) -> pd.DataFrame:
    """One FDC release -> fdc_id, source, name, category, alt_names, COLUMNS (per 100 g)."""
    food = _read(zip_path, "food.csv", usecols=["fdc_id", "data_type", "description", "food_category_id",
                                                 "publication_date"])
    if source == "foundation":
        food = food[food["data_type"] == "foundation_food"]   # the rest are lab samples of these foods
    food = food.rename(columns={"description": "name"})
    food["name"] = food["name"].astype(str).str.strip()
    if source == "branded":
        cat = _read(zip_path, "branded_food.csv", usecols=["fdc_id", "branded_food_category"])
        food["category"] = food["fdc_id"].map(cat.set_index("fdc_id")["branded_food_category"])
    elif source == "fndds":
        cat = _read(zip_path, "wweia_food_category.csv")
        food["category"] = food["food_category_id"].map(cat.set_index(cat.columns[0])[cat.columns[1]])
    else:
        cat = _read(zip_path, "food_category.csv")
        food["category"] = food["food_category_id"].map(cat.set_index("id")["description"])
    food["source"] = source
    food["alt_names"] = food["fdc_id"].map(_alt_names(zip_path, food)) if source != "branded" else None
    food = food.join(_nutrients(zip_path, pd.Index(food["fdc_id"])), on="fdc_id")
    return food[["fdc_id", "source", "name", "category", "alt_names", "publication_date", *COLUMNS]]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop physically impossible rows, blank impossible sub-nutrients, dedupe branded names."""
    df = df.dropna(subset=["kcal", "protein", "fat", "carb"]).copy()
    v = df[list(COLUMNS)]
    mass = v[["protein", "fat", "carb"]].sum(axis=1) + v["water"].fillna(0) + v["alcohol"].fillna(0)
    ok = (v[["kcal", "protein", "fat", "carb"]] >= 0).all(axis=1) & (mass <= 105) & (v["kcal"] <= 950)
    atwater = 4 * v["protein"] + 9 * v["fat"] + 4 * v["carb"] + 7 * v["alcohol"].fillna(0)
    label_noise = (df["source"] == "branded") & ((v["kcal"] - atwater).abs() > np.maximum(40, 0.3 * v["kcal"]))
    df = df[ok & ~label_noise & (df["name"].str.len() >= 3)]
    for sub, parent, slack in (("sugars", "carb", 1.0), ("fiber", "carb", 1.0), ("sat_fat", "fat", 0.5)):
        df.loc[df[sub] > df[parent] + slack, sub] = np.nan
    df.loc[(df["sodium"] < 0) | (df["sodium"] > 40_000), "sodium"] = np.nan        # table salt = 38,758 mg
    df.loc[(df["water"] < 0) | (df["water"] > 100), "water"] = np.nan
    branded = df["source"] == "branded"
    key = df.loc[branded, "name"].str.lower().str.replace(r"\s+", " ", regex=True)
    newest = df.loc[branded].assign(key=key).sort_values(["publication_date", "fdc_id"]).drop_duplicates("key", keep="last")
    return pd.concat([df[~branded], newest.drop(columns="key")], ignore_index=True)


def build_table(zips: dict[str, Path]) -> pd.DataFrame:
    """All releases in `zips` -> one cleaned table (see clean())."""
    return clean(pd.concat([load_source(s, p) for s, p in zips.items()], ignore_index=True))


# --------------------------------------------------------------------------- Nutrition5k -> USDA
_MACRO_EPS = {"kcal": 20.0, "fat": 1.0, "carb": 2.0, "protein": 1.0}   # per 100 g


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", name.lower()).strip()


def macro_error(per_gram: pd.DataFrame, usda: pd.DataFrame) -> np.ndarray:
    """Mean relative difference of kcal/fat/carb/protein (rows of per_gram x 100 vs usda), 0 = same."""
    err = [np.abs(per_gram[c].to_numpy() * 100 - usda[c].to_numpy())
           / (np.maximum(per_gram[c].to_numpy() * 100, usda[c].to_numpy()) + e) for c, e in _MACRO_EPS.items()]
    return np.mean(err, axis=0)


def suggest_matches(vocab: pd.DataFrame, usda: pd.DataFrame, k: int = 5, shortlist: int = 400,
                    macro_weight: float = 2.0) -> pd.DataFrame:
    """Top-k generic USDA foods per Nutrition5k ingredient: a wide shortlist by char n-gram TF-IDF
    similarity of the names, re-ranked mostly by the macro error against the ingredient's own per-gram
    values. Nutrition5k took many of its values straight from SR Legacy rows, so the exact source row
    (macro error ~0) is usually in the shortlist even when its name is worded differently."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    gen = usda[usda["source"].isin(GENERIC)].reset_index(drop=True)
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
    g = tfidf.fit_transform(gen["name"].map(_norm))
    q = tfidf.transform(vocab["name"].map(_norm))
    sim = (q @ g.T).toarray()
    rows = []
    for i, ing in enumerate(vocab.itertuples()):
        top = np.argsort(-sim[i])[:shortlist]
        cand = gen.iloc[top]
        err = macro_error(pd.DataFrame([ing._asdict()] * len(top))[list(_MACRO_EPS)], cand)
        score = sim[i, top] - macro_weight * err
        for rank, j in enumerate(np.argsort(-score)[:k]):
            c = cand.iloc[j]
            rows.append({"ingr_id": ing.ingr_id, "name": ing.name, "rank": rank, "fdc_id": int(c["fdc_id"]),
                         "fdc_name": c["name"], "source": c["source"], "text_sim": round(float(sim[i, top[j]]), 3),
                         "macro_err": round(float(err[j]), 3), "score": round(float(score[j]), 3)})
    return pd.DataFrame(rows)


def extended_densities(vocab: pd.DataFrame, fdc_map: pd.DataFrame, usda: pd.DataFrame) -> pd.DataFrame:
    """Per-gram fiber / sugars / sat_fat (g) and sodium (mg) for every vocab row.

    Fiber and sugars keep the matched USDA food's share of carbohydrate and saturated fat its share of
    fat, applied to the Nutrition5k values, so they never exceed the exact macros (sugars <= carb,
    sat_fat <= fat). Sodium is taken per gram as is. Rows without a match get 0 and source "none".
    """
    m = vocab[["ingr_id", "fat", "carb"]].merge(fdc_map[["ingr_id", "fdc_id"]], on="ingr_id", how="left")
    m = m.merge(usda.drop_duplicates("fdc_id").set_index("fdc_id")[["fat", "carb", "fiber", "sugars", "sat_fat",
                                                                     "sodium"]].add_prefix("u_"),
                left_on="fdc_id", right_index=True, how="left")
    share = lambda part, whole: (m[part] / m[whole]).where(m[whole] > 0).clip(0, 1)
    out = pd.DataFrame({"ingr_id": m["ingr_id"]})
    out["fiber"] = (share("u_fiber", "u_carb") * m["carb"]).fillna(0.0)
    out["sugars"] = (share("u_sugars", "u_carb") * m["carb"]).fillna(0.0)
    out["sat_fat"] = (share("u_sat_fat", "u_fat") * m["fat"]).fillna(0.0)
    out["sodium"] = (m["u_sodium"] / 100).fillna(0.0)
    out["source"] = np.where(m["fdc_id"].notna(), "usda_match", "none")
    out["fdc_id"] = m["fdc_id"]
    return out
