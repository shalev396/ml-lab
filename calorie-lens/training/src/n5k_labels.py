"""Nutrition5k labels: parse the metadata, fix the known label bugs, group re-scans of one plate, split.

Pure pandas on ~2.4 MB of CSV/txt, so it runs anywhere in seconds. Facts this module relies on
(checked against the metadata, see check()):
  * the dish CSVs have no header: dish_id, total kcal/mass/fat/carb/protein, then 7 fields per
    ingredient (ingr_id, name, grams, kcal, fat, carb, protein);
  * every ingredient's nutrients are exactly grams x the per-gram row of ingredients_metadata.csv,
    and dish totals are the sum of the ingredients (Cafe 2 totals are all logged as 0);
  * dish ids are unix timestamps; plates were built one item at a time with a scan after every
    item, so many dishes are earlier states of a later dish. Those scans share a plate group and
    a group never straddles train / validation, nor train / test_clean.
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset"
METADATA_FILES = (
    "metadata/dish_metadata_cafe1.csv", "metadata/dish_metadata_cafe2.csv",
    "metadata/ingredients_metadata.csv",
    "dish_ids/splits/rgb_train_ids.txt", "dish_ids/splits/rgb_test_ids.txt",
    "dish_ids/splits/depth_train_ids.txt", "dish_ids/splits/depth_test_ids.txt",
)
NUTRIENTS = ("kcal", "fat", "carb", "protein")          # per-gram table columns, this order everywhere
TOTALS = ("kcal", "mass", "fat", "carb", "protein")     # dish-level targets (the paper's five)
_TABLE_COLUMNS = {"ingr": "name", "id": "ingr_id", "cal/g": "kcal", "fat(g)": "fat",
                  "carb(g)": "carb", "protein(g)": "protein"}

# No single ingredient on a cafeteria plate weighs a kilogram: these are x1000 logging errors
# (the scale wrote 7974 for 7.974 g). Bianco et al. (Nutrients 2025) confirm the corrected values.
SCALE_ERROR_GRAMS = 1000.0
KNOWN_SCALE_ERRORS = {
    ("dish_1551567508", "olives"), ("dish_1551567573", "olives"), ("dish_1551567604", "olives"),
    ("dish_1551389458", "lemon"), ("dish_1551389551", "lemon"), ("dish_1551389588", "lemon"),
    ("dish_1551381990", "asparagus"), ("dish_1551382149", "asparagus"), ("dish_1551382179", "asparagus"),
    ("dish_1560974769", "white rice"),
}
# The two "deprecated" table rows are duplicates of live ones: 453 is apple with its carb and protein
# columns swapped (0.52 kcal/g, 0.14 g protein/g), 458 is watermelon.
MERGE_IDS = {453: 54, 458: 121}
PLATE_ONLY_ID = 423                      # "plate only": the empty plate's weight, 0 kcal -> 0 g of food
DROP_DISHES = {"dish_1559172356"}        # the overhead photo shows a phone, not the plate (GitHub issue #10)
GROUP_WINDOW_S = 30 * 60                 # incremental scans of one plate happen within minutes
GROUP_TOL_G = 1.0


@dataclass
class Labels:
    dishes: pd.DataFrame      # one row per dish: corrected + original totals, group, split, test_clean
    items: pd.DataFrame       # one row per (dish, ingredient): k (vocab index), grams, nutrients
    vocab: pd.DataFrame       # K rows: ingr_id, name, per-gram nutrients, train frequency, median grams
    fixes: pd.DataFrame       # every change made to the raw labels, with the reason
    official: dict            # the official split id sets


def download_metadata(root: Path) -> Path:
    """Fetch the metadata CSVs + split lists (~2.4 MB, anonymous HTTPS) into root; skips files present."""
    root = Path(root)
    for rel in METADATA_FILES:
        dst = root / rel
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(f"{BASE_URL}/{rel}", timeout=60) as r:
                dst.write_bytes(r.read())
    return root


def read_table(root: Path) -> pd.DataFrame:
    """ingredients_metadata.csv -> per-gram nutrients, indexed by integer ingredient id."""
    t = pd.read_csv(Path(root) / "metadata/ingredients_metadata.csv").rename(columns=_TABLE_COLUMNS)
    t["name"] = t["name"].str.strip()
    return t.set_index("ingr_id")[["name", *NUTRIENTS]]


def read_dishes(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The two headerless dish CSVs -> (dishes, items) exactly as logged."""
    dishes, items = [], []
    for cafe in (1, 2):
        for line in (Path(root) / f"metadata/dish_metadata_cafe{cafe}.csv").read_text().splitlines():
            f = line.split(",")
            dishes.append([f[0], cafe, *map(float, f[1:6])])
            rest = f[6:]
            if len(rest) % 7:
                raise ValueError(f"{f[0]}: {len(rest)} ingredient fields is not a multiple of 7")
            for pos in range(len(rest) // 7):
                ingr_id, name, *values = rest[7 * pos: 7 * pos + 7]
                items.append([f[0], pos, int(ingr_id.removeprefix("ingr_")), name.strip(), *map(float, values)])
    dishes = pd.DataFrame(dishes, columns=["dish_id", "cafe", *TOTALS])
    dishes["ts"] = dishes["dish_id"].str.removeprefix("dish_").astype(np.int64)
    items = pd.DataFrame(items, columns=["dish_id", "pos", "ingr_id", "name", "grams", *NUTRIENTS])
    return dishes, items


def read_official_splits(root: Path) -> dict[str, set[str]]:
    return {name: set((Path(root) / f"dish_ids/splits/{name}_ids.txt").read_text().split())
            for name in ("rgb_train", "rgb_test", "depth_train", "depth_test")}


def fix_labels(dishes: pd.DataFrame, items: pd.DataFrame, table: pd.DataFrame
               ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Corrected (dishes, items) + a log of every change.

    Ingredient nutrients are recomputed as grams x table and dish totals (mass included) as the sum
    of the ingredients, so the labels are internally consistent by construction. The logged dish
    totals stay available as orig_* (needed to report numbers comparable with published results).
    """
    fixes = []
    items = items.copy()
    items["orig_grams"] = items["grams"]

    # 1. dishes that are not food photos
    items = items[~items["dish_id"].isin(DROP_DISHES)]
    dishes = dishes[~dishes["dish_id"].isin(DROP_DISHES)].copy()
    fixes += [(d, "dish", "", "dropped", "photo shows a phone, not the plate") for d in DROP_DISHES]

    # 2. x1000 scale errors
    bad = items["grams"] >= SCALE_ERROR_GRAMS
    for row in items[bad].itertuples():
        fixes.append((row.dish_id, f"grams[{row.name}]", row.grams, row.grams / 1000, "x1000 scale error"))
    items.loc[bad, "grams"] = items.loc[bad, "grams"] / 1000

    # 3. the empty plate is not food
    plate = items["ingr_id"] == PLATE_ONLY_ID
    for row in items[plate].itertuples():
        fixes.append((row.dish_id, "grams[plate only]", row.grams, 0.0, "empty-plate weight, not food"))
    items = items[~plate]

    # 4. duplicate table rows + names from the table (some rows are logged with an empty name)
    for old, new in MERGE_IDS.items():
        n = int((items["ingr_id"] == old).sum())
        fixes.append(("*", f"ingr_id {old}", old, new, f"duplicate of '{table.at[new, 'name']}' ({n} items)"))
    items["ingr_id"] = items["ingr_id"].replace(MERGE_IDS)
    renamed = items["name"] != items["ingr_id"].map(table["name"])
    for (iid, name), n in items[renamed].groupby(["ingr_id", "name"]).size().items():
        fixes.append(("*", f"name[{iid}]", name, table.at[iid, "name"], f"name taken from the table ({n} items)"))
    items["name"] = items["ingr_id"].map(table["name"])

    # 5. nutrients = grams x table, totals = sum of ingredients
    for n in NUTRIENTS:
        items[n] = items["grams"] * items["ingr_id"].map(table[n])
    sums = items.groupby("dish_id")[["grams", *NUTRIENTS]].sum().rename(columns={"grams": "mass"})
    dishes = dishes.rename(columns={t: f"orig_{t}" for t in TOTALS})
    dishes = dishes.join(sums, on="dish_id")
    dishes[list(TOTALS)] = dishes[list(TOTALS)].fillna(0.0)
    dishes["n_items"] = dishes["dish_id"].map(items.groupby("dish_id").size()).fillna(0).astype(int)
    for t in TOTALS:
        changed = ~np.isclose(dishes[t], dishes[f"orig_{t}"], atol=0.5, rtol=0.0)
        for r in dishes[changed].itertuples():
            reason = ("scale total != sum of ingredient grams" if t == "mass" else
                      "Cafe 2 totals logged as 0" if r.cafe == 2 else "follows the ingredient fixes")
            fixes.append((r.dish_id, t, getattr(r, f"orig_{t}"), getattr(r, t), reason))
    fixes = pd.DataFrame(fixes, columns=["dish_id", "field", "old", "new", "reason"])
    return dishes.reset_index(drop=True), items.reset_index(drop=True), fixes


def plate_groups(dishes: pd.DataFrame, items: pd.DataFrame, window_s: int = GROUP_WINDOW_S,
                 tol_g: float = GROUP_TOL_G) -> pd.Series:
    """Plate group id per dish, by union-find over two kinds of link:
      (a) incremental scans: each dish links to ONE predecessor, the largest earlier dish (within
          window_s) whose ingredients all reappear in it with the same grams +- tol_g. Linking every
          earlier subset instead chains unrelated plates through standard single-item portions;
      (b) re-scans: consecutive dishes with identical labels (grams to 0.1 g), within window_s, or at
          any time when they have >= 2 ingredients (a coincidence is then implausible; one standard
          portion, e.g. 100 g of rice, repeats across days on different plates)."""
    parent = {d: d for d in dishes["dish_id"]}

    def find(d):
        while parent[d] != d:
            parent[d] = parent[parent[d]]
            d = parent[d]
        return d

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    content = {d: dict(zip(g["ingr_id"], g["grams"])) for d, g in items.groupby("dish_id")}
    order = dishes[dishes["dish_id"].isin(content)].sort_values("ts")[["dish_id", "ts"]].to_numpy()
    ts = dict(order)

    for j, (b, tb) in enumerate(order):                                   # (a) one predecessor each
        cb, best = content[b], None
        for a, ta in order[:j][::-1]:
            if tb - ta > window_s:
                break
            ca = content[a]
            if (len(ca) < len(cb) and (best is None or len(ca) > len(content[best]))
                    and all(k in cb and abs(cb[k] - g) <= tol_g for k, g in ca.items())):
                best = a
        if best is not None:
            union(best, b)

    same_label: dict[tuple, list] = {}                                    # (b) identical re-scans
    for d, c in content.items():
        same_label.setdefault(tuple(sorted((i, round(g, 1)) for i, g in c.items())), []).append(d)
    for label, ds in same_label.items():
        ds.sort(key=ts.get)
        for a, b in zip(ds, ds[1:]):
            if ts[b] - ts[a] <= window_s or len(label) >= 2:
                union(a, b)
    roots = dishes["dish_id"].map(find)
    return roots.map({r: i for i, r in enumerate(sorted(roots.unique()))}).rename("group")


def make_splits(dishes: pd.DataFrame, official: dict[str, set[str]], val_frac: float = 0.1,
                seed: int = 42) -> pd.DataFrame:
    """Adds `split` (train | val | test) and `test_clean` to dishes (which must carry `group`).

    test = the official rgb_test ids (its dishes with an overhead photo are exactly depth_test), so
    test numbers stay comparable with published results. val = whole plate groups (~val_frac of the
    train dishes) drawn from groups with no test dish. Cafe 2 (in no official split) trains.
    test_clean = test dishes whose plate group has no train / val dish.
    """
    out = dishes.copy()
    out["split"] = np.where(out["dish_id"].isin(official["rgb_test"]), "test", "train")
    test_groups = set(out.loc[out["split"] == "test", "group"])
    candidates = out[(out["split"] == "train") & ~out["group"].isin(test_groups)]
    sizes = candidates.groupby("group").size()
    rng = np.random.default_rng(seed)
    target, picked, total = val_frac * (out["split"] == "train").sum(), [], 0
    for g in rng.permutation(sizes.index.to_numpy()):
        if total >= target:
            break
        picked.append(g)
        total += sizes[g]
    out.loc[out["group"].isin(picked), "split"] = "val"
    seen = set(out.loc[out["split"] != "test", "group"])
    out["test_clean"] = (out["split"] == "test") & ~out["group"].isin(seen)
    return out


def build_vocab(items: pd.DataFrame, table: pd.DataFrame, dishes: pd.DataFrame) -> pd.DataFrame:
    """The K ingredients the model predicts (every id used by a dish, sorted by id) with their per-gram
    nutrients, how many train dishes contain them and their median grams there (head bias init)."""
    ids = sorted(items["ingr_id"].unique())
    train = items[items["dish_id"].isin(dishes.loc[dishes["split"] == "train", "dish_id"])]
    vocab = table.loc[ids].reset_index()
    vocab["train_dishes"] = vocab["ingr_id"].map(train.groupby("ingr_id")["dish_id"].nunique()).fillna(0).astype(int)
    median = train.groupby("ingr_id")["grams"].median().combine_first(items.groupby("ingr_id")["grams"].median())
    vocab["median_grams"] = vocab["ingr_id"].map(median)
    vocab.index.name = "k"
    return vocab


def load_labels(root: Path, val_frac: float = 0.1, seed: int = 42) -> Labels:
    """Everything above in one call: read -> fix -> group -> split -> vocab."""
    table = read_table(root)
    raw_dishes, raw_items = read_dishes(root)
    official = read_official_splits(root)
    dishes, items, fixes = fix_labels(raw_dishes, raw_items, table)
    dishes["group"] = plate_groups(dishes, items).to_numpy()
    dishes = make_splits(dishes, official, val_frac, seed)
    vocab = build_vocab(items, table, dishes)
    items["k"] = items["ingr_id"].map(pd.Series(vocab.index, index=vocab["ingr_id"]))
    return Labels(dishes, items, vocab, fixes, official)


def check(labels: Labels, root: Path) -> dict:
    """Assert the label invariants (verification step V1); returns a printable summary."""
    d, it, v, off = labels.dishes, labels.items, labels.vocab, labels.official
    table = read_table(root)
    raw_dishes, raw_items = read_dishes(root)
    # the dataset property everything relies on: logged nutrients = logged grams x table
    m = raw_items.join(table[list(NUTRIENTS)], on="ingr_id", rsuffix="_g")
    for n in NUTRIENTS:
        assert np.allclose(m[n], m["grams"] * m[f"{n}_g"], atol=1e-3), f"logged {n} != grams x table"
    # corrected labels are consistent and plausible
    sums = it.groupby("dish_id")[["grams", *NUTRIENTS]].sum()
    dd = d.set_index("dish_id")
    assert np.allclose(dd.loc[sums.index, "mass"], sums["grams"]) and \
        all(np.allclose(dd.loc[sums.index, n], sums[n]) for n in NUTRIENTS), "totals != sum of items"
    assert (d.loc[(d["cafe"] == 2), "kcal"] > 0).all(), "a Cafe 2 dish still has 0 kcal"
    assert it["grams"].max() < SCALE_ERROR_GRAMS, "an ingredient still weighs >= 1 kg"
    shifted = {(r.dish_id, r.name) for r in raw_items[raw_items["grams"] >= SCALE_ERROR_GRAMS].itertuples()}
    assert shifted == KNOWN_SCALE_ERRORS, f"scale errors changed: {shifted ^ KNOWN_SCALE_ERRORS}"
    assert not (it["name"].eq("") | it["name"].eq("deprecated")).any(), "empty / deprecated names left"
    assert it["k"].notna().all() and v["ingr_id"].is_unique
    # splits
    n = d["split"].value_counts().to_dict()
    assert n["test"] == len(off["rgb_test"]), "test != official rgb_test"
    overhead_test = off["depth_test"]
    assert overhead_test <= set(d.loc[d["split"] == "test", "dish_id"])
    assert n["train"] + n["val"] == len(off["rgb_train"] - DROP_DISHES) + (raw_dishes["cafe"] == 2).sum()
    groups = {s: set(d.loc[d["split"] == s, "group"]) for s in ("train", "val", "test")}
    assert not groups["train"] & groups["val"], "a plate group straddles train / val"
    assert not set(d.loc[d["test_clean"], "group"]) & (groups["train"] | groups["val"]), "test_clean leaks"
    return {
        "dishes": len(d), "ingredients_K": len(v), "plate_groups": d["group"].nunique(),
        "splits": n, "test_clean": int(d["test_clean"].sum()),
        "test_dishes_sharing_a_plate_with_train": int(((d["split"] == "test") & ~d["test_clean"]).sum()),
        "overhead_test (depth_test)": len(overhead_test), "fixes": labels.fixes["reason"].value_counts().to_dict(),
        "rare_ingredients (<5 train dishes)": int((v["train_dishes"] < 5).sum()),
    }
