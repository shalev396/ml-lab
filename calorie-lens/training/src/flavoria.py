"""FlavoriaFoodWeight1700 (University of Turku lunch line; Sarapisto et al., Zenodo 2022, CC BY 4.0): overhead
photos of lunch plates whose components were weighed by the serving line's scales. A second cafeteria and camera,
with plates mostly heavier than Nutrition5k's: weighed plate MASS (it has no calories) as extra training data and
a second, independent test set.

  build()      once, on the Colab runtime (CPU work, ~2.6 GB download): parse + clean the weighing events, match
               each tray's "identification" photo (taken while the tray stands at the RFID reader; the "enter"
               photos often show a hand) to its session, keep the plates whose weighed items are all on the plate,
               resize to 512 x 512 JPEG, split by lunch day, and add flavoria/ + two tables to the dataset repo
  published()  whether the dataset repo already has it
  update_card()  the Flavoria section + viewer config in the dataset card

Cleaning (every count is written to the card):
  * a session's items are its weighing events summed per component;
  * trays with soup or dessert are dropped: those are served in a bowl / on a side plate, outside the photo;
  * scale glitches: a tray with one component over MAX_COMPONENT_G or a plate over MAX_PLATE_G is dropped (a pan
    lifted or refilled shows up as 1-2 kg of beetroot or stew);
  * sauces from the self-service station and bread are not weighed: the plate weight is a slight lower bound;
  * labels/flavoria_component_map.csv maps the 246 component names to Nutrition5k ingredients where one plain
    food matches (rice, boiled potato, tomato, cucumber, carrot, ...): those give grams per ingredient; mixed dishes
    count in the plate weight only.
"""
from __future__ import annotations

import io
import json
import re
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import utils

RECORD = "https://zenodo.org/records/5850856/files/"
CSV_TAR, IMAGES_TAR = "dataset_csv.tar", "dataset_images_and_csv.tar"
MAP_FILE = utils.TRAINING_DIR / "labels" / "flavoria_component_map.csv"
TABLES = ("flavoria_sessions.parquet", "flavoria_items.parquet")
SPLITS = ("train", "test")                    # no validation split: selection stays on Nutrition5k validation
MAX_COMPONENT_G, MAX_PLATE_G, MIN_PLATE_G = 600.0, 1200.0, 30.0
PRESENT_G = 10.0                              # an ingredient counts as on the plate (grams supervised) from here
TZ_OFFSETS_H = (0, 2, 3)                      # the CSV files mix UTC with Finnish summer (+3) / winter (+2) time
MATCH_S = 5.0                                 # the identification photo and the RFID time agree to the second
IMAGE_PX, JPEG_QUALITY = 512, 90
TEST_EVERY = 5                                # every 5th lunch day (sorted) is a test day: menus repeat within a day
CITATION = ("Sarapisto, T., Koivunen, L., Mäkilä, T., Klami, A., Ojansivu, P. (2022). FlavoriaFoodWeight1700: Automated "
            "Lunch Line Meal Pictures with Automatic Measurement of Weight and Contents. Zenodo. "
            "https://doi.org/10.5281/zenodo.5850856")
_CARD_START, _CARD_END = "<!-- flavoria:start -->", "<!-- flavoria:end -->"


# --------------------------------------------------------------------------- labels
def read_events(csv_tar: Path) -> pd.DataFrame:
    """Every weighing event of the CSVs in dataset_csv.tar: session, tray, identification time, component, grams."""
    frames = []
    with tarfile.open(csv_tar) as tf:
        for m in sorted(tf.getmembers(), key=lambda m: m.name):
            if not m.name.endswith(".csv"):
                continue
            d = pd.read_csv(tf.extractfile(m), encoding="utf-8-sig")
            d = d.loc[:, [c for c in d.columns if not c.startswith("Unnamed")]]
            d["file"] = Path(m.name).stem
            frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    time_of = lambda s: pd.to_datetime(s.astype(str).str.replace(".", "-", regex=False))
    return pd.DataFrame({"session": d["session_id"].astype(str) + "@" + d["file"],   # ids repeat across files
                         "tray": d["tray_id"].astype(str), "tid": time_of(d["user_identification_time"]),
                         "component": d["component_name"].astype(str).str.strip(),
                         "grams": d["weight_of_a_component"].astype(float)})


def sessions(events: pd.DataFrame, cmap: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """-> (kept sessions: session, tray, tid, date, mass, n_items, mapped_g; their items: session, component,
    grams, ingredient; counts of what was dropped and why)."""
    items = events.groupby(["session", "component"], as_index=False).agg(grams=("grams", "sum"))
    unknown = set(items["component"]) - set(cmap["component"])
    if unknown:
        raise ValueError(f"components missing from {MAP_FILE.name}: {sorted(unknown)[:5]}")
    items = items.merge(cmap[["component", "ingredient", "on_plate"]], on="component", how="left")
    items["ingredient"] = items["ingredient"].fillna("")
    per = items.groupby("session").agg(mass=("grams", "sum"), biggest=("grams", "max"), off=("on_plate", lambda s: (s == 0).any()),
                                       n_items=("component", "size"))
    first = events.groupby("session").agg(tray=("tray", "first"), tid=("tid", "first"))
    per = per.join(first)
    drop = {"soup or dessert (served outside the plate)": per["off"],
            f"one component over {MAX_COMPONENT_G:.0f} g (scale glitch)": ~per["off"] & (per["biggest"] > MAX_COMPONENT_G),
            f"plate over {MAX_PLATE_G:.0f} g or under {MIN_PLATE_G:.0f} g": ~per["off"] & (per["biggest"] <= MAX_COMPONENT_G)
            & ((per["mass"] > MAX_PLATE_G) | (per["mass"] < MIN_PLATE_G))}
    keep = ~np.logical_or.reduce(list(drop.values()))
    stats = {"sessions": int(len(per)), **{k: int(v.sum()) for k, v in drop.items()}, "kept_sessions": int(keep.sum())}
    kept = per[keep].copy()
    kept["date"] = kept["tid"].dt.strftime("%Y-%m-%d")
    mapped = items[items["ingredient"] != ""].groupby("session")["grams"].sum()
    kept["mapped_g"] = mapped.reindex(kept.index).fillna(0.0)
    kept = kept.drop(columns=["off", "biggest"]).reset_index()
    return kept, items[items["session"].isin(kept["session"])].drop(columns="on_plate").reset_index(drop=True), stats


def split_days(dates: pd.Series) -> pd.Series:
    """train / test by lunch day: every TEST_EVERY-th day of the sorted list (2nd, 7th, ...) is a test day."""
    days = sorted(dates.unique())
    test = {d for i, d in enumerate(days) if i % TEST_EVERY == 1}   # one day only (tiny smoke build): all train
    return dates.map(lambda d: "test" if d in test else "train")


# --------------------------------------------------------------------------- images
def download(url: str, dest: Path, workers: int = 16, chunk: int = 32 << 20, max_bytes: int | None = None) -> Path:
    """Parallel ranged download (Zenodo serves one stream slowly), resumable: finished chunks are recorded in
    <dest>.parts.json. max_bytes: only the first bytes (a partial tar, for smoke runs)."""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    probe = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=60, allow_redirects=True)
    if probe.status_code != 206 or "Content-Range" not in probe.headers:
        raise IOError(f"{url}: no range support (HTTP {probe.status_code})")
    size = int(probe.headers["Content-Range"].rsplit("/", 1)[1])
    size = min(size, max_bytes) if max_bytes else size
    parts_file = dest.with_suffix(dest.suffix + ".parts.json")
    done = set(json.loads(parts_file.read_text())) if parts_file.exists() and dest.exists() else set()
    if not dest.exists() or dest.stat().st_size != size:
        with open(dest, "wb") as f:
            f.truncate(size)
        done = set()
    starts = [s for s in range(0, size, chunk) if s not in done]

    def get(start: int) -> int:
        end = min(start + chunk, size) - 1
        for attempt in range(6):
            try:
                r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=300)
                if r.status_code != 206 or len(r.content) != end - start + 1:
                    raise IOError(f"range {start}-{end}: HTTP {r.status_code}, {len(r.content)} bytes")
                with open(dest, "r+b") as f:
                    f.seek(start)
                    f.write(r.content)
                return start
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)

    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        for i, start in enumerate(pool.map(get, starts), 1):
            done.add(start)
            parts_file.write_text(json.dumps(sorted(done)))
            if i % 10 == 0 or i == len(starts):
                mb = (len(done) * chunk) / 2 ** 20
                print(f"  {min(mb, size / 2 ** 20):.0f} / {size / 2 ** 20:.0f} MB ({mb / max(time.time() - t0, 1e-6):.1f} MB/s)",
                      flush=True)
    return dest


def identification_photos(tar_path: Path):
    """(tray, unix time, PNG bytes) of every "identification" photo in the image tar (a truncated tar: up to the
    last complete member)."""
    pattern = re.compile(r"(tray\d+)_(\d+)_[\d.]+_identification_")
    try:
        with tarfile.open(tar_path, "r|") as tf:
            for m in tf:
                hit = pattern.search(Path(m.name).name)
                if m.isfile() and hit and m.name.endswith(".png"):
                    data = tf.extractfile(m).read()
                    if len(data) == m.size:
                        yield hit.group(1), int(hit.group(2)), data
    except (tarfile.ReadError, EOFError, OSError):
        return


def match(tray: str, unix: int, kept: pd.DataFrame) -> str | None:
    """The session of this tray whose identification time equals the photo's time (in one of the CSV time zones)."""
    cand = kept[kept["tray"] == tray]
    if cand.empty:
        return None
    t = pd.Timestamp(unix, unit="s")
    best = min(((abs((cand["tid"] - t - pd.Timedelta(hours=h)).dt.total_seconds())).min(), h) for h in TZ_OFFSETS_H)
    if best[0] > MATCH_S:
        return None
    dt = abs((cand["tid"] - t - pd.Timedelta(hours=best[1])).dt.total_seconds())
    return str(cand.loc[dt.idxmin(), "session"])


def to_jpeg(png: bytes, px: int = IMAGE_PX, quality: int = JPEG_QUALITY) -> bytes:
    from PIL import Image

    with Image.open(io.BytesIO(png)) as img:
        out = io.BytesIO()
        img.convert("RGB").resize((px, px), Image.BICUBIC).save(out, "JPEG", quality=quality)
        return out.getvalue()


# --------------------------------------------------------------------------- build + publish
def build(out_dir: Path, ingredients: pd.DataFrame, repo_id: str | None = None, raw_dir: Path | None = None,
          tar_path: str | Path | None = None, max_bytes: int | None = None) -> dict:
    """Labels + photos -> out_dir/flavoria/<split>-00000.parquet, flavoria_sessions.parquet, flavoria_items.parquet,
    the card section; uploaded to repo_id if given. tar_path: an image tar already on disk (else downloaded into
    raw_dir; max_bytes = only its start, for smoke runs)."""
    from .dataset_build import _write_shard, check_write_access, fetch, publish

    out_dir, raw_dir = Path(out_dir), Path(raw_dir or out_dir.parent / "flavoria_raw")
    if repo_id:
        print("publishing as", check_write_access(repo_id), flush=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_tar = raw_dir / CSV_TAR
    if not csv_tar.exists():
        csv_tar.write_bytes(fetch(RECORD + CSV_TAR))
    cmap = pd.read_csv(MAP_FILE)
    cmap["ingredient"] = cmap["ingredient"].fillna("")
    k_of = dict(zip(ingredients["name"], ingredients["k"]))
    missing = set(cmap.loc[cmap["ingredient"] != "", "ingredient"]) - set(k_of)
    if missing:
        raise ValueError(f"{MAP_FILE.name} names ingredients the model does not have: {sorted(missing)}")
    kept, items, stats = sessions(read_events(csv_tar), cmap)
    print("sessions:", stats, flush=True)

    tar = Path(tar_path) if tar_path else download(RECORD + IMAGES_TAR, raw_dir / IMAGES_TAR, max_bytes=max_bytes)
    photos, n_seen = {}, 0
    for tray, unix, png in identification_photos(tar):
        n_seen += 1
        session = match(tray, unix, kept)
        if session is not None and session not in photos:
            photos[session] = to_jpeg(png)
    stats.update(identification_photos=n_seen, matched_plates=len(photos))
    kept = kept[kept["session"].isin(photos)].copy()
    kept["split"] = split_days(kept["date"])
    kept["dish_id"] = "flavoria_" + kept["session"].str.replace("@", "_", regex=False)
    items = items[items["session"].isin(kept["session"])].copy()
    items["k"] = items["ingredient"].map(k_of).fillna(-1).astype(int)
    stats.update({f"{s}_plates": int((kept["split"] == s).sum()) for s in SPLITS},
                 lunch_days=int(kept["date"].nunique()), median_plate_g=float(kept["mass"].median()),
                 plates_over_600g=int((kept["mass"] > 600).sum()),
                 mapped_share_of_grams=round(float(kept["mapped_g"].sum() / kept["mass"].sum()), 3))
    print("photos:", {k: stats[k] for k in ("identification_photos", "matched_plates", "train_plates", "test_plates")}, flush=True)

    written = []
    for split in SPLITS:
        rows = [{"dish_id": d, "image": photos[s]} for d, s in kept.loc[kept["split"] == split, ["dish_id", "session"]].values]
        if rows:
            path = out_dir / "flavoria" / f"{split}-00000.parquet"
            _write_shard(rows, path, "flavoria")
            written.append(str(path.relative_to(out_dir)).replace("\\", "/"))
    dish_of = dict(zip(kept["session"], kept["dish_id"]))
    kept[["dish_id", "session", "tray", "date", "split", "mass", "n_items", "mapped_g"]].to_parquet(out_dir / TABLES[0], index=False)
    items.assign(dish_id=items["session"].map(dish_of))[["dish_id", "component", "grams", "ingredient", "k"]].to_parquet(
        out_dir / TABLES[1], index=False)
    update_card(out_dir, stats)
    if repo_id:
        publish(out_dir, repo_id, written + list(TABLES) + ["README.md"], "Add FlavoriaFoodWeight1700 (weighed lunch plates)")
    return stats


def published(repo_id: str) -> bool:
    from huggingface_hub import HfApi

    return HfApi().file_exists(repo_id, TABLES[0], repo_type="dataset")


def update_card(out_dir: Path, stats: dict) -> None:
    """Add (or refresh) the Flavoria section and its viewer config in the dataset card."""
    path = Path(out_dir) / "README.md"
    text = path.read_text(encoding="utf-8") if path.exists() else "---\nlicense: cc-by-4.0\nconfigs:\n---\n"
    config = ("- config_name: flavoria\n  data_files:\n" + "\n".join(f"  - split: {s}\n    path: flavoria/{s}-*.parquet" for s in SPLITS)
              + "\n- config_name: flavoria_sessions\n  data_files: flavoria_sessions.parquet"
              + "\n- config_name: flavoria_items\n  data_files: flavoria_items.parquet\n")
    if "config_name: flavoria\n" not in text:
        head, sep, rest = text.partition("\n---\n")          # end of the YAML front matter
        text = head + "\n" + config.rstrip("\n") + sep + rest
    dropped = "\n".join(f"  - {k}: {v}" for k, v in stats.items() if k.startswith(("soup", "one component", "plate over")))
    section = f"""{_CARD_START}
## FlavoriaFoodWeight1700 (second cafeteria, weighed plates)
[FlavoriaFoodWeight1700]({RECORD.rsplit('/files/', 1)[0]}) (University of Turku, **CC BY 4.0**): an automated lunch line
photographs each tray from above and weighs every serving it takes. Built by
[`training/src/flavoria.py`](https://github.com/shalev396/ml-lab/tree/main/calorie-lens/training/src):

| Config | Rows | What |
|---|---|---|
| `flavoria` | {stats.get('matched_plates', 0)} | `dish_id`, `image` (the "identification" photo, 1024 px PNG -> 512 x 512 JPEG q{JPEG_QUALITY}) |
| `flavoria_sessions` | {stats.get('matched_plates', 0)} | plate weight `mass` (g, sum of the weighed servings), `date`, `split`, `n_items`, `mapped_g` |
| `flavoria_items` | per plate | each weighed component, its grams, and the Nutrition5k ingredient (`k`) where one plain food matches |

- Weight only: the source has no calories. {stats.get('mapped_share_of_grams', 0):.0%} of the weighed grams map to a
  Nutrition5k ingredient (`labels/flavoria_component_map.csv`, reviewed by hand); mixed dishes count in the plate weight only.
- Splits by lunch day ({stats.get('lunch_days', 0)} days; every {TEST_EVERY}th is a test day): train {stats.get('train_plates', 0)},
  test {stats.get('test_plates', 0)} plates. Median plate {stats.get('median_plate_g', 0):.0f} g; {stats.get('plates_over_600g', 0)} plates over 600 g.
- Cleaning, of {stats.get('sessions', 0)} sessions:
{dropped}
  then {stats.get('identification_photos', 0)} identification photos matched to {stats.get('matched_plates', 0)} kept plates (tray RFID +
  time to the second). Sauces from the self-service station and bread are not weighed: weights are slightly low.

{CITATION}
{_CARD_END}"""
    if _CARD_START in text:
        text = re.sub(re.escape(_CARD_START) + r".*?" + re.escape(_CARD_END), lambda _: section, text, flags=re.S)
    else:
        marker = "\n## Citation"
        text = text.replace(marker, "\n" + section + "\n" + marker, 1) if marker in text else text.rstrip() + "\n\n" + section + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
