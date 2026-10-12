"""Build the calorie-lens dataset once and publish it as a Hugging Face dataset repo.

CPU-only work (a cheap Colab CPU runtime is enough; nothing here needs a GPU):
  * every Nutrition5k overhead RGB-D scan (rgb.png -> JPEG q95, depth_raw.png kept as uint16 PNG);
  * every side-angle video (raw H.264, 1920x1080, ~30 frames): FRAMES_PER_CAMERA frames spread over
    the middle of the sweep, resized to 910x512 (16:9 kept: the cameras are close and a crop cuts food);
  * the cleaned labels (n5k_labels), the USDA table (usda) and the reviewed ingredient -> USDA map.
~195 GB are streamed over anonymous HTTPS and discarded; ~8 GB of parquet shards remain.

Resumable: build/manifest.json records every finished dish and is uploaded with the shards, so a
new runtime reads it back from the Hub and carries on where the last one stopped.
"""
from __future__ import annotations

import io
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd

from . import n5k_labels, usda

GCS_LIST = "https://storage.googleapis.com/storage/v1/b/nutrition5k_dataset/o"
GCS_FILE = "https://storage.googleapis.com/nutrition5k_dataset/"
IMAGERY = "nutrition5k_dataset/imagery/"
CAMERAS = ("A", "B", "C", "D")
SIDE_SIZE = (910, 512)                    # (width, height)
FRAMES_PER_CAMERA = 5
MAX_SIDE_FRAMES = 32_767                  # frame indices are int16 in the shards; a sweep has ~40-150 frames
SPLIT_FILES = {"train": "train", "val": "validation", "test": "test"}
MANIFEST = "build/manifest.json"

_session = None


def _http():
    global _session
    if _session is None:
        import requests
        from requests.adapters import HTTPAdapter

        _session = requests.Session()
        _session.mount("https://", HTTPAdapter(pool_connections=8, pool_maxsize=8))
    return _session


def _reset_http() -> None:
    """Worker initializer: a forked child must not reuse the parent's keep-alive (TLS) connections."""
    global _session
    _session = None


class FetchError(RuntimeError):
    """A download that failed after every retry: the dish is retried later, not marked done."""


def fetch(url: str, retries: int = 5) -> bytes:
    """GET with exponential back-off (GCS occasionally drops a connection under load)."""
    for attempt in range(retries):
        try:
            r = _http().get(url, timeout=120)
            r.raise_for_status()
            return r.content
        except Exception as e:
            if attempt == retries - 1:
                raise FetchError(f"{url}: {type(e).__name__}: {e}") from e
            time.sleep(2 ** attempt)


def list_imagery() -> pd.DataFrame:
    """Every imagery file in the bucket -> dish_id, kind (overhead | side), key, path, size."""
    rows, token = [], None
    while True:
        params = {"prefix": IMAGERY, "fields": "items(name,size),nextPageToken", "maxResults": 1000}
        if token:
            params["pageToken"] = token
        page = _http().get(GCS_LIST, params=params, timeout=60).json()
        for item in page.get("items", []):
            parts = item["name"].split("/")
            if len(parts) != 5:
                continue
            _, _, kind, dish, fname = parts
            key = fname.rsplit(".", 1)[0]
            if (kind == "realsense_overhead" and key in ("rgb", "depth_raw")) or \
                    (kind == "side_angles" and key in {f"camera_{c}" for c in CAMERAS}):
                rows.append((dish, "overhead" if kind == "realsense_overhead" else "side", key,
                             item["name"], int(item["size"])))
        token = page.get("nextPageToken")
        if not token:
            break
    return pd.DataFrame(rows, columns=["dish_id", "kind", "key", "path", "size"])


def pick_frames(n: int, k: int = FRAMES_PER_CAMERA, margin: float = 0.1) -> list[int]:
    """k frame indices spread evenly over the middle (1 - 2*margin) of an n-frame sweep."""
    if n <= k:
        return list(range(n))
    lo, hi = margin * (n - 1), (1 - margin) * (n - 1)
    return sorted({int(round(x)) for x in np.linspace(lo, hi, k)})


def decode_side(data: bytes, k: int = FRAMES_PER_CAMERA, size=SIDE_SIZE, quality: int = 90,
                max_frames: int = MAX_SIDE_FRAMES) -> tuple[int, list[tuple[int, bytes]]]:
    """Raw H.264 stream -> (frame count, [(frame index, JPEG bytes)]) for the picked frames."""
    import av
    from PIL import Image

    with av.open(io.BytesIO(data), format="h264") as c:     # pass 1: count frames (no decoding)
        n = sum(1 for p in c.demux(video=0) if p.size)
    if n > max_frames:          # a camera left recording (dish_1568048053 camera_C: 2.4 GB), not a sweep
        raise ValueError(f"{n} frames, more than a turntable sweep (max {max_frames})")
    keep, out = set(pick_frames(n, k)), []
    with av.open(io.BytesIO(data), format="h264") as c:
        c.streams.video[0].thread_type = "AUTO"
        for i, frame in enumerate(c.decode(video=0)):
            if i in keep:
                rgb = frame.reformat(width=size[0], height=size[1], format="rgb24").to_ndarray()
                buf = io.BytesIO()
                Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
                out.append((i, buf.getvalue()))
                if len(out) == len(keep):
                    break
    return n, out


def overhead_to_jpeg(png: bytes, quality: int = 95) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.open(io.BytesIO(png)).convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def process_dish(dish_id: str, files: dict[str, str], frames: int = FRAMES_PER_CAMERA) -> dict:
    """Download + convert one dish (runs in a worker process). files: key -> bucket path."""
    out = {"dish_id": dish_id, "overhead": None, "side": [], "errors": [], "bytes": 0, "retry": False}
    try:
        if "rgb" in files and "depth_raw" in files:
            rgb, depth = fetch(GCS_FILE + files["rgb"]), fetch(GCS_FILE + files["depth_raw"])
            out["bytes"] += len(rgb) + len(depth)
            out["overhead"] = {"dish_id": dish_id, "image": overhead_to_jpeg(rgb), "depth": depth}
    except FetchError as e:                                  # network: retry the whole dish later
        out["errors"].append(f"overhead: {e}")
        out["retry"] = True
    except Exception as e:                                   # a broken scan must not stop the build
        out["errors"].append(f"overhead: {type(e).__name__}: {e}")
    for cam in CAMERAS:
        path = files.get(f"camera_{cam}")
        if path is None:
            continue
        try:
            data = fetch(GCS_FILE + path)
            out["bytes"] += len(data)
            n, picked = decode_side(data, frames)
            out["side"] += [{"dish_id": dish_id, "camera": cam, "frame": i, "n_frames": n, "image": jpg}
                            for i, jpg in picked]
        except FetchError as e:
            out["errors"].append(f"camera_{cam}: {e}")
            out["retry"] = True
        except Exception as e:                               # a corrupt video: skipped, recorded
            out["errors"].append(f"camera_{cam}: {type(e).__name__}: {e}")
    return out


def probe(imagery: pd.DataFrame, n_streams: int | None = None, workers: int | None = None) -> dict:
    """Time side videos end to end and project the full build on this machine. Defaults to two videos per
    worker: with fewer videos than workers the pool is never full and the projection comes out too long."""
    workers = workers or default_workers()
    n_streams = n_streams or max(20, 2 * workers)
    side = imagery[imagery["kind"] == "side"].sample(n_streams, random_state=0)
    t, frames, nbytes = time.time(), [], 0
    with ProcessPoolExecutor(workers, initializer=_reset_http) as ex:
        results = list(ex.map(process_dish, side["dish_id"], [{r.key: r.path} for r in side.itertuples()]))
    seconds = time.time() - t
    for r in results:
        frames += [row["n_frames"] for row in r["side"][:1]]
        nbytes += r["bytes"]
    n_side = int((imagery["kind"] == "side").sum())
    return {"streams": n_streams, "workers": workers, "seconds": round(seconds, 1),
            "frames_per_stream": float(np.mean(frames)) if frames else None,
            "MB_per_s": round(nbytes / 1e6 / seconds, 1),
            "projected_hours": round(seconds / n_streams * n_side / 3600, 2)}


def default_workers() -> int:
    """Downloads overlap with decoding: twice the cores (env BUILD_WORKERS overrides)."""
    return int(os.getenv("BUILD_WORKERS", 2 * (os.cpu_count() or 1)))


# --------------------------------------------------------------------------- shards + manifest
def _features(table: str):
    from datasets import Features, Image, Value

    if table == "overhead":
        return Features({"dish_id": Value("string"), "image": Image(), "depth": Image()})
    if table == "flavoria":
        return Features({"dish_id": Value("string"), "image": Image()})
    return Features({"dish_id": Value("string"), "camera": Value("string"), "frame": Value("int16"),
                     "n_frames": Value("int16"), "image": Image()})


def _write_shard(rows: list[dict], path: Path, table: str) -> None:
    from datasets import Dataset, disable_progress_bars

    disable_progress_bars()
    cols = {k: [r[k] for r in rows] for k in rows[0]}
    for k in ("image", "depth"):
        if k in cols:
            cols[k] = [{"bytes": b, "path": None} for b in cols[k]]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    Dataset.from_dict(cols, features=_features(table)).to_parquet(str(tmp))
    tmp.replace(path)


def _load_manifest(out_dir: Path, repo_id: str | None) -> dict:
    path = out_dir / MANIFEST
    if not path.exists() and repo_id:
        try:
            from huggingface_hub import hf_hub_download

            hf_hub_download(repo_id, MANIFEST, repo_type="dataset", local_dir=out_dir)
        except Exception:
            pass                                             # nothing published yet
    if path.exists():
        return json.loads(path.read_text())
    return {"done": {}, "shards": {}, "uploaded": [], "errors": {}}


def _drop_unlisted_shards(out_dir: Path, manifest: dict) -> None:
    """A build that died inside a flush can leave a shard the manifest never recorded. Delete it: its
    dishes are not marked done, so this run redoes them (and would otherwise reuse the file name)."""
    for path in out_dir.glob("*/*.parquet"):
        name, _, index = path.stem.rpartition("-")
        if int(index) >= manifest["shards"].get(f"{path.parent.name}/{name}", 0):
            print("removing unrecorded shard", path.relative_to(out_dir), flush=True)
            path.unlink()


def _save_manifest(out_dir: Path, manifest: dict) -> None:
    path = out_dir / MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest))
    tmp.replace(path)


def check_write_access(repo_id: str) -> str:
    """Fail at the start, not at the first upload 30 min in: publishing needs a token with write access."""
    from huggingface_hub import HfApi

    try:
        info = HfApi().whoami()
    except Exception as e:
        raise PermissionError(f"publishing {repo_id} needs a Hugging Face token with write access: a Colab "
                              "secret HF_TOKEN with notebook access, or `hf auth login`") from e
    if info.get("auth", {}).get("accessToken", {}).get("role") == "read":
        raise PermissionError(f"the Hugging Face token is read-only; publishing {repo_id} needs write access")
    return info["name"]


def publish(out_dir: Path, repo_id: str, paths: list[str], message: str) -> None:
    """Upload the given files (relative to out_dir) in one commit."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=str(out_dir), repo_id=repo_id, repo_type="dataset",
                      allow_patterns=paths, commit_message=message)


def build_images(out_dir: Path, labels: n5k_labels.Labels, imagery: pd.DataFrame, *, dishes=None,
                 workers: int | None = None, shard_rows: int = 1000, repo_id: str | None = None,
                 publish_every_s: float = 1800, log_every: int = 100) -> dict:
    """Download + convert every dish's imagery into parquet shards (resumable, see module docstring).

    Shards are written per split: overhead/<split>-NNNNN.parquet and side/<split>-NNNNN.parquet. A
    dish counts as done once its rows are in a written shard. With repo_id, new shards + the manifest
    are uploaded every publish_every_s seconds and at the end.
    """
    out_dir = Path(out_dir)
    if repo_id:
        print("publishing as", check_write_access(repo_id), flush=True)
    manifest = _load_manifest(out_dir, repo_id)
    _drop_unlisted_shards(out_dir, manifest)
    split_of = labels.dishes.set_index("dish_id")["split"]
    files = {d: dict(zip(g["key"], g["path"])) for d, g in imagery.groupby("dish_id") if d in split_of.index}
    todo = [d for d in (dishes if dishes is not None else sorted(files)) if d in files and d not in manifest["done"]]
    workers = workers or default_workers()
    buffers = {s: {"overhead": [], "side": [], "dishes": {}} for s in SPLIT_FILES}
    last_publish, t0, done_now, nbytes = time.time(), time.time(), 0, 0

    def flush(split: str) -> None:
        b = buffers[split]
        if not b["dishes"]:
            return
        for table in ("overhead", "side"):
            if b[table]:
                key = f"{table}/{SPLIT_FILES[split]}"
                n = manifest["shards"].get(key, 0)
                _write_shard(b[table], out_dir / f"{key}-{n:05d}.parquet", table)
                manifest["shards"][key] = n + 1
        manifest["done"].update(b["dishes"])
        buffers[split] = {"overhead": [], "side": [], "dishes": {}}
        _save_manifest(out_dir, manifest)

    def maybe_publish(final: bool = False) -> None:
        nonlocal last_publish
        if not repo_id or (not final and time.time() - last_publish < publish_every_s):
            return
        new = sorted(str(p.relative_to(out_dir)).replace(os.sep, "/") for p in out_dir.glob("*/*.parquet"))
        new = [p for p in new if p not in set(manifest["uploaded"])]
        if new:
            publish(out_dir, repo_id, new, f"Add {len(new)} image shards ({len(manifest['done'])} dishes)")
        manifest["uploaded"] += new                       # only once the upload went through
        _save_manifest(out_dir, manifest)
        publish(out_dir, repo_id, [MANIFEST], f"Build manifest: {len(manifest['done'])} dishes done")
        last_publish = time.time()

    failed: list[str] = []
    with ProcessPoolExecutor(workers, initializer=_reset_http) as ex:
        pending, queue, retried = set(), iter(todo), False
        while True:
            while len(pending) < 4 * workers:
                d = next(queue, None)
                if d is None:
                    break
                pending.add(ex.submit(process_dish, d, files[d]))
            if not pending:
                if failed and not retried:                # one more pass over network failures
                    print(f"retrying {len(failed)} dishes whose downloads failed", flush=True)
                    queue, failed, retried = iter(failed), [], True
                    continue
                break
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in finished:
                r = f.result()
                if r["retry"]:                            # not done: picked up again by the next pass / run
                    failed.append(r["dish_id"])
                    manifest["errors"][r["dish_id"]] = r["errors"]
                    continue
                split = split_of[r["dish_id"]]
                b = buffers[split]
                if r["overhead"]:
                    b["overhead"].append(r["overhead"])
                b["side"] += r["side"]
                b["dishes"][r["dish_id"]] = {"overhead": int(r["overhead"] is not None), "side": len(r["side"])}
                if r["errors"]:                           # e.g. a corrupt video: done, with the error kept
                    manifest["errors"][r["dish_id"]] = r["errors"]
                else:
                    manifest["errors"].pop(r["dish_id"], None)
                nbytes += r["bytes"]
                done_now += 1
                if len(b["side"]) >= shard_rows or len(b["overhead"]) >= shard_rows:
                    flush(split)
                    maybe_publish()
                if done_now % log_every == 0:
                    rate = done_now / (time.time() - t0)
                    print(f"{done_now}/{len(todo)} dishes  {rate:.2f}/s  {nbytes / 1e6 / (time.time() - t0):.0f} MB/s  "
                          f"eta {(len(todo) - done_now) / rate / 60:.0f} min", flush=True)
    for split in SPLIT_FILES:
        flush(split)
    maybe_publish(final=True)
    return {"dishes_done": len(manifest["done"]), "this_run": done_now, "seconds": round(time.time() - t0),
            "errors": len(manifest["errors"]), "not_done_network_failures": failed, "shards": manifest["shards"]}


# --------------------------------------------------------------------------- label tables + card
def write_tables(out_dir: Path, labels: n5k_labels.Labels, usda_table: pd.DataFrame, fdc_map: pd.DataFrame
                 ) -> list[str]:
    """dishes / items / ingredients / fixes / usda tables next to the image shards."""
    out_dir = Path(out_dir)
    manifest = _load_manifest(out_dir, None)
    avail = pd.DataFrame.from_dict(manifest["done"], orient="index")
    dishes = labels.dishes.copy()
    dishes["has_overhead"] = dishes["dish_id"].map(avail["overhead"] if len(avail) else {}).fillna(0).astype(bool)
    dishes["n_side_frames"] = dishes["dish_id"].map(avail["side"] if len(avail) else {}).fillna(0).astype(int)
    ext = usda.extended_densities(labels.vocab, fdc_map, usda_table)
    ingredients = labels.vocab.reset_index().merge(ext.drop(columns="ingr_id"), left_index=True, right_index=True)
    ingredients = ingredients.merge(fdc_map[["ingr_id", "fdc_name"]], on="ingr_id", how="left")
    fixes = labels.fixes.astype({"old": str, "new": str})      # mixed numbers / names
    paths = {"dishes.parquet": dishes, "items.parquet": labels.items, "fixes.parquet": fixes,
             "usda.parquet": usda_table}
    for name, df in paths.items():
        df.to_parquet(out_dir / name, index=False)
    ingredients.to_csv(out_dir / "ingredients.csv", index=False)
    return [*paths, "ingredients.csv"]


def write_card(out_dir: Path, labels: n5k_labels.Labels, repo_id: str) -> str:
    """Dataset card: configs for the viewer, sources + licences, what was changed and how it was built."""
    d = labels.dishes
    n = d["split"].value_counts()
    configs = "\n".join(
        [f"- config_name: {t}\n  data_files:\n" + "\n".join(
            f"  - split: {s}\n    path: {t}/{s}-*.parquet" for s in SPLIT_FILES.values()) for t in ("overhead", "side")]
        + [f"- config_name: {t}\n  data_files: {t}.parquet" for t in ("dishes", "items", "usda")]
        + ["- config_name: ingredients\n  data_files: ingredients.csv"])
    fixes = labels.fixes["reason"].str.replace(r" \(\d+ items\)", "", regex=True).value_counts()
    card = f"""---
license: cc-by-4.0
pretty_name: Calorie Lens data (cleaned Nutrition5k + USDA FoodData Central)
task_categories: [image-classification, tabular-regression]
tags: [food, nutrition, calories, portion-estimation, ml-lab]
size_categories: [100K<n<1M]
configs:
{configs}
---

# Calorie Lens data

Training data of [shalev396/calorie-lens](https://huggingface.co/shalev396/calorie-lens): a photo of a meal ->
the grams of each of {len(labels.vocab)} ingredients -> nutrition. Built by
[`training/src/dataset_build.py`](https://github.com/shalev396/ml-lab/tree/main/calorie-lens/training/src) from:

- **[Nutrition5k](https://github.com/google-research-datasets/Nutrition5k)** (Thames et al., CVPR 2021, **CC BY 4.0**):
  cafeteria dishes weighed ingredient by ingredient, an overhead RGB-D scan and four side-angle videos per dish.
- **[USDA FoodData Central](https://fdc.nal.usda.gov/)** (public domain, CC0): Foundation Foods, SR Legacy,
  FNDDS 2021-2023 and Branded Foods, nutrients per 100 g.

## Contents
| Config | Rows | What |
|---|---|---|
| `overhead` | one per dish with a scan | `image` (640x480 JPEG q95), `depth` (uint16 PNG, value / 10000 = metres; 0 = no reading) |
| `side` | {FRAMES_PER_CAMERA} per camera | `camera` A-D, `frame` index of `n_frames`, `image` (910x512 JPEG q90 from the 1080p video) |
| `dishes` | {len(d)} | corrected totals (kcal, mass g, fat, carb, protein), the logged ones as `orig_*`, `group`, `split`, `test_clean` |
| `items` | {len(labels.items)} | one row per (dish, ingredient): `k` (model output index), grams, nutrients |
| `ingredients` | {len(labels.vocab)} | per-gram kcal / fat / carb / protein (Nutrition5k, exact), fiber / sugars / sat_fat / sodium (mg) from the reviewed USDA match |
| `usda` | see config | name, category, alternative names, nutrients per 100 g (sodium in mg) |

## Splits
- `test` = the official Nutrition5k `rgb_test` dishes ({n.get('test', 0)}); those with an overhead scan are exactly the official
  `depth_test` split, so results compare with published numbers. `test_clean` marks the {int(d['test_clean'].sum())} test dishes
  whose plate never appears in train / validation.
- `validation` = {n.get('val', 0)} dishes: whole plate groups drawn from train. `train` = {n.get('train', 0)} dishes (Cafe 2 included).
- **Plate groups**: plates were built one item at a time with a scan after each, so many dishes are earlier states of a
  later one. A group never straddles train and validation.

## Changes to the original labels
Every Nutrition5k ingredient's nutrients are grams x a per-gram table; these tables keep that exactly. Fixes (counts):
{chr(10).join(f'- {reason}: {count}' for reason, count in fixes.items())}

## Citation
```bibtex
@inproceedings{{thames2021nutrition5k,
  title={{Nutrition5k: Towards Automatic Nutritional Understanding of Generic Food}},
  author={{Thames, Quin and Karpur, Arjun and Norris, Wade and Xia, Fangting and Panait, Liviu and Weyand, Tobias and Sim, Jack}},
  booktitle={{CVPR}}, pages={{8903--8911}}, year={{2021}}
}}
```
U.S. Department of Agriculture, Agricultural Research Service. FoodData Central, 2019-2026. fdc.nal.usda.gov.
"""
    (Path(out_dir) / "README.md").write_text(card, encoding="utf-8")
    return "README.md"


def smoke_dishes(labels: n5k_labels.Labels, imagery: pd.DataFrame, per_split: int = 2) -> list[str]:
    """A tiny build for smoke runs: the first dishes of each split that have both a scan and videos."""
    both = set(imagery.loc[imagery["kind"] == "overhead", "dish_id"]) & set(imagery.loc[imagery["kind"] == "side", "dish_id"])
    d = labels.dishes
    return [x for s in ("train", "val", "test") for x in d[(d["split"] == s) & d["dish_id"].isin(both)]["dish_id"].head(per_split)]
