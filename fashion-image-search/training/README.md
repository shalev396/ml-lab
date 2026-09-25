# Training: Fashion Image Search

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shalev396/ml-lab/blob/main/fashion-image-search/training/notebook.ipynb)

`notebook.ipynb` builds the search index that the [Space](https://huggingface.co/spaces/shalev396/fashion-image-search)
serves and writes the Hugging Face model repo `../model/`. There is no gradient training: a frozen,
ImageNet-pretrained MobileNetV2 embeds the catalog, and five representations are compared on held-out
query photos with Precision@K. All logic lives in `src/` and `../model/model.py`; the notebook calls it
step by step.

## Notebook sections

| # | Section | What it does |
|---|---|---|
| 1 | Setup | 1a finds the project files (clones the repo on a fresh runtime), 1b `%pip install -q -r ../../requirements.txt`, 1c picks the TensorFlow `device` (GPU if present, else CPU) |
| 2 | Config | `Config()` from `src/config.py` (every hyperparameter), `SMOKE_TEST` switch, `PUSH_TO_HUB` flag |
| 3 | Data | kagglehub download (cached) -> metadata -> stratified catalog / val / test split -> label tables, sample grid, decode the photos |
| 4 | Load model | `model.build_embedder()` (Rescaling + MobileNetV2, 2,257,984 params, frozen) and the list of variants |
| 5 | Training | embed every split, build every variant's index (baselines, full embedding, PCA fit on the catalog), Precision@K curves on validation |
| 6 | Evaluation | val + test Precision@1/5/10 for every variant, selection on validation, sklearn cross-check, per-category table, comparison chart, query examples |
| 7 | Inference | `export.save_model` writes the servable files, writes the Space examples (held-out test photos), then `model.load(dir).predict(image, k)`: exactly what the Space runs |
| 8 | Export | `metrics.json`, `assets/*.png`, refreshed model card; optional `utils.upload_model` when `PUSH_TO_HUB = True` |

## Run it

- **Colab**: click the badge. The setup cells clone the repo and install the requirements.
- **Locally**: `pip install -r ../../requirements.txt` (repo root), then open `notebook.ipynb` from this folder.
- **Headless smoke test** (300-product catalog, 40 + 40 queries, exports to `outputs/smoke/model/`, never touches `../model`):
  ```bash
  SMOKE_TEST=1 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp
  ```
- **Full run**: the same command with `SMOKE_TEST=0` (the default inside the notebook). It overwrites `../model/`
  and `../space/examples/`.

Hardware and time: the full run embeds ~9,000 photos. The original CPU run took 256 s end to end
(desktop CPU, idle machine); the rebuild recorded in `../model/metrics.json` (`train_time_s`) was slower
because the machine was shared with other jobs. A GPU runtime (Colab T4) takes about a minute (estimate,
not measured). The smoke run needs a few minutes, mostly the TensorFlow import and graph tracing.

## Data

- Primary: Kaggle [`paramaggarwal/fashion-product-images-small`](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small),
  downloaded anonymously with `kagglehub` (~570 MB, once). `styles.csv` is copied to `data/`; the images stay in
  the kagglehub cache and `data/kaggle_path.txt` points at them. `styles.csv` has a few malformed rows, which are skipped.
- Fallback: if Kaggle is unreachable, CIFAR-10 (`keras.datasets.cifar10`) is written to `data/cifar_images/` and the
  same pipeline runs with the class name as every label (numbers are not comparable).
- Split: master categories with < 20 products are dropped; stratified by master category into 8,000 catalog products,
  500 validation queries and 500 test queries (seed 42). Queries are never in the catalog.

## Outputs

| Where | What |
|---|---|
| `../model/` | `backbone.keras`, `embeddings.npy` (float16), `catalog.parquet` (metadata + thumbnails), `projection.npz` (PCA 1280->256, fit on the catalog embeddings; used by the deployed `mobilenetv2_pca256`), `config.json`, `metrics.json`, `assets/*.png`, `README.md` (card refreshed) |
| `../space/examples/` | one held-out test photo per master category |
| `outputs/` | notebook plots (gitignored); `outputs/smoke/` holds the complete smoke model repo |

Publishing: set `PUSH_TO_HUB = True`. The token comes from `hf auth login` (local) or an `HF_TOKEN`
secret / environment variable. Smoke runs are never uploaded.

## `src/` map

| File | Contents |
|---|---|
| `__init__.py` | puts `../model` on `sys.path` so `import model` is the model repo's `model.py` |
| `config.py` | `@dataclass Config`: every tunable, smoke shrinking, `model_dir` / `outputs_dir` / `examples_dir` |
| `data_setup.py` | kagglehub download + cache, CIFAR-10 fallback, metadata, stratified split, image decoding, Space examples |
| `model_builder.py` | the embedder from `model.py`, raw-pixel and colour-histogram baselines, PCA fit, variant names |
| `engine.py` | embedding every split, building the variants, Precision@K / curves, selection, sklearn cross-check |
| `export.py` | `save_model` (servable files + round-trip check), `export` (metrics.json, card, graphs), plots |
| `utils.py` | shared ml-lab helpers (same file in every project) |
