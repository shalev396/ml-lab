# Training — Credit Card Fraud Detector

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shalev396/ml-lab/blob/main/credit-card-fraud/training/notebook.ipynb)

Builds `../model/` from scratch. Twelve candidate models are fit on the train split. Each gets a decision
threshold tuned for max F1 on the validation split, and the one with the best **validation PR-AUC** becomes
the default. The test split is only used to report every candidate once, at the end.

## Notebook (`notebook.ipynb`) — the controller
Every section calls functions from `src/` (or `../model/model.py`) one step at a time.

| Section | What it does | Calls |
|---|---|---|
| 1. Setup | 1a fetches the repo / `model` submodule if missing (fresh Colab runtime), 1b installs `../../requirements.txt`, 1c sets `device` (`"GPU"` / `"CPU"`), 1d imports `src` | — |
| 2. Config | `SMOKE_TEST` toggle, `PUSH_TO_HUB`, `cfg = Config()`, TF setup + seeds | `Config`, `utils.setup_tf`, `utils.set_seeds` |
| 3. Data | download + cache, duplicate removal, stratified 60/20/20 split, class-balance / Amount / Time plots, features | `data_setup.load_dataframe`, `data_setup.split`, `export.plot_eda`, `engine.prepare_features` |
| 4. Load model | the 11 scikit-learn / imbalanced-learn variants and the Keras MLP from `model.build_mlp` (parameter count) | `model_builder.build_variants`, `model_builder.build_mlp`, `engine.compile_mlp` |
| 5. Training | fits every candidate on the train split; MLP learning curves | `engine.fit_variants`, `engine.fit_mlp` |
| 6. Evaluation | threshold tuning + validation/test metrics, selection by validation PR-AUC, comparison table, PR curves, threshold plot, confusion matrix | `engine.evaluate_all`, `engine.select`, `engine.comparison_table`, `export.figures` |
| 7. Inference | saves the selected models, checks the reload reproduces the training predictions, then runs `model.load(dir).predict(...)` on `../space/examples/*.csv` and the endpoint `handler.EndpointHandler` | `export.save_models`, `export.check_roundtrip` |
| 8. Export | `metrics.json`, `assets/*.png`, model card refresh, optional Hub upload | `export.export`, `utils.upload_model` |

## Run it
- **Colab**: click the badge. The first cells clone the repo and install the requirements.
- **Local**: `pip install -r ../../requirements.txt`, then open `notebook.ipynb` from this folder and run all
  cells. Headless smoke check:
  ```bash
  SMOKE_TEST=1 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp
  ```
  **A full run (`SMOKE_TEST=0`, the default) overwrites `../model/`**. A smoke run writes a complete model
  repo to `outputs/smoke/model/` and never touches `../model/`.

| Run | Rows | Time (measured, see conditions) |
|---|---|---|
| smoke | 20,000 stratified subset, small models | 0.5-2 min for the whole notebook, including imports (headless `nbconvert` runs measured 31 s and 1 min 55 s, depending on load from other jobs) |
| full | 283,726 | 15.6 min wall time for the whole notebook on a shared desktop CPU limited to 6 threads (fitting the 12 candidates: 383 s, the `train_time_s` in `metrics.json`); an earlier 20-thread run of the old `train.py` pipeline took about 3 min end to end |

Everything runs on CPU. The MLP is tiny, so a GPU (picked up by `utils.setup_tf()` when present) barely
helps. On a 2-core Colab CPU, expect several minutes, mostly for SMOTE + kNN and SMOTE + random forest.

## Data
- Primary source: Kaggle [`mlg-ulb/creditcardfraud`](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
  via `kagglehub` (anonymous, ~150 MB CSV). Cached at `data/creditcard.csv`; the download is skipped when
  the file is present.
- Fallback: OpenML `data_id=1597` (the same data) via `sklearn.datasets.fetch_openml`.
- 1,081 exact duplicate rows are dropped before splitting, which leaves 283,726 rows and 473 frauds.

## Outputs
- `../model/` (full run): `scaler.joblib`, `model.joblib` (best scikit-learn variant, fitted classifier
  without its resampler), `model.keras` (MLP), `config.json` (schema, thresholds, default model, versions),
  `metrics.json`, `assets/*.png` (comparison of all variants, MLP training curves, PR curves, threshold
  tuning, confusion matrix). The model card's YAML and both of its tables are refreshed from `metrics.json`.
  `model.py`, `handler.py` and `requirements.txt` are hand-written and not generated.
- `outputs/smoke/model/`: the smoke run's complete model repo (card, `model.py`, `handler.py` included, so
  `MODEL_DIR=training/outputs/smoke/model python app.py` serves it).
- Publish: `PUSH_TO_HUB = True` in section 2. The token comes from `hf auth login` or an `HF_TOKEN`
  env var / Colab secret.

## `src/`
| File | Role |
|---|---|
| `__init__.py` | puts `../model` on `sys.path`, so `import model as M` works everywhere |
| `config.py` | `Config` dataclass: every hyperparameter + smoke shrinking + output paths |
| `data_setup.py` | download (kagglehub, then OpenML) → cache → dedupe → stratified 60/20/20 split |
| `model_builder.py` | the 11 scikit-learn / imblearn variants + the MLP from `model.build_mlp`; strips samplers for inference |
| `engine.py` | features, metrics, max-F1 threshold tuning, fitting / evaluating every candidate, selection by validation PR-AUC, Keras compile/fit (class weights + early stopping) |
| `export.py` | `save_models` (weights + `config.json`), `check_roundtrip`, `export` (`metrics.json`, plots, card) and the plot functions |
| `utils.py` | shared ml-lab helpers (identical in every project) |
