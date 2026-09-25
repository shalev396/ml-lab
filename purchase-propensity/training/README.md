# Training — Purchase Propensity + RFM

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shalev396/ml-lab/blob/main/purchase-propensity/training/notebook.ipynb)

Builds `../model/` from scratch: a class-balanced logistic regression and a PyTorch MLP that predict whether a
website session ends in an order, plus the RFM customer segments of the UCI Online Retail shop.
`notebook.ipynb` is the controller. It calls the functions in `src/` step by step, and `../model/model.py`
(architecture, preprocessing, `Predictor`) is the same file the Space and the Inference Endpoint run.

## Notebook (`notebook.ipynb`)
1. **Setup**
   - 1a clones the repo / fetches the `model` submodule when the files are missing (a fresh Colab runtime).
   - 1b installs `../../requirements.txt`.
   - 1c picks `device` (cuda → mps → cpu).
   - 1d imports `src` + `model` and prints library versions.
2. **Config**: the `Config` dataclass, the `SMOKE_TEST` switch and `PUSH_TO_HUB`.
3. **Data**:
   - sessions: download + cache, stratified 68/12/20 split, `StandardScaler` fit on train.
   - order rate per flag (EDA).
   - UCI Online Retail: clean + cache, then RFM scores and segments.
4. **Load model**: `LogisticRegression(class_weight="balanced")` and `model.PropensityMLP` with their parameter
   counts (24 and 3,649).
5. **Training**: fits the logistic regression, then trains the MLP with pos_weight BCE, AdamW and early stopping on
   validation ROC-AUC. Shows the loss / ROC-AUC curves.
6. **Evaluation**:
   - thresholds tuned for max F1 on validation.
   - default model = best validation PR-AUC.
   - test comparison table, ROC/PR curves, confusion matrix.
   - LR coefficients + MLP permutation importance, RFM segment chart.
7. **Inference**: `export.save_model(...)` writes the weights and checks the round trip, then
   `model.load(cfg.model_dir, device).predict(...)` runs on `../space/examples/visitors.csv`, followed by an RFM
   lookup for a new customer.
8. **Export**: `metrics.json`, the 6 card plots in `assets/`, the refreshed model card, and (full run only) the Space
   examples. `PUSH_TO_HUB = True` uploads the folder.

## Run it
- **Colab**: click the badge. The first cells clone the repo and install the requirements.
- **Local**: `pip install -r ../../requirements.txt`, then open `notebook.ipynb` from this folder.
- **Headless smoke test** (25k-session subset, tiny MLP, writes only to `outputs/smoke/model/`):
  ```bash
  SMOKE_TEST=1 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp
  ```
  Leaving `SMOKE_TEST` unset (or `0`) runs the full training and writes to `../model/`.

| Run | Sessions | Time (desktop CPU, 6 threads) |
|---|---|---|
| smoke | 25,000 stratified subset, MLP 23-16-1, 3 epochs | 82 s measured for the headless notebook run, kernel start included |
| full | 455,401 | 147 s training (LR 1.6 s, MLP 141 s, 18 epochs) + about 1 min for evaluation and plots |

A GPU is picked up automatically for the MLP, but the network is so small that the CPU is fine. On a 2-core
Colab CPU, expect a few minutes for the full run.

## Data
- **Sessions:** Kaggle [`benpowis/customer-propensity-to-purchase-data`](https://www.kaggle.com/datasets/benpowis/customer-propensity-to-purchase-data),
  `training_sample.csv` (39 MB), downloaded anonymously with `kagglehub` and cached as
  `data/propensity_training_sample.csv`. The download is skipped when the file is present.
  Fallback: download `training_sample.csv` by hand from the Kaggle page and save it under that name.
- **Retail invoices:** [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail) xlsx (23 MB) from
  the UCI archive (two alternate URLs: the legacy `machine-learning-databases/00352/` file and the
  `static/public/352` zip). It is cleaned once (the openpyxl parse takes about a minute) and cached as
  `data/online_retail_clean.parquet`.

## Outputs
- `../model/` (full run):
  - `logreg.joblib`, `scaler.joblib`
  - `model.safetensors` + `config.json` (features, MLP architecture, thresholds, default model, versions)
  - `rfm_segment_stats.json`, `metrics.json`, `assets/*.png`
  - the model card's YAML and both tables are refreshed from `metrics.json`.
- `../space/examples/visitors.csv` (full run only): 8 real test-split sessions.
- `outputs/smoke/model/` (smoke run): a complete model repo (card, `model.py` and `handler.py` included), so
  `MODEL_DIR=training/outputs/smoke/model python app.py` serves it.
- Publish: `PUSH_TO_HUB = True` in the notebook, or `python -c "from src import utils; utils.upload_model()"`.
  The token comes from `hf auth login` or an `HF_TOKEN` env var / Colab secret.

## `src/`
| File | Role |
|---|---|
| `__init__.py` | puts `../model` on `sys.path`, so `import model as M` works everywhere |
| `config.py` | `Config` dataclass: every hyperparameter, smoke shrinking and output paths |
| `data_setup.py` | session download → cache → stratified split + scaler, loaders, examples; retail download → clean → RFM scores, segments, artifact |
| `model_builder.py` | the two experiments (`build_logreg`, `build_mlp` from `model.PropensityMLP`) and their parameter counts |
| `engine.py` | LR fit, MLP train loop with early stopping, threshold tuning, metrics, selection, permutation importance |
| `export.py` | `save_model` (weights + config + round-trip check), `build_metrics`, `write_report` (metrics.json, plots, card), `write_examples` |
| `utils.py` | shared ml-lab helpers (identical in every project) |
