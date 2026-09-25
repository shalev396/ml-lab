# Training: California Housing Price Predictor

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shalev396/ml-lab/blob/main/california-housing/training/notebook.ipynb)

`notebook.ipynb` is the controller: it trains and compares four regressors on sklearn's California Housing
dataset, then exports XGBoost + the PyTorch MLP to [`../model`](../model) (the Hugging Face model repo). All
logic lives in `src/`; the feature engineering, the MLP class and the `Predictor` come from `../model/model.py`,
the same file the Space runs.

## Notebook sections
1. **Setup**: 1a finds the project files (clones the repo on a fresh runtime), 1b installs the root
   `requirements.txt`, 1c picks `device` (cuda → mps → cpu), 1d imports `src/` and `model`.
2. **Config**: `Config()` from `src/config.py`, the `SMOKE_TEST` switch and `PUSH_TO_HUB`.
3. **Data**: download + cache, 80/20 train/test split (seed 42) with 15 % of train as validation, EDA plots.
4. **Load model**: builds LinearRegression, RandomForest (200 trees), XGBoost and `model.HousingMLP`
   (11-256-128-64-1, 44,289 parameters); fits the MLP's `StandardScaler` on the train split.
5. **Training**: fits every experiment. XGBoost uses early stopping on validation RMSE (50 rounds) and is trimmed
   to its best round; the MLP uses AdamW + MSE with early stopping on validation loss (patience 10).
6. **Evaluation**: validation + test RMSE / MAE / R² of all four; the default model is the exported model with
   the lowest **validation** RMSE (the test split never influences a choice); predicted-vs-actual plot.
7. **Inference**: saves the models to a scratch folder, reloads them with `model.load(dir, device)`, prices the
   districts in `../space/examples/districts.csv`, and checks the reloaded files reproduce the test RMSE.
8. **Export**: writes weights, `config.json`, `metrics.json` and `assets/*.png` to `../model`, refreshes the model
   card; `PUSH_TO_HUB = True` uploads it (token from `hf auth login` or an `HF_TOKEN` secret / env var).

## Run it
- **Colab**: click the badge (a CPU runtime is enough).
- **Locally**: `pip install -r ../../requirements.txt`, then open `notebook.ipynb` from this folder.
- **Headless smoke test** (3,000 rows, tiny models, writes only to `outputs/smoke/`):
  `SMOKE_TEST=1 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp`
- **Full run headless** (overwrites `../model`): same command with `SMOKE_TEST=0`.
- On a shared machine, `N_JOBS=6` limits the RandomForest / XGBoost threads (default: all cores).

## Hardware and time
Everything runs on a CPU; a GPU only speeds up the small MLP and is not needed. The full run of 2026-09-25 was
measured on a shared 20-thread desktop CPU while several other training jobs were running (`N_JOBS=6`), so it is
much slower than an idle machine would be: XGBoost 1,205 s, MLP 956 s (91 epochs), RandomForest 8 s, about 37 min
end to end (the fit times are in `../model/metrics.json`). The smoke run takes about a minute of compute on an idle
CPU, plus the `%pip` check.

## Data
- **Primary**: `sklearn.datasets.fetch_california_housing` (StatLib copy of the 1990 US census data, 20,640 block
  groups), cached as `data/california_housing.csv` and reused on later runs.
- **Fallback**: `housing.csv` from [ageron/handson-ml2](https://github.com/ageron/handson-ml2) (the same districts
  as raw totals), converted to the same per-household features; its 207 rows without `total_bedrooms` are dropped.

## Outputs
- `../model/`: `xgb.ubj`, `model.safetensors` + `config.json`, `scaler.joblib`, `metrics.json`, `assets/`, card.
- `outputs/staged_model/`: the inference check of section 7 (gitignored).
- `outputs/smoke/`: everything a smoke run writes (gitignored).

## `src/` map
| file | contents |
|---|---|
| `__init__.py` | puts `../model` on `sys.path` (`import model as M`) |
| `config.py` | `Config` dataclass: every hyperparameter + smoke handling + output folders |
| `data_setup.py` | download/cache, fallback source, splits, `xy()`, example districts for the Space |
| `model_builder.py` | builds the four experiments and the MLP scaler |
| `engine.py` | fit helpers, the PyTorch train loop, prediction, metrics, model selection |
| `export.py` | `config.json`, weights, `metrics.json`, graphs, model-card tables |
| `utils.py` | shared ml-lab helpers (devices, seeds, JSON, model card, Hub upload) |
