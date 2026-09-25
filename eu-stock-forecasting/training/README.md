# Training — EU Stock Market Forecasting

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shalev396/ml-lab/blob/main/eu-stock-forecasting/training/notebook.ipynb)

Builds `../model/` from scratch. Five forecasters (ARIMA, Holt-Winters, VAR, Keras MLP, Keras LSTM) plus
the naive baseline are compared on each of the four indices. They are fit on the fit window, and the
classical model with the lowest **validation** MAPE is deployed for that index. The test window (last 100
business days) is only reported. A multi-step backtest then replays exactly what the Space serves.

## Notebook (`notebook.ipynb`)
1. **Setup**: 1a finds the project files (clones the repo + `model` submodule on a fresh runtime), 1b
   installs `../../requirements.txt`, 1c picks the `device` (TensorFlow GPU if present, else CPU). Then
   imports `src` and prints library versions.
2. **Config**: `Config()` from `src/config.py`, the `SMOKE_TEST` toggle and `PUSH_TO_HUB`.
3. **Data**: `data_setup.load_eustocks()` (cache or download), the fit / validation / test split, a plot
   of the four series and the ADF stationarity test (levels vs first differences, which gives d = 1).
4. **Load model**: the classical forecasters from `../model/model.py` and the two Keras nets from
   `src/model_builder.py`, with their parameter counts.
5. **Training**: VAR lag by AIC (once, all four indices). Per index: the ARIMA(p,1,q) AIC grid,
   Holt-Winters, and the MLP + LSTM with early stopping. Shows the training curves.
6. **Evaluation**: one-step-ahead walk-forward predictions over validation + test, the R `accuracy()`
   table (ME, RMSE, MAE, MPE, MAPE, MASE) for every variant and index, per-index selection, the
   comparison chart, the holdout overlay and the 20-day multi-step backtest.
7. **Inference**: `export.save_model` writes `config.json` + `eustocks.csv`. Then
   `model.load(cfg.model_dir).predict(cutoff, horizon, index)` runs on `../space/examples/requests.json`,
   and `predict_series` on the example CSV. This is the same code the Space runs.
8. **Export**: `export.export` writes `metrics.json`, `assets/*.png` and refreshes the model card. It lists
   the repo and optionally uploads it with `utils.upload_model`.

## Run it
- **Colab**: click the badge. The first cells clone the repo and install the requirements.
- **Local**: `pip install -r ../../requirements.txt`, then open `notebook.ipynb`. Headless:
  ```bash
  SMOKE_TEST=1 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp   # smoke -> outputs/smoke/model/
  SMOKE_TEST=0 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp   # full  -> ../model/
  ```

| Run | What | Time (shared 20-thread desktop CPU, measured 2026-09-25) |
|---|---|---|
| smoke | last 400 rows, 30/30-day val/test, tiny grids, nets 1 epoch on DAX only | 3.5 min (about 1.5 min of it is the TensorFlow import + `%pip` check) |
| full | 1,860 rows x 4 indices, 15-model ARIMA grid per index, 8 Keras fits | 17.4 min (796 s fitting: ~7 min ARIMA grids, ~6 min Keras) |

The machine was shared with other training jobs during these runs, so an idle CPU should be faster. The
original single-index version took about 1 min. Only the Keras baselines can use a GPU (picked up automatically
when TensorFlow sees one); they are tiny, so a GPU barely helps. The statsmodels fits are CPU-only.

## Data
- Primary source: R's `datasets::EuStockMarkets` through `statsmodels.datasets.get_rdataset("EuStockMarkets", "datasets")`.
- Fallback: the Rdatasets CSV mirror
  `https://vincentarelbundock.github.io/Rdatasets/csv/datasets/EuStockMarkets.csv`.
- Cached at `data/EuStockMarkets.csv` (the download is skipped when present). `data/DATA_SOURCE.txt`
  records which source was used.
- 1,860 rows x 4 indices. The R ts has no calendar (start 1991.496, frequency 260), so a **synthetic
  business-day index from 1991-07-01** is attached (`model.business_days`): Mon-Fri, holidays not skipped,
  last row 1998-08-14.

## Outputs
- `../model/` (full run): `config.json` (deployed method per index, ARIMA orders + coefficients, VAR lag,
  serving limits, versions), `eustocks.csv` (the data the Space forecasts from), `metrics.json`,
  `assets/*.png`. The model card's YAML and its three tables are refreshed from `metrics.json`. The Keras
  nets are not exported: they are baselines and they lose.
- `outputs/`: `comparison_<index>.csv` and the notebook's plots. `outputs/smoke/`: the smoke run's complete
  model repo (card + `model.py` + `handler.py` included, so `MODEL_DIR=training/outputs/smoke/model python app.py`
  serves it).
- Publish: `PUSH_TO_HUB = True` in the notebook, or `python -c "from src import utils; utils.upload_model()"`.
  The token comes from `hf auth login` or an `HF_TOKEN` env var / Colab secret.

## `src/`
| File | Role |
|---|---|
| `__init__.py` | puts `../model` on `sys.path`, so `import model as M` works everywhere |
| `config.py` | `Config` dataclass: every hyperparameter, smoke shrinking, output paths |
| `data_setup.py` | download + cache, `Splits` (fit / validation / test), ADF table, NN lag windows |
| `model_builder.py` | Keras MLP + LSTM builders, classical model descriptions, parameter counts |
| `engine.py` | ARIMA AIC grid, VAR lag selection, NN fitting, one-step walk-forward, R accuracy table, selection, backtest |
| `export.py` | `save_model` (config.json + eustocks.csv), `export` (metrics.json, plots, card tables) |
| `utils.py` | shared across ml-lab (identical in every project): paths, seeding, JSON, card updater, Hub upload |
