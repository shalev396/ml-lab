# Training: Stock Price LSTM

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shalev396/ml-lab/blob/main/stock-lstm-forecast/training/notebook.ipynb)

`notebook.ipynb` is the controller. It calls the reusable code in `src/` step by step and exports the
Hugging Face model repo (`../model/`) that the Space serves.

## What the notebook does
1. **Setup**: 1a finds the project files (it clones the repo and the `model/` submodule on a fresh runtime),
   1b runs `%pip install -q -r ../../requirements.txt`, 1c sets `device` (TensorFlow: `"GPU"` or `"CPU"`),
   1d imports `src/` and `../model/model.py`.
2. **Config**: `cfg = Config()` holds every hyperparameter, the `SMOKE_TEST` switch and `PUSH_TO_HUB = False`.
3. **Data**: `data_setup.load_prices` reads AAPL daily OHLCV (cache, then yfinance), then
   `data_setup.prepare_tracks` builds three representations with a chronological 70/15/15 split. The
   scalers are fit on the train rows only.
4. **Load model**: `model_builder.build_variants` builds the four networks. The deployed one comes from
   `model.build_lstm`. The cell prints the parameter counts.
5. **Training**: `engine.compile_model` and `engine.fit` for each variant (Adam + MSE, early stopping and
   learning-rate halving on validation loss), then the loss curves.
6. **Evaluation**: `engine.evaluate` computes one-step-ahead RMSE / MAE / MAPE on the $ scale, and
   `engine.evaluate_naive` does the same for the persistence baseline. `engine.select_best` picks the model
   by **validation** RMSE. `engine.horizon_eval` then scores recursive multi-day forecasts from many test
   cutoffs, the way the Space uses the model.
7. **Inference**: `export.save_model` writes `model.keras`, `config.json` and the frozen `prices.csv`, and
   checks that the reloaded model reproduces the in-memory forecast. Then
   `model.load(dir, "cpu").predict(cutoff, horizon)` runs on the pairs in `../space/examples/cutoffs.csv`.
8. **Export**: `export.export_report` writes `metrics.json` and `assets/*.png` and refreshes the model card.
   The last cell uploads the model repo when `PUSH_TO_HUB = True` (never for smoke runs).

## How to run
- **Colab**: click the badge. The setup cells clone the repo and install the requirements.
- **Locally**: `pip install -r ../../requirements.txt` from this folder, then open `notebook.ipynb`.
- **Headless smoke test** (5 min 17 s measured on a CPU shared with several other training jobs; it trains 8-unit models for 1 epoch on 400 days; exports to `outputs/smoke/model/` and never touches `../model/`):
  ```bash
  SMOKE_TEST=1 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp
  ```
- **Full run** (`SMOKE_TEST=0`, the default): it **overwrites `../model/`** (weights, prices, metrics,
  card, plots). The measured training time is `train_time_s` in `../model/metrics.json`: 575 s for the committed
  run, on a CPU shared with several other jobs. The earlier uncontended run took 72 s. The models are tiny
  (4k to 51k parameters), so a GPU brings little.
- Publishing: set `PUSH_TO_HUB = True`. The token comes from `hf auth login` or an `HF_TOKEN`
  secret / environment variable.

## Data
- **Primary**: [yfinance](https://github.com/ranaroussi/yfinance), `AAPL` daily OHLCV from 2015-01-01
  (`auto_adjust=False`: `Close` is split-adjusted, not dividend-adjusted). It is cached in `data/AAPL.csv`
  and read from there on every later run. The committed model was trained on the cache downloaded on
  2026-08-01, which covers 2015-01-02 to 2026-07-31 (2,911 trading days).
- **Fallback**: if yfinance fails and nothing is cached, a seeded geometric-Brownian-motion series is used.
  `data/DATA_SOURCE.txt` records which source was used, and `metrics.json` repeats it in `dataset`.
- The Space uses the frozen `model/prices.csv` (Date, Close), written by `export.save_model`.

## `src/`
| file | purpose |
|---|---|
| `__init__.py` | puts `../model` on `sys.path` (`import model as M`) |
| `config.py` | `Config` dataclass: every hyperparameter + smoke handling + `model_dir` |
| `data_setup.py` | download/cache, RSI/EMA indicators, the three `Track`s, windows and splits |
| `model_builder.py` | the four variants (the deployed LSTM from `model.py`, SimpleRNN, stacked LSTM) |
| `engine.py` | compile/fit, $-scale metrics, naive baseline, selection, recursive multi-step evaluation |
| `export.py` | `save_model` (weights, config, prices), `export_report` (metrics, plots, card) |
| `utils.py` | shared ml-lab helpers (identical in every project) |

`data/` and `outputs/` are gitignored.
