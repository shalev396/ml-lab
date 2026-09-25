# 📈 Stock Price LSTM

> Pick a date, and an LSTM forecasts Apple's stock price for the following weeks, next to the naive "no change" forecast and the real prices, so you can see how hard it is to beat a random walk.

<p>
  <a href="https://huggingface.co/spaces/shalev396/stock-lstm-forecast"><img alt="Space" src="https://img.shields.io/badge/🤗%20Space-live%20demo-blue"></a>
  <a href="https://huggingface.co/shalev396/stock-lstm-forecast"><img alt="Model" src="https://img.shields.io/badge/🤗%20Model-shalev396%2Fstock--lstm--forecast-yellow"></a>
  <a href="https://colab.research.google.com/github/shalev396/ml-lab/blob/main/stock-lstm-forecast/training/notebook.ipynb"><img alt="Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>

| | |
|---|---|
| **Task** | Time-series forecasting: next-day AAPL close, applied recursively for 1 to 60 business days |
| **Framework** | TensorFlow 2.21 / Keras 3 (float32) |
| **Architecture** | LSTM(64) on standardized daily log-returns, 60-day window (16,961 parameters) |
| **Dataset** | AAPL daily OHLCV from [yfinance](https://github.com/ranaroussi/yfinance), 2015-01-02 to 2026-07-31 (2,911 days), chronological 70/15/15 split |
| **Result** | test RMSE = $4.48 next-day (naive persistence: $4.47, so the model does **not** beat it) |
| **Runs on** | Space: CPU basic · Training: CPU, GPU or Colab |

## The problem
Given the closing prices up to a cutoff date, forecast the closes of the following business days. The
reference is **naive persistence**: tomorrow's close equals today's. For liquid stocks this baseline is
famously hard to beat, because daily prices behave almost like a random walk. The project reports every
number against it.

## The data
Daily Open/High/Low/Close/Volume for `AAPL` from 2015, downloaded with yfinance (`auto_adjust=False`: the
close is split-adjusted, not dividend-adjusted). The data is cached in `training/data/AAPL.csv`. The cache
used here was downloaded on 2026-08-01 and covers 2015-01-02 to 2026-07-31. When yfinance is unreachable
and nothing is cached, a seeded synthetic random walk takes its place, and `DATA_SOURCE.txt` records that.

The split is chronological 70/15/15 by the day being predicted:
- train: 1,966 windows, 2015-04-22 to 2023-02-09, closes $22.58 to $182.01
- validation: 435 windows, 2023-02-10 to 2024-11-01
- test: 435 windows, 2024-11-04 to 2026-07-31, closes $172.42 to $340.08

The test period trades far above anything seen in training. That matters for any model that predicts price levels.

## Architecture
The deployed network (`model/model.py: build_lstm`) is `Input(60, 1) -> LSTM(64) -> Dropout(0.2) -> Dense(1)`.
It reads the last 60 log-returns `ln(C_t / C_(t-1))`, standardized with the training mean and std, and
predicts the next one. For a multi-day forecast, each prediction is appended to the window and the model
runs again. Prices are rebuilt as `C_cutoff × exp(cumulative returns)`.

## Training & experiments
Four networks were trained with the same recipe: Adam 1e-3, MSE, batch 32, up to 40 epochs, with early
stopping and learning-rate halving on validation loss. They were compared with the naive baseline on
next-day predictions in $, over the same test days:

| variant | params | val RMSE | test RMSE | test MAE | test MAPE |
|---|---|---|---|---|---|
| naive persistence (tomorrow = today) | - | 2.583 | **4.469** | 3.020 | 1.233 % |
| **LSTM(64) on log-returns** (deployed) | 16,961 | 2.588 | 4.483 | 3.026 | 1.237 % |
| SimpleRNN(64) on scaled closes | 4,289 | 4.562 | 12.875 | 10.803 | 4.159 % |
| LSTM(64) on scaled closes | 16,961 | 5.896 | 14.482 | 10.904 | 4.173 % |
| 2x LSTM(64) on Close/Volume/RSI/EMA20/EMA50 | 51,009 | 6.544 | 21.975 | 18.107 | 6.854 % |

The deployed model is selected by validation RMSE among the univariate models, which are the ones the
Space can run recursively from closes alone. The test split never influences a choice.

- The **level models** fail because their MinMax scaler, fit on train, must extrapolate to test prices up to
  almost twice the training maximum.
- **Log-returns** stay in the training range, and the model then lands exactly on the random-walk baseline.
  It learned roughly "predict the average daily return".
- **Recursive 20-day forecasts** from 84 test cutoffs have a mean MAE of $10.67, against $10.87 for
  repeating the cutoff close. That small edge is the upward drift during a rising market, not skill.

## Results
Deployed model on the test split (2024-11-04 to 2026-07-31), next-day: **RMSE $4.483, MAE $3.026,
MAPE 1.237 %**. Naive persistence scores $4.469 / $3.020 / 1.233 %. The model does not beat naive
persistence; it ties it. All numbers come from `model/metrics.json` (retrained 2026-09-25 on CPU). They
match the earlier version of this project exactly, since the data and seed are the same.

## Deployment
- **Model repo**: https://huggingface.co/shalev396/stock-lstm-forecast (weights, `model.py`, `handler.py`,
  frozen `prices.csv`)
- **Space / API**: https://huggingface.co/spaces/shalev396/stock-lstm-forecast — `POST /gradio_api/call/predict`
  with `{"data": ["2025-06-30", 30]}` (cutoff, horizon). The result is
  `{model, history, forecast, actual, naive, mae, naiveMae}`, followed by seconds and device. The Space never
  calls yfinance, because Yahoo blocks Space IPs. It forecasts from the frozen prices, so valid cutoffs run
  from 2015-03-31 to 2026-07-31, and other dates are clamped to that range.
- **Inference Endpoint**: deploy the model repo from its page (Deploy → Inference Endpoints). `handler.py`
  takes `{"inputs": {"cutoff": "2025-06-30", "horizon": 30}}`.

## Project structure
```
stock-lstm-forecast/
├── model/     HF model repo: model.keras, model.py, handler.py, config.json, prices.csv, metrics.json, assets/, README.md
├── space/     HF Space: app.py, space_utils.py, examples/cutoffs.csv, README.md (API docs)
└── training/  notebook.ipynb (controller) + src/ (config, data_setup, model_builder, engine, export, utils)
```

## Reproduce
```bash
pip install -r requirements.txt                     # from the ml-lab root
cd stock-lstm-forecast/training
jupyter notebook notebook.ipynb                     # full run: overwrites ../model/
SMOKE_TEST=1 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp   # smoke run, a few minutes on CPU
cd ../space && python app.py                        # local Space on ../model
```

## Limitations
- It does not beat naive persistence. Treat it as time-series engineering practice, **not** a trading signal
  or financial advice.
- It was trained on one ticker (AAPL) and one period, and was not validated on other stocks.
- The Space's data is frozen at 2026-07-31. New dates need a retrain and a new `prices.csv`.
- Recursive forecasts converge to a smooth drift. There are no uncertainty bands, and no news, earnings or
  volatility information.
