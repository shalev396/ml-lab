# 📊 EU Stock Market Forecasting

> Pick any date between 1991 and 1998 and a European stock index: a classical time-series model that
> only sees the history up to that date forecasts the next weeks, next to what really happened and the naive baseline.

<p>
  <a href="https://huggingface.co/spaces/shalev396/eu-stock-forecasting"><img alt="Space" src="https://img.shields.io/badge/🤗%20Space-live%20demo-blue"></a>
  <a href="https://huggingface.co/shalev396/eu-stock-forecasting"><img alt="Model" src="https://img.shields.io/badge/🤗%20Model-shalev396%2Feu--stock--forecasting-yellow"></a>
  <a href="https://colab.research.google.com/github/shalev396/ml-lab/blob/main/eu-stock-forecasting/training/notebook.ipynb"><img alt="Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>

| | |
|---|---|
| **Task** | Time-series forecasting of daily stock-index closes (one-step and multi-step) |
| **Framework** | statsmodels (deployed) · TensorFlow/Keras 3 (compared) |
| **Architecture** | Holt-Winters (4 parameters) on DAX, SMI and FTSE, VAR(10) (164 parameters) on CAC: per index, the fewest-parameter model within 2 % of the best validation MAPE among ARIMA, Holt-Winters and VAR; MLP and LSTM compared |
| **Dataset** | [EuStockMarkets](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/EuStockMarkets.html) (R `datasets`): DAX, SMI, CAC, FTSE, 1,860 business days 1991-1998, split 1,660 fit / 100 validation / 100 test |
| **Result** | DAX one-step MAPE = 1.047 % (test, Holt-Winters), vs 1.048 % for naive persistence |
| **Runs on** | Space: CPU basic · Training: CPU (GPU/Colab optional, only the Keras baselines use it) |

## The problem
This is a Python port of a classic R time-series course (ProjectPro's forecasting project on `EuStockMarkets`).
The course walks through the full classical workflow: look at the series, test stationarity (ADF), pick model
orders by AIC, fit ARIMA / exponential smoothing / VAR, and score one-step-ahead forecasts with R's `accuracy()`
table. It then compares the classical models with small neural nets. The goal here is to do that workflow
cleanly (no look-ahead, selection on validation, test reported once) and to serve it as an interactive API: "forecast
the DAX from 1997-10-01 for 30 business days, and show me how that compared with reality."

## The data
`EuStockMarkets` holds the daily closing prices of four indices: **DAX** (Germany), **SMI** (Switzerland), **CAC**
(France) and **FTSE** (UK), 1,860 rows each. It is loaded with `statsmodels.datasets.get_rdataset("EuStockMarkets")`,
with the Rdatasets CSV mirror as a fallback, and cached in `training/data/`.

**The dates are synthetic.** The R object is a `ts` with no calendar, only "start 1991.496, frequency 260". The
data therefore get a business-day index starting on **1991-07-01** (Mon-Fri, holidays not skipped), so the last row is
1998-08-14. The order and spacing of the data are real; the calendar dates are approximate. The model, the Space
and this README all use that calendar.

All four indices roughly triple over the period, and the last year (the validation and test windows) is the
steepest part of the 1990s bull market, followed by the August 1998 sell-off.

![The four series and the split](model/assets/series.png)

## Architecture
Five forecasters, each fit on the fit window only:

| Model | What it is | Parameters (DAX) |
|---|---|---|
| ARIMA(p,1,q) | (p, q) ≤ 3 grid by AIC; d = 1 from the ADF test. DAX (2,1,3), SMI (2,1,3), CAC (3,1,3), FTSE (3,1,2) | 6 |
| **Holt-Winters** | exponential smoothing, additive trend, no seasonality (deployed: DAX, SMI, FTSE) | 4 |
| **VAR(10)** | vector autoregression on all four indices together; lag 10 by AIC (max 12) (deployed: CAC) | 164 |
| Keras MLP | 20 lagged closes -> Dense 64 -> Dense 32 -> next close (MinMax scaled) | 3,457 |
| Keras LSTM | the same 20-day window as a sequence -> LSTM(64) -> next close | 16,961 |
| Naive persistence | tomorrow = today (reference) | 0 |

The deployed "model" is deliberately small. Classical fits take a fraction of a second, so the model repo ships
the chosen method + orders per index (`config.json`) and the data (`eustocks.csv`, ~82 KB), not pickled fits.
Every request fits on the history up to the cutoff: VAR and Holt-Winters are refit by OLS / MLE, and ARIMA keeps
its trained coefficients and re-runs the Kalman filter (MLE refits of these ARIMA orders take 1-4 s and often fail to
converge). All of this lives in [`model/model.py`](model/model.py), which training, the Space and the Inference
Endpoint share.

## Training & experiments
- **Split** (per index, chronological, nothing shuffled): fit = 1991-07-01 .. 1997-11-07 (1,660 days),
  validation = the next 100 days, test = the last 100 days (1998-03-30 .. 1998-08-14).
- **Stationarity**: the ADF test cannot reject a unit root in any index level (p ≥ 0.76), while first differences
  are clearly stationary (p < 0.001), so d = 1.
- **One-step-ahead walk-forward** over validation + test with frozen parameters: each forecast sees the real closes
  up to the day before. Metrics are the R `accuracy()` table: ME, RMSE, MAE, MPE, MAPE, MASE.
- **Selection** (a rule fixed before the test window is looked at; test metrics are never used): per index, rank
  ARIMA, Holt-Winters and VAR by **validation** MAPE; among all models within **2 % (relative)** of the best, deploy
  the one with the **fewest fitted parameters** (ties go to the lower validation MAPE). VAR(10) has the best
  validation MAPE everywhere, but Holt-Winters (4 parameters vs 164) is within 2 % of it on DAX (+1.6 %), SMI
  (+1.5 %) and FTSE (+0.4 %); on CAC it is 6.3 % behind, so VAR(10) stays. (The original course picked the best
  model on the test window itself; here the test window is only reported.)
- **Multi-step backtest**: exactly what the Space serves. From 9 origins in the test window (one every 10 days),
  `model.forecast` sees only the history up to the origin and forecasts 20 business days ahead.

The full run took about 3.5 min of wall time on a shared 20-thread desktop CPU (168 s of fitting, most of it the
ARIMA grids and the Keras nets). See [`training/`](training/).

## Results
One-step-ahead test MAPE (%) for every variant on every index, deployed in bold (run of 2026-09-25, `model/metrics.json`):

| Model | DAX | SMI | CAC | FTSE |
|---|---|---|---|---|
| ARIMA | 1.054 | 0.934 | 0.989 | 0.795 |
| Holt-Winters | **1.047** | **0.943** | 0.969 | **0.788** |
| VAR(10) | 1.061 | 0.951 | **1.012** | 0.812 |
| Keras MLP | 2.657 | 2.954 | 1.596 | 2.265 |
| Keras LSTM | 2.622 | 2.246 | 1.282 | 2.475 |
| Naive persistence | 1.048 | 0.943 | 0.968 | 0.788 |

DAX in detail: the deployed Holt-Winters scores RMSE 72.9 / MAE 57.8 index points, MAPE 1.047 %, MASE 3.20. On the
20-day backtest its MAE is 174.3 points (MAPE 3.12 %), against 195.7 for naive.

![All variants](model/assets/model_comparison.png)
![DAX multi-step backtest](model/assets/forecast_backtest.png)

What this shows, honestly:
- **Daily index closes are close to a random walk.** ARIMA, Holt-Winters and VAR all land within 0.05 MAPE points of
  naive persistence on every index (0.02 on DAX). The fitted Holt-Winters has smoothing level ≈ 1 and trend smoothing ≈ 0, which *is* a random walk
  with drift.
- **Selection on 100 validation days is noise-level.** VAR had the lowest validation MAPE on all four indices, but it
  was the weakest classical model on every test window. That is the case for the parsimony rule: when the validation
  gap is under 2 %, the 4-parameter Holt-Winters is deployed instead of the 164-parameter VAR. Holt-Winters also happens
  to be the best DAX model on the test window (1.047 %, matching the original course's 1.046 % run), but the rule never
  looks at the test set; on CAC it keeps VAR(10), which is the worst classical model there on test (1.012 % vs 0.968 %
  for naive).
- **The neural nets lose clearly** (1.3-3.0 %). Their scaler is fit on the fit window, and the 1998 levels lie far above
  it (the DAX fit window tops out at about 4,460; the test window reaches 6,186). The nets cannot extrapolate. The original
  course run, whose scaler and early stopping also saw the 100 days that are validation here, got 1.3-1.5 % on DAX.
- Over 20 days the errors grow to ~3 % MAPE. The deployed models beat naive on DAX and SMI but not on CAC or FTSE in
  this short window.

The per-variant tables, the one-step overlay and the NN training curves are in the
[model card](model/README.md).

## Deployment
- **Model repo**: https://huggingface.co/shalev396/eu-stock-forecasting (`config.json`, `eustocks.csv`, `model.py`, `handler.py`)
- **Space / API**: https://huggingface.co/spaces/shalev396/eu-stock-forecasting — `POST /gradio_api/call/predict`
  with `{"data": ["1997-10-01", 30, "DAX"]}` returns `[{model, history, forecast, actual, naive, mae, naiveMae}, seconds, device]`
  (details in [space/README.md](space/README.md)).
- **Inference Endpoint**: deploy the model repo from its page (Deploy → Inference Endpoints, CPU). Body:
  `{"inputs": {"cutoff": "1997-10-01", "horizon": 30, "index": "DAX"}}`.

## Project structure
```
eu-stock-forecasting/
├── model/     HF model repo: model.py (forecasters + Predictor), handler.py, config.json, eustocks.csv,
│              metrics.json, assets/*.png, README.md (model card)
├── space/     HF Space: app.py (Gradio: /predict + bring-your-own-CSV tab), space_utils.py,
│              examples/ (requests.json, dax_last_300_days.csv), README.md (API docs)
└── training/  notebook.ipynb (the controller) + src/ (config, data_setup, model_builder, engine, export, utils)
```

## Reproduce
```bash
git clone --recurse-submodules https://github.com/shalev396/ml-lab && cd ml-lab
pip install -r requirements.txt
cd eu-stock-forecasting/training
SMOKE_TEST=1 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp   # smoke -> outputs/smoke/model
SMOKE_TEST=0 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp   # full  -> ../model
cd ../space && python app.py                                                               # local Space on ../model
```
Or open the notebook in [Colab](https://colab.research.google.com/github/shalev396/ml-lab/blob/main/eu-stock-forecasting/training/notebook.ipynb).

## Limitations
- This is not a trading model. Nothing here meaningfully beats "tomorrow = today".
- The calendar is synthetic (weekdays from 1991-07-01, no holidays); treat dates as approximate.
- The data are historical only (1991-1998). A cutoff after 1998-08-14 simply forecasts from the last row.
- Point forecasts only, no prediction intervals.
- The choice between the classical models rests on small validation differences, and would likely change with another window.
