# 🏠 California Housing Price Predictor

> Describe a California census district from 1990 and get its predicted median house value in dollars.

<p>
  <a href="https://huggingface.co/spaces/shalev396/california-housing"><img alt="Space" src="https://img.shields.io/badge/🤗%20Space-live%20demo-blue"></a>
  <a href="https://huggingface.co/shalev396/california-housing"><img alt="Model" src="https://img.shields.io/badge/🤗%20Model-shalev396%2Fcalifornia--housing-yellow"></a>
  <a href="https://colab.research.google.com/github/shalev396/ml-lab/blob/main/california-housing/training/notebook.ipynb"><img alt="Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>

| | |
|---|---|
| **Task** | Tabular regression: district median house value from 8 census features |
| **Framework** | XGBoost (default) + PyTorch (MLP alternative), scikit-learn baselines |
| **Architecture** | XGBoost, 1,085 hist trees of depth 6 · MLP 11-256-128-64-1 (44,289 parameters) |
| **Dataset** | [California Housing](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) (1990 census, 20,640 districts), 14,035 train / 2,477 val / 4,128 test |
| **Result** | R² = 0.853, RMSE = $43,966, MAE = $28,540 (XGBoost, test) |
| **Runs on** | Space: CPU basic · Training: CPU, GPU or Colab |

## The problem
Given the aggregate statistics of a California census block group (a "district" of typically 600 to 3,000
people), predict the median value of its houses. It is a classic small tabular regression benchmark, and this
project uses it to compare a linear baseline, two tree ensembles and a neural network on equal terms.

## The data
sklearn's `fetch_california_housing` (StatLib copy of the 1990 US census): 20,640 districts with 8 features:
`MedInc` (median income in $10k), `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`,
`Longitude`. The target `MedHouseVal` is in $100k and capped at 5.00001 ($500,001). Three ratios of the classic
recipe are added (`rooms_per_household`, `bedrooms_per_room`, `population_per_household`). Because sklearn's
version is already per household, two of them equal `AveRooms` and `AveOccup`; only `bedrooms_per_room` is new.
Split: 80/20 train/test (seed 42), and 15 % of train as validation. A converted `housing.csv` from
*Hands-On Machine Learning* is the documented fallback source.

## Architecture
- **XGBoost** (default): `XGBRegressor`, hist trees, depth 6, learning rate 0.05, subsample / column subsample
  0.8, early stopping on validation RMSE (best round 1,085 of at most 2,000). It uses the raw features and is saved
  natively as `xgb.ubj`.
- **PyTorch MLP**: `HousingMLP` in [`model/model.py`](model/model.py), a `PyTorchModelHubMixin` module with
  [Linear, ReLU, Dropout 0.15] x 3 (256, 128, 64 units) and a linear output: 44,289 parameters. Its inputs are
  standardized by a `StandardScaler` fit on the train split.
- `model.py` holds the feature engineering and the `Predictor`, so training, the Space and the Inference Endpoint
  all run the same code.

## Training & experiments
Four experiments, all fit on the train split: LinearRegression, RandomForest (200 trees), XGBoost, and the MLP
(AdamW lr 1e-3, weight decay 1e-4, batch 256, MSE, early stopping with patience 10: best epoch 81). The default
model is the exported one with the lowest validation RMSE; the test split is only scored afterwards.

| model | val RMSE | test RMSE | test MAE | test R² | test RMSE ($) |
|---|---|---|---|---|---|
| **XGBoost** (default) | **0.4671** | **0.4397** | **0.2854** | **0.8525** | **$43,966** |
| PyTorch MLP | 0.5269 | 0.5186 | 0.3487 | 0.7947 | $51,865 |
| RandomForest | 0.5299 | 0.5096 | 0.3327 | 0.8018 | $50,964 |
| LinearRegression | 0.7241 | 0.7277 | 0.5252 | 0.5958 | $72,774 |

RMSE / MAE in $100k unless marked. Numbers from the retrain of 2026-09-25 (`model/metrics.json`).

## Results
XGBoost is the clear winner: its test RMSE is about $7k lower than the RandomForest's and $8k lower than the MLP's,
and it explains 85 % of the variance. Median income is by far its most important feature (42 % of total gain),
followed by location. The earlier version of this project, with the same split and XGBoost settings, reported a
test RMSE of 0.4423 and R² of 0.851, so the retrain reproduces it. Graphs (model comparison, training curves,
predicted vs actual, feature importance) are in [`model/assets/`](model/assets) and on the model card.

## Deployment
- **Model repo**: https://huggingface.co/shalev396/california-housing (weights, `model.py`, `handler.py`)
- **Space / API**: https://huggingface.co/spaces/shalev396/california-housing — `POST /gradio_api/call/predict`
  with the 8 features + model name → `[{"price_usd", "model"}, seconds, device]`
- **Inference Endpoint**: deploy the model repo from its page (Deploy → Inference Endpoints); `handler.py`
  accepts `{"inputs": {<8 features>}, "parameters": {"model": "best"}}`

## Project structure
```
california-housing/
├── README.md             this write-up
├── model/                Hugging Face model repo: model.py, handler.py, xgb.ubj, model.safetensors,
│                         config.json, scaler.joblib, metrics.json, assets/, model card
├── space/                Hugging Face Space: app.py (Gradio), space_utils.py, examples/districts.csv
└── training/             notebook.ipynb (controller) + src/ (config, data, models, engine, export, utils)
```

## Reproduce
```bash
git clone --recurse-submodules https://github.com/shalev396/ml-lab && cd ml-lab
pip install -r requirements.txt
cd california-housing/training
SMOKE_TEST=1 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp   # ~1 min sanity run
# full run: open notebook.ipynb and run all (SMOKE_TEST=0 overwrites ../model)
cd ../space && python app.py                                                              # local Space
```
The full run needs only a CPU. It took about 37 minutes on a shared desktop CPU that other training jobs were
loading at the same time; an idle machine is much faster.

## Limitations
- The prices are 1990 district medians in 1990 dollars. They are not current prices and not single-house values.
- The target is capped at $500,001, so capped districts are often underestimated.
- The inputs are block-group averages; inputs far outside the training data (for example coordinates outside
  California) still return a number, but it means nothing.
- Two of the three engineered ratios duplicate raw columns (kept to mirror the classic recipe).
