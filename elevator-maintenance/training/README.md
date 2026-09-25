# Training: Elevator Predictive Maintenance

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shalev396/ml-lab/blob/main/elevator-maintenance/training/notebook.ipynb)

`notebook.ipynb` is the controller. It builds the model in [`../model`](../model) step by step, calling the
reusable code in `src/`. The features and the served predictor live in [`../model/model.py`](../model/model.py),
so training, the Space and the Inference Endpoint all run the same code.

> ⚠️ **The data is synthetic.** `src/data_setup.py` generates a seeded month of minutely readings from 11
> elevator sensors, matched to the ProjectPro brief. The public Kaggle set usually linked to this brief,
> `shivamb/elevator-predictive-maintenance-dataset` (112,001 rows of `ID, revolutions, humidity, vibration,
> x1..x5`), has no failure/status label, so it cannot train a failure classifier. All metrics describe how
> well the models learn the generator.

## Notebook sections
1. **Setup**: 1a finds the project files (clones the repo on a fresh runtime), 1b installs the root
   `requirements.txt`, 1c picks the TensorFlow `device` (GPU if present, else CPU).
2. **Config**: `Config()` from `src/config.py`, the `SMOKE_TEST` switch and `PUSH_TO_HUB`.
3. **Data**: generate or load the cached month (`data/elevator.csv`, 44,640 rows). EDA plots: the month with
   its 5 episodes, the daily usage profile, and a Butterworth low-pass filter on vibration. Then the time-based
   70/15/15 split, sliding windows (60 min, label 10 min after the window, stride 5; no window crosses a split
   boundary), and 63 features per window from `model.window_features`.
4. **Load model**: build the RandomForest (constructor in `model.py`), the Keras 1D-CNN and the Keras LSTM,
   and print their sizes.
5. **Training**: fit the forest on the train features. Fit both nets on the standardised raw windows (class
   weights, early stopping on validation PR-AUC). Plot the loss and PR-AUC curves.
6. **Evaluation**: tune each model's threshold for max F1 on validation, choose the model by validation F1,
   score the test segment once. Comparison table, PR curves, confusion matrix, month timeline, feature
   importance.
7. **Inference**: save `rf.joblib` + `config.json`, cut the three example windows from the test days into
   `../space/examples/`, and score them with `model.load(dir).predict(csv)`, as the Space does.
8. **Export**: check that the saved model reproduces the in-memory predictions, then write `metrics.json`,
   `assets/*.png` and the refreshed model card. `PUSH_TO_HUB = True` uploads `../model` (token from
   `hf auth login` or an `HF_TOKEN` secret / environment variable).

## Run it
- **Colab**: click the badge. The setup cells clone the repo and install the requirements.
- **Locally**: `pip install -r ../../requirements.txt`, then open `notebook.ipynb` from this folder.
- **Headless smoke test** (every 20th window, 50 trees, tiny nets, 1 epoch; writes only to `outputs/smoke/`):
  `SMOKE_TEST=1 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp`

| Hardware | Full run |
|---|---|
| Desktop CPU | a few minutes (the forest fits in seconds; most of the time is the two Keras nets) |
| GPU / Colab | about the same: the job is small and not GPU-bound |

Timings from this machine are in `../model/metrics.json` (`train_time_s`, per-model `train_time_s`).

## Outputs
- Full run: `../model/` (`rf.joblib`, `config.json`, `metrics.json`, `assets/`, `README.md`) and
  `../space/examples/*.csv`.
- Smoke run: `outputs/smoke/model/` and `outputs/smoke/examples/`. `../model` is never touched.
- The Keras nets are compared but not exported: the forest wins on validation, and serving it keeps the Space
  free of TensorFlow.

## `src/`
| File | What it does |
|---|---|
| `__init__.py` | puts `../model` on `sys.path` so `import model as M` works |
| `config.py` | `Config` dataclass: every hyperparameter + smoke-mode shrinking + output folders |
| `data_setup.py` | synthetic generator, cache, episodes, windows + split, features, Space examples |
| `model_builder.py` | RandomForest (from `model.py`), Keras 1D-CNN, Keras LSTM |
| `engine.py` | fitting, class weights, threshold tuning, metrics, model selection |
| `export.py` | `save_model` (weights + config), `export` (metrics, plots, card), all plots |
| `utils.py` | shared ml-lab helpers (identical in every project) |
