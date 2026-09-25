# 🛗 Elevator Predictive Maintenance

> From the last hour of elevator sensor data, predict whether the elevator will be failing 10 minutes from now.
> **Trained on synthetic data.**

<p>
  <a href="https://huggingface.co/spaces/shalev396/elevator-maintenance"><img alt="Space" src="https://img.shields.io/badge/🤗%20Space-live%20demo-blue"></a>
  <a href="https://huggingface.co/shalev396/elevator-maintenance"><img alt="Model" src="https://img.shields.io/badge/🤗%20Model-shalev396%2Felevator--maintenance-yellow"></a>
  <a href="https://colab.research.google.com/github/shalev396/ml-lab/blob/main/elevator-maintenance/training/notebook.ipynb"><img alt="Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>

> ⚠️ **The data is SYNTHETIC.** The public Kaggle dataset usually linked to this task
> (`shivamb/elevator-predictive-maintenance-dataset`) has no failure label, so this project generates a seeded,
> spec-matched month of sensor data instead. Every number below measures how well the models learn that
> generator. None of it says how they would do on a real elevator.

| | |
|---|---|
| **Task** | Time-series binary classification: failure state 10 min after a 60-min window of 11 sensors |
| **Framework** | scikit-learn (deployed) · TensorFlow/Keras (comparison models) |
| **Architecture** | RandomForest, 300 trees, on 63 engineered window features incl. FFT (10,408 tree nodes) |
| **Dataset** | **Synthetic** month of minutely readings: 44,640 rows, 11 sensors, 5 degradation episodes; time-based 70/15/15 → 6,236 / 1,326 / 1,326 windows |
| **Result** | F1 = 0.988 (test, synthetic); PR-AUC 0.989 |
| **Runs on** | Space: CPU basic · Training: CPU, GPU or Colab (a few minutes on CPU) |

## The problem
Predictive maintenance: service an elevator before it breaks instead of after. The project follows the
ProjectPro brief "time series project for elevator predictive maintenance with IoT sensor data": take one
month of per-minute readings from 11 sensors, cut them into sliding windows, and train a classifier that says
from the **last 60 minutes** whether the elevator will be in a failure state **10 minutes later**. The brief
calls for windowing, lag features, rolling statistics, frequency-domain features and low-pass filtering.

## The data
The brief's dataset has a binary `Status` target. The Kaggle set usually linked to it has 112,001 rows of
`ID, revolutions, humidity, vibration, x1..x5`: a vibration-regression table with **no failure label**. It
cannot train a failure classifier, so it was rejected.

Instead, `training/src/data_setup.py` generates a **seeded synthetic month** (seed 42, cached to
`training/data/elevator.csv`):
- 31 days x 1,440 minutes = **44,640 rows** of 11 sensors: temperature, humidity, vibration_rms,
  vibration_peak, motor_current, motor_rpm, door_cycles, load_kg, acoustic_db, oil_level, power_kw.
- A daily usage profile (morning, lunch and evening peaks, quiet nights, quieter weekends) drives every sensor,
  plus noise and rare vibration spikes. The oil level decays slowly.
- **Five degradation episodes**: 6-14 h of linear drift (more vibration, current, heat, noise and power, lower
  rpm, faster oil loss), then a 45-120 min failure, then a maintenance reset with fresh oil.
  `Status = 1` during degradation + failure, which is **8.5 %** of minutes.
- The episodes are placed so that train holds three, and validation and test hold one each.

The generator reproduces the cached file exactly (checked: identical to the last digit).

## Architecture
- **Windows**: the rows are split by time 70/15/15. Inside each segment, a window covers minutes `[t-59, t]`
  and its label is `Status` at `t+10`. Window and label must lie in one segment, so nothing crosses a split
  boundary. Stride 5 minutes.
- **Features** (63, in [`model/model.py`](model/model.py) `window_features`, shared by training and serving):
  mean, std, min, max and slope of every sensor; the top-5 FFT magnitudes of `vibration_rms` and its dominant
  frequency bin; hour and day of week.
- **Deployed model**: `RandomForestClassifier(n_estimators=300, class_weight="balanced")`. Maintenance is
  flagged when P(failure) >= 0.377, the max-F1 threshold on validation.
- **Comparison models** (raw windows, standardised with a train-fit scaler): a Keras 1D-CNN (52,993 weights)
  and a Keras LSTM (23,681 weights), trained with class weights and early stopping.

## Training & experiments
The notebook fits all three models, tunes each one's threshold for max F1 on the validation days, picks the
model by **validation F1**, and only then scores the test days. Local CPU run of 2026-09-25 (`model/metrics.json`):

| experiment | size | val F1 | threshold | test F1 | test precision | test recall | test PR-AUC | train time |
|---|---|---|---|---|---|---|---|---|
| **RandomForest (deployed)** | 10,408 tree nodes | **0.989** | 0.377 | **0.988** | **0.988** | **0.988** | **0.989** | 2.9 s |
| CNN-1D | 52,993 weights | 0.912 | 0.189 | 0.958 | 0.976 | 0.941 | 0.938 | 64.5 s |
| LSTM | 23,681 weights | 0.938 | 0.166 | 0.933 | 0.968 | 0.900 | 0.920 | 146.1 s |

The engineered-feature forest wins clearly. Rolling means, slopes and maxima pick up a slow drift almost
directly, and this generator's degradation is exactly that. Both nets peaked in their first epoch on validation
and then overfit (early stopping restored the epoch-1 weights).

## Results
On the 1,326 test windows (the last 4.65 days, with one unseen episode), the deployed forest catches **168 of
170** failure windows with **2 false alarms** among 1,156 healthy windows: F1 0.988, PR-AUC 0.989,
ROC-AUC 0.993. On the three Space examples, cut from the test days, it gives P(failure) = 0.003 (healthy),
0.92 (halfway into the degradation) and 0.81 (during the failure). Plots (PR curves, confusion matrix, the
month timeline, feature importance, Keras curves) are in [`model/assets/`](model/assets) and on the model card.

**Fix from the earlier version**: the old app served the 1D-CNN even though the RandomForest scored best. The
Space now serves the forest, which is also why it needs no TensorFlow.

## Deployment
- **Model repo**: https://huggingface.co/shalev396/elevator-maintenance (weights, `model.py`, `handler.py`)
- **Space / API**: https://huggingface.co/spaces/shalev396/elevator-maintenance — `POST /gradio_api/call/predict`
  with a CSV file → `[label {failure, healthy}, seconds, device]`. The threshold slider and the sensor plot
  are UI-only.
- **Inference Endpoint**: deploy the model repo from its page (Deploy → Inference Endpoints); send
  `{"inputs": "<CSV text>"}`.

## Project structure
```
elevator-maintenance/
├── model/      HF model repo: model.py, handler.py, rf.joblib, config.json, metrics.json, assets/, README.md
├── space/      HF Space: app.py, space_utils.py, examples/{healthy,degrading,failing}.csv, README.md
└── training/   notebook.ipynb + src/ (config, data_setup, model_builder, engine, export, utils)
```

## Reproduce
```bash
pip install -r requirements.txt                       # from the ml-lab root
cd elevator-maintenance/training
jupyter notebook notebook.ipynb                       # full run: a few minutes on CPU, writes ../model
SMOKE_TEST=1 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp   # quick check
cd ../space && python app.py                          # local Space, uses ../model
```

## Limitations
- **Synthetic data**: the generator defines what degradation looks like (smooth, simultaneous linear drifts).
  Real failures are noisier, rarer and more varied. This is a demo of the method, not a maintenance tool.
- Validation and test hold one episode each, so the scores rest on 178 and 170 overlapping failure windows.
  A different episode could move them by several points.
- The label covers the whole 6-14 h drift, so the model recognises "inside a degradation episode" more than
  it gives an early warning hours ahead.
- Hour and day-of-week features encode the generator's schedule; uploads without timestamps are scored as a
  weekday noon.
