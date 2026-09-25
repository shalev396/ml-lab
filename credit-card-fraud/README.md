# 💳 Credit Card Fraud Detector

> Upload a CSV of card transactions and get a fraud probability and a fraud / legit decision for every row.

<p>
  <a href="https://huggingface.co/spaces/shalev396/credit-card-fraud"><img alt="Space" src="https://img.shields.io/badge/🤗%20Space-live%20demo-blue"></a>
  <a href="https://huggingface.co/shalev396/credit-card-fraud"><img alt="Model" src="https://img.shields.io/badge/🤗%20Model-shalev396%2Fcredit--card--fraud-yellow"></a>
  <a href="https://colab.research.google.com/github/shalev396/ml-lab/blob/main/credit-card-fraud/training/notebook.ipynb"><img alt="Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>

| | |
|---|---|
| **Task** | Binary classification on a heavily imbalanced table (0.17 % fraud) |
| **Framework** | scikit-learn + imbalanced-learn (training only), TensorFlow/Keras |
| **Architecture** | Default: SMOTE + RandomForest (100 trees). Alternative: Keras MLP 30-64-32-16-1 (4,609 parameters) |
| **Dataset** | [ULB Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), 284,807 transactions (283,726 after de-duplication), stratified 60/20/20 split |
| **Result** | PR-AUC = 0.802 (test), F1 0.812 at the tuned threshold 0.57: 69 of 95 frauds caught, 6 false alarms |
| **Runs on** | Space: CPU basic · Training: CPU (a GPU barely helps), locally or on Colab |

## The problem
Card fraud is rare: in this dataset, 492 of 284,807 transactions are fraud. A model that calls everything
"legit" is 99.8 % accurate and useless. The job is to rank transactions so that the few frauds come first,
and then to pick a cut-off that catches most frauds without burying analysts in false alarms. That is why
every model here is judged by **PR-AUC** (average precision), and why the decision threshold is tuned
instead of fixed at 0.5.

## The data
[ULB Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud): card transactions made by
European cardholders over two days in September 2013, released by Worldline and the ULB Machine Learning
Group (Dal Pozzolo et al., IEEE CIDM 2015).

- `V1..V28`: the bank's original features after PCA (the only form it released), `Time`: seconds since the
  first transaction, `Amount`: value in EUR, `Class`: 1 = fraud.
- Downloaded by the notebook with `kagglehub` (anonymous), with OpenML `data_id=1597` as fallback, and cached
  in `training/data/`.
- 1,081 exact duplicate rows are dropped before splitting, so no transaction appears in two splits. That
  leaves 283,726 rows and 473 frauds.
- Stratified split: train 170,235 rows (284 frauds) · validation 56,745 (94) · test 56,746 (95).

## Architecture
Preprocessing is shared by training, the Space and the endpoint (`model/model.py: to_features`): `V1..V28`
as-is, plus `Amount` and `Time` scaled by a `RobustScaler` fit on the train rows. That gives 30 float32
features.

- **`smote_rf`** (deployed): a `RandomForestClassifier` (100 trees) fit on train rows that SMOTE balanced
  to 50/50. SMOTE only runs during training, so `model.joblib` holds just the forest.
- **`keras_mlp`** (exported alternative): Dense 64-32-16 with ReLU and dropout 0.3, sigmoid head, trained
  with balanced class weights and early stopping (`model.keras`).
- `config.json` stores the input schema, both models' tuned thresholds and the default model.
  `model.load(dir).predict(data, model="best", threshold=None)` returns per-row probabilities and decisions.

## Training & experiments
The notebook (`training/notebook.ipynb`) drives every step through functions in `training/src/`.
Twelve candidates were trained on the train split:

| Group | Variants |
|---|---|
| Baselines (imbalanced data) | logistic regression, random forest |
| NearMiss-1 undersampling | logistic regression, kNN, decision tree, random forest, calibrated RBF SVM |
| SMOTE oversampling | logistic regression, kNN, decision tree, random forest |
| Neural net | Keras MLP with class weights |

Protocol, where the test split never influences a choice:
1. Fit every candidate on the train split. Resampling sits inside an imblearn `Pipeline`, so it only
   touches the rows a model is fit on.
2. On validation, set each model's decision threshold to the value that maximises F1.
3. Select by **validation PR-AUC**. The best scikit-learn variant (`smote_rf`, 0.884) and the MLP (0.733)
   are exported, and the higher one becomes the default.
4. Score the test split once, for every variant, at its own tuned threshold.

Top of the comparison (full table and graphs in the [model card](model/README.md#experiments)):

| variant | val PR-AUC | test PR-AUC | test F1 | test precision | test recall |
|---|---|---|---|---|---|
| **smote_rf** (deployed) | **0.8839** | **0.8016** | **0.8118** | **0.9200** | **0.7263** |
| baseline_rf | 0.8769 | 0.7983 | 0.8161 | 0.8987 | 0.7474 |
| baseline_logreg | 0.8024 | 0.6939 | 0.7727 | 0.8395 | 0.7158 |
| smote_logreg | 0.7658 | 0.7179 | 0.7977 | 0.8846 | 0.7263 |
| keras_mlp | 0.7333 | 0.6838 | 0.7912 | 0.8276 | 0.7579 |

![Model comparison](model/assets/model_comparison.png)

## Results
The deployed `smote_rf` on the test split at threshold 0.57: **PR-AUC 0.802**, ROC-AUC 0.960, F1 0.812,
precision 0.920, recall 0.726. That is 69 of 95 frauds caught and 6 false alarms among 56,651 legit
transactions (`model/metrics.json`, run of 2026-09-25).

- `smote_rf` and `baseline_rf` are statistically tied: a validation PR-AUC of 0.884 vs 0.877 is within
  noise for 94 validation frauds, and on test they score 0.802 vs 0.798.
- ROC-AUC hides the differences: `smote_logreg` has a test ROC-AUC of 0.958, close to `smote_rf`, but a
  PR-AUC of only 0.718.
- NearMiss undersampling hurts every model: it throws away almost all legit rows, and precision collapses.
- The original version of this project chose its model by *test* ROC-AUC at a fixed 0.5 threshold and
  shipped SMOTE + logistic regression with a precision of 0.059 (numbers from a documented run of the old
  pipeline, on a different split). Selecting on validation PR-AUC with a tuned threshold fixes that.

![Precision-recall curves](model/assets/pr_curve.png)

## Deployment
- **Model repo**: https://huggingface.co/shalev396/credit-card-fraud (weights, `model.py`, `handler.py`)
- **Space / API**: https://huggingface.co/spaces/shalev396/credit-card-fraud — `POST /gradio_api/call/predict`
  with `(csv_file, model, threshold)`, returns `[result, seconds, device]` (see [space/README.md](space/README.md))
- **Inference Endpoint**: deploy the model repo from its page (Deploy → Inference Endpoints). The body is
  `{"inputs": <CSV text | list of row dicts | {"rows": [...]}>, "parameters": {"model": ..., "threshold": ...}}`.

## Project structure
```
credit-card-fraud/
├── model/      HF model repo: model.py, handler.py, requirements.txt, scaler.joblib, model.joblib,
│               model.keras, config.json, metrics.json, assets/*.png, README.md (model card)
├── space/      HF Space: app.py, space_utils.py, requirements.txt, examples/*.csv, README.md (API docs)
└── training/   notebook.ipynb (the controller), src/ (config, data_setup, model_builder, engine,
                export, utils), README.md
```

## Reproduce
```bash
pip install -r requirements.txt                      # from the ml-lab root
cd credit-card-fraud/training
jupyter notebook notebook.ipynb                      # run all cells; SMOKE_TEST=0 overwrites ../model
SMOKE_TEST=1 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp   # 0.5-2 min check
cd ../space && python app.py                         # the Space locally, served from ../model
```
Or open the notebook in [Colab](https://colab.research.google.com/github/shalev396/ml-lab/blob/main/credit-card-fraud/training/notebook.ipynb).
The last full run took 15.6 min wall time on a desktop CPU limited to 6 threads and shared with other jobs
(fitting the 12 candidates took 383 s of that). An earlier 20-thread run of the old `train.py` pipeline took
about 3 min end to end.

## Limitations
- Works only on this dataset's feature space: `V1..V28` come from an unpublished PCA, so transactions from
  any other source cannot be scored.
- Two days of European card traffic from 2013. Fraud patterns drift, and nothing was validated on later data.
- 95 test frauds: every missed fraud moves recall by about 1 point, so the metrics have wide error bars.
- `fraud_probability` is a ranking score, not a calibrated probability (SMOTE-balanced forest votes, a
  class-weighted MLP).
- The max-F1 threshold is one operating point. Choose your own when a missed fraud and a false alarm cost
  different amounts.
- Educational portfolio project. Not for real payment decisions.
