# 🛒 Purchase Propensity + RFM

> Predicts whether a website visitor will place an order from 23 session flags, and segments shop customers by
> recency, frequency and spend.

<p>
  <a href="https://huggingface.co/spaces/shalev396/purchase-propensity"><img alt="Space" src="https://img.shields.io/badge/🤗%20Space-live%20demo-blue"></a>
  <a href="https://huggingface.co/shalev396/purchase-propensity"><img alt="Model" src="https://img.shields.io/badge/🤗%20Model-shalev396%2Fpurchase--propensity-yellow"></a>
  <a href="https://colab.research.google.com/github/shalev396/ml-lab/blob/main/purchase-propensity/training/notebook.ipynb"><img alt="Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>

| | |
|---|---|
| **Task** | Binary classification (will the session end in an order?) + RFM customer segmentation |
| **Framework** | scikit-learn + PyTorch |
| **Architecture** | LogisticRegression, balanced (24 parameters, deployed) vs PyTorch MLP 23 → 64 → 32 → 1 (3,649 parameters) |
| **Dataset** | [Kaggle customer propensity to purchase](https://www.kaggle.com/datasets/benpowis/customer-propensity-to-purchase-data): 455,401 sessions, 4.19 % order, 68/12/20 stratified split · [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail): 397,884 cleaned invoice lines, 4,338 customers |
| **Result** | ROC-AUC = 0.9974 · PR-AUC = 0.8995 · F1 = 0.9245 at the tuned threshold (test, 91,081 sessions) |
| **Runs on** | Space: CPU basic · Training: CPU, GPU or Colab (about 2.5 min on a desktop CPU) |

## The problem
An online shop wants to know, while a visitor is browsing, how likely the session is to end in an order. Then it
can act on it: prioritize support, show an offer, or measure how a change to the funnel shifts purchase intent.
Only about 1 in 24 sessions orders, so the classes are heavily imbalanced, and accuracy tells you nothing: a model
that always says "no order" is 95.8 % accurate. The second half of the project is the classic CRM view of *existing*
customers: an **RFM** analysis groups them by how recently they bought (R), how often (F) and how much they spent (M).

## The data
- **Sessions:** each row is one visitor-day on an online retailer's website (93 % flagged `loc_uk`). It has 23 binary flags (basket clicks
  and adds, sort / image / size interactions, promo and account clicks, `checked_delivery_detail`, `sign_in`,
  `saw_checkout`, device type, `returning_user`, `loc_uk`) and the label `ordered`. There are 455,401 sessions and
  19,093 orders. The flags take only 9,086 distinct combinations. They are split 309,672 / 54,648 / 91,081
  (train / validation / test), stratified with seed 42.
- **Retail invoices:** 541,909 invoice lines from a UK online gift shop (Dec 2010 – Dec 2011). Cleaning removes lines
  without a customer, cancellations and non-positive quantities or prices, which leaves 397,884 lines, 4,338
  customers and GBP 8.91M of revenue.

## Architecture
- **Logistic regression:** `LogisticRegression(class_weight="balanced")` on standardized flags: 23 weights +
  1 intercept.
- **PyTorch MLP** (`model.PropensityMLP`, a `PyTorchModelHubMixin`): Linear(23, 64) → ReLU → Dropout(0.2) →
  Linear(64, 32) → ReLU → Dropout(0.2) → Linear(32, 1), trained with `BCEWithLogitsLoss(pos_weight = 22.85)`.
- **RFM:** per customer, days since the last order, number of invoices and total spend. Each gets a rank-based
  quintile score of 1–5, and the (R, F) pair maps onto the standard segment grid: Champions, Loyal Customers,
  Potential Loyalists, At Risk, Hibernating and 5 more.

`model/model.py` holds the feature schema, the MLP, the input encoding and a `Predictor`:
- `predict(features, model="best")` → `{"buy": p, "no_buy": 1 - p}`
- `explain`, `predict_batch` (CSV / DataFrame)
- `segments`, `rfm_segment(recency, frequency, monetary)`

## Training & experiments
Both models train on the same standardized train split. The MLP uses AdamW (lr 1e-3), batch 4,096 and early stopping
on validation ROC-AUC. It stopped after 18 of 30 epochs and kept epoch 13. Each model's decision threshold is tuned
for max F1 on validation. The **default model is the one with the higher validation PR-AUC**, and the test split is
scored once, at the end.

| experiment | params | val PR-AUC | threshold | test ROC-AUC | test PR-AUC | test F1 | test precision | test recall |
|---|---|---|---|---|---|---|---|---|
| **logistic_regression** (deployed) | 24 | **0.8999** | 0.949 | **0.9974** | 0.8995 | **0.9245** | 0.8689 | **0.9877** |
| pytorch_mlp | 3,649 | 0.8968 | 0.965 | 0.9973 | **0.8997** | 0.9237 | **0.8703** | 0.9840 |

The two are tied within noise. The 24-parameter linear model matches the MLP because the signal is nearly additive,
and it is carried by two late-funnel flags: every order in the data saw the checkout, and 66 % of sessions that
checked the delivery details ordered. Plots (comparison, training curves, ROC/PR, confusion matrix, feature
importance, RFM segments) are in [`model/assets/`](model/assets/).

## Results
Deployed logistic regression on the 91,081 test sessions, threshold 0.949:
- It finds 3,772 of 3,819 buyers (recall 98.8 %).
- 86.9 % of the sessions it flags order (569 false alarms among 87,262 non-buyers).
- ROC-AUC is 0.9974 and PR-AUC 0.8995.

The original version of this project picked its model on the test ROC-AUC. The retrain picks it on validation
PR-AUC instead, and the logistic regression reproduces the original run's numbers exactly.

RFM: Champions are 14.6 % of the customers and bring 48.7 % of the revenue. Hibernating customers are the largest
group (24.8 %) but bring only 5.9 %. All numbers come from [`model/metrics.json`](model/metrics.json) (retrained
2026-09-25 on a local CPU).

## Deployment
- **Model repo**: https://huggingface.co/shalev396/purchase-propensity (weights, `model.py`, `handler.py`)
- **Space / API**: https://huggingface.co/spaces/shalev396/purchase-propensity — `POST /gradio_api/call/predict`
  with `{"data": [["saw_checkout", "sign_in"], "best"]}` → `[label, seconds, device]`
- **Inference Endpoint**: deploy the model repo from its page (Deploy → Inference Endpoints). The handler accepts
  `{"inputs": [...active flags...], "parameters": {"model": "best"}}`.

## Project structure
```
purchase-propensity/
├── README.md
├── model/      HF model repo: model.py, handler.py, logreg.joblib, scaler.joblib, model.safetensors, config.json,
│               rfm_segment_stats.json, metrics.json, assets/, README.md (model card)
├── space/      HF Space: app.py (score a visitor · batch CSV · RFM explorer), space_utils.py, examples/visitors.csv
└── training/   notebook.ipynb (controller) + src/ (config, data_setup, model_builder, engine, export, utils)
```

## Reproduce
```bash
git clone --recurse-submodules https://github.com/shalev396/ml-lab && cd ml-lab
pip install -r requirements.txt
cd purchase-propensity/training
jupyter notebook notebook.ipynb                     # full run -> ../model/
SMOKE_TEST=1 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp   # quick check
cd ../space && python app.py                        # local Space, uses ../model
```

## Limitations
- **Late-funnel leakage.** `saw_checkout` and `checked_delivery_detail` are recorded in the same session as the
  order, so the near-perfect scores answer "is this session about to buy?". They are not an early prediction.
- **Repeated patterns.** The data has only 9,086 distinct flag combinations, so most test sessions have an
  identical twin in training.
- **Uncalibrated probabilities.** The class weighting pushes p up, so use the tuned thresholds.
- **One retailer per dataset**, one period each. The RFM quintile edges used for new customers are approximate for
  tied values.
