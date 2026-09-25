# 📰 Fake News Detector

> Paste a news headline and article: a bidirectional LSTM tells you whether it reads like fake or real news.

<p>
  <a href="https://huggingface.co/spaces/shalev396/fake-news-detector"><img alt="Space" src="https://img.shields.io/badge/🤗%20Space-live%20demo-blue"></a>
  <a href="https://huggingface.co/shalev396/fake-news-detector"><img alt="Model" src="https://img.shields.io/badge/🤗%20Model-shalev396%2Ffake--news--detector-yellow"></a>
  <a href="https://colab.research.google.com/github/shalev396/ml-lab/blob/main/fake-news-detector/training/notebook.ipynb"><img alt="Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>

| | |
|---|---|
| **Task** | Binary text classification: fake vs real news article |
| **Framework** | TensorFlow / Keras 3 |
| **Architecture** | Embedding(20k, 100) -> Bidirectional LSTM(64) -> Dense(32) -> sigmoid (2,088,641 parameters) |
| **Dataset** | [ISOT Fake and Real News](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset), balanced 20k articles, 70/15/15 split |
| **Result** | accuracy = 0.985, F1 = 0.985, ROC-AUC = 0.9987 (test, 3,000 articles) |
| **Runs on** | Space: CPU basic · Training: CPU, GPU or Colab |

## The problem
Given a headline and the article text, estimate whether the article is fake news or real reporting. This is a
sequence classification task: the order of the words matters, which is what recurrent networks are built for.
The project compares four recurrent architectures on the same data and serves the best one.

## The data
The ISOT dataset has about 23.5k fake articles (`Fake.csv`, from sites flagged by PolitiFact and Wikipedia) and
21.4k real ones (`True.csv`, Reuters), mostly US politics from 2015-2018. After removing duplicate title+text pairs,
a balanced sample of 20,000 articles is used (19,998 after dropping near-empty rows) and split, stratified, into
14,448 train / 2,550 validation / 3,000 test articles.

The dataset has two well-known shortcuts, and both are removed so the model has to read the content:
1. Every real article starts with a `CITY (Reuters) -` dateline. It is stripped with a regex, and every
   remaining `reuters` token is removed.
2. The `subject` column separates the classes on its own (for example `politicsNews` only appears in real
   articles). Only `title` and `text` are used.

Cleaning (`model.clean_text`, shared by training and the Space): lowercase, remove URLs, digits and punctuation,
drop the NLTK English stopwords and 1-letter words.

## Architecture
`model.build_model(variant)`: `TextVectorization` (20,000 words, 300 tokens, adapted on the training split) ->
`Embedding(20,000, 100, mask_zero=True)` trained from scratch -> recurrent layer (64 units) -> `Dropout(0.3)` ->
`Dense(32, relu)` -> `Dense(1, sigmoid)` = P(fake). The recurrent layer is the experiment: SimpleRNN, LSTM, GRU or
a bidirectional LSTM (which reads the article in both directions and concatenates the two 64-d states).

## Training & experiments
Adam (lr 1e-3), binary cross-entropy, batch 64, up to 4 epochs, early stopping on validation loss (patience 2,
best weights restored), the same seed for every variant. The variant with the best **validation** accuracy is
deployed; the test split is only used for the report.

| variant | params | epochs | val accuracy | test accuracy | test precision | test recall | test F1 | test ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| **BiLSTM** (deployed) | **2,088,641** | **3** | **0.9914** | **0.9850** | **0.9906** | 0.9793 | **0.9849** | **0.9987** |
| LSTM | 2,044,353 | 4 | 0.9753 | 0.9747 | 0.9666 | **0.9833** | 0.9749 | 0.9954 |
| SimpleRNN | 2,012,673 | 4 | 0.9686 | 0.9667 | 0.9762 | 0.9567 | 0.9663 | 0.9885 |
| GRU | 2,033,985 | 4 | 0.9525 | 0.9580 | 0.9623 | 0.9533 | 0.9578 | 0.9892 |

All four clear 95% test accuracy. The BiLSTM is the best on validation accuracy and on test accuracy, precision,
F1 and ROC-AUC; the LSTM has slightly higher test recall (0.983 vs 0.979). The BiLSTM reached its lowest validation
loss after one epoch, while the single-direction models began to overfit after epoch 2. The full four-model run took
499 s on a desktop CPU.

## Results
On the 3,000 test articles the BiLSTM gets 2,955 right: 14 real articles are flagged as fake and 31 fake articles
pass as real (accuracy 0.985, precision 0.991, recall 0.979, ROC-AUC 0.9987). Graphs and the full numbers are in
the [model card](model/README.md) and [`model/metrics.json`](model/metrics.json).

The deployed weights come from the earlier version of this project (trained 2026-08-01 on CPU with the same data
pipeline and architecture). They were converted to the Keras 3.15 format without retraining and re-evaluated with
this repo's code on the reconstructed split, which reproduced the original test numbers exactly.

## Deployment
- **Model repo**: https://huggingface.co/shalev396/fake-news-detector (weights, `model.py`, `handler.py`)
- **Space / API**: https://huggingface.co/spaces/shalev396/fake-news-detector — `POST /gradio_api/call/predict`
  with `{"data": ["<headline>\n\n<article>"]}` returns `[label, seconds, device]`
- **Inference Endpoint**: deploy the model repo from its page (Deploy → Inference Endpoints)

## Project structure
```
fake-news-detector/
├── model/      HF model repo: model.keras, vocab.json, stopwords.json, config.json, model.py, handler.py, metrics.json, assets/
├── space/      HF Space: app.py (Gradio + /predict), space_utils.py, examples/*.txt
└── training/   notebook.ipynb + src/ (config, data_setup, model_builder, engine, export, utils)
```

## Reproduce
```bash
pip install -r requirements.txt                      # from the ml-lab root
cd fake-news-detector/training
SMOKE_TEST=1 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp   # ~2 min check
jupyter notebook notebook.ipynb                      # SMOKE_TEST=0: full retrain (~8 min CPU), overwrites ../model
cd ../space && python app.py                         # local Space on http://127.0.0.1:7860, uses ../model
```

## Limitations
- The model learned the style of two sources (Reuters vs flagged sites, 2015-2018, mostly US politics) as much as
  truthfulness. Expect lower accuracy on other outlets, topics, years or languages.
- It does not check facts. A false claim written like a wire story can score as real; sensational but true writing
  can score as fake.
- Only the first 300 cleaned tokens are read; words outside the 20,000-word vocabulary become one unknown token.
- The probabilities are not calibrated.
