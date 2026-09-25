# 📬 Email Spam Classifier

> Paste an email and a fine-tuned DistilBERT tells you whether it belongs in the inbox or the spam folder.

<p>
  <a href="https://huggingface.co/spaces/shalev396/email-spam-classifier"><img alt="Space" src="https://img.shields.io/badge/🤗%20Space-live%20demo-blue"></a>
  <a href="https://huggingface.co/shalev396/email-spam-classifier"><img alt="Model" src="https://img.shields.io/badge/🤗%20Model-shalev396%2Femail--spam--classifier-yellow"></a>
  <a href="https://colab.research.google.com/github/shalev396/ml-lab/blob/main/email-spam-classifier/training/notebook.ipynb"><img alt="Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>

| | |
|---|---|
| **Task** | Binary text classification: spam vs ham (legitimate email) |
| **Framework** | PyTorch + Hugging Face `transformers` (custom `PyTorchModelHubMixin` model) |
| **Architecture** | DistilBERT-base-uncased, last 2 of 6 blocks fine-tuned, `Dropout(0.3) -> Linear(768, 1)` on `[CLS]` (66,363,649 parameters, 14,176,513 trained) |
| **Dataset** | [Enron-Spam](https://huggingface.co/datasets/SetFit/enron_spam) (SetFit copy), 33,665 cleaned emails, stratified 70/15/15: 23,565 / 5,050 / 5,050 |
| **Result** | F1 (spam) = 0.9938, accuracy = 0.9937 (test, 32 errors in 5,050 emails) |
| **Runs on** | Space: CPU basic · Training: CPU, GPU or Colab (the full fine-tune is a GPU job) |

## The problem
Spam filtering is a classic text-classification task: given the subject and body of an email, decide whether it is
unsolicited bulk mail (advertising, scams, phishing) or a normal message. False alarms (a real email sent to spam)
are usually more costly than a missed spam, so precision on the spam class matters as much as recall.

## The data
[Enron-Spam](https://huggingface.co/datasets/SetFit/enron_spam) (Metsis, Androutsopoulos & Paliouras, 2006) mixes the
real mailboxes of six Enron employees (ham) with spam collected from several spam traps. The SetFit copy on the Hub
has 33,716 emails, each with the subject and body joined into one `text`. The training code cleans every email with
`model.clean_text` (HTML unescape, strip tags and URLs, lowercase, keep letters, digits and `. , ! ? $ % ' -`),
drops the 51 emails that end up empty and makes a stratified 70/15/15 split (seed 42). The classes are balanced,
about 51% spam in every split. The [SMS Spam Collection](https://huggingface.co/datasets/ucirvine/sms_spam) can be
used instead with `Config(dataset="sms")`.

## Architecture
A pretrained [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) encoder (6 transformer blocks,
hidden size 768) reads up to 256 WordPiece tokens. The hidden state of the `[CLS]` token goes through
`Dropout(0.3) -> Linear(768, 1)`, and a sigmoid turns that logit into P(spam). The model class
(`model.SpamClassifier`) lives in [`model/model.py`](model/model.py) together with the text cleaning and the
`Predictor`, so training, the Space and the Inference Endpoint all use the same code. The encoder config is stored in
`config.json`, so inference rebuilds the network offline and never downloads the base weights.

## Training & experiments
Recipe: all encoder weights are frozen except the last 2 blocks. The loss is `BCEWithLogitsLoss(pos_weight = n_ham / n_spam)`.
AdamW uses differential learning rates (head 5e-4, encoder 2e-5), with 10% linear warmup then linear decay. Batch 32,
up to 5 epochs, early stopping on validation loss (patience 2). The original run (CUDA GPU, mixed precision) kept epoch 5.

| test split (5,050 emails) | accuracy | precision | recall | F1 | errors |
|---|---|---|---|---|---|
| **DistilBERT fine-tuned (deployed)** | **0.9937** | **0.9930** | 0.9945 | **0.9938** | **32** |
| TF-IDF 1-2-grams + logistic regression (baseline) | 0.9913 | 0.9880 | **0.9949** | 0.9915 | 44 |

The baseline is strong on this corpus. The transformer mainly cuts false alarms (18 ham emails flagged as spam
instead of 31). Graphs (training curves, experiment comparison, confusion matrix, ROC / PR curves) are in the
[model card](https://huggingface.co/shalev396/email-spam-classifier).

## Results
F1 (spam) = 0.9938, accuracy = 0.9937, precision = 0.9930, recall = 0.9945, ROC-AUC = 0.9998 on the 5,050-email test split
([`model/metrics.json`](model/metrics.json)). The deployed weights are the checkpoint of the original training run
(2026-07-13), converted 1:1 to safetensors and re-evaluated on CPU through `model.Predictor`. The confusion matrix
(2464 / 18 / 14 / 2554) matches the original run's log exactly.

## Deployment
- **Model repo**: https://huggingface.co/shalev396/email-spam-classifier (weights, tokenizer, `model.py`, `handler.py`)
- **Space / API**: https://huggingface.co/spaces/shalev396/email-spam-classifier — `POST /gradio_api/call/predict`
  with one string `"Subject: <subject>\n\n<body>"`, returns `[{"label", "confidences": spam / ham}, seconds, device]`
- **Inference Endpoint**: deploy the model repo from its page (Deploy → Inference Endpoints)

## Project structure
```
email-spam-classifier/
├── model/      HF model repo: model.safetensors, config.json, tokenizer files, model.py, handler.py,
│               requirements.txt, metrics.json, assets/*.png, README.md (model card)
├── space/      HF Space: app.py (Gradio, /predict), space_utils.py, requirements.txt, examples/*.txt, README.md
└── training/   notebook.ipynb (controller) + src/ (config, data_setup, model_builder, engine, export, utils)
```

## Reproduce
```bash
git clone https://github.com/shalev396/ml-lab && cd ml-lab
pip install -r requirements.txt
cd email-spam-classifier/training
SMOKE_TEST=1 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp   # ~2-3 min sanity run
jupyter notebook notebook.ipynb    # full run (GPU recommended): retrains and overwrites ../model
```
Or open it in [Colab](https://colab.research.google.com/github/shalev396/ml-lab/blob/main/email-spam-classifier/training/notebook.ipynb).
Run the Space locally with `cd space && python app.py` (it uses `../model`).

## Limitations
- The ham is one company's mail from around 2000-2002, and the spam is from the same era. Modern inboxes (phishing, newsletters) look different, so treat 99.4% as an in-distribution number.
- A random split puts emails from the same mailboxes and period in train and test, which makes the score optimistic.
- Only the text is used: no headers, sender, URLs (removed in cleaning) or attachments. Only the first 256 tokens are read, and the model is English and uncased.
- The 0.5 threshold was not tuned for the cost of false alarms.
