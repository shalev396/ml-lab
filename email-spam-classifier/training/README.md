# 📬 Email Spam Classifier: training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shalev396/ml-lab/blob/main/email-spam-classifier/training/notebook.ipynb)

Fine-tunes DistilBERT (uncased) to tell spam from ham on the Enron-Spam corpus, trains a TF-IDF +
logistic-regression baseline next to it, and exports a Hugging Face model folder with the same layout
as [`../model`](../model).

## What the notebook does
`notebook.ipynb` is the controller. Every step calls a function in `src/`:
1. **Setup**: 1a finds the project files (clones the repo on a fresh runtime), 1b installs `../../requirements.txt`, 1c picks `device` (cuda -> mps -> cpu).
2. **Config**: `Config()` from `src/config.py`, the `SMOKE_TEST` toggle and the `PUSH_TO_HUB` flag.
3. **Data**: `data_setup.load_splits` (download, clean with `model.clean_text`, stratified 70/15/15, CSV cache), class balance, `pos_weight`, words-per-email histogram.
4. **Load model**: `model_builder.build_model` builds `model.SpamClassifier` with pretrained DistilBERT, freezes the encoder except the last 2 blocks, prints the parameter counts.
5. **Training**: `engine.train_baseline` (TF-IDF + logistic regression), then `engine.fit` (DistilBERT, early stopping) and `engine.save_checkpoint`, plus the loss / F1 curves.
6. **Evaluation**: test metrics of both experiments through `model.load(checkpoint).predict_proba`, next to the deployed model's `metrics.json`. Confusion matrix, ROC / PR curves, comparison chart.
7. **Inference**: `model.load(dir, device).predict(text)` on `../space/examples/*.txt`, exactly what the Space and the endpoint run.
8. **Export**: `export.export` writes weights + tokenizer + `metrics.json` + `assets/` + the refreshed card into `cfg.model_dir`; optional Hub upload.

## Recipe (defaults in `src/config.py`)
| | |
|---|---|
| Encoder | `distilbert/distilbert-base-uncased` (6 blocks, dim 768), all frozen except the last 2 blocks |
| Head | `[CLS]` -> `Dropout(0.3)` -> `Linear(768, 1)`; sigmoid = P(spam), threshold 0.5 |
| Parameters | 66,363,649 total, 14,176,513 trainable (2 blocks + head) |
| Input | `model.prepare_text`: drop a leading `Subject:`, `clean_text` (HTML unescape, strip tags/URLs, lowercase, keep `a-z0-9 .,!?$%'-`), WordPiece, max 256 tokens |
| Loss | `BCEWithLogitsLoss(pos_weight = n_ham / n_spam)` (0.966 on the Enron train split) |
| Optimiser | AdamW, lr 5e-4 (head) / 2e-5 (encoder), weight decay 0.01, 10% linear warmup then linear decay |
| Schedule | batch 32, up to 5 epochs, early stopping on validation loss (patience 2), best epoch restored, seed 42 |
| Mixed precision | on CUDA only (`utils.autocast`: bf16 on Ampere+, fp16 + GradScaler on older GPUs) |
| Baseline | word 1-2-gram TF-IDF (50k features, sublinear tf, min_df 2) + `LogisticRegression(C=10, class_weight="balanced")` |

## Run it
- **Colab:** click the badge (a GPU runtime is preselected), then *Run all*.
- **Local:** `pip install -r ../../requirements.txt`, then open `notebook.ipynb` from this folder.
- **Headless smoke test** (from `training/`):
  ```bash
  SMOKE_TEST=1 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp
  ```
  128 train / 64 val / 64 test emails, 64 tokens, 1 epoch. Exports to `outputs/smoke/model` and never touches `../model`.

## Hardware and time
| run | CPU | GPU |
|---|---|---|
| smoke | about 1-2 min, most of it loading libraries and the pretrained encoder | seconds |
| baseline (TF-IDF + LR, full data) | under a minute | n/a (CPU) |
| test-set evaluation (5,050 emails, 256 tokens) | 17-41 min on a shared 20-core desktop, 6 threads (measured 1,005 s and 2,452 s) | seconds |
| full DistilBERT fine-tune (23,565 emails, 5 epochs) | several hours (estimate, not measured) | the original run used a CUDA GPU with AMP; its time was not recorded |

## Data
- **Primary:** [Enron-Spam](https://huggingface.co/datasets/SetFit/enron_spam) (Metsis, Androutsopoulos & Paliouras, 2006), Hub copy by SetFit:
  33,716 emails, `text` = subject + body, `label` 1 = spam. After cleaning and dropping empty texts: 33,665 emails,
  split 23,565 train / 5,050 val / 5,050 test (stratified, seed 42, about 51% spam in every split).
- **Fallback:** if the `datasets` loader fails, the same `train.jsonl` / `test.jsonl` files are read straight from the Hub.
- **Alternative:** `Config(dataset="sms")` uses the [SMS Spam Collection](https://huggingface.co/datasets/ucirvine/sms_spam) (5,574 SMS) instead.
- **Cache:** `training/data/enron_{train,val,test}.csv` (cleaned text, about 47 MB, gitignored) and the Hub cache in `training/data/hf/`.

## Outputs
- Full run: `outputs/run/checkpoint/` (best epoch), then `../model/` gets `model.safetensors` + `config.json`
  (`PyTorchModelHubMixin.save_pretrained`), `tokenizer.json` + `tokenizer_config.json`, `metrics.json` and
  `assets/{training_curves,confusion_matrix,roc_pr_curves,comparison}.png`. The YAML and the metrics table in
  `../model/README.md` are refreshed by `utils.update_model_card`; the prose is left untouched.
- Smoke run: the same files in `outputs/smoke/model/`, plus copies of the card, `model.py`, `handler.py` and
  `requirements.txt`, so the folder is self-contained. From `space/`, `MODEL_DIR=../training/outputs/smoke/model python app.py` serves it.
- Publish: set `PUSH_TO_HUB = True` in the notebook, or call `utils.upload_model()`. Authenticate with
  `hf auth login` or an `HF_TOKEN` secret / env var.

## `src/` file map
| file | role |
|---|---|
| `__init__.py` | puts `../model` on `sys.path`, so `import model as M` is the deployed `model.py` |
| `config.py` | `@dataclass Config`: every hyperparameter, smoke handling, `run_dir` / `checkpoint_dir` / `model_dir` |
| `data_setup.py` | download with fallback, cleaning, stratified split + CSV cache, smoke sampling, `pos_weight`, dataset + dynamic-padding `DataLoader`s |
| `model_builder.py` | tokenizer, `build_model()` (pretrained encoder, partial unfreezing), AdamW parameter groups, the TF-IDF baseline |
| `engine.py` | `train_one_epoch`, `evaluate`, `fit` (early stopping), `save_checkpoint`, `predict_texts`, baseline helpers, `binary_metrics`, `confusion` |
| `export.py` | `metrics.json` builder, checkpoint -> model folder copy, plots, model-card refresh |
| `utils.py` | shared ml-lab helpers (devices, seeds, JSON, card update, Hub upload), identical in every project |

## Provenance of the deployed weights
`../model/model.safetensors` is **not** from a run of this folder. It is the checkpoint of the original run
(`train.py` of the first version of this project, CUDA GPU with AMP, same recipe as `src/config.py`, best epoch 5),
converted 1:1 into `model.SpamClassifier` (all 102 tensors matched, `strict=True`) and re-evaluated on the test split
on CPU with `model.Predictor`. The per-epoch curves in `../model/assets/training_curves.png` come from that run's log.
