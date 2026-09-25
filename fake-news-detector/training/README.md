# Fake News Detector — training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shalev396/ml-lab/blob/main/fake-news-detector/training/notebook.ipynb)

`notebook.ipynb` is the controller: every step calls a function from `src/` (cleaning, the vectorizer and the
architectures come from `../model/model.py`, the same file the Space runs).

| Section | What it does |
|---|---|
| 1. Setup | 1a finds the project files (clones the repo on a fresh runtime), 1b installs `../../requirements.txt`, 1c picks the `device` (`GPU` if TensorFlow sees one, else `CPU`), 1d imports `src` + `model` |
| 2. Config | `cfg = Config()` (every hyperparameter), `SMOKE_TEST`, `PUSH_TO_HUB` |
| 3. Data | download (cached) -> dedup -> balanced 20k subsample -> leakage-safe cleaning -> stratified 70/15/15 split -> `TextVectorization` adapted on train only |
| 4. Load model | builds each variant with `model.build_model` and prints its parameter count |
| 5. Training | trains SimpleRNN, LSTM, GRU and BiLSTM with the same seed and recipe (Adam 1e-3, batch 64, <= 4 epochs, early stopping on val loss) and plots the curves |
| 6. Evaluation | comparison table, best variant by **validation** accuracy, comparison chart, confusion matrix, ROC |
| 7. Inference | saves the chosen model (`export.save_model`, with a reload check) and runs `model.load(dir).predict(text)` on `../space/examples` |
| 8. Export | `metrics.json`, `assets/*.png` and the refreshed model card (`export.write_report`); optional Hub upload |

## Run it
- **Colab**: click the badge (a GPU runtime is selected by default; CPU works too).
- **Locally**: `pip install -r ../../requirements.txt`, then open `notebook.ipynb` from this folder.
- **Headless smoke test** (about 2 minutes on CPU, writes only to `outputs/smoke/`):
  `SMOKE_TEST=1 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp`

With `SMOKE_TEST=0` the notebook retrains all four variants and **overwrites `../model`**. The earlier full run of
this pipeline took 499 s (about 8 minutes) on a desktop CPU. On a GPU it should take a few minutes (estimate, not
measured). The smoke run uses 800 articles, a 5,000-word vocabulary, 100 tokens, 16 units and 1 epoch per variant.

## Data
- Primary: Kaggle [`clmentbisaillon/fake-and-real-news-dataset`](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
  (ISOT, `Fake.csv` + `True.csv`, about 45k articles), downloaded anonymously with `kagglehub` into `data/`.
- Fallback: Hugging Face [`GonzaloA/fake_news`](https://huggingface.co/datasets/GonzaloA/fake_news) (a mirror of the same
  articles; its label 1 = true is remapped to our 0 = real).
- NLTK English stopwords are cached in `data/nltk_data/`.
- Leakage: `subject` and `date` are dropped, and the `CITY (Reuters) -` dateline and every `reuters` token are removed.

## Outputs
- `../model/` (full run): `model.keras`, `vocab.json`, `stopwords.json`, `config.json`, `metrics.json`, `assets/`,
  refreshed `README.md`. Publish with `PUSH_TO_HUB = True` (token from `hf auth login` or `HF_TOKEN`).
- `outputs/` (gitignored): notebook plots; `outputs/smoke/model/` is a complete model folder from a smoke run
  (`MODEL_DIR=training/outputs/smoke/model python space/app.py` serves it).

## `src/`
| File | Contents |
|---|---|
| `__init__.py` | puts `../model` on `sys.path` (`import model as M`) |
| `config.py` | `Config` dataclass + smoke settings + `model_dir` |
| `data_setup.py` | download / fallback, stopwords, `prepare`, `split`, `build_vectorizer` |
| `model_builder.py` | `build(variant, cfg)` from `model.build_model`, `describe` |
| `engine.py` | compile, fit with early stopping, scores, confusion, `evaluate`, `select_best` |
| `export.py` | `save_model`, `write_report` (metrics.json, plots, model card) |
| `utils.py` | shared ml-lab helpers (same file in every project) |
