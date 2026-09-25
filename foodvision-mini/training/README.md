# FoodVision Mini: training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shalev396/ml-lab/blob/main/foodvision-mini/training/notebook.ipynb)

Reproduces the FoodVision Mini recipe from the PyTorch Deep Learning bootcamp (notebook 09).
An ImageNet-pretrained torchvision EfficientNet-B2 is frozen as a feature extractor, and a new
`Dropout(0.3) -> Linear(1408, 3)` head is trained for pizza / steak / sushi. The new run is compared
with the checkpoint that is currently deployed, and the better of the two is exported as a Hugging
Face model folder, the same layout as [`../model`](../model).

## What the notebook does
`notebook.ipynb` is the controller: every section calls functions from `src/`, step by step.

1. **Setup**: 1a finds the project files (clones the repo + the `model/` submodule code on a fresh
   runtime), 1b installs `../../requirements.txt`, 1c picks `device` (cuda -> mps -> cpu).
2. **Config**: `Config()` from `src/config.py`, the `SMOKE_TEST` switch and the `PUSH_TO_HUB` flag.
3. **Data**: `data_setup.create_dataloaders` downloads + caches the images; class counts, sample photos, raw image sizes.
4. **Load model**: `model_builder.build_model` (`model.FoodVisionNet`, ImageNet weights, backbone frozen,
   parameter count) and `model_builder.load_deployed` (the checkpoint deployed now, from `../model` or the Hub).
5. **Training**: `engine.train` for 10 epochs, then the loss/accuracy curves (`export.plot_history`).
6. **Evaluation**: `engine.evaluate` on the test split for both variants, the comparison table,
   `export.select_best`, the confusion matrix of the best and the comparison chart.
7. **Inference**: `model.load(dir, device).predict(image)` on `../space/examples`, exactly what the Space runs.
8. **Export**: `export.build_metrics` + `export.export` write the best variant, `metrics.json`, the plots and
   the refreshed card to `cfg.model_dir`; optional upload to the Hub.

## Recipe (defaults in `src/config.py`)
| | |
|---|---|
| Backbone | torchvision `efficientnet_b2`, `IMAGENET1K_V1` weights, all `features` frozen |
| Head | `Dropout(0.3) -> Linear(1408, 3)`, 4,227 trainable parameters (7,705,221 total) |
| Preprocessing | `model.get_transform()`: resize 288 (bicubic), center-crop 288, ImageNet mean/std (train = test, no augmentation) |
| Optimisation | `Adam(lr=1e-3)` over the head, plain `CrossEntropyLoss`, batch 32, 10 epochs, seed 42 |
| BatchNorm | the whole net is in `train()` mode while training, as in the original, so BN running stats adapt |
| Selection | none inside a run (the last epoch is kept); across runs the deployed checkpoint is replaced only if the new run is strictly better on test accuracy (then macro F1) |

## Run it
- **Colab:** click the badge (a GPU runtime is preselected), then *Run all*.
- **Local:** `pip install -r ../../requirements.txt`, then open `notebook.ipynb` from this folder and run all cells.
- **Headless smoke test** (from `training/`, about a minute on CPU):
  ```bash
  SMOKE_TEST=1 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp
  ```
  Smoke runs use 8 images per class and 1 epoch and write only to `outputs/smoke/` (never `../model`).

## Hardware and time
| run | CPU (measured, desktop, torch 2.14) | GPU |
|---|---|---|
| smoke notebook (24 train / 24 test images, 1 epoch) | ~45 s end to end | seconds |
| full notebook (450 / 150 images, 10 epochs) | ~9-10 min (training alone ~47 s/epoch) | roughly 1-2 min (estimate, not measured) |

On CUDA, mixed precision is used automatically (`utils.autocast`).

## Data
- **Source:** `pizza_steak_sushi_20_percent.zip` from
  [mrdbourke/pytorch-deep-learning](https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip).
  It is a 20% sample of the pizza, steak and sushi classes of [Food-101](https://huggingface.co/datasets/ethz/food101):
  450 train (154 / 146 / 150) and 150 test (46 / 58 / 46) images.
- **Fallback:** if the GitHub URL fails, the same file is fetched from `raw.githubusercontent.com`.
- **Cache:** extracted to `training/data/pizza_steak_sushi_20_percent/{train,test}/<class>/*.jpg` (about 32 MB, gitignored).
  Later runs skip the download.

## Outputs
- Full run: `../model/` gets the best variant's `model.safetensors` + `config.json` (only rewritten when the
  new run wins), `metrics.json` (deployed metrics + `comparison` of every variant),
  `assets/{training_curves,comparison,confusion_matrix}.png`, and a refreshed card
  (YAML, metrics table and Experiments table; the prose is left untouched).
- Scratch: `outputs/run/` (or `outputs/smoke/`) holds this run's plots and the staged best model.
- Smoke run: the same files in `outputs/smoke/model/`, plus copies of the card, `model.py`, `handler.py`
  and `requirements.txt`, so the folder is self-contained. From `space/`,
  `MODEL_DIR=../training/outputs/smoke/model python app.py` serves it.
- Publish: set `PUSH_TO_HUB = True` in section 2, or call `utils.upload_model()`. Authenticate with
  `hf auth login` or an `HF_TOKEN` secret / env var.

## `src/` file map
| file | role |
|---|---|
| `__init__.py` | puts `../model` on `sys.path`, so `import model as M` is the deployed `model.py` |
| `config.py` | `@dataclass Config` with every hyperparameter + smoke handling + `run_dir` / `model_dir` |
| `data_setup.py` | download with fallback, cache, `ImageFolder` datasets, smoke subset, `DataLoader`s |
| `model_builder.py` | `build_model()` (frozen backbone) and `load_deployed()` (the current checkpoint as a baseline) |
| `engine.py` | `make_optimizer`, `train_one_epoch`, `train`, `evaluate`, `classification_metrics` |
| `export.py` | variants, `select_best`, `metrics.json` builder, plots, weights/metrics/assets writer, card refresh |
| `utils.py` | shared ml-lab helpers (devices, seeds, JSON, card update, Hub upload), identical in every project |

## Provenance of the deployed weights
`../model/model.safetensors` is the original bootcamp checkpoint (trained on Apple MPS, published
2026-05-08), converted 1:1 to `FoodVisionNet` (identical probabilities on the example images). Every
full notebook run re-evaluates it next to a fresh retrain; see `../model/metrics.json` (`comparison`)
and the Experiments section of the [model card](../model/README.md).
