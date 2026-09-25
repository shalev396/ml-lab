# FoodVision Big: training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shalev396/ml-lab/blob/main/foodvision-big/training/notebook.ipynb)

Reproduces the FoodVision Big run from the PyTorch Deep Learning bootcamp (`foodvision_big/train.py`).
An ImageNet-pretrained torchvision EfficientNet-B2 gets a new `Dropout(0.3) -> Linear(1408, 101)` head
and is fine-tuned end to end on all of Food-101. The new run is compared with the checkpoint that is
currently deployed, and the better of the two is exported as a Hugging Face model folder, the same
layout as [`../model`](../model).

## What the notebook does
`notebook.ipynb` is the controller: every section calls functions from `src/`, step by step.

1. **Setup**: 1a finds the project files (clones the repo + the `model/` submodule code on a fresh
   runtime), 1b installs `../../requirements.txt`, 1c picks `device` (cuda -> mps -> cpu).
2. **Config**: `Config()` from `src/config.py`, the `SMOKE_TEST` switch and the `PUSH_TO_HUB` flag.
3. **Data**: `data_setup.create_dataloaders` downloads + caches Food-101 and builds the loaders;
   images per class, sample photos, what TrivialAugmentWide does, raw image sizes.
4. **Load model**: `model_builder.build_model` (`model.FoodVisionNet`, ImageNet weights, parameter count),
   `model_builder.maybe_compile`, and `model_builder.load_deployed` (the checkpoint deployed now, from `../model` or the Hub).
5. **Training**: `engine.train` for 5 epochs (keeps the lowest-test-loss epoch), then the loss/accuracy curves (`export.plot_history`).
6. **Evaluation**: `engine.evaluate` per image on the test split for both variants (top-1, top-5, macro F1),
   the comparison table, `export.select_best`, the confusion matrix, per-class accuracy, the most frequent mistakes
   and the comparison chart.
7. **Inference**: `model.load(dir, device).predict(image, top_k=5)` on `../space/examples`, exactly what the Space runs.
8. **Export**: `export.build_metrics` + `export.export` write the best variant, `metrics.json`, the plots and
   the refreshed card to `cfg.model_dir`; optional upload to the Hub.

## Recipe (defaults in `src/config.py`)
| | |
|---|---|
| Backbone | torchvision `efficientnet_b2`, `IMAGENET1K_V1` weights, every layer trainable (7,843,303 parameters) |
| Head | `Dropout(0.3) -> Linear(1408, 101)` |
| Preprocessing | `model.get_transform()`: resize 288 (bicubic), center-crop 288, ImageNet mean/std; training images get `TrivialAugmentWide` first |
| Optimisation | `Adam(lr=1e-4)`, constant LR, `CrossEntropyLoss(label_smoothing=0.1)`, batch 32, 5 epochs, seed 42 |
| Speed-ups | AMP on CUDA (`utils.autocast`: bf16 on Ampere+, fp16 + GradScaler on older GPUs); `use_compile=True` enables `torch.compile` on CUDA (the original run used it) |
| Selection | inside a run, the epoch with the lowest test loss (Food-101 has no validation split; in the original run it was the last epoch); across runs the deployed checkpoint is replaced only if the new run is strictly better on test top-1 (then top-5) |

## Run it
- **Colab:** click the badge (a GPU runtime is preselected), then *Run all*. A T4 needs roughly 1-1.5 hours
  for the full run (estimate, not measured) plus the ~5 GB download.
- **Local:** `pip install -r ../../requirements.txt`, then open `notebook.ipynb` from this folder and run all cells.
- **Headless smoke test** (from `training/`, about 1.5 min on CPU):
  ```bash
  SMOKE_TEST=1 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp
  ```
  Smoke runs use a 64-image train slice + 64-image test slice and 1 epoch and write only to
  `outputs/smoke/` (never `../model`).
- **Full run, headless** (GPU): `jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp --ExecutePreprocessor.timeout=-1`

- **Re-evaluate only the deployed checkpoint on the full test split** (no training; ~30 min on a desktop
  CPU after the ~5 GB download, minutes on a GPU), from `training/`:
  ```python
  from src import data_setup, engine, model_builder
  from src.config import Config
  import model as M
  cfg = Config()                                    # source="hub": full Food-101, cached in data/hf/
  _, test_loader, _ = data_setup.create_dataloaders(cfg, M.get_transform(), "cpu")
  net, _, _ = model_builder.load_deployed("cpu")
  result = engine.evaluate(net, test_loader, engine.make_loss(cfg.label_smoothing), "cpu")
  print(engine.classification_metrics(result))      # per-image top-1 / top-5 / macro F1
  ```

## Hardware and time
| run | CPU (desktop, torch 2.14) | GPU |
|---|---|---|
| smoke notebook (64 / 64 images, 1 epoch) | ~1.5 min end to end (measured) | seconds |
| full notebook (75,750 / 25,250 images, 5 epochs + 2 full test evaluations) | not practical: evaluation alone runs at ~15 images/s, so a single test pass takes ~28 min and training many hours | original run: ~22 min for 5 epochs on an RTX 2080 Ti (AMP + `torch.compile`), ~4 min per epoch |

## Data
- **Source:** [ethz/food101](https://huggingface.co/datasets/ethz/food101) on the Hub (101 dishes, 750 train
  + 250 test photos each; the Hub's `validation` split is the official test split).
- **Full run:** `datasets.load_dataset` downloads the parquet shards once (~5 GB) into `training/data/hf/`.
- **Smoke run / quick checks (`source="slice"`):** only a few 100-image parquet row groups, spread evenly over
  the split, are read with HTTP range requests and cached as one small parquet file in `training/data/slices/`.
  The Hub files are grouped by class, so a slice covers about one to two dishes per row group.
- **Fallback:** if the Hub fails, `torchvision.datasets.Food101` downloads the original ETH Zurich tarball
  (~5 GB) into `training/data/food-101/`.
- **Label order:** the model uses torchvision's alphabetical order. The Hub lists `cheesecake` before
  `cheese_plate`, so Hub labels are remapped by name (`data_setup.class_names_and_map`).

## Outputs
- Full run: `../model/` gets the best variant's `model.safetensors` + `config.json` (only rewritten when the
  new run wins), `metrics.json` (deployed metrics + `comparison` of every variant),
  `assets/{training_curves,comparison,confusion_matrix,per_class_accuracy}.png`, and a refreshed card
  (YAML, metrics table and Experiments table). The card's prose describes the current (converted)
  checkpoint and its slice re-check; update it by hand after a full run.
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
| `data_setup.py` | Hub download / row-group slices / torchvision fallback, label remap, augmentation, `DataLoader`s |
| `model_builder.py` | `build_model()`, `maybe_compile()` and `load_deployed()` (the current checkpoint as a baseline) |
| `engine.py` | `make_loss`, `make_optimizer`, `train_one_epoch`, `train`, `evaluate` (top-1/top-5), metrics |
| `export.py` | variants, `select_best`, `metrics.json` builder, plots, weights/metrics/assets writer, card refresh |
| `utils.py` | shared ml-lab helpers (devices, seeds, JSON, card update, Hub upload), identical in every project |

## Provenance of the deployed weights
`../model/model.safetensors` is the original bootcamp checkpoint (`foodvision_big.pth`, trained on
2026-05-09 on an RTX 2080 Ti), converted 1:1 to `FoodVisionNet` (identical probabilities on the example
images). Its headline number, 87.46% top-1 on all 25,250 test images, comes from that run's TensorBoard
log. A local CPU re-check on a 1,010-image test slice gave 87.9% top-1 / 97.5% top-5. See
`../model/metrics.json` (`history`, `sanity_check`) and the [model card](../model/README.md).
