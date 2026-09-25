# Age & Gender Estimator: training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shalev396/ml-lab/blob/main/age-estimator/training/notebook.ipynb)

Fine-tunes an ImageNet EfficientNet-B2 on UTKFace with two heads (90 age bins + gender) and
exports it as a Hugging Face model folder, the same layout as [`../model`](../model). The face
detector is the one trained in the face-recognition project (`../model/face_detector.safetensors`);
it is not trained here.

## What the notebook does
1. **Setup**: finds the project files (clones the repo on a fresh runtime), installs `../../requirements.txt`, picks `device` (cuda → mps → cpu).
2. **Config**: `Config()` from `src/config.py`, the `SMOKE_TEST` switch and the `PUSH_TO_HUB` flag.
3. **Data**: downloads + caches the UTKFace parquet shards, 90/10 split, age histogram, sample faces, DataLoaders.
4. **Load model**: `model.AgeGenderNet(pretrained=True)` from `../model/model.py`, parameter count.
5. **Training**: `engine.fit()`, one validation pass per epoch, keeps the epoch with the lowest age MAE, loss/MAE/accuracy curves.
6. **Evaluation**: age MAE / RMSE / ±5-year accuracy / gender accuracy, compared with argmax decoding, a no-learning baseline and the deployed model. Plots: predicted vs true age, error by age group, gender confusion matrix, comparison chart.
7. **Inference**: saves the trained net as a loadable folder and runs `model.load(dir).predict(image)` + Grad-CAM on `../space/examples`, exactly what the Space runs.
8. **Export**: weights, config, `metrics.json` and plots to `../model` (or `outputs/smoke/model`), refreshes the card, optional upload.

## Recipe (defaults in `src/config.py`)
| | |
|---|---|
| Network | EfficientNet-B2 (`IMAGENET1K_V1`) trunk, fully fine-tuned · heads `Dropout(0.3) → Linear(1408, 90)` and `Dropout(0.3) → Linear(1408, 2)` |
| Targets | age bin (`0-1`, `2` … `89`, `90+`) and gender (male/female) |
| Loss | CE(age) + 1.0 × CE(gender) |
| Optimisation | `Adam(lr=1e-3)`, batch 64, 10 epochs, seed 42, mixed precision on CUDA (`utils.autocast`) |
| Augmentation | horizontal flip, colour jitter (brightness/contrast/saturation 0.2) on top of resize 288 → center-crop 288 → ImageNet norm |
| Model selection | best validation age MAE (expected-age decoding) |
| Split | random 90/10 (seed 42), 21,334 / 2,371 faces, identical to `datasets.train_test_split(test_size=0.1, seed=42)` |

## Run it
- **Colab:** click the badge (a GPU runtime is preselected), then *Run all*.
- **Local:** `pip install -r ../../requirements.txt`, then open `notebook.ipynb` from this folder.
- **Smoke test:** `SMOKE_TEST=1 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp`
  (96 faces from the first shard, 1 epoch; everything goes to `outputs/smoke/`, `../model` is never touched).

## Hardware and time
| run | CPU | GPU |
|---|---|---|
| smoke (86 train / 10 validation faces from shard 1, 1 epoch) | 63–112 s for the whole notebook (measured on a busy shared desktop CPU) | well under a minute |
| full (21,334 / 2,371 faces, 10 epochs) | many hours; not practical | roughly 1 h on a T4-class GPU (estimate, not measured) |

The deployed weights came from a run in the project's original repo. Its hardware and time were not recorded.

## Data
- **Source:** [`nu-delta/utkface`](https://huggingface.co/datasets/nu-delta/utkface) on the Hugging Face Hub, a copy of
  [UTKFace](https://susanqq.github.io/UTKFace/) (aligned + cropped faces, 200×200), 3 parquet shards, about 1 GB.
  UTKFace is licensed for non-commercial research only.
- **Fallback:** if the parquet files of the repo can't be fetched, the Hub's auto-converted parquet branch
  (`refs/convert/parquet`) of the same dataset is used.
- **Cache:** `training/data/utkface/data/train-0000{0,1,2}-of-00003.parquet` (gitignored). Later runs skip the download.

## Outputs
- Full run: `../model/` gets `model.safetensors` + `config.json` (via `PyTorchModelHubMixin.save_pretrained`),
  `metrics.json` and `assets/*.png`. The metrics table and YAML in `../model/README.md` are refreshed by
  `utils.update_model_card`. Update the prose (Evaluation section) by hand if the numbers change.
- Scratch: `outputs/run/` (full) or `outputs/smoke/` (smoke) hold the EDA plot, the training curves and a loadable copy
  of the trained model (`trained_model/`). From `space/`, `MODEL_DIR=../training/outputs/smoke/model python app.py` serves it.
- Publish: set `PUSH_TO_HUB = True` in the notebook, or call `utils.upload_model()`. Authenticate with
  `hf auth login` or an `HF_TOKEN` secret / env var.

## `src/` file map
| file | role |
|---|---|
| `__init__.py` | puts `../model` on `sys.path`, so `import model as M` is the deployed `model.py` |
| `config.py` | `@dataclass Config` with every hyperparameter + smoke handling + `run_dir` / `model_dir` |
| `data_setup.py` | download with fallback, parquet loading, split, representative ages, augmentation, `Dataset`/`DataLoader`s |
| `model_builder.py` | `build_model()` (ImageNet-initialised `AgeGenderNet`) and the median/majority baseline |
| `engine.py` | `train_one_epoch`, `evaluate`, `fit` (best-epoch selection), `scores`, `mae_by_age_group` |
| `export.py` | `metrics.json` builder, loadable-folder writer, card refresh, all plots |
| `utils.py` | shared ml-lab helpers (devices, AMP, seeds, JSON, card update, Hub upload), identical in every project |

## Provenance of the deployed weights
`../model/model.safetensors` is **not** from a run of this folder. It is the checkpoint of the original
training run (the same recipe in the project's earlier `train.py`), converted 1:1 to the mixin format. It was re-evaluated
here on the same validation split. See `../model/metrics.json` (`source`, `comparison`).
