# Calorie Lens: training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shalev396/ml-lab/blob/main/calorie-lens/training/notebook.ipynb)

Builds the whole model: a meal photo -> the grams of each of 253 ingredients (stage 1) -> nutrition through a
frozen per-gram table (stage 2a), plus NutriText, a food-name -> nutrients model trained from scratch
(stage 2b). Two backbones are compared with the same recipe, the validation winner is retrained with an
auxiliary depth head, and the result is exported as a Hugging Face model folder, the same layout as
[`../model`](../model).

## What the notebook does
`notebook.ipynb` is the controller: every section calls functions from `src/`, step by step.

1. **Setup**: 1a finds the project files (clones the repo + the `model/` submodule code on a fresh
   runtime), 1b installs `../../requirements.txt`, 1c picks `device` (cuda -> mps -> cpu).
2. **Config**: `Config()` from `src/config.py`, the `SMOKE_TEST` / `BUILD_DATA` switches, `PUSH_TO_HUB`,
   TF32 + cuDNN autotuning on CUDA.
3. **Data**: 3a builds and publishes the dataset (by itself while it is not on the Hub, or with
   `BUILD_DATA=1`); 3b downloads it, converts it to memory-mapped Arrow files and builds the loaders; a few
   augmented samples.
4. **Load model**: `model_builder.build_model` for each backbone (parameter breakdown) and
   `engine.throughput`: images/s of the loader and of a full training step on this GPU.
5. **Training**: 5a NutriText (vs TF-IDF kNN and the mean, plus zero-shot on the Nutrition5k names);
   5b `engine.train` once per backbone, `engine.pick_winner`; 5c the winner + depth head; training curves.
6. **Evaluation**: mass blend + display threshold fitted on validation; the test split once (overhead scans,
   side frames, side dishes, `test_clean`, original labels), bootstrap CI, ablation, every experiment on test,
   the graphs.
7. **Inference**: `export.assemble` -> save -> `model.load(dir, device)`, then `predict` / `analyze` /
   `calculate` on example photos, exactly what the Space runs.
8. **Export**: `export.build_metrics` + `export.export` write the model, `metrics.json`, the graphs, NOTICE
   (DINOv2 only) and the refreshed card to `cfg.model_dir`; optional upload to the Hub.

## Recipe (defaults in `src/config.py`)
| | |
|---|---|
| Input | any photo fitted (aspect kept) into a 378 x 504 canvas (4:3; 27 x 36 DINOv2 patches) |
| Backbones | E1: torchvision `convnext_base` (ImageNet, drop-path 0.1) · E2: `facebook/dinov2-with-registers-base` (Apache-2.0) · E3: the winner + depth head |
| Head | 253 ingredient queries, 2 cross-attention layers (d 256, no query self-attention), per-ingredient presence + log-grams outputs; totals head on the pooled features; output biases from the training data |
| Fusion | grams = sigmoid(logit) x grams, rescaled to the plate mass (direct head, blended in log space with the grams' sum; blend fitted on validation) -> frozen 253 x 8 NutritionLayer |
| Losses | ASL presence, log-grams of present ingredients, soft grams vs truth, log totals, NutritionLayer totals vs truth (end to end), E3: depth (scale-invariant log + L1); side frames count 0.5 on the mass losses |
| Sampling | 12,000 samples per epoch, half overhead scans (any rotation), half side frames (5 per camera) |
| Optimisation | AdamW (fused), layer-wise LR decay (ConvNeXt 1e-4 x 0.8 per stage, DINOv2 6e-5 x 0.75 per layer), head 1e-3, 1 warmup epoch + cosine, wd 0.05, EMA 0.999, clip 1.0, batch 64, <= 30 epochs, patience 6 |
| Speed-ups | bf16 autocast, TF32, cuDNN autotune, channels-last (ConvNeXt), `torch.compile` (eager fallback), uint8 decoding in the workers + augmentation on the GPU, memory-mapped Arrow data, 64-image steps on >= 70 GB GPUs (32 x 2 accumulation below; batches split further on out-of-memory) |
| Selection | validation score = mean of the overhead and side-dish calorie % errors; best EMA epoch per run; between backbones a paired plate-group bootstrap (a ConvNeXt within the CI of a DINOv2 wins: cheaper CPU fallback, no extra licence) |
| NutriText | own BPE tokenizer (8,000) + 4-layer d-256 transformer, ~470k USDA names (branded x 0.3), AdamW 1e-3, batch 512, <= 20 epochs, physically bounded outputs |

## Run it
1. Add a Colab secret `HF_TOKEN` with write access (and notebook access).
2. Open the notebook on a **GPU runtime** (A100 / H100 / G4), set `PUSH_TO_HUB = True` to upload the result,
   and *Run all*.
   - **First run:** section 3a builds the dataset before training, because `shalev396/calorie-lens-data`
     is not on the Hub yet. It streams ~195 GB from Google Cloud Storage, keeps ~8 GB and uploads shards every
     30 min; the probe prints the projected time. If the runtime dies, run it again: the build resumes from
     the manifest on the Hub.
   - **Later runs** download the published dataset. `BUILD_DATA=1` forces a rebuild.
   - After a disconnect, re-run: every experiment resumes from its last finished epoch
     (`outputs/run/checkpoints/`, runtime disk).
   - Every training line and the traceback of any failed cell also go to `outputs/run/train.log`
     (`!tail -50 outputs/run/train.log`), which survives cleared outputs and kernel restarts.
   - Checkpoints and log live on the runtime disk: a runtime Colab deletes (for example after an idle timeout
     while the PC sleeps, on plans without background execution) loses them, and the run starts over.
- **Local:** `pip install -r ../../requirements.txt`, then open `notebook.ipynb` from this folder.
- **Headless smoke test** (from `training/`, ~1 min on a desktop CPU once the mini build exists):
  ```bash
  SMOKE_TEST=1 BUILD_DATA=1 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp
  ```
  The first smoke run downloads ~3 MB of metadata, ~150 MB of imagery for 6 dishes and ~14 MB of USDA tables.
  Smoke runs use a random tiny ConvNeXt at 96 x 128 px and write only to `outputs/smoke/` (never `../model`).

## Hardware and time
| run | CPU (desktop) | GPU |
|---|---|---|
| smoke notebook (6 dishes, 2 epochs, NutriText 1 epoch) | ~65 s end to end (measured) | seconds |
| dataset build (25,466 files, ~195 GB streamed) | ~1.5 h on a 20-thread desktop at ~1 Gbit/s (measured on the first 400 dishes) | runs on the GPU runtime's CPUs in the first session, ~1-2 h (section 3a's probe projects it) |
| full notebook (NutriText + E1 + E2 + E3) | not practical | ~2.7 h on an A100, ~1.6 h on an H100 (estimate from FLOPs; section 4 measures it) |

## Data
- **[Nutrition5k](https://github.com/google-research-datasets/Nutrition5k)** (Thames et al., CVPR 2021, CC BY 4.0),
  anonymous HTTPS: 5,006 dishes, 3,490 overhead RGB-D scans, 18,486 side-angle H.264 videos (1080p, ~31-41 frames).
- **[USDA FoodData Central](https://fdc.nal.usda.gov/download-datasets/)** (public domain): Foundation 04/2026,
  SR Legacy 04/2018, FNDDS 2021-2023, Branded 04/2026. FNDDS stores legacy nutrient *numbers* in
  `food_nutrient.nutrient_id`; `usda._nutrients` maps them.
- **Label fixes** (`src/n5k_labels.py`, asserted by `check()`): Cafe 2 totals recomputed from the ingredients;
  10 x1000 scale errors (the scale logged 7974 for 7.974 g); "plate only" (the empty plate's weight) set to 0 g;
  the two "deprecated" table rows merged into apple (453, carb/protein swapped) and watermelon (458); empty
  names taken from the table; the phone photo (dish_1559172356) dropped; totals = sum of ingredients.
- **Plate groups:** each dish links to its largest earlier subset within 30 min (one predecessor: linking
  every subset chains unrelated plates through standard single-item portions) and to identical re-scans.
  2,346 groups; validation is drawn as whole groups; `test_clean` = 565 of the 709 test dishes.
- **Ingredient -> USDA map** (`labels/ingredient_fdc_map.csv`, committed): one generic USDA food per
  ingredient for fiber / sugars / saturated fat / sodium. `usda.suggest_matches` ranked candidates by name
  similarity and, mostly, by agreement with the ingredient's known macros (Nutrition5k took many values
  straight from SR Legacy rows); every row was reviewed, 44 corrected by hand (`status`).

## Outputs
- Full run: `../model/` gets `model.safetensors` + `config.json` + `nutritext_tokenizer.json`, `metrics.json`,
  `assets/*.png` (training curves, experiments, ablation, calorie scatter, error by plate size, NutriText,
  heatmaps), `NOTICE` + `LICENSE-APACHE-2.0.txt` if DINOv2 ships, and a refreshed card (YAML, metrics table,
  Experiments table). Update the card's prose and the project README with the numbers by hand.
- Scratch: `outputs/run/` (checkpoints, graphs, the staged best model, example photos).
- Smoke run: the same files in `outputs/smoke/model/` plus copies of the card and code, so the folder is
  self-contained. From `space/`: `MODEL_DIR=../training/outputs/smoke/model FORCE_CPU=1 python app.py`.

## `src/`
| file | what it does |
|---|---|
| `config.py` | every hyperparameter, the experiments, smoke settings, paths |
| `n5k_labels.py` | Nutrition5k metadata: parse, fix, plate groups, splits, ingredient vocabulary, `check()` |
| `usda.py` | FoodData Central -> one per-100 g table; ingredient match suggestions; extended per-gram nutrients |
| `dataset_build.py` | resumable image build (GCS listing, H.264 frame extraction, shards, manifest, Hub upload) + tables + dataset card |
| `data_setup.py` | download, Arrow conversion, labels, Dataset, mixture sampler, GPU augmentation, examples |
| `model_builder.py` | builds `model.CalorieLens` with output-bias priors, depth head, training wrapper, param groups, micro-batch |
| `engine.py` | losses, training loop (EMA, resume, compile, OOM splitting), evaluation, metrics, blend / threshold fitting, bootstrap, model selection |
| `nutritext.py` | stage 2b: data, tokenizer, training, baselines, zero-shot test |
| `export.py` | assembles the deployed model, metrics.json, NOTICE, graphs, model card |
| `utils.py` | the same file in every ml-lab project (device, seeding, JSON, card updater, Hub upload) |
