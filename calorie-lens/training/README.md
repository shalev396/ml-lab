# Calorie Lens: training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shalev396/ml-lab/blob/main/calorie-lens/training/notebook.ipynb)

Builds the whole model: a meal photo -> the grams of each of 253 ingredients (stage 1) -> nutrition through a
frozen per-gram table (stage 2a), plus NutriText, a food-name -> nutrients model trained from scratch
(stage 2b). Two backbones are trained with the same recipe on Nutrition5k plus the Flavoria lunch plates; then
each candidate (each network and the shipped v1 model, with 1 / 2 / 4 test-time views and its own fitted readout)
is scored on validation, and the winner is exported as a Hugging Face model folder, the same layout as
[`../model`](../model).

## What the notebook does
`notebook.ipynb` is the controller: every section calls functions from `src/`, step by step.

1. **Setup**: 1a finds the project files (clones the repo + the `model/` submodule code on a fresh
   runtime), 1b installs `../../requirements.txt`, 1c picks `device` (cuda -> mps -> cpu).
2. **Config**: `Config()` from `src/config.py`, the `SMOKE_TEST` / `BUILD_DATA` switches, `PUSH_TO_HUB`,
   TF32 + cuDNN autotuning on CUDA.
3. **Data**: 3a builds and publishes the dataset (by itself while it is not on the Hub, or with
   `BUILD_DATA=1`), then the Flavoria plates the same way (`flavoria.build`); 3b downloads it, converts it to memory-mapped Arrow files and builds the loaders; a few
   augmented samples.
4. **Load model**: `model_builder.build_model` for each backbone (parameter breakdown) and
   `engine.throughput`: images/s of the loader and of a full training step on this GPU.
5. **Training**: 5a NutriText (vs TF-IDF kNN and the mean, plus zero-shot on the Nutrition5k names);
   5b `engine.train` once per backbone, training curves; 5c candidates: `engine.evaluate_views` (4 views per
   network on validation), `engine.fit_candidates` (readout per candidate), `engine.pick_candidate`.
6. **Evaluation**: display threshold fitted on validation; the test split once (overhead scans, side frames,
   side dishes, `test_clean`, original labels), bootstrap CI, ablation, every candidate on test, the graphs.
7. **Inference**: `export.assemble` (one `CalorieLens`) -> save -> `model.load(dir,
   device)`; asserts that its forward pass reproduces the readout scored on validation, then `predict` /
   `analyze` / `calculate` on example photos, exactly what the Space runs.
8. **Export**: `export.build_metrics` + `export.export` write the model, `metrics.json`, the graphs, NOTICE
   (DINOv2 only) and the refreshed card to `cfg.model_dir`; optional upload to the Hub.

## Recipe (defaults in `src/config.py`)
| | |
|---|---|
| Input | any photo fitted (aspect kept) into a 378 x 504 canvas (4:3; 27 x 36 DINOv2 patches) |
| Backbones | E1: `convnext_base` from timm's ImageNet-22k checkpoint (`convnext_base.fb_in22k_ft_in1k_384`, Apache-2.0, renamed into torchvision's layout; drop-path 0.1) · E2: `facebook/dinov2-with-registers-base` (Apache-2.0) · plus the shipped v1 model (torchvision ImageNet-1k ConvNeXt-B) as a candidate |
| Head | 253 ingredient queries, 2 cross-attention layers (d 256, no query self-attention), per-ingredient presence + log-grams outputs; totals head on pooled features of two depths (ConvNeXt stages 3 + 4, DINOv2 CLS + mean patch); output biases from the training data |
| Fusion | grams = sigmoid(logit) x grams, rescaled to the plate mass (direct head blended in log space with the grams' sum), tilted toward the direct head's kcal per gram (minimum-KL exponential tilt, same mass, rows still add up), the rest optionally closed by a rescale -> frozen 253 x 8 NutritionLayer. The three readout weights are fitted on validation |
| Losses | ASL presence, log-grams of present ingredients, soft grams vs truth, totals head in log space (0.5) and on the grams / kcal scale (1.0, normalised by the train mean: the metric's scale), NutritionLayer totals vs truth (end to end); side frames count 0.5 on the mass losses |
| Sampling | 12,000 samples per epoch: 15 % Flavoria plates (about their share of distinct meals), the rest half Nutrition5k overhead scans (any rotation), half side frames (5 per camera, mirrored, no zoom); within each pool every image equally likely |
| Flavoria | each plate trains only what it labels: presence + grams of the ingredients that map to one plain food, and the mass of the totals head; never the soft-grams or NutritionLayer losses (its composition is incomplete). Validation stays Nutrition5k only |
| Optimisation | AdamW (fused), layer-wise LR decay (ConvNeXt 1e-4 x 0.8 per stage, DINOv2 6e-5 x 0.75 per layer), head 1e-3, 1 warmup epoch + cosine, wd 0.05, EMA 0.999, clip 1.0, batch 64, <= 40 epochs, patience 8 |
| Speed-ups | bf16 autocast, TF32, cuDNN autotune, channels-last (ConvNeXt), `torch.compile` (eager fallback), uint8 decoding in the workers + augmentation on the GPU, memory-mapped Arrow data, 64-image steps on >= 70 GB GPUs (32 x 2 accumulation below; batches split further on out-of-memory) |
| Selection | validation score = mean of the overhead and side-dish calorie % errors; best EMA epoch per run; then every candidate (each network and the v1 model x 1 / 2 / 4 test-time views, each with its own mass blend / kcal blend / kcal scale fitted by coordinate descent) on validation: the lowest score ships, with its paired plate-group bootstrap CI against the runner-up |
| NutriText | own BPE tokenizer (8,000) + 4-layer d-256 transformer, ~470k USDA names (branded x 0.3), AdamW 1e-3, batch 512, <= 20 epochs, physically bounded outputs |

## Run it
1. Add a Colab secret `HF_TOKEN` with write access (and notebook access).
2. Open the notebook on a **GPU runtime** (A100 / H100 / G4), set `PUSH_TO_HUB = True` to upload the result,
   and *Run all*.
   - **First run:** section 3a builds the dataset before training, because `shalev396/calorie-lens-data`
     is not on the Hub yet. It streams ~195 GB from Google Cloud Storage, keeps ~8 GB and uploads shards every
     30 min; the probe prints the projected time. If the runtime dies, run it again: the build resumes from
     the manifest on the Hub.
   - **Flavoria** (once, while `flavoria_sessions.parquet` is not in the dataset): downloads the 2.6 GB image
     archive from Zenodo in 16 parallel ranges (resumable), keeps the cleaned plates (~100 MB) and adds them to
     the dataset repo with its card section.
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
| Flavoria build (3,345 trays, 2.6 GB download) | the download only (~120 KB/s per stream from Zenodo here; 16 streams) | ~10-30 min on Colab, depending on Zenodo |
| full notebook (NutriText + E1 + E2 + candidates) | not practical | ~3 h estimated on a 40 GB+ Colab GPU: 40 epochs at the v1 per-epoch times (ConvNeXt 90 s, DINOv2 146 s, micro-batch 32, peak 30 GB) + ~15 min of 4-view evaluation. v1 (30 epochs + a depth run) took 3.1 h. A 22 GB L4 runs out of memory in DINOv2's throughput probe |

## Data
- **[Nutrition5k](https://github.com/google-research-datasets/Nutrition5k)** (Thames et al., CVPR 2021, CC BY 4.0),
  anonymous HTTPS: 5,006 dishes, 3,490 overhead RGB-D scans, 18,486 side-angle H.264 videos (1080p, ~31-41 frames).
- **[FlavoriaFoodWeight1700](https://zenodo.org/records/5850856)** (Sarapisto et al., University of Turku, Zenodo 2022,
  CC BY 4.0): an automated lunch line photographs each tray from above (1024 px, plate cropped to a circle) and
  its scales weigh every serving: 12,284 weighings in 3,345 sessions, 246 component names, no calories.
  `src/flavoria.py` keeps the "identification" photo (the tray at rest at the RFID reader; the "enter" photos often
  show a hand), matched to its session by tray and time to the second (the CSVs mix UTC with Finnish summer /
  winter time), and drops trays with soup or dessert (served beside the plate, outside the photo: 95), scale
  glitches (one component over 600 g: 378) and plates over 1,200 g or under 30 g (384, nearly all a single tiny
  weighing). `labels/flavoria_component_map.csv` (reviewed by hand) maps 52 component names to a Nutrition5k
  ingredient where one plain food matches (rice, boiled potato, tomato, cucumber, carrot, ...); mixed dishes
  count in the plate weight only. Unweighed self-service sauces and bread make weights slight lower bounds.
  Train / test by lunch day (every 5th day is a test day).
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
  `assets/*.png` (training curves, candidates, ablation, calorie scatter, error by plate size, NutriText,
  heatmaps), the Flavoria test plate-weight error (`test.flavoria`), `NOTICE` + `LICENSE-APACHE-2.0.txt` if a shipped backbone came from Apache-2.0 weights (DINOv2, the
  timm ConvNeXt), and a refreshed card (YAML, metrics table, Experiments table). Update the card's prose and the project README with the numbers by hand.
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
| `data_setup.py` | download, Arrow conversion, labels, Dataset, mixture sampler (size-weighted), GPU augmentation, examples |
| `model_builder.py` | builds `model.CalorieLens` with output-bias priors, depth head, training wrapper, param groups, micro-batch |
| `engine.py` | losses (masked to the labels each sample has), training loop (EMA, resume, compile, OOM splitting), per-view evaluation, metrics (+ Flavoria plate weight), readout / threshold fitting, candidates (network x views), bootstrap, model selection |
| `nutritext.py` | stage 2b: data, tokenizer, training, baselines, zero-shot test |
| `export.py` | assembles the deployed model, metrics.json, NOTICE, graphs, model card |
| `flavoria.py` | FlavoriaFoodWeight1700: weighing events -> cleaned plates, photo matching, parallel Zenodo download, shards + tables + card section |
| `utils.py` | the same file in every ml-lab project (device, seeding, JSON, card updater, Hub upload) |
