# 🥗 Calorie Lens

> Photograph a meal and get every ingredient, its weight in grams, and the calories, protein, fat, carbs and more of each ingredient and of the whole plate.

<p>
  <a href="https://huggingface.co/spaces/shalev396/calorie-lens"><img alt="Space" src="https://img.shields.io/badge/🤗%20Space-live%20demo-blue"></a>
  <a href="https://huggingface.co/shalev396/calorie-lens"><img alt="Model" src="https://img.shields.io/badge/🤗%20Model-shalev396%2Fcalorie--lens-yellow"></a>
  <a href="https://huggingface.co/datasets/shalev396/calorie-lens-data"><img alt="Dataset" src="https://img.shields.io/badge/🤗%20Dataset-calorie--lens--data-orange"></a>
  <a href="https://colab.research.google.com/github/shalev396/ml-lab/blob/main/calorie-lens/training/notebook.ipynb"><img alt="Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>

| | |
|---|---|
| **Task** | Meal photo -> ingredients + grams (multi-label + regression) -> nutrition; food name -> nutrients |
| **Framework** | PyTorch |
| **Architecture** | Two models fused: an image backbone (ConvNeXt-B vs DINOv2-B, benchmarked) with 253 ingredient queries whose last layer is 253 gram neurons, feeding a frozen per-gram nutrition layer; plus NutriText, a food-name transformer trained from scratch |
| **Dataset** | [Nutrition5k](https://github.com/google-research-datasets/Nutrition5k) (CC BY 4.0, cleaned: 5,005 dishes, 253 ingredients, overhead scans + side-angle frames) and [USDA FoodData Central](https://fdc.nal.usda.gov/) (public domain, ~470k food names), published as [shalev396/calorie-lens-data](https://huggingface.co/datasets/shalev396/calorie-lens-data) |
| **Result** | Pending the first full training run (Colab GPU); the target is a calorie percent error below 18 % on the official overhead test split (paper's RGB baseline 26.1 %, best published RGB-only 14.1 %) |
| **License** | MIT, Copyright (c) 2026 Shalev Ben Moshe; if the DINOv2 backbone wins, its fine-tuned weights also carry Apache-2.0 (NOTICE in the model repo) |
| **Runs on** | Space: ZeroGPU with a CPU fallback · Training: Colab A100 / H100 / G4 (~1.6-2.7 h, plus ~1-2 h to build the dataset in the first run) |

## The problem
Counting calories from a photo has two hard parts: **what** is on the plate, and **how much** of it. The
second is the real bottleneck: in the Nutrition5k paper, knowing the true weight of a dish and predicting only
its composition gives a 9.5 % calorie error, while estimating everything from one photo gives 26 %. Vision
language models asked the same question land at 33-55 %.

Calorie Lens splits the problem the way a nutritionist would, inside one network:
1. **Stage 1** looks at the photo and outputs the grams of each of 253 ingredients.
2. **Stage 2a** turns grams into nutrients with a per-gram table. It is a linear layer whose 253 inputs are
   stage 1's 253 outputs, so the two stages train end to end on calories, fat, carbs and protein.
3. **Stage 2b**, NutriText, covers foods outside those 253: type any food and its grams.

Every number the model gives is explainable: an ingredient list that adds up to the totals, and a heatmap of
where it saw each ingredient. If the plate was weighed, the user can enter its weight and the scale error
disappears entirely.

## The data
**Nutrition5k** (Google, 2021): ~5,000 dishes from two Google cafeterias. Every ingredient was weighed on a
scale, and each plate was filmed by an overhead depth camera and four rotating side cameras. What this
project found and fixed (`training/src/n5k_labels.py`, all asserted in code):
- **The labels are exactly grams x a per-gram table**: for all 28,455 ingredient entries, the logged
  calories, fat, carbs and protein equal grams x `ingredients_metadata.csv`. So a model that gets the grams
  right gets the nutrition right, and stage 2a can be that exact table, frozen.
- **Label bugs:** Cafe 2's 238 dish totals are logged as 0 (recomputed from the ingredients); 10 ingredient
  weights are x1000 scale errors (the scale wrote 7974 for 7.974 g of olives); "plate only" is the empty
  plate's weight; two "deprecated" table rows are duplicates of apple (with carbs and protein swapped) and
  watermelon; some names are empty; one overhead photo shows a phone.
- **Plates were scanned once per added item**, so many dishes are earlier states of a later dish. Re-scans
  are linked into 2,346 plate groups; validation is drawn as whole groups, and `test_clean` marks the 565
  official test dishes whose plate never appears in training (144 do).
- **Images:** 3,490 overhead RGB-D scans and 5 frames from each side-angle video (~92k frames), streamed from
  ~195 GB of raw H.264 and published as an ~8 GB Hugging Face dataset with a full card.

**USDA FoodData Central** (public domain): ~13,600 generic foods (Foundation, SR Legacy, FNDDS) with ~10,700
alternative names, and ~448,000 branded products (label values, weighted 0.3). It trains NutriText and
supplies fiber, sugars, saturated fat and sodium for the 253 ingredients through a reviewed one-to-one match
(`training/labels/ingredient_fdc_map.csv`: candidates ranked by name and by agreement with each ingredient's
known macros, 44 of 253 corrected by hand).

Rejected on purpose: every other weighed-portion image set (NutritionVerse, Food Portion Benchmark) is
non-commercial, and the large "nutrition-labelled" photo sets (MM-Food-100K, CalorieBench) take their numbers
from an LLM.

## Architecture
```
photo ─► backbone ─► image tokens ─► 253 ingredient queries (2 cross-attention layers)
                  │                    ├─ presence logit_k
                  │                    └─ log-grams_k (if present)
                  └─► pooled ─► totals head ─► plate mass (+ kcal, fat, carbs, protein)
grams_k = sigmoid(logit_k) x grams_k, rescaled to the plate mass ──► 253 neurons
253 neurons ─► NutritionLayer: frozen 253 x 8 per-gram table ─► nutrients per ingredient + totals
food name ─► BPE tokens ─► NutriText (4-layer transformer) ─► nutrients per 100 g
```
- **Backbones** (benchmarked with the same recipe): torchvision ConvNeXt-B (ImageNet) and DINOv2-B with
  registers (self-supervised, Apache-2.0). The winner on validation is retrained with an **auxiliary depth
  head** that predicts the overhead depth map (training only), to sharpen its sense of volume.
- **Why queries:** each ingredient has its own learned query and its own output weights, which also gives
  the per-ingredient heatmaps in the Space.
- **Why rescaling:** a model that sums 253 soft gram guesses drifts; one that predicts the plate's mass
  directly is steadier. The grams are rescaled to a log-space blend of the two (the blend is fitted on
  validation), and a user-supplied weight replaces it exactly.
- **NutriText's outputs are physical by construction:** composition is a softmax x 100 g (never more than
  100 g), sugars <= available carbs, saturated fat <= fat, and kcal stays within -26 % / +35 % of the Atwater
  energy of the composition.

## Training & experiments
| Experiment | What it tests |
|---|---|
| NutriText vs TF-IDF kNN vs mean | does a from-scratch transformer beat nearest-neighbour lookup on unseen USDA foods and on the 253 Nutrition5k names? |
| E1 ConvNeXt-B | supervised ImageNet backbone |
| E2 DINOv2-B (registers) | self-supervised backbone with stronger geometry features |
| E3 winner + depth head | does predicting depth during training help portion size? |
| Ablations (every model, free) | totals head alone vs summed grams vs grams rescaled to the mass |

Recipe: AdamW with layer-wise learning-rate decay, warmup + cosine, bf16, EMA, early stopping; augmentation on
the GPU (any rotation and flips for overhead scans, which keeps the fixed camera height as a scale cue; flips
and mild zoom for side frames). Validation picks every checkpoint, the backbone (paired plate-group bootstrap;
a ConvNeXt within the CI wins the tie), the mass blend and the display threshold. The test split is looked at
once, at the end. Details: [training/README.md](training/README.md).

## Results
Pending the first full training run. The notebook reports MAE and percent MAE for kcal, mass, fat, carbs and
protein on the official overhead test split (corrected and original labels), side views, `test_clean`, a
bootstrap CI, every experiment and the ablation.

## Deployment
- **Model repo**: https://huggingface.co/shalev396/calorie-lens (weights, `model.py`, `handler.py`)
- **Space / API**: https://huggingface.co/spaces/shalev396/calorie-lens — `POST /gradio_api/call/predict`
  (photo + optional food weight -> `[result, seconds, device]`); a calculator tab for typed foods
- **Inference Endpoint**: deploy the model repo from its page (Deploy -> Inference Endpoints); the handler
  takes a photo or a `[[food, grams], ...]` list
- **Dataset**: https://huggingface.co/datasets/shalev396/calorie-lens-data

## Project structure
```
calorie-lens/
├── model/      -> HF model repo: model.py (all three stages + Predictor), handler.py, weights, metrics, card
├── space/      -> HF Space: app.py (photo tab + calculator tab), cpu_backend.py, space_utils.py, examples/
└── training/   GitHub only: notebook.ipynb + src/ (labels, USDA, dataset build, data, model, engine, NutriText, export)
                labels/ingredient_fdc_map.csv (reviewed ingredient -> USDA match)
```

## Reproduce
1. Colab secret `HF_TOKEN` with write access; open the notebook on a GPU runtime, `PUSH_TO_HUB = True`, *Run all*.
   The first run builds and publishes the dataset before training (section 3a); later runs download it.
2. Smoke test anywhere: `SMOKE_TEST=1 BUILD_DATA=1 jupyter nbconvert --to notebook --execute training/notebook.ipynb --output-dir /tmp`

## Limitations
- All photos come from two cafeterias with a fixed overhead camera: the model's sense of scale is tied to that
  setup and 253 ingredients. On home photos the grams are less reliable unless the food's weight is entered.
- Oil, butter, dressings and sauces are largely invisible; the model can only infer them from appearance.
- Only kcal, fat, carbs and protein are Nutrition5k's own values; fiber, sugars, saturated fat and sodium come
  from the matched USDA food. NutriText estimates for unfamiliar names are approximate.
- Corrected-label results are not directly comparable with published numbers; both are reported.
- Not a medical device: for a rough idea, not for dietary decisions.
