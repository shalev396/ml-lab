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
| **Architecture** | Two models fused: an image backbone (DINOv2-B, picked over ConvNeXt-B on validation) with 253 ingredient queries whose last layer is 253 gram neurons, feeding a frozen per-gram nutrition layer; plus NutriText, a food-name transformer trained from scratch |
| **Dataset** | [Nutrition5k](https://github.com/google-research-datasets/Nutrition5k) (CC BY 4.0, cleaned: 5,005 dishes, 253 ingredients, overhead scans + side-angle frames), [FlavoriaFoodWeight1700](https://zenodo.org/records/5850856) (CC BY 4.0: 856 weighed lunch plates from a Finnish cafeteria) and [USDA FoodData Central](https://fdc.nal.usda.gov/) (public domain, ~470k food names), published as [shalev396/calorie-lens-data](https://huggingface.co/datasets/shalev396/calorie-lens-data) |
| **Result** | **16.01 % calorie error** (40.9 kcal) on the official overhead test split, 95 % CI 14.1-17.9 % (v1: 18.35 %; the paper's RGB model: 26.1 %, its RGB-D model: 18.8 %, best published RGB-only: 14.1 %); plate mass 12.6 % |
| **License** | MIT, Copyright (c) 2026 Shalev Ben Moshe; the backbone was fine-tuned from DINOv2 weights (Apache-2.0), so the model repo ships their `NOTICE` |
| **Runs on** | Space: ZeroGPU (0.2 s per photo with a GPU attached, up to ~2.5 s when ZeroGPU hands one over) with a CPU fallback · Training: 2.9 h on a Colab GPU with 40 GB+ of memory (a 22 GB L4 runs out of memory on DINOv2), plus the dataset build in the first run |

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

**FlavoriaFoodWeight1700** (University of Turku, 2022): an automated lunch line in Finland photographs each
tray from above and weighs every serving taken. It adds the large plates Nutrition5k lacks (median 472 g, 138
plates over 600 g) and a second kind of kitchen and camera. What it took (`training/src/flavoria.py`):
- **Labels are weight only.** For 52 plain foods (rice, potatoes, tomato, cucumber, ...) the grams map to a
  Nutrition5k ingredient (`training/labels/flavoria_component_map.csv`, reviewed by hand); mixed dishes count
  in the plate weight only. Missing targets are masked out of the loss, not treated as zero.
- **Cleaning, of 3,345 weighing sessions:** soups and desserts are served outside the photo (95 dropped);
  some weighings are scale glitches of 1-2 kg from one serving (378 trays with a component over 600 g
  dropped); near-empty trays and trays over 1.2 kg (384 dropped). 856 plates matched an identification photo
  (tray RFID + time to the second; the other photo taken at the line often shows a hand).
- **Split by lunch day** (every 5th of 32 days is a test day): 555 train, 301 test plates. 15 % of the
  training samples.

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
grams_k = sigmoid(logit_k) x grams_k, rescaled to the plate mass, tilted toward its kcal per gram ──► 253 neurons
253 neurons ─► NutritionLayer: frozen 253 x 8 per-gram table ─► nutrients per ingredient + totals
food name ─► BPE tokens ─► NutriText (4-layer transformer) ─► nutrients per 100 g
```
- **Backbones** (trained with the same recipe): ConvNeXt-B from timm's ImageNet-22k weights (converted into
  torchvision's layout) and DINOv2-B with registers (self-supervised, Apache-2.0). The totals head pools two
  scales: ConvNeXt's last two stages, or DINOv2's CLS token plus its mean patch token.
- **Why queries:** each ingredient has its own learned query and its own output weights, which also gives
  the per-ingredient heatmaps in the Space.
- **The readout** (all settings fitted on validation):
  - **Rescale:** a model that sums 253 soft gram guesses drifts; one that predicts the plate's mass directly
    is steadier. The grams are rescaled to a log-space blend of the two, and a user-supplied weight replaces
    it exactly.
  - **Tilt:** the grams are then tilted toward a kcal-per-gram target that blends the totals head's with the
    ingredients' own. Denser foods gain grams and lighter ones lose them, at the same plate mass.
  - **Views:** the photo and its mirror image are averaged (1, 2 and 4 views were tried).
- **NutriText's outputs are physical by construction:** composition is a softmax x 100 g (never more than
  100 g), sugars <= available carbs, saturated fat <= fat, and kcal stays within -26 % / +35 % of the Atwater
  energy of the composition.

## Training & experiments
| Experiment | What it tests |
|---|---|
| NutriText vs TF-IDF kNN vs mean | does a from-scratch transformer beat nearest-neighbour lookup on unseen USDA foods and on the 253 Nutrition5k names? |
| E1 ConvNeXt-B (ImageNet-22k) | supervised backbone, pretrained on 14M images |
| E2 DINOv2-B (registers) | self-supervised backbone with stronger geometry features |
| v1 model as a candidate | is the new training better than just the new readout on the old network? |
| Ablations (every model, free) | totals head alone vs summed grams vs rescaled vs tilted vs averaged views |

Recipe: AdamW with layer-wise learning-rate decay, warmup + cosine, bf16, EMA, 40 epochs with early stopping;
augmentation on the GPU (any rotation and flips for overhead scans, which keeps the fixed camera height as a
scale cue; flips for side frames, with no zoom so their scale cue stays too). The totals head is trained on
both log error and absolute error, so a miss on a big plate costs in proportion to its calories. Validation picks every
checkpoint, the candidate (lowest score, with a paired plate-group bootstrap CI reported), the readout, the
number of views and the display threshold. The test split is looked at once, at the end. Details:
[training/README.md](training/README.md).

## Results
Official overhead test split (507 scans), percent MAE = MAE / mean of the truth:

| | kcal | mass | fat | carbs | protein |
|---|---|---|---|---|---|
| percent error | **16.01 %** | 12.64 % | 23.32 % | 21.65 % | 21.72 % |
| absolute error | 40.9 kcal | 25.1 g | 3.0 g | 4.3 g | 3.7 g |
| v1, percent error | 18.35 % | 14.29 % | 24.26 % | 22.37 % | 27.89 % |

- **Against published results:** the Nutrition5k paper's RGB model gets 26.1 % and its RGB-D model 18.8 %;
  the best published RGB-only result is 14.1 %. The original labels give the same 16.01 % on these scans
  (the label fixes barely touch these scans), so the comparison is direct.
- **Other views:** 15.9 % when all side-angle frames of a dish are averaged, 16.3 % on `test_clean` (plates
  never seen in training).
- **Experiments:** validation picked DINOv2-B (score 14.92 vs ConvNeXt-B 15.17, inside the CI; on test 16.01 %
  vs 16.11 %). The v1 network with the new readout reaches 17.46 %: the readout is worth about 0.9 points and
  the new training about 1.5 more.

![Training loss and validation score of every experiment](model/assets/training_curves.png)

![Predicted vs true kcal on the overhead test scans](model/assets/kcal_scatter.png)

- **By plate size:** 12 kcal error under 100 kcal; over 600 kcal (38 dishes) the error fell from 167 to
  146 kcal, but those plates are still underestimated.
- **Another cafeteria:** on the 301 Finnish test plates (lunch days never trained on), the plate-weight error
  fell from 45.4 % (v1, 207 g light on average) to 25.5 %, and on plates over 600 g from 410 to 86 g. The
  totals head alone (`direct_estimate`) does better still there: 16.1 % with almost no bias. The itemised
  readout runs 104 g heavy on these photos because the model marks dozens of ingredients present: Flavoria
  labels say which foods are on a plate but never which are absent.
- **Ablation:** the totals head alone scores 15.7 %, the itemised readout 16.0 % (in v1: 16.4 vs 18.35 %), the
  plain sum of the soft grams 21.9 %. The ingredient list now costs about 0.3 points of calorie accuracy, and
  the totals head's own estimate is returned alongside it (`direct_estimate`).
- **Ingredients:** presence mAP 0.69, F1 0.75; 10.1 g error per present ingredient.
- **NutriText** on 633 unseen USDA foods: kcal R² 0.89 vs 0.78 for TF-IDF nearest neighbours (31.2 vs
  47.4 kcal per 100 g), better on all eight nutrients; zero-shot on the 253 Nutrition5k names, 52 vs 87 kcal
  per 100 g.

![Calorie error of each way to read the model](model/assets/ablation.png)

![Where each ingredient's query looked](model/assets/heatmaps.png)

![NutriText vs a TF-IDF nearest-neighbour lookup](model/assets/nutritext.png)

## Versions
| | v1 (2026-10-12) | v2 (2026-10-13) |
|---|---|---|
| Backbone | ConvNeXt-B (ImageNet-1k) | DINOv2-B (picked over ConvNeXt-B ImageNet-22k) |
| Data | Nutrition5k | + 555 Finnish lunch plates (weight only) |
| Readout | grams rescaled to the plate mass | + tilt toward the kcal per gram, 2 averaged views |
| Training | 30 epochs, depth head tried (no gain) | 40 epochs, totals head also on absolute error, no side-frame zoom |
| Test calorie error | 18.35 % | **16.01 %** |
| Finnish plate weight error | 45.4 % | 25.5 % (totals head: 16.1 %) |

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
└── training/   GitHub only: notebook.ipynb + src/ (labels, USDA, Flavoria, dataset build, data, model, engine, NutriText, export)
                labels/ingredient_fdc_map.csv (reviewed ingredient -> USDA match)
                labels/flavoria_component_map.csv (reviewed Flavoria component -> ingredient map)
```

## Reproduce
1. Colab secret `HF_TOKEN` with write access and *Notebook access* switched on; open the notebook on a GPU
   runtime, `PUSH_TO_HUB = True`, *Run all*. The first run builds and publishes the dataset before training
   (section 3a); later runs download it.
2. Smoke test anywhere: `SMOKE_TEST=1 BUILD_DATA=1 jupyter nbconvert --to notebook --execute training/notebook.ipynb --output-dir /tmp`

## Limitations
- The Nutrition5k photos come from two cafeterias with a fixed overhead camera, and the Finnish plates add one
  more, for weight only: the model's sense of scale is tied to those setups and to 253 ingredients. On home
  photos the grams are less reliable unless the food's weight is entered.
- On photos that don't look like Nutrition5k the ingredient list is unreliable (a Finnish rice-and-ratatouille
  plate comes out as lasagna, mashed potatoes, pasta, rice, ...); the plate weight in `direct_estimate` holds
  up better.
- Oil, butter, dressings and sauces are largely invisible; the model can only infer them from appearance.
- Only kcal, fat, carbs and protein are Nutrition5k's own values; fiber, sugars, saturated fat and sodium come
  from the matched USDA food. NutriText estimates for unfamiliar names are approximate, and some are off
  (it gives grilled salmon 6.2 g of carbs per 100 g).
- Plates over ~600 kcal are still underestimated.
- Corrected-label results are not directly comparable with published numbers; both are reported.
- Not a medical device: for a rough idea, not for dietary decisions.
