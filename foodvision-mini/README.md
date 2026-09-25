# 🍣 FoodVision Mini

> Upload a food photo and a frozen ImageNet EfficientNet-B2 with a small trained head tells you whether it shows pizza, steak or sushi.

<p>
  <a href="https://huggingface.co/spaces/shalev396/foodvision-mini"><img alt="Space" src="https://img.shields.io/badge/🤗%20Space-live%20demo-blue"></a>
  <a href="https://huggingface.co/shalev396/foodvision-mini"><img alt="Model" src="https://img.shields.io/badge/🤗%20Model-shalev396%2Ffoodvision--mini-yellow"></a>
  <a href="https://colab.research.google.com/github/shalev396/ml-lab/blob/main/foodvision-mini/training/notebook.ipynb"><img alt="Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>

| | |
|---|---|
| **Task** | 3-class image classification (pizza / steak / sushi) |
| **Framework** | PyTorch + torchvision (`PyTorchModelHubMixin`, safetensors) |
| **Architecture** | EfficientNet-B2 feature extractor: frozen ImageNet backbone + `Dropout(0.3) -> Linear(1408, 3)` (7,705,221 parameters, 4,227 trained) |
| **Dataset** | [Food-101](https://huggingface.co/datasets/ethz/food101) pizza / steak / sushi 20% subset ([zip](https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip)): 450 train / 150 test images |
| **Result** | accuracy = 0.960 (test, 144/150) · macro F1 = 0.961 |
| **Runs on** | Space: CPU basic · Training: CPU, GPU or Colab |

## The problem
Given a single photo of a dish, say which of three foods it is: pizza, steak or sushi. It is the
"mini" version of a food classifier: few classes and few images, so it shows how far transfer
learning goes when you only have a few hundred labelled photos and a CPU. The deployed model answers
in about 0.1-0.2 s per image on a CPU.

## The data
- `pizza_steak_sushi_20_percent` from the PyTorch Deep Learning course: a random 20% sample of the
  pizza, steak and sushi classes of Food-101 (user-uploaded restaurant and home photos, some label noise).
- **Train:** 450 images (pizza 154, steak 146, sushi 150). **Test:** 150 images (46 / 58 / 46).
  There is no separate validation split.
- Raw photos are up to 512 px on a side (median 512x512, smallest 289x262). Every image is resized to 288
  (bicubic), center-cropped to 288x288 and normalized with ImageNet mean/std, the exact preprocessing
  EfficientNet-B2 was pretrained with. No augmentation.
- The notebook downloads the zip once into `training/data/` (GitHub URL, with the raw.githubusercontent.com
  CDN as fallback).

## Architecture
torchvision `efficientnet_b2`, initialised from `EfficientNet_B2_Weights.IMAGENET1K_V1`. All
convolutional `features` are frozen, and the 1000-class ImageNet head is replaced by
`Dropout(0.3) -> Linear(1408, 3)`. Only those 4,227 head parameters are trained. The network, its
preprocessing (`get_transform()`), `load(dir, device)` and `Predictor.predict(image)` all live in one
file, [`model/model.py`](model/model.py), which training, the Space and the Inference Endpoint share.
The output is a softmax over all three classes: `{"pizza": p, "steak": p, "sushi": p}`.

## Training & experiments
Recipe (the original bootcamp notebook 09): plain cross-entropy, `Adam(lr=1e-3)` over the head,
batch 32, 10 epochs, seed 42, whole net in `train()` mode so the frozen backbone's BatchNorm statistics
adapt to food photos, last epoch kept.

The notebook trains the recipe from scratch, evaluates it next to the checkpoint that is deployed now,
and exports the better one (ties keep the deployed one). Two variants were evaluated on the same 150
test images ([`model/metrics.json`](model/metrics.json) → `comparison`):

| variant | accuracy | macro F1 | pizza | steak | sushi |
|---|---|---|---|---|---|
| **original bootcamp checkpoint (deployed)** | **0.960** | **0.961** | 0.957 | 0.948 | 0.978 |
| retrain with `training/` (desktop CPU, 2026-09-25) | 0.940 | 0.941 | 0.978 | 0.983 | 0.848 |

- The deployed weights come from the original bootcamp run (Apple MPS, published 2026-05-08),
  converted 1:1 to safetensors (identical probabilities on the example images). Its training time
  and curves were not recorded.
- The retrain took 857 s of training on a CPU shared with other jobs (an earlier run on 2026-09-24
  took 520 s and landed on exactly the same 0.940). It is 3 images lower (141 vs 144), which is within
  the noise of a 150-image test set, so the original stays deployed. The retrain misses more sushi
  (7 of 46), the original misses more steak (3 of 58).

![Test metrics of every variant](model/assets/comparison.png)
![Training curves of the retrain](model/assets/training_curves.png)

## Results
The deployed model gets **144 of 150** test images right: accuracy 0.960, macro F1 0.961. Per class:
pizza 0.957 (44/46), steak 0.948 (55/58), sushi 0.978 (45/46). Half of the six errors are steak photos
predicted as sushi. The bootcamp notebook printed 96.25% for this model; that figure is a mean of
per-batch accuracies (the last batch holds only 22 images), while 96.0% is the per-image accuracy.

![Confusion matrix of the deployed model](model/assets/confusion_matrix.png)

## Deployment
- **Model repo**: https://huggingface.co/shalev396/foodvision-mini (weights, `model.py`, `handler.py`)
- **Space / API**: https://huggingface.co/spaces/shalev396/foodvision-mini — `POST /gradio_api/call/predict`
  (image in, `[label, seconds, device]` out; curl and `@gradio/client` examples in [`space/README.md`](space/README.md))
- **Inference Endpoint**: deploy the model repo from its page (Deploy → Inference Endpoints). `handler.py`
  accepts raw image bytes or a base64 string and returns `{class: probability}`; it uses `cuda` when the
  endpoint has a GPU.

## Project structure
```
foodvision-mini/
├── README.md                 this write-up
├── model/                    HF model repo (git submodule) shalev396/foodvision-mini
│   ├── model.py              FoodVisionNet + get_transform() + load() + Predictor
│   ├── handler.py            Inference Endpoint entry point (PIL / bytes / base64 / path)
│   ├── requirements.txt      deps of model.py + handler.py
│   ├── model.safetensors     weights · config.json (class_names, dropout, pretrained)
│   ├── metrics.json          deployed metrics + comparison of every variant
│   ├── assets/               training_curves.png, comparison.png, confusion_matrix.png
│   └── README.md             model card
├── space/                    HF Space (git submodule) shalev396/foodvision-mini
│   ├── app.py                Gradio Blocks UI + the public /predict endpoint
│   ├── space_utils.py        shared ml-lab Space helpers
│   ├── examples/             3 test photos
│   └── README.md             Space config + API docs
└── training/                 GitHub only
    ├── notebook.ipynb        the controller: setup → config → data → model → train → evaluate → inference → export
    ├── src/                  config, data_setup, model_builder, engine, export, utils
    └── README.md             how to run, time on CPU/GPU, data sources
```

## Reproduce
```bash
git clone --recurse-submodules https://github.com/shalev396/ml-lab && cd ml-lab
pip install -r requirements.txt
cd foodvision-mini/training
SMOKE_TEST=1 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp   # ~1 min sanity run
jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp                # full run, ~10-15 min on CPU
```
Or open the notebook in [Colab](https://colab.research.google.com/github/shalev396/ml-lab/blob/main/foodvision-mini/training/notebook.ipynb)
and *Run all* (it clones the repo itself). The full run writes to `model/` only if the new run beats the
deployed checkpoint (weights), and always refreshes `metrics.json`, the graphs and the card.
Run the app locally with `cd space && python app.py` (it loads `../model`).

## Limitations
- **Three classes only.** Every image gets a pizza/steak/sushi label, even a photo of a dog: there is no "other" class or rejection threshold.
- **Small data.** 150 test images put roughly a ±1.6 percentage-point standard error on the accuracy; the two variants are not reliably different.
- **No validation split.** Curves are monitored on the test split and the deployed variant is chosen on it (two candidates, so the optimism is small but not zero).
- **Domain.** Food-101 photos only; packaged food, drawings or other cuisines are untested.
