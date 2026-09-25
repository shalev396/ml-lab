# 🍔 FoodVision Big

> Upload a food photo and an EfficientNet-B2, fine-tuned end to end on all of Food-101, names the dish out of 101 and shows its top-5 guesses.

<p>
  <a href="https://huggingface.co/spaces/shalev396/foodvision-big"><img alt="Space" src="https://img.shields.io/badge/🤗%20Space-live%20demo-blue"></a>
  <a href="https://huggingface.co/shalev396/foodvision-big"><img alt="Model" src="https://img.shields.io/badge/🤗%20Model-shalev396%2Ffoodvision--big-yellow"></a>
  <a href="https://colab.research.google.com/github/shalev396/ml-lab/blob/main/foodvision-big/training/notebook.ipynb"><img alt="Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>

| | |
|---|---|
| **Task** | 101-class image classification (Food-101 dishes) |
| **Framework** | PyTorch + torchvision (`PyTorchModelHubMixin`, safetensors) |
| **Architecture** | EfficientNet-B2 from ImageNet weights, head `Dropout(0.3) -> Linear(1408, 101)`, every layer fine-tuned (7,843,303 parameters) |
| **Dataset** | [Food-101](https://huggingface.co/datasets/ethz/food101): 75,750 train / 25,250 test images |
| **Result** | top-1 accuracy = 0.8746 (all 25,250 test images, original training run) · local re-check on 1,010 test images: top-1 0.879, top-5 0.975 |
| **Runs on** | Space: CPU basic · Training: GPU (Colab or local); a CPU runs the smoke test |

## The problem
Given one photo of a dish, say which of 101 common dishes it is: apple pie, bibimbap, ramen, tiramisu
and 97 others. It is the "big" sibling of [FoodVision Mini](../foodvision-mini) (3 classes, 450
images): same backbone, but 101 classes, the full dataset, and the whole network fine-tuned instead
of only a new head. Many classes look alike (steak, filet mignon, prime rib, pork chop), so the Space
shows the top 5 guesses, not just one.

## The data
- [Food-101](https://huggingface.co/datasets/ethz/food101) (Bossard et al., 2014): 101 dishes, 1,000
  photos each from foodspotting.com, split 750 train / 250 test per class (75,750 / 25,250). The
  training photos keep some label noise on purpose; the test photos were cleaned by hand. There is
  no separate validation split.
- Photos are rescaled so their longer side is at most 512 px. Every image is resized to 288
  (bicubic), center-cropped to 288x288 and normalized with ImageNet mean/std, the preprocessing
  EfficientNet-B2 was pretrained with. Training images also get `TrivialAugmentWide` (a random
  colour/geometry operation per image).
- The notebook downloads the Hub parquet files once (~5 GB) into `training/data/`. Smoke runs read
  only a few 100-image parquet row groups over HTTP. Fallback: `torchvision.datasets.Food101`.
- The Hub dataset's label order puts `cheesecake` before `cheese_plate`; the model uses torchvision's
  alphabetical order, so labels are remapped by name.

## Architecture
torchvision `efficientnet_b2`, initialised from `EfficientNet_B2_Weights.IMAGENET1K_V1`, with the
1000-class ImageNet head replaced by `Dropout(0.3) -> Linear(1408, 101)`. All 7,843,303 parameters are
trained. The network, its preprocessing (`get_transform()`), `load(dir, device)` and
`Predictor.predict(image, top_k=None)` live in one file, [`model/model.py`](model/model.py), which
training, the Space and the Inference Endpoint share. The output is a softmax over all 101 dishes.

## Training & experiments
Recipe (the original bootcamp `foodvision_big/train.py`): `Adam(lr=1e-4)`, cross-entropy with label
smoothing 0.1, batch 32, 5 epochs, seed 42, AMP (fp16) and `torch.compile` on one RTX 2080 Ti, keep the
epoch with the lowest test loss. The run took about 22 minutes (~4 min per epoch, train + test).

| epoch | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| train accuracy | 0.546 | 0.734 | 0.787 | 0.820 | 0.843 |
| test accuracy | 0.806 | 0.849 | 0.861 | 0.865 | **0.875** |
| test loss (label-smoothed) | 1.445 | 1.300 | 1.248 | 1.226 | **1.202** |

Values are from the run's TensorBoard log (per-epoch means of per-batch values). Test loss fell every
epoch, so epoch 5 is the checkpoint deployed now. Train accuracy stays below test accuracy because
training images are augmented and dropout is active; both were still improving at epoch 5, so a longer
run would probably gain a bit more.

![Training curves of the original run](model/assets/training_curves.png)

The checkpoint was converted 1:1 from the original `.pth` into `FoodVisionNet` + safetensors, with
identical probabilities on the example images. The notebook can retrain the recipe (a GPU job); it
then evaluates the new run and the deployed checkpoint per image on the full test split and deploys
the better one. That retrain has not been run yet, so the Experiments table in the
[model card](model/README.md) has one variant.

## Results
- **Top-1 accuracy 0.8746** on all 25,250 test images, from the original run's log (mean of per-batch
  accuracies; per-image accuracy can differ by at most ~0.1 point).
- **Local re-check** of the converted weights, per image on CPU, on 1,010 test images (101 row groups
  spread over the Hub test split, all 101 classes present): top-1 **0.879** (888/1,010), top-5 **0.975**,
  macro F1 0.869. That is within the slice's ±1-point sampling error of the logged number.
- Most frequent mistakes on the slice: filet mignon -> steak, pork chop <-> steak, prime rib -> steak,
  ice cream -> chocolate mousse, lobster bisque -> clam chowder.

![Test accuracy: original log vs local re-check](model/assets/comparison.png)
![Confusion matrix on the test slice](model/assets/confusion_matrix.png)

## Deployment
- **Model repo**: https://huggingface.co/shalev396/foodvision-big (weights, `model.py`, `handler.py`)
- **Space / API**: https://huggingface.co/spaces/shalev396/foodvision-big — `POST /gradio_api/call/predict`
  (image in, `[top-5 label, seconds, device]` out; curl and `@gradio/client` examples in [`space/README.md`](space/README.md))
- **Inference Endpoint**: deploy the model repo from its page (Deploy → Inference Endpoints). `handler.py`
  accepts raw image bytes or a base64 string (optional `parameters.top_k`) and returns `{dish: probability}`;
  it uses `cuda` when the endpoint has a GPU.

## Project structure
```
foodvision-big/
├── README.md                 this write-up
├── model/                    HF model repo (git submodule) shalev396/foodvision-big
│   ├── model.py              FoodVisionNet + get_transform() + load() + Predictor
│   ├── handler.py            Inference Endpoint entry point (PIL / bytes / base64 / path, top_k)
│   ├── requirements.txt      deps of model.py + handler.py
│   ├── model.safetensors     weights · config.json (101 class_names, dropout, pretrained)
│   ├── metrics.json          deployed metrics, original-run history, slice re-check
│   ├── assets/               training_curves, comparison, confusion_matrix, per_class_accuracy (.png)
│   └── README.md             model card
├── space/                    HF Space (git submodule) shalev396/foodvision-big
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
cd foodvision-big/training
SMOKE_TEST=1 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp   # ~1.5 min sanity run on CPU
jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp --ExecutePreprocessor.timeout=-1   # full run: needs a GPU
```
Or open the notebook in [Colab](https://colab.research.google.com/github/shalev396/ml-lab/blob/main/foodvision-big/training/notebook.ipynb)
with a GPU runtime and *Run all* (it clones the repo itself). The full run writes new weights to `model/`
only if they beat the deployed checkpoint, and always refreshes `metrics.json`, the graphs and the card.
Run the app locally with `cd space && python app.py` (it loads `../model`).

## Limitations
- **101 dishes only.** Every photo gets one of the 101 labels, even a photo of a dog: there is no "other" class or rejection threshold.
- **Look-alike dishes** (the steak family, soups, chocolate desserts) cause most errors; the top-5 list helps there.
- **No validation split.** The recipe keeps the lowest-test-loss epoch, so the test number is slightly optimistic (it was the last epoch anyway).
- **Logged, not re-measured, headline.** The 0.8746 comes from the original run's log; the local check covers 1,010 of the 25,250 test images. A full per-image re-evaluation takes about 30 min on a CPU (snippet in [`training/README.md`](training/README.md)).
- **Domain.** Food-101 photos only; packaged food, drawings or dishes outside the 101 classes are untested.
