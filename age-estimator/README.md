# 🎂 Age & Gender Estimator

> Finds every face in a photo, estimates each person's age and gender, and shows which facial regions drove each estimate.

<p>
  <a href="https://huggingface.co/spaces/shalev396/age-estimator"><img alt="Space" src="https://img.shields.io/badge/🤗%20Space-live%20demo-blue"></a>
  <a href="https://huggingface.co/shalev396/age-estimator"><img alt="Model" src="https://img.shields.io/badge/🤗%20Model-shalev396%2Fage--estimator-yellow"></a>
  <a href="https://colab.research.google.com/github/shalev396/ml-lab/blob/main/age-estimator/training/notebook.ipynb"><img alt="Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>

| | |
|---|---|
| **Task** | Face detection + apparent-age estimation (regression via 90 age bins) + binary gender classification, with Grad-CAM explanations |
| **Framework** | PyTorch (torchvision, `PyTorchModelHubMixin`) |
| **Architecture** | Our face detector (from the face-recognition project) + EfficientNet-B2 with an age head and a gender head (7,830,622 parameters) |
| **Dataset** | [UTKFace](https://huggingface.co/datasets/nu-delta/utkface), 23,705 faces, random 90/10 split: 21,334 train / 2,371 validation |
| **Result** | age MAE = 4.59 years · 67.2 % within ±5 years · gender accuracy = 93.3 % (validation) |
| **Runs on** | Space: ZeroGPU (CPU fallback) · Training: CPU, GPU or Colab |

## The problem
Given any photo, find the people in it and say roughly how old each one looks and whether they
look male or female. It should also *show* its reasoning, so a viewer can see whether the model
looked at the face (wrinkles, eyes, hairline) or at something irrelevant.

## The data
[UTKFace](https://susanqq.github.io/UTKFace/) as published on the Hub by
[`nu-delta/utkface`](https://huggingface.co/datasets/nu-delta/utkface): 23,705 aligned and cropped
200×200 face photos, each labelled with age (1–116), gender and ethnicity. Ages are heavily skewed:
many infants and 20–35 year-olds, few people over 80. The training code downloads the 3 parquet
shards (~1 GB) once into `training/data/` and splits them 90/10 at random (seed 42).
UTKFace is licensed for **non-commercial research only**.

## Architecture
```
photo ──> FaceDetector (single-shot CNN)   ──> face boxes, score ≥ 0.5, most confident first
            └─ each box + 20 px context ──> resize 288 / crop 288 / ImageNet norm
                 └─ EfficientNet-B2 trunk (1408-d) ─┬─ age head: Dropout → Linear(1408, 90) → softmax over bins
                                                    │     age = Σ pᵢ · repᵢ, 80 % range, top-5 bins
                                                    └─ gender head: Dropout → Linear(1408, 2) → {male, female}
                 └─ Grad-CAM (blocks 5 + 8, geometric mean) ──> age / gender evidence heatmaps
```
- **Why bins instead of plain regression?** The softmax over 90 bins (`0-1`, every year `2`…`89`,
  `90+`) exposes the model's uncertainty (probability spread over neighbouring ages). The headline
  age is the expected value over the bins, each bin represented by the mean true age of its training
  faces. That is 0.74 years more accurate than taking the most likely bin.
- **One shared trunk:** a single forward pass gives both answers, and the face features help both tasks.
- **Face detector:** the `FaceDetector` trained from scratch in the
  [face-recognition](../face-recognition/) project (Open Images + LFW faces), reused as-is: the
  same code in `model.py` and the same `face_detector.safetensors`.
- Everything (architecture, preprocessing, detection, decoding, Grad-CAM, annotation) is in
  [`model/model.py`](model/model.py), used by training, the Space and the endpoint alike.

## Training & experiments
ImageNet-initialised EfficientNet-B2, fully fine-tuned with CE(age bins) + CE(gender), Adam 1e-3,
batch 64, 10 epochs, flips + mild colour jitter, mixed precision on CUDA. The epoch with the lowest
validation age MAE is kept (epoch 9 of 10 for the deployed weights).

| variant (validation, 2,371 faces) | age MAE (y) | within ±5 y | gender accuracy |
|---|---|---|---|
| **EfficientNet-B2, expected age (deployed)** | **4.59** | **67.2 %** | **93.3 %** |
| same weights, argmax bin | 5.33 | 65.1 % | 93.3 % |
| baseline: median age 29 + majority gender | 15.02 | 34.0 % | 51.5 % |

The deployed weights come from the project's original training run (same recipe) and were
converted 1:1 to safetensors. Re-evaluating them here reproduced the metrics stored in the checkpoint.

## Results
Age MAE 4.59 years (RMSE 6.80), 67.2 % of faces within ±5 years, gender accuracy 93.3 %, all on
the validation split. The error grows with age: 2.0 y for 0–12, 3.2 y for 20–29, about 7 y from 40 to 79.
Plots and the full table are on the [model card](https://huggingface.co/shalev396/age-estimator)
(`model/assets/`).

## Deployment
- **Model repo**: https://huggingface.co/shalev396/age-estimator (weights, `model.py`, `face_detector.safetensors`, `handler.py`)
- **Space / API**: https://huggingface.co/spaces/shalev396/age-estimator — `POST /gradio_api/call/predict`
  returns `[{"faces": [{box, age, age_range, age_top, gender, confidence, detected}], "n_faces"}, seconds, device]`
- **Inference Endpoint**: deploy the model repo from its page (Deploy → Inference Endpoints)

## Project structure
- [`model/`](model/) — Hugging Face model repo (git submodule): `model.safetensors`, `config.json`, `model.py`,
  `face_detector.safetensors`, `handler.py`, `metrics.json`, `assets/`, model card.
- [`space/`](space/) — Hugging Face Space (git submodule): Gradio app (upload or webcam, annotated photo,
  Grad-CAM toggle), `/predict` API, examples.
- [`training/`](training/) — `notebook.ipynb` (the controller) + `src/` (config, data, model builder, engine, export).

## Reproduce
```bash
pip install -r requirements.txt                         # from the ml-lab root
cd age-estimator/training
SMOKE_TEST=1 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp   # ~2 min sanity run
jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp                # full run (use a GPU)
cd ../space && python app.py                            # local Space on http://127.0.0.1:7860
```
Or open the notebook in [Colab](https://colab.research.google.com/github/shalev396/ml-lab/blob/main/age-estimator/training/notebook.ipynb).
A full run replaces `model/` (weights + metrics) only when its validation age MAE is lower than the deployed one; set `OVERWRITE_IF_WORSE = True` in the notebook to replace it anyway.

## Limitations
- It estimates **apparent** age from a single photo. A third of validation faces are off by more than 5 years.
- Gender is a binary male/female label assigned from appearance in UTKFace. It says nothing about gender identity.
- UTKFace is imbalanced in age and ethnicity. Performance on under-represented groups is not measured.
- Metrics are on the validation split that also selected the epoch (no separate test split).
- The detector can miss very small, profile or occluded faces. Then the whole image is classified, which is much less reliable.
- Not for age verification or any decision about a person. UTKFace allows non-commercial research use only.
