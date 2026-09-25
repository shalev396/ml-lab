# Face Recognition: training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shalev396/ml-lab/blob/main/face-recognition/training/notebook.ipynb)

Builds [`../model`](../model), every network of it: a face detector (trained on Open Images +
LFW) -> 5-point alignment of every LFW photo -> two face embedders trained with CosFace
(EfficientNet-B2 from ImageNet weights, ResNet-18 from scratch) -> MLP identity head on the
concatenated embeddings -> a Hugging Face model folder. No pretrained face network is part of the
model and no third-party code is used: the detector's landmark labels are a committed data file
(`labels/landmark_labels.json.gz`), made once with MTCNN (facenet-pytorch).

## What the notebook does
1. **Setup**: finds the project files (clones the repo on a fresh runtime), installs `../../requirements.txt`, picks the device (cuda -> mps -> cpu), imports `src/` and `../model/model.py`.
2. **Config**: `Config()` from `src/config.py`; `SMOKE_TEST` toggle; `PUSH_TO_HUB` flag.
3. **Data**: downloads LFW and makes the splits; downloads the Open Images face photos; reads the landmark labels; builds the detector's training and held-out records and shows a few.
4. **Load model**: builds `model.FaceDetector` and `model.FaceRecognizer` (one member per `cfg.embedders` entry + the head), prints parameter counts.
5. **Training**: trains the detector (best held-out epoch kept), aligns every LFW photo with it (cached), trains each embedder (CosFace, best validation epoch kept), then the head on the frozen embeddings (lowest validation loss kept), picks the verification threshold on validation pairs, shows the curves.
6. **Evaluation**: detection AP of ours (held-out Open Images, WIDER FACE val) next to the reference detector's recorded scores; accuracy / macro F1 / top-3 on train, validation and test; verification on the test pairs and on the unseen people; the experiments table and chart, confusion matrix, distance histogram, t-SNE.
7. **Inference**: saves the model, then `model.load(dir).predict / verify / embed` on `../space/examples`, exactly what the Space runs.
8. **Export**: `model.safetensors` + `config.json`, `metrics.json`, `assets/*.png`, model-card refresh; optional upload.

## Recipe (defaults in `src/config.py`)
| | |
|---|---|
| Data | LFW funneled, all 13,233 photos (5,749 people), downloaded by `sklearn.datasets.fetch_lfw_people` |
| Detector | `model.FaceDetector` (single-shot CNN, feature pyramid 8/16/32, 10 anchors, face / box / 5 landmarks), random init; 640 px crops / zoom-outs / 2x2 mosaics; AdamW lr 2e-3, wd 5e-4, batch 16, 30 epochs, cosine; hard-negative mining 3:1; best epoch by AP on 600 held-out Open Images photos |
| Detector data | Open Images V7 validation + test photos with "Human face" boxes (~12,000, CC BY) + 5,000 LFW training photos; landmarks from `labels/landmark_labels.json.gz`; group boxes and faces only the reference detector found are ignored |
| Alignment | our detector (most central large face) -> 5 landmarks -> similarity transform onto the standard template, 128 x 128 |
| Splits | the 42 people with >= 25 photos: 75/25 train/test per person (seed 42), 15% of train -> validation; 10% of the other people with >= 2 photos are held out ("unseen"); the embedders train on the 42 people's train photos + everyone else |
| Embedder 1 | EfficientNet-B2, torchvision ImageNet weights as the start, 128 px, 25 epochs |
| Embedder 2 | ResNet-18, random initialisation, 128 px, 80 epochs |
| Embedder training | `backbone -> Dropout(0.2) -> Linear(512) -> BatchNorm`; CosFace (s=30, m=0.35); AdamW lr 1e-3 (ImageNet backbone 0.3x), wd 5e-4, batch 128, 2 warm-up epochs + cosine; AMP on CUDA; GPU augmentation (zoom, shift, rotation, flip, colour, grayscale, erasing); best epoch by validation nearest-centroid accuracy + 0.1 x unseen-people AUC |
| Head | MLP `1024 -> 256 -> 42`, dropout 0.3, AdamW lr 1e-3, wd 1e-4, batch 64, up to 200 epochs, the epoch with the lowest validation loss kept |
| Verification | cosine distance < threshold = same person; threshold picked on validation pairs from the grid 0.2 to 1.2 (best balanced accuracy) |
| Experiments | `src/config.EXPERIMENTS`: the 11 embedders/ensembles that were compared (same code, splits and head; run on MTCNN-aligned faces before our detector existed) |

## Run it
- **Colab:** click the badge, *Runtime -> Change runtime type -> T4 GPU*, then *Run all*.
- **Local:** `pip install -r ../../requirements.txt`, then open `notebook.ipynb` from this folder.
- **Headless:** `SMOKE_TEST=1 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp`
  (60 Open Images photos, 6 known + 60 other LFW people, 1 detector epoch, 1-2 epochs of two tiny
  ResNet-18s, no WIDER; exports to `outputs/smoke/model`, never touches `../model`). Drop `SMOKE_TEST=1` for the full run, which overwrites `../model`'s
  weights, metrics and graphs.

## Hardware and time
| run | GPU (RTX 2080 Ti, measured) | CPU |
|---|---|---|
| downloads, first run (LFW 233 MB, Open Images ~1.2 GB, WIDER val 363 MB) | ~30-40 min | longer |
| full notebook | 191 min of training (detector + embedders + head) | many hours: use a GPU |
| smoke | seconds | ~1 min |

## Data
- **Source:** [LFW](http://vis-www.cs.umass.edu/lfw/) (funneled images) through scikit-learn, which
  downloads `lfw-funneled.tgz` (~233 MB) from figshare into `training/data/sklearn_lfw/`.
- **Fallback:** if the figshare download fails, `data_setup.lfw_dir` fetches the same archive from the
  LFW website (`http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz`).
- **Open Images** (`detector.download_open_images`): the box files from Google's bucket and every
  validation + test photo with a "Human face" box from the public S3 bucket, resized to <= 800 px,
  into `training/data/openimages/` (photos with rotation metadata are skipped).
- **WIDER FACE val** (`detector.wider_items`, evaluation only, CC BY-NC-ND): from the Hub dataset
  `CUHK-CSE/wider_face` into `training/data/wider_face/`. Set `cfg.eval_wider = False` to skip it.
- **Landmark labels:** `labels/landmark_labels.json.gz` (committed, 4 MB): for every Open Images
  and LFW training photo, the faces a reference detector (MTCNN, facenet-pytorch) found, as numbers
  (box, score, 5 points). Only the data is used; no third-party code or weights.
- **Cache:** `training/data/lfw_aligned_128.npz` (~650 MB) holds the aligned crops, whether a face
  was found and the names; it is keyed to the detector weights and recomputed when they change.

## Outputs
- Full run: `../model/` gets `model.safetensors` + `config.json`, `metrics.json` and
  `assets/{training_curves,model_comparison,confusion_matrix,verification_distances,tsne_embeddings}.png`;
  the metrics table and YAML in `../model/README.md` are refreshed by `utils.update_model_card`.
- Smoke run: the same files in `outputs/smoke/model/`, plus copies of the card, code and detector
  weights, so the folder is self-contained. From `space/`, `MODEL_DIR=../training/outputs/smoke/model python app.py` serves it.
- Publish: `PUSH_TO_HUB = True` in the notebook, or `utils.upload_model()`. Authenticate with
  `hf auth login` or an `HF_TOKEN` secret / env var.

## `src/` file map
| file | role |
|---|---|
| `__init__.py` | puts `../model` on `sys.path`, so `import model as M` is the deployed `model.py` |
| `config.py` | `@dataclass Config`: every hyperparameter, the embedders, smoke handling, cache and export paths; `EXPERIMENTS` |
| `data_setup.py` | LFW download (+ fallback), smoke subset, splits, alignment cache |
| `detector.py` | Open Images download, landmark labels, detector dataset + augmentation, loss with hard-negative mining, training loop, AP evaluation (held-out Open Images, WIDER FACE) |
| `model_builder.py` | detector, aligner, `FaceRecognizer` with ImageNet starts, parameter counts |
| `engine.py` | GPU augmentation, CosFace, embedder and head training loops, embeddings, classification and verification metrics |
| `export.py` | `metrics.json` builder, weights/metrics/plots writer, model-card refresh |
| `utils.py` | shared ml-lab helpers (devices, seeds, JSON, card update, Hub upload), identical in every project |
