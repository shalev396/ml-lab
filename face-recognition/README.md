# 🧑 Face Recognition

> Upload a photo and find out which of 42 public figures it shows, or check whether two photos show the same person.

<p>
  <a href="https://huggingface.co/spaces/shalev396/face-recognition"><img alt="Space" src="https://img.shields.io/badge/🤗%20Space-live%20demo-blue"></a>
  <a href="https://huggingface.co/shalev396/face-recognition"><img alt="Model" src="https://img.shields.io/badge/🤗%20Model-shalev396%2Fface--recognition-yellow"></a>
  <a href="https://colab.research.google.com/github/shalev396/ml-lab/blob/main/face-recognition/training/notebook.ipynb"><img alt="Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>

| | |
|---|---|
| **Task** | Face identification (42 people) + face verification (same person or not) |
| **Framework** | PyTorch |
| **Architecture** | Our face detector (single-shot CNN, face + 5 landmarks) + 5-point alignment + an ensemble of two face embedders trained with CosFace (EfficientNet-B2 from ImageNet weights, ResNet-18 from scratch) + MLP head 1024 -> 256 -> 42 (22.8M parameters, all trained here) |
| **Dataset** | [LFW](http://vis-www.cs.umass.edu/lfw/) funneled: 13,233 photos of 5,749 people. The 42 people with >= 25 photos are named (1,649 train / 292 val / 647 test); every other person trains the embedders; 163 people are held out entirely. The detector also learns from ~12,000 [Open Images](https://storage.googleapis.com/openimages/web/index.html) face photos |
| **Result** | test accuracy = 0.9969 (645 of 647 photos), macro F1 = 0.9961; verification ROC AUC 0.9973 on the 42 people, 0.9750 on never-seen people; face detection AP on WIDER FACE (large / medium / small faces) 0.897 / 0.825 / 0.655 vs 0.813 / 0.763 / 0.640 for MTCNN |
| **License** | MIT, Copyright (c) 2026 Shalev Ben Moshe: all code and all weights (no pretrained face network is part of the model) |
| **Runs on** | Space: CPU basic · Training: GPU recommended (~4.5 h on an RTX 2080 Ti), CPU or Colab |

## The problem
Given a photo of a face, say who it is. This is *closed-set identification*: the model knows a fixed
list of 42 people. A second task, *verification*, needs no list at all: given two photos, decide
whether they show the same person. Both come from the same idea: map every face to an embedding in
which photos of the same person are close together and photos of different people are far apart.

The earlier version of this project borrowed that embedding from FaceNet (an InceptionResnetV1
pretrained on 3.3 million VGGFace2 photos, whose terms only allow non-commercial research) and its
face detector from facenet-pytorch (MTCNN). This version **trains every network itself**: the face
detector, the embedders and the head, and is MIT-licensed end to end.

## The data
[Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) is a set of news photos of public
figures: 13,233 photos of 5,749 people, most of whom appear once or twice.
- **The 42 people with at least 25 photos** are the classes the head names. They keep the original
  75/25 split (stratified per person, seed 42), so the test split is the same 647 photos as before;
  15% of their training photos are the validation split that every choice is made on.
- **Every other person** is extra training material for the embedders (5,586 identities in total):
  learning to tell ~5,600 people apart is what teaches a network what makes faces differ.
- **Unseen people:** 10% of the other people with at least 2 photos (163 people, 662 photos)
  are never trained on. They show whether the embeddings work for faces the model has never seen,
  which is what the Verify and Enroll tabs of the Space rely on.

## Architecture
1. **Face detector** (`FaceDetector`, trained here): one fully-convolutional pass over the photo
   with a feature pyramid (strides 8/16/32) and anchors; for every anchor it predicts face / not
   face, a box and five landmarks (eyes, nose tip, mouth corners). It takes MTCNN's ideas (three
   tasks per candidate, PReLU, online hard-negative mining), not its code: MTCNN's image pyramid and
   three cascaded networks are replaced by a single network. It learns from ~12,000 Open Images face
   photos + 5,000 LFW photos; the 5-point labels (a data file in `training/labels/`) were made once
   with MTCNN.
2. **Alignment:** a similarity transform puts the five landmarks on a fixed template, so every
   128 x 128 face crop has its eyes and mouth at the same pixels.
3. **Two embedders**, each a CNN + `Linear -> BatchNorm` projection to 512 numbers, trained with a
   **CosFace** loss (a softmax over identities with a margin on the cosine similarity):
   an EfficientNet-B2 that starts from torchvision's ImageNet weights, and a ResNet-18 trained from
   random weights. Their L2-normalised embeddings are concatenated into one 1024-d embedding.
4. **MLP head** `Linear(1024, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 42)` names the person.
5. **Verification** uses the cosine distance of two embeddings: below 0.675 = same person
   (picked on validation pairs).

## Training & experiments
Stage 0 trains the face detector (30 epochs, best epoch on 600 held-out Open Images photos) and
aligns every LFW photo with it. Stage 1 trains each embedder on all training identities (GPU augmentation: zoom, shift, rotation,
flip, colour, erasing); after every epoch it is scored on the validation split and on the unseen
people, and the best epoch is kept. Stage 2 trains the head on the frozen embeddings, early-stopped
on validation. The test split is looked at once, at the end.

Eleven embedders were compared (on MTCNN-aligned faces, before our detector existed), all with the
same code, splits and head, chosen on **validation accuracy, then unseen-people AUC, then size**
(test was not used for any choice):

| embedder | val acc. | test acc. | unseen-people AUC |
|---|---|---|---|
| ResNet-18 scratch, 42 people only, unaligned | 0.9281 | 0.9243 | 0.8381 |
| ResNet-18 scratch, all LFW, unaligned | 0.9760 | 0.9567 | 0.9399 |
| ResNet-50 ImageNet init, unaligned | 0.9863 | 0.9815 | 0.9404 |
| ResNet-18 scratch, aligned 112 px | 0.9932 | 0.9753 | 0.9475 |
| ResNet-18 scratch, aligned 128 px | 0.9932 | 0.9768 | 0.9614 |
| ResNet-50 ImageNet init, aligned | 0.9897 | 0.9892 | 0.9442 |
| ResNet-50 ImageNet init, aligned, 15 epochs | 0.9863 | 0.9861 | 0.9406 |
| EfficientNet-B2 ImageNet init, aligned | 0.9897 | 0.9845 | 0.9519 |
| Ensemble: ResNet-50 ImageNet init + ResNet-18 scratch | 0.9966 | 0.9938 | 0.9753 |
| Ensemble: ResNet-50 ImageNet init + EfficientNet-B2 ImageNet init | 0.9966 | 0.9938 | 0.9631 |
| Ensemble: EfficientNet-B2 ImageNet init + ResNet-18 scratch **(chosen)** | **0.9966** | **0.9954** | **0.9772** |

- Training on the 42 people alone memorises them (train 100%, test 92%); adding the other LFW
  people as extra classes is what makes the embeddings generalise.
- Aligning faces to the 5-point template is the largest single gain (+2 points).
- ImageNet initialisation helps on the 42 people but not on unseen faces; a ResNet-50 peaked after
  ~9 epochs and then memorised the training identities.
- Ensembling an ImageNet-initialised network with a from-scratch one closes most of the remaining
  gap: their mistakes differ.

For the detector, a single-shot network matched and beat MTCNN on WIDER FACE, and a
second, O-Net-like refining stage made it worse, so it was dropped.

Graphs (detector training and WIDER AP, training curves for both embedders and the head, the
experiment chart, confusion matrix, t-SNE, verification distances) are in the
[model card](https://huggingface.co/shalev396/face-recognition).

## Results
- **Identification:** 645 of 647 test photos correct (accuracy 0.9969, macro F1 0.9961);
  train / validation / test accuracy 1.0000 / 0.9932 / 0.9969. The mistakes: Arnold Schwarzenegger predicted as David Beckham; George W Bush predicted as Hans Blix.
- **Detection** on WIDER FACE val (AP at IoU 0.5, large / medium / small faces): 0.897 / 0.825 / 0.655,
  vs 0.813 / 0.763 / 0.640 for MTCNN (which was trained on WIDER FACE itself; ours never saw it).
- **Verification** over all 208,981 test pairs: ROC AUC 0.9973, balanced accuracy 0.9916 at the
  0.675 threshold. On the never-seen people: ROC AUC 0.9750.
- **Compared with the FaceNet version** (0.9985 on the same 647 photos, 1 mistake): 0.9969
  (645 of 647) without any face data beyond LFW and Open Images, better verification (balanced
  accuracy 0.9916 vs 0.9882), and with weights that are this project's own (MIT).

## Deployment
- **Model repo**: https://huggingface.co/shalev396/face-recognition (weights, `model.py`, `handler.py`)
- **Space / API**: https://huggingface.co/spaces/shalev396/face-recognition — `POST /gradio_api/call/predict`
  (photo -> top-3 people, seconds, device). The Verify and Enroll tabs are UI-only; Enroll keeps
  embeddings in the visitor's session and stores nothing.
- **Inference Endpoint**: deploy the model repo from its page (Deploy → Inference Endpoints);
  `handler.py` serves identification, verification and raw embeddings.

## Project structure
```
face-recognition/
├── model/      HF model repo: model.py, handler.py, face_detector.safetensors, model.safetensors, config.json, metrics.json, assets/
├── space/      HF Space: app.py (Identify / Verify / Enroll), space_utils.py, examples/
└── training/   notebook.ipynb (the controller) + src/ (config, data_setup, detector, model_builder, engine, export, utils) + labels/
```

## Reproduce
```bash
git clone https://github.com/shalev396/ml-lab && cd ml-lab
pip install -r requirements.txt
cd face-recognition/training
SMOKE_TEST=1 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp   # few-minute sanity run
jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp                # full run (GPU: ~4.5 h) -> ../model
cd ../space && python app.py                                                             # local demo
```
Or open the notebook in [Colab](https://colab.research.google.com/github/shalev396/ml-lab/blob/main/face-recognition/training/notebook.ipynb) with a GPU runtime.
Details: [training/README.md](training/README.md).

## Limitations
- **Only 42 people, no "unknown" class.** Anybody else is matched to the closest of the 42.
- **Small training set.** About 12,000 photos, mostly one or two per person: expect lower accuracy on
  photos unlike LFW (profile views, low light, children, masks).
- **LFW is easy and near-duplicate-prone.** One person's photos often come from the same events.
- **Bias.** LFW is mostly adult, male, white public figures; errors across demographic groups were
  not measured.
- **Sensitive use.** Face recognition can be used for surveillance. This is a learning project: use it
  only on public-figure photos or on people who agreed to it.
