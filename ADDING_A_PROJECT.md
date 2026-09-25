# Adding a project to ml-lab

Every project in this lab has the same shape. Follow this guide top to bottom when you add a new
one (or bring an old one up to date). Templates for every file are at the [end](#templates).

```
<slug>/                    kebab-case, identical to the Hugging Face repo names
├── README.md              detailed project write-up (problem, data, architecture, results, deployment)
├── model/   → git submodule = Hugging Face MODEL repo  huggingface.co/shalev396/<slug>
├── space/   → git submodule = Hugging Face SPACE        huggingface.co/spaces/shalev396/<slug>
└── training/              lives only on GitHub: notebook + reusable code that builds the model
```

| Folder | Lives on | Purpose |
|---|---|---|
| `model/` | Hugging Face model repo | The saved model: weights, `model.py`, `handler.py`, experiment graphs, model card. Deployable as an API ([Inference Endpoint](https://huggingface.co/docs/inference-endpoints/guides/custom_handler)). |
| `space/` | Hugging Face Space | The Gradio app + free public API (`/predict`) the portfolio calls. Pulls code + weights from `model/`. |
| `training/` | GitHub only | `notebook.ipynb` (the controller) + `src/` (reusable code). Runs on CPU, a local GPU or Google Colab. |

---

## 1. Connect `model/` and `space/` to Hugging Face (git submodules)

`model/` and `space/` are **git submodules**: each one is its own git repository whose remote is the
Hugging Face repo. GitHub only stores a pointer (a commit id) for them, so weights never touch GitHub;
they live on Hugging Face (LFS/Xet storage). Committing inside `model/` commits to the HF model repo;
committing inside `space/` commits to the HF Space.

### One-time machine setup
```bash
pip install -U huggingface_hub                    # provides the `hf` CLI
hf auth login --add-to-git-credential             # write token from huggingface.co/settings/tokens
git lfs install                                   # large files (weights, images)
winget install git-xet && git xet install         # faster uploads to HF (Xet storage); macOS/Linux: see docs
git config --global push.recurseSubmodules on-demand   # `git push` in ml-lab also pushes submodule commits
git config --global submodule.recurse true             # pull/checkout update submodules too
git config --global status.submoduleSummary true       # `git status` shows submodule changes
```
Docs: [Getting started with repositories](https://huggingface.co/docs/hub/repositories-getting-started) ·
[Xet storage](https://huggingface.co/docs/hub/xet/using-xet-storage) ·
[git submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules)

### Create the two Hugging Face repos and link them
```bash
SLUG=my-new-project
hf repos create shalev396/$SLUG --type model
hf repos create shalev396/$SLUG --type space --sdk gradio          # add --flavor zero-a10g for ZeroGPU

cd ml-lab
git submodule add https://huggingface.co/shalev396/$SLUG         $SLUG/model
git submodule add https://huggingface.co/spaces/shalev396/$SLUG  $SLUG/space
git commit -m "Add $SLUG submodules"
```
If the folders already contain files (e.g. you built them first), turn each one into a repo and point
it at Hugging Face instead:
```bash
cd $SLUG/model
git init -b main && git remote add origin https://huggingface.co/shalev396/$SLUG
git add . && git commit -m "Initial model"
git push --force -u origin main        # only on a brand-new repo (it holds just HF's auto-commit)
cd ../.. && git submodule add https://huggingface.co/shalev396/$SLUG $SLUG/model
```

### Day-to-day workflow
```bash
cd $SLUG/space && git add -A && git commit -m "Improve UI"     # 1. commit inside the submodule
cd ../.. && git add $SLUG/space && git commit -m "Bump $SLUG space"   # 2. record the new pointer
git push                                                        # pushes the HF repo(s) first, then GitHub
```
When the training notebook uploads a new model straight to the Hub (e.g. from Colab), sync it back:
`git submodule update --remote $SLUG/model && git add $SLUG/model && git commit -m "Sync $SLUG model"`.

Cloning everything: `git clone --recurse-submodules https://github.com/shalev396/ml-lab`
(add `GIT_LFS_SKIP_SMUDGE=1` in front to skip downloading weights).

### GPU on Hugging Face
- **Space**: pick hardware in *Settings → Hardware*. PyTorch Spaces can use
  [ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu) (PRO: up to 10 Spaces, Gradio SDK,
  Python 3.10/3.12, PyTorch only): `import spaces` first, move models to `cuda` at module level, wrap
  inference in `@spaces.GPU(duration=…)`. Same code runs on CPU Spaces (`spaces.GPU` is a no-op).
  Other options: [GPU Spaces](https://huggingface.co/docs/hub/spaces-gpus). TensorFlow / scikit-learn
  projects run on **CPU basic**.
- **Model**: `handler.py` makes the repo deployable as an
  [Inference Endpoint](https://huggingface.co/docs/inference-endpoints/guides/custom_handler)
  on CPU or GPU hardware; the handler picks `cuda` when the endpoint has a GPU. (Hugging Face's free
  serverless inference does not run custom models, so the free public API is the Space.)

---

## 2. `training/` — build the model (GitHub only)

- [ ] `training/notebook.ipynb`: **the controller**. It triggers every step (data, training, evaluation,
      inference, saving). Sections, in order:
  1. **Setup**
     - project name (`SLUG`) and a check that fetches the repo / submodule code if it's missing
       (for example on a fresh Colab runtime)
     - **Install dependencies** from the root [`requirements.txt`](requirements.txt), the single source
       of truth: `%pip install -q -r ../../requirements.txt`. Colab does not install a repo's
       requirements on its own, so this cell always runs.
     - `device`: picked from what the machine has, in the order `cuda` → `mps` → `cpu`. Nothing is
       special-cased for any platform, and no extra platform-specific libraries are installed.
  2. **Config**: every hyperparameter, plus the `SMOKE_TEST` switch for a 2-minute sanity run.
  3. **Data**: download → cache in `training/data/` → preprocess → splits.
  4. **Load model**: build the architecture from `model/model.py` and print the parameter count.
  5. **Training**: train every experiment/variant and show the loss/metric curves.
  6. **Evaluation**: test metrics, the comparison table across experiments, and task plots
     (confusion matrix, ROC/PR, forecasts…).
  7. **Inference**: `model.load(dir, device).predict(...)` on `space/examples/`, exactly what the
     Space and the endpoint run.
  8. **Export**: save the best model + `metrics.json` + graphs into `model/`, refresh the model card,
     and optionally upload (`PUSH_TO_HUB = True`).
- [ ] `training/src/`: reusable code only. The notebook calls it; there are no big logic blobs in cells.
  - `__init__.py` (makes `../model` importable)
  - `config.py`
  - `data_setup.py`
  - `model_builder.py`
  - `engine.py`
  - `export.py`
  - `utils.py`: the same file in every project (device helpers, seeding, JSON, model-card updater, Hub upload)
- [ ] `training/data/`: dataset cache (gitignored, created by the notebook).
- [ ] `training/outputs/`: scratch outputs + smoke runs (gitignored). Smoke runs never write into `model/`.
- [ ] `training/README.md`: what each section does, how to run it (Colab badge / locally), time on
      CPU vs GPU, data sources.

## 3. `space/` — the app + public API (Hugging Face Space)

- [ ] `README.md`: Space front matter (`sdk: gradio`, `sdk_version`, `python_version: "3.12"`,
      `models: [shalev396/<slug>]`, `preload_from_hub`, `short_description` ≤ 60 chars, `tags: [ml-lab]`),
      then API docs (curl + `@gradio/client`). [Config reference](https://huggingface.co/docs/hub/spaces-config-reference)
- [ ] `app.py`: a Gradio Blocks UI in the FoodVision style (title, input column / output column,
      examples, "Model" + "Training" spec cards, links row).
  - It loads `model.py` + weights from the model repo (`space_utils.import_model`).
  - Exactly one public endpoint, `api_name="predict"`, returning `[result, seconds, device]`. Every
    other event is `api_visibility="private"`.
  - On a GPU (ZeroGPU) it runs through `@spaces.GPU`, with a CPU fallback when the visitor has no GPU
    quota left.
- [ ] `space_utils.py`: the same file in every Space (ZeroGPU shim, model loading, UI cards, CSS).
- [ ] `requirements.txt`: lower bounds only. No `gradio` (it comes from `sdk_version`) and no `spaces`
      (preinstalled). PyTorch Spaces use `torch>=2.8,<2.14`, the ZeroGPU range.
- [ ] `packages.txt` (apt packages, only if needed), `examples/` (3–5 small inputs).
- [ ] `.gitattributes` (LFS rules, including images) and `.gitignore`.

## 4. `model/` — the saved model (Hugging Face model repo)

- [ ] Weights: `model.safetensors` (PyTorch, via `PyTorchModelHubMixin.save_pretrained`) ·
      `model.keras` (Keras) · `*.joblib` (scikit-learn) · `xgb.ubj` (XGBoost).
- [ ] `config.json`: everything needed to rebuild and run the model (labels, preprocessing, threshold,
      library versions).
- [ ] `model.py`: the single source of truth for architecture + preprocessing + `load(dir, device)` +
      `Predictor.predict(...)`. Training, the Space and the endpoint all use it.
- [ ] `handler.py` + `requirements.txt`: the [Inference Endpoint](https://huggingface.co/docs/inference-endpoints/guides/custom_handler)
      entry point. It picks `cuda` when available.
- [ ] `metrics.json`: the numbers of the deployed model plus a comparison of every experiment.
- [ ] `assets/*.png`: graphs from the experiments, e.g. training curves, the comparison chart of all
      variants, and the confusion matrix / ROC / forecast plot of the best one.
- [ ] `README.md` (model card): YAML metadata (incl. the `ml_lab:` block the portfolio reads via the
      Hub API), then Model · Usage (Python, Space API, Endpoint) · Training · **Experiments** (table of
      every variant, best highlighted, graphs) · Evaluation · Limitations.
- [ ] `.gitattributes`, `.gitignore`.

## 5. Finish

- [ ] `<slug>/README.md`: a detailed write-up (template below).
- [ ] A ≤ 5-line entry in the root [README](README.md).
- [ ] Run the notebook headless: `SMOKE_TEST=1 jupyter nbconvert --to notebook --execute training/notebook.ipynb --output-dir /tmp`
- [ ] Run the Space locally: `cd space && python app.py`. Check `/predict` in the "Use via API" page.
- [ ] Commit inside `model/` and `space/`, then commit the pointers and push (§1).

---

## Templates

### `<slug>/README.md`
````markdown
# <emoji> <Title>

> <one-sentence pitch>

<p>
  <a href="https://huggingface.co/spaces/shalev396/<slug>"><img alt="Space" src="https://img.shields.io/badge/🤗%20Space-live%20demo-blue"></a>
  <a href="https://huggingface.co/shalev396/<slug>"><img alt="Model" src="https://img.shields.io/badge/🤗%20Model-shalev396%2F<slug>-yellow"></a>
  <a href="https://colab.research.google.com/github/shalev396/ml-lab/blob/main/<slug>/training/notebook.ipynb"><img alt="Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>

| | |
|---|---|
| **Task** | … |
| **Framework** | … |
| **Architecture** | … (<params> parameters) |
| **Dataset** | [name](link), size, split |
| **Result** | <primary metric> = <value> (test) |
| **Runs on** | Space: CPU basic / ZeroGPU · Training: CPU, GPU or Colab |

## The problem
## The data
## Architecture
## Training & experiments
## Results
## Deployment
- **Model repo**: https://huggingface.co/shalev396/<slug> (weights, `model.py`, `handler.py`)
- **Space / API**: https://huggingface.co/spaces/shalev396/<slug> — `POST /gradio_api/call/predict`
- **Inference Endpoint**: deploy the model repo from its page (Deploy → Inference Endpoints)
## Project structure
## Reproduce
## Limitations
````

### `model/README.md` (model card) front matter
```yaml
---
license: mit
library_name: <keras | sklearn | xgboost | peft | omit for PyTorch>
pipeline_tag: <image-classification | text-classification | tabular-classification | …>
tags: [ml-lab, <framework>, <task>]
datasets: [<hub dataset ids>]
base_model: <hub id, only for fine-tunes>
metrics: [<names>]
ml_lab:
  title: <Title>
  order: <n>
  summary: <one plain sentence>
  framework: <…>
  architecture: <…>
  dataset: <…>
  space: shalev396/<slug>
  runtime: <cpu-basic | zerogpu>
  ui_kind: <image-classifier | text-classifier | spam-inbox | timeseries | custom>
  colab: https://colab.research.google.com/github/shalev396/ml-lab/blob/main/<slug>/training/notebook.ipynb
  github: https://github.com/shalev396/ml-lab/tree/main/<slug>
---
```
`training/src/utils.py:update_model_card()` fills in `model-index`, `ml_lab.metric`, `ml_lab.params`
and the table between `<!-- metrics:start -->` / `<!-- metrics:end -->` from `metrics.json`.

### `model/handler.py`
```python
"""Hugging Face Inference Endpoints entry point — deploy this repo as a CPU/GPU API."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model as M  # noqa: E402


class EndpointHandler:
    def __init__(self, path: str = ""):
        self.predictor = M.load(path or HERE, "cuda" if M.cuda_available() else "cpu")

    def __call__(self, data: dict):
        inputs = data.pop("inputs", data)
        parameters = data.pop("parameters", None) or {}
        return self.predictor.predict(inputs, **parameters)
```

### `space/README.md` front matter
```yaml
---
title: <Title>
emoji: <emoji>
colorFrom: <topic color>     # red|yellow|green|blue|indigo|purple|pink|gray — pick per topic,
colorTo: <topic color>       # it's the card gradient AND the app theme (space_utils reads it)
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: <≤ 60 chars>
models: [shalev396/<slug>]
preload_from_hub: [shalev396/<slug>]
tags: [ml-lab, <framework>, <task>]
---
```

### `space/app.py`
```python
from space_utils import GPU, LAUNCH_KWARGS, card, header, import_model, is_quota_error, links, read_json  # first import

import time
import gradio as gr

SLUG = "<slug>"
M, MODEL_DIR = import_model(f"shalev396/{SLUG}")
CPU = M.load(MODEL_DIR, "cpu")
CUDA = M.load(MODEL_DIR, "cuda") if M.cuda_available() else None   # module level (ZeroGPU)


@GPU(duration=10)
def _predict_gpu(x):
    return CUDA.predict(x)


def predict(x):
    start = time.perf_counter()
    if CUDA is not None:
        try:
            result, device = _predict_gpu(x), "gpu"
        except gr.Error as err:            # out of ZeroGPU quota -> CPU
            if not is_quota_error(err):
                raise
            result, device = CPU.predict(x), "cpu"
    else:
        result, device = CPU.predict(x), "cpu"
    return result, round(time.perf_counter() - start, 4), device


with gr.Blocks(title="<Title>") as demo:
    ...
    btn.click(predict, [inp], [out, sec, dev], api_name="predict")

if __name__ == "__main__":
    demo.launch(**LAUNCH_KWARGS)
```

### Notebook setup cells
```python
# 1a. Project + files: fetch the repo / model code only if they aren't here (fresh Colab runtime, etc.)
import os, subprocess
from pathlib import Path
SLUG = "<slug>"
if not Path("src").is_dir():
    if not Path("ml-lab").is_dir():
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/shalev396/ml-lab.git"], check=True)
    os.chdir(f"ml-lab/{SLUG}/training")
if not Path("../model/model.py").exists():
    subprocess.run(["git", "submodule", "update", "--init", "--depth", "1", f"{SLUG}/model"],
                   cwd="../..", check=True, env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"})
```
```python
# 1b. Dependencies: the root requirements.txt is the single source of truth
%pip install -q -r ../../requirements.txt
```
```python
# 1c. Device: whatever this machine has (cuda -> mps -> cpu)
import torch
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)
```
