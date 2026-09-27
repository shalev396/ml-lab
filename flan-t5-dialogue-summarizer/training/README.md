# Training — FLAN-T5 Dialogue Summarizer + RAG

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shalev396/ml-lab/blob/main/flan-t5-dialogue-summarizer/training/notebook.ipynb)

Builds `../model/`: a LoRA adapter for `google/flan-t5-base` fine-tuned on DialogSum, evaluated against prompting
the frozen base model zero-shot, one-shot and few-shot, plus the RAG retriever evaluation.

## Notebook (`notebook.ipynb`)
1. **Setup**: 1a finds the project files (clones the repo and the `model` submodule on a fresh runtime), 1b
   installs `../../requirements.txt`, 1c picks the `device` (cuda → mps → cpu), 1d imports `src/` and `model.py`.
2. **Config**: the `Config` dataclass, the `SMOKE_TEST` switch and `PUSH_TO_HUB`.
3. **Data**: `data_setup.load_dialogsum()` → seeded train / validation / test subsets → length stats →
   tokenization and dynamic-padding DataLoaders.
4. **Load model**: `model_builder.load_base()` + `add_lora()`, and the parameter counts.
5. **Training**: `engine.train()` (resumable per epoch), the loss curve, then `export.save_model()` writes the
   adapter, tokenizer and `config.json`. Also a short note on why RLHF is not used.
6. **Evaluation**: `engine.evaluate_variants()` (ROUGE for base zero/one/few-shot and the fine-tune), sample
   summaries side by side, and `engine.evaluate_retrieval()` + grounded vs ungrounded RAG answers.
7. **Inference**: `model.load(cfg.model_dir, device)` on `../space/examples/*.txt` and one RAG question, which is
   exactly what the Space runs.
8. **Export**: `export.export()` checks the round trip through `model.py` and writes `metrics.json`, `assets/*.png`
   and the model card. Optional `utils.upload_model()`.

## Run it
- **Colab**: click the badge and pick a GPU runtime. The first cells clone the repo and install the requirements.
- **Local**: `pip install -r ../../requirements.txt`, then open `notebook.ipynb`, or run it headless from this folder:
  ```bash
  SMOKE_TEST=1 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp        # smoke
  jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp --ExecutePreprocessor.timeout=-1  # full
  ```
  The smoke run exports to `outputs/smoke/model/` and never touches `../model/`.

| Run | What | Time |
|---|---|---|
| smoke | flan-t5-small, 64 train / 16 val / 12 test dialogues, 1 epoch (8 steps) | about 3-5 min on a CPU (measured, busy 20-thread desktop) |
| full, CPU | flan-t5-base, 12,460 dialogues, up to 6 epochs × 1,558 steps + 4 × 500 test generations | days. See the measured seconds per step below |
| full, GPU | same | measured on Google Colab, A100 (bf16): 1,788 s for 6 epochs (0.19 s/step) + 268 s evaluation. RTX 2080 Ti (fp32): 0.52 s/step, about 14 min per epoch |

Measured on the development desktop CPU (6 threads, while other training jobs shared the machine): flan-t5-base
LoRA took **30-145 s per step** at batch size 8, and greedy generation of 16 summaries took over 10 minutes. On
that machine a full CPU run would take well over 12 hours, which is why a GPU is recommended. If a run is
interrupted, running the notebook again continues from the last finished epoch (`outputs/checkpoints/last.pt`,
used only when every training setting matches).

## Data
- Primary source: [`knkarthick/dialogsum`](https://huggingface.co/datasets/knkarthick/dialogsum) via 🤗 `datasets`,
  cached in `data/hf_datasets/` (the download is skipped when it is present).
- Fallback: the authors' JSONL files from [github.com/cylnlp/dialogsum](https://github.com/cylnlp/dialogsum),
  cached in `data/mirror/`.
- The Hub test split has 1,500 rows: 500 dialogues × 3 human summaries. `data_setup.group_references()` turns it
  into 500 dialogues with 3 references each, and ROUGE averages over the three.
- The RAG knowledge base ships with the model repo (`../model/assets/kb.json`). The 18 retrieval test questions
  are in `src/data_setup.py` (`RAG_EVAL`).

## Precision and devices
Float32 by default. On GPUs with bf16 support (`utils.amp_dtype(device) == torch.bfloat16`, Ampere or newer)
the forward pass runs under bf16 autocast. fp16 is never used, even where `utils.amp_dtype` would choose it
(e.g. a T4), because T5 overflows in fp16 and the loss is NaN. A non-finite loss stops training with an error.

## Outputs
- `../model/` (full run): `adapter_config.json`, `adapter_model.safetensors`, the tokenizer files, `config.json`
  (base model id, generation and RAG settings, prompt templates, versions), `metrics.json`, `assets/*.png`. The
  model card's YAML, metrics table and Experiments block are refreshed from `metrics.json`, and so is the Result
  row of `../README.md`.
- `outputs/`: `loss_curve.png` and `checkpoints/last.pt`. `outputs/smoke/`: the smoke run's complete model repo,
  with card, code and knowledge base, so `MODEL_DIR=training/outputs/smoke/model python app.py` serves it.
- Publish: `PUSH_TO_HUB = True` in the notebook, or `python -c "from src import utils; utils.upload_model()"`. The
  token comes from `hf auth login` or an `HF_TOKEN` env var / Colab secret.

## `src/`
| File | Role |
|---|---|
| `__init__.py` | puts `../model` on `sys.path`, so `import model as M` works everywhere |
| `config.py` | `Config` dataclass: every hyperparameter, smoke shrinking and output paths |
| `data_setup.py` | DialogSum download (Hub, then GitHub mirror) → subsets → tokenization → DataLoaders; few-shot prompts; KB + RAG test questions |
| `model_builder.py` | FLAN-T5 in float32 + LoRA (`peft`), parameter counts |
| `engine.py` | training loop (AdamW, linear decay, bf16-or-fp32, epoch checkpoints), validation loss, ROUGE, prompt variants, retrieval metrics |
| `export.py` | `save_model()` (adapter, tokenizer, `config.json`) and `export()` (round-trip check, `metrics.json`, plots, card) |
| `utils.py` | shared ml-lab helpers (identical in every project) |
