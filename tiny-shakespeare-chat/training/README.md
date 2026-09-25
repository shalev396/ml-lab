# Tiny Shakespeare Chat: training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shalev396/ml-lab/blob/main/tiny-shakespeare-chat/training/notebook.ipynb)

Builds the chat model from nothing: a character-level GPT (`model.GPT` in [`../model/model.py`](../model/model.py))
is pretrained on the raw Tiny Shakespeare text (stage A), then chat-tuned on consecutive dialogue-line
pairs (stage B). The result is exported as a Hugging Face model folder, the same layout as [`../model`](../model).

## What the notebook does
1. **Setup**: 1a finds the project files (clones the repo on a fresh runtime), 1b installs `../../requirements.txt`, 1c picks the `device` (cuda, then mps, then cpu), 1d imports `src/` and `model`.
2. **Config**: `Config()` from `src/config.py`, the `SMOKE_TEST` toggle and the `PUSH_TO_HUB` flag.
3. **Data**: downloads + caches the corpus, builds the 68-token vocabulary, makes the stage A and stage B splits, prints a chat sample and the chat-val overlap (see Caveat).
4. **Load model**: builds `model.GPT` via `model_builder.build_gpt`, prints the parameter count, loads the deployed model for comparison.
5. **Training**: `engine.train_stage` for stage A (then a raw `ROMEO:` continuation), `engine.train_stage` again for stage B, then the loss curves.
6. **Evaluation**: `engine.evaluate` (deterministic, every validation character scored once) for the bigram baseline, the stage A snapshot, this run and the deployed model, on both splits, plus the comparison chart and the loss-vs-context plot.
7. **Inference**: stages the weights, then `model.load(dir, device).predict(...)` on `../space/examples/messages.json` (exactly what the Space runs) and one call to `handler.EndpointHandler`.
8. **Export**: `export.export` writes weights, `config.json`, `metrics.json`, `assets/*.png` and refreshes the model card; optional Hub upload.

## Recipe (defaults in `src/config.py`)
| | |
|---|---|
| Tokenizer | the 65 characters of the corpus (sorted) + `<\|user\|>`, `<\|bot\|>`, `<\|end\|>` as single ids = 68 |
| Model | 6 layers, 6 heads, 384-d, 256-char context, dropout 0.2, pre-LayerNorm, GELU MLP, tied LM head: 10,751,232 parameters |
| Optimiser | AdamW (betas 0.9/0.95, weight decay 0.1 on matrices only), grad clip 1.0, batch 64 x 256 chars, AMP on CUDA |
| Stage A | 5,000 iters, warmup 100, cosine 3e-4 -> 3e-5, on 90 % of the raw text |
| Stage B | 2,000 iters, warmup 50, cosine 1e-4 -> 1e-5, on 95 % of the chat-formatted stream |
| Chat format | `<\|user\|> {line} <\|end\|> <\|bot\|> {next line} <\|end\|>`, plus a variant with the reply prefixed by `SPEAKER:` |
| Sampling | temperature 0.8, top-k 40, stop at `<\|end\|>`, last 3 turns kept in the prompt |

## Run it
- **Colab:** click the badge (a GPU runtime is preselected), then *Run all*.
- **Local notebook:** `pip install -r ../../requirements.txt`, then open `notebook.ipynb` from this folder.
- **Headless:** `SMOKE_TEST=1 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp`.
  `SMOKE_TEST=1` trains a 2-layer / 64-d / 64-char model for 200 + 60 iterations, scores the first
  8,192 characters of each split and exports to `outputs/smoke/model/` (never to `../model`).
  With `SMOKE_TEST=0` the notebook exports to `../model` and **overwrites the deployed weights**.

## Hardware and time
| run | CPU | GPU |
|---|---|---|
| smoke | about 1-2 min (desktop CPU, shared with other jobs) | seconds |
| full (5,000 + 2,000 iters) | several hours (not measured) | 541 s measured (CUDA, legacy run that produced the deployed weights) |

## Data
- **Source:** [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
  from karpathy/char-rnn, 1,115,394 characters.
- **Fallback:** `https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt`
  (the TensorFlow tutorial copy, checked to be byte-identical).
- **Cache:** `training/data/tinyshakespeare.txt` (gitignored). Later runs skip the download.

## Caveat: the chat validation split leaks
The chat samples are built from the whole play, and the two variants of each pair are shuffled together
before the 95/5 split. Of the chat-val samples, 95.8 % have their twin (same dialogue pair, other variant)
in chat-train (`data_setup.chat_val_overlap`). The chat-tuned model has also seen the stage A validation
text inside chat pairs. So chat-val loss is optimistic, and text-val loss after stage B is not held out.
The split is kept this way so that the code reproduces the deployed weights. A clean protocol would take
the chat-val pairs from the last 10 % of the play and keep both variants on the same side.

## Outputs
- Full run: `../model/` gets `model.safetensors` + `config.json` (via `PyTorchModelHubMixin.save_pretrained`;
  vocabulary and `block_size` are in `config.json`, and the tied weight is stored once), `metrics.json`,
  `assets/training_curves.png`, `assets/comparison.png`, `assets/loss_by_position.png`. The YAML and the metrics
  table of `../model/README.md` are refreshed by `utils.update_model_card`; the prose is left untouched.
- Smoke run: the same files in `outputs/smoke/model/`, plus copies of the card, `model.py`, `handler.py`
  and `requirements.txt`. From `space/`, `MODEL_DIR=../training/outputs/smoke/model python app.py` serves it.
- Scratch: `outputs/run/` (or `outputs/smoke/`) holds the plots and the staged weights of section 7.
- Publish: `PUSH_TO_HUB = True` in the notebook, or `utils.upload_model()`. Authenticate with
  `hf auth login` or an `HF_TOKEN` secret / env var.

## `src/` file map
| file | role |
|---|---|
| `__init__.py` | puts `../model` on `sys.path`, so `import model as M` is the deployed `model.py` |
| `config.py` | `@dataclass Config`: every hyperparameter, smoke handling, `model_dir`, `run_dir` |
| `data_setup.py` | download with fallback, tokenizer, stage A / stage B splits, dialogue parsing, batches, `chat_val_overlap` |
| `model_builder.py` | `build_gpt` (from `model.GPT`), `load_deployed`, `snapshot`, the `BigramLM` baseline |
| `engine.py` | `make_optimizer`, `get_lr`, `train_stage`, `estimate_loss`, deterministic `evaluate`, `reply`, `continue_text` |
| `export.py` | `metrics.json` builder, `save_weights`, `export`, the three plots |
| `utils.py` | shared ml-lab helpers (devices, seeds, JSON, card update, Hub upload), identical in every project |

## Provenance of the deployed weights
`../model/model.safetensors` is **not** from a run of this folder. It is the chat checkpoint of the earlier
full run (legacy project `13_tiny_shakespeare_chat_pt`, CUDA, 541 s, the same recipe and code as here),
converted into `model.GPT`: every tensor loads with `strict=True`, the logits match the legacy code
exactly, and seeded generations are identical. Its metrics were re-computed locally with `engine.evaluate`.
