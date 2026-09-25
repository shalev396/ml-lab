# 🎭 Tiny Shakespeare Chat

> Chat with a small GPT that was written and trained from scratch on Shakespeare and answers in play-style verse.

<p>
  <a href="https://huggingface.co/spaces/shalev396/tiny-shakespeare-chat"><img alt="Space" src="https://img.shields.io/badge/🤗%20Space-live%20demo-blue"></a>
  <a href="https://huggingface.co/shalev396/tiny-shakespeare-chat"><img alt="Model" src="https://img.shields.io/badge/🤗%20Model-shalev396%2Ftiny--shakespeare--chat-yellow"></a>
  <a href="https://colab.research.google.com/github/shalev396/ml-lab/blob/main/tiny-shakespeare-chat/training/notebook.ipynb"><img alt="Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>

| | |
|---|---|
| **Task** | Character-level language modelling, then chat-style reply generation |
| **Framework** | PyTorch (no pretrained weights, no tokenizer library) |
| **Architecture** | nanoGPT-style decoder: 6 layers, 6 heads, 384-d, 256-character context, tied embeddings (10,751,232 parameters) |
| **Dataset** | [Tiny Shakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare), 1,115,394 characters; 90/10 split for pretraining, 7,096 dialogue-line pairs (95/5) for chat-tuning |
| **Result** | chat val loss = 1.054 nats/char (perplexity 2.87, optimistic, see below); clean held-out text val loss after pretraining = 1.495 |
| **Runs on** | Space: CPU basic · Training: GPU recommended (541 s on CUDA), CPU works but takes hours |

## The problem
Can a language model be built end to end, with no borrowed weights, and still hold a (Shakespearean)
conversation? This project writes the whole stack by hand: a character tokenizer with chat markers, a
GPT-style transformer, the training loop, the sampler and the chat prompt format. A 10.75M-parameter model
trained for nine minutes on one GPU learns the look of a play (speaker names, blank-verse lines, archaic
words) and answers a message with the next line of an imaginary scene. It is a teaching model: it
produces poetry, not facts.

## The data
[Tiny Shakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare) from
karpathy/char-rnn: 1.1 MB of Shakespeare plays in one text file (40,000 lines). The notebook downloads it
into `training/data/` (fallback: the byte-identical TensorFlow tutorial copy).

- **Vocabulary:** the 65 distinct characters of the corpus plus three chat markers, `<|user|>`, `<|bot|>` and
  `<|end|>`, each a single reserved id: 68 ids.
- **Stage A (pretrain) split:** the raw text, first 90 % train (1,003,855 characters), last 10 % val (111,539).
- **Stage B (chat-tune) data:** the play is parsed into consecutive `(speaker, utterance)` turns. Every
  neighbouring pair becomes one chat sample, `<|user|> {line} <|end|> <|bot|> {next line} <|end|>`, in two
  variants (plain reply, and reply prefixed with `SPEAKER:`). That gives 7,096 pairs. The samples are shuffled
  (seed 42) and joined, and the last 5 % of the stream is val (4,088,379 train / 215,177 val tokens).

## Architecture
All of it lives in one file, [`model/model.py`](model/model.py), used by training, the Space and the endpoint.

- `CharTokenizer`: splits on the marker strings first (each marker = one id), char-encodes the rest.
- `GPT` (`nn.Module` + `PyTorchModelHubMixin`): token + learned position embeddings, 6 pre-LayerNorm
  blocks (6-head causal self-attention through `scaled_dot_product_attention`, 4x GELU MLP, dropout 0.2),
  final LayerNorm and an LM head tied to the token embedding. GPT-2 initialisation with scaled residual
  projections. The vocabulary and sizes are constructor arguments, so `save_pretrained` writes them to
  `config.json`.
- `sample`: temperature + top-k sampling that stops at `<|end|>`.
- `Predictor.predict(message, history=None, max_new_tokens=200, temperature=0.8, top_k=40)` returns the
  reply string. `Predictor.stream(...)` yields it one character at a time (the Space's chat UI). The prompt
  keeps up to 3 earlier turns and is cropped to fit the 256-character context.

## Training & experiments
The notebook ([`training/notebook.ipynb`](training/notebook.ipynb)) drives every step through `training/src/`:

1. **Stage A, pretrain:** 5,000 iterations of next-character prediction on the raw play, batch 64 x 256,
   AdamW (betas 0.9/0.95, weight decay 0.1 on matrices only), warmup 100 then cosine LR 3e-4 -> 3e-5,
   gradient clipping 1.0, fp16 autocast on CUDA.
2. **Stage B, chat-tune:** 2,000 more iterations on the chat-formatted pairs, warmup 50, cosine LR 1e-4 -> 1e-5.
3. **Evaluate:** a deterministic pass over each validation split (non-overlapping 256-character windows,
   every character scored once) for every variant.

| variant | chat val loss | text val loss |
|---|---|---|
| bigram baseline (character counts, add-one smoothing) | 2.451 | 2.482 |
| stage A only: pretrained GPT, 5,000 iters | 1.563 | 1.495 |
| **stage A + B: chat-tuned GPT (deployed)** | **1.054** | **1.296** |

Cross-entropy in nats per character, lower is better (`model/metrics.json`). Graphs and discussion are in the
[model card](model/README.md#experiments).

![Validation loss of every variant](model/assets/comparison.png)

![Training curves of both stages](model/assets/training_curves.png)

## Results
The deployed chat-tuned model scores **1.054 nats/char on chat val** (1.52 bits per character, perplexity
2.87) and 1.296 on text val.

These two numbers are **optimistic**, and the project says so rather than hiding it:
- Both variants of each dialogue pair are shuffled together before the 95/5 chat split, so 95.8 % of chat-val
  samples have their twin in chat-train (`data_setup.chat_val_overlap`).
- The chat pairs cover the whole play, so after stage B the model has also seen the text val lines.

The clean held-out number is stage A on text val: **1.495 nats/char** (2.16 bits per character), against
2.482 for the bigram baseline. The split was kept as it is so that the code reproduces the deployed weights.
A clean protocol would take the chat-val pairs from the last 10 % of the play and keep both variants of a
pair on the same side.

Sample replies (sampled at temperature 0.8, so every run differs):

```
USER: Speak to me of love
BOT:  BIONDELLO: Half an unkindness! the matter of my mind
      Make our envious stoutes, our springings of accusation.

USER: What say you of the moon?
BOT:  CAMILLO: Your people, sir,--
```

![Loss vs. context length](model/assets/loss_by_position.png)

The loss levels off after roughly 30-50 characters of context: the model mostly uses about a line.

## Deployment
- **Model repo**: https://huggingface.co/shalev396/tiny-shakespeare-chat (weights, `model.py`, `handler.py`)
- **Space / API**: https://huggingface.co/spaces/shalev396/tiny-shakespeare-chat — `POST /gradio_api/call/predict`
  with `(message, history_json, temperature, max_new_tokens)`, returns `[reply, seconds, device]`
  (see [space/README.md](space/README.md)). The streaming chat UI uses private events.
- **Inference Endpoint**: deploy the model repo from its page (Deploy → Inference Endpoints). The body is
  `{"inputs": "<message>" | {"message": ..., "history": [[user, bot], ...]}, "parameters": {"max_new_tokens", "temperature", "top_k", "seed"}}`,
  and the response is `[{"generated_text": "<reply>"}]`.

## Project structure
```
tiny-shakespeare-chat/
├── model/      HF model repo: model.py, handler.py, requirements.txt, model.safetensors, config.json,
│               metrics.json, assets/*.png, README.md (model card)
├── space/      HF Space: app.py (gr.Blocks + gr.Chatbot), space_utils.py, requirements.txt,
│               examples/messages.json, README.md (API docs)
└── training/   notebook.ipynb (the controller), src/ (config, data_setup, model_builder, engine,
                export, utils), README.md
```

## Reproduce
```bash
pip install -r requirements.txt                      # from the ml-lab root
cd tiny-shakespeare-chat/training
jupyter notebook notebook.ipynb                      # run all cells; SMOKE_TEST=0 overwrites ../model
SMOKE_TEST=1 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp   # 1-2 min check
cd ../space && python app.py                         # the Space locally, served from ../model
```
Or open the notebook in [Colab](https://colab.research.google.com/github/shalev396/ml-lab/blob/main/tiny-shakespeare-chat/training/notebook.ipynb)
(a GPU runtime is preselected). The deployed weights come from an earlier full run of the same recipe
(CUDA, 541 s for both stages), converted into `model.GPT` with bit-identical logits; see
[training/README.md](training/README.md#provenance-of-the-deployed-weights).

## Limitations
- A toy model: 10.75M parameters and 1.1 MB of text. It imitates the style but invents words, loses the
  thread within a few lines and does not understand the message. Nothing it says is factual.
- The chat validation numbers leak (see Results). Only stage A's text val loss is a clean held-out figure.
- Only the corpus's 68 symbols exist: digits other than 3, brackets, double quotes, accents and emoji are
  dropped from the message.
- A 256-character context: only the last 3 turns are kept, and long prompts are cropped.
- Random sampling: replies differ each time (pass `seed` for repeatable output) and can be empty or cut off.
- The plays contain archaic, violent or offensive language, and the model can reproduce it.
