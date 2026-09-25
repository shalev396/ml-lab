# 💬 FLAN-T5 Dialogue Summarizer + RAG

> Paste a conversation and get a short summary from a LoRA-tuned FLAN-T5. Ask a fictional store a question and
> the same model answers from the store's own documentation.

<p>
  <a href="https://huggingface.co/spaces/shalev396/flan-t5-dialogue-summarizer"><img alt="Space" src="https://img.shields.io/badge/🤗%20Space-live%20demo-blue"></a>
  <a href="https://huggingface.co/shalev396/flan-t5-dialogue-summarizer"><img alt="Model" src="https://img.shields.io/badge/🤗%20Model-shalev396%2Fflan--t5--dialogue--summarizer-yellow"></a>
  <a href="https://colab.research.google.com/github/shalev396/ml-lab/blob/main/flan-t5-dialogue-summarizer/training/notebook.ipynb"><img alt="Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>

| | |
|---|---|
| **Task** | Dialogue summarization (seq2seq), plus retrieval-augmented Q&A over a small knowledge base |
| **Framework** | PyTorch · 🤗 Transformers · PEFT (LoRA, training only) |
| **Architecture** | `google/flan-t5-base` (247.6M parameters, frozen) + LoRA r=16 on attention q/v (1,769,472 trained parameters) |
| **Dataset** | [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum): 3,000 train / 200 validation / 200 test dialogues (seeded subsets) |
| **Result** | <!-- result:start -->ROUGE-L = 35.32 (200 test dialogues), +14.63 over zero-shot flan-t5-base<!-- result:end --> |
| **Runs on** | Space: ZeroGPU with CPU fallback · Training: GPU (recommended), CPU or Colab |

## The problem
Chat logs, support calls and meeting notes pile up faster than anyone can read them. A good summary says who wanted
what and how it ended, in one or two sentences. Large instruction-tuned models can do this zero-shot, but a small
model that runs cheaply does it poorly without help. This project measures how far each kind of help goes for
FLAN-T5-base: prompting it with examples, or fine-tuning a tiny LoRA adapter.

The second half is retrieval-augmented generation (RAG). A language model does not know a store's return policy.
If you ask it anyway, it invents one. Retrieving the relevant policy text and putting it in the prompt grounds the
answer in real documents.

## The data
- **[DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum)** (Chen et al., 2021): 13,460 two-speaker
  English dialogues about everyday topics (shopping, travel, work, doctors) with human-written third-person
  summaries. Speakers are anonymised as `#Person1#`, `#Person2#`. The splits are 12,460 train, 500 validation
  and 500 test dialogues, and every test dialogue has **3** independent human summaries.
- Used here: a seeded subset of **3,000** train dialogues for fine-tuning, **200** validation dialogues for the
  loss after each epoch, and **200** test dialogues, scored once at the end against all 3 references.
- Downloaded from the Hub and cached in `training/data/`. The fallback is the authors' JSONL files on
  [GitHub](https://github.com/cylnlp/dialogsum).
- **Nova Gadgets knowledge base** (`model/assets/kb.json`): 41 short hand-written documents about a fictional
  electronics shop, covering shipping, returns, warranty, product specs, payments, account and support.
  18 held-out questions, each labelled with its answering document, measure the retriever.

## Architecture
```
dialogue ─► "Summarize the following conversation.\n\n{dialogue}\n\nSummary: "
         ─► FLAN-T5-base encoder-decoder (frozen)  +  LoRA ΔW = B·A (rank 16) on every attention q and v
         ─► greedy decoding (≤ 96 tokens) ─► summary

question ─► MiniLM-L6 embedding ─► cosine top-3 of 41 KB documents ─► "Answer ... using only the context"
         ─► the same FLAN-T5 ─► {"answer", "sources"}
```
- **LoRA** (Hu et al., 2021) freezes the pretrained weights and learns a low-rank update `ΔW = (α/r)·B·A` for
  selected matrices. With r=16 on the query and value projections of all 36 attention modules (12 encoder
  self-attention, 12 decoder self-attention, 12 decoder cross-attention), that is 72 matrices × 16·(768+768)
  = 1,769,472 parameters, **0.71 %** of the model, stored in a ~7 MB file.
- At load time `model.py` merges the adapter into the base weights on the CPU (`merge_and_unload`) and only then
  moves the model to the GPU. The served model is a plain FLAN-T5 with no adapter overhead. Merging on the CPU
  also avoids adapter-device problems under ZeroGPU.
- **RAG retrieval** uses `sentence-transformers/all-MiniLM-L6-v2` (22.7M parameters) on the CPU with normalized
  embeddings and a dot product. No vector database is needed for 41 documents.

## Training & experiments
- **Recipe**: AdamW with lr 1e-3, weight decay 0.01, linear decay to 0, batch size 8, 3 epochs (1,125 steps),
  gradient clipping at 1.0, and per-batch dynamic padding (inputs ≤ 512 tokens, summaries ≤ 128).
- **Precision**: float32, or bf16 autocast on GPUs with bf16 support (Ampere or newer). fp16 is never used: T5
  overflows in fp16 and the loss is NaN from the first step. A Colab T4 therefore trains in float32.
- **Experiments** (same 200 test dialogues, greedy decoding):
  1. base FLAN-T5, zero-shot (the instruction only)
  2. base, one-shot (one solved train example in the prompt)
  3. base, few-shot (k=2)
  4. **LoRA fine-tuned** (deployed)
- **RLHF** is explained in the notebook but not trained. DialogSum has reference summaries, not human
  preference pairs, so supervised fine-tuning is the right stopping point.
- **Resumable**: the adapter and optimizer state are checkpointed after every epoch, so a long CPU run that gets
  interrupted continues from the last finished epoch.

## Results
ROUGE x100 on 200 DialogSum test dialogues, each scored against its 3 human summaries. Greedy decoding for
every variant.

| variant | ROUGE-1 | ROUGE-2 | ROUGE-L | avg. words |
|---|---|---|---|---|
| base zero-shot | 24.41 | 7.31 | 20.69 | 13.9 |
| base one-shot | 24.57 | 7.02 | 20.76 | 15.5 |
| base few-shot (k=2) | 24.47 | 6.85 | 20.81 | 15.1 |
| **LoRA fine-tuned** (deployed) | **43.69** | **17.76** | **35.32** | 22.0 |

- Prompting alone barely moves the base model (one/few-shot +0.1 ROUGE-L); the adapter adds **+14.6 ROUGE-L**.
- No overfitting: validation loss fell every epoch (1.834 before training → 1.124 → 1.116 → 1.096) and ended
  level with the training loss (1.095).
- RAG retriever on 18 held-out store questions: hit@1 0.94 · hit@3 1.00 · MRR 0.97.
- Trained on an RTX 2080 Ti in fp32: 580 s (0.52 s/step), evaluation 663 s. Graphs and samples are in the
  [model card](https://huggingface.co/shalev396/flan-t5-dialogue-summarizer).

## Deployment
- **Model repo**: https://huggingface.co/shalev396/flan-t5-dialogue-summarizer (adapter, tokenizer, `model.py`,
  `handler.py`, knowledge base)
- **Space / API**: https://huggingface.co/spaces/shalev396/flan-t5-dialogue-summarizer.
  `POST /gradio_api/call/predict` (dialogue -> summary) and `POST /gradio_api/call/rag` (question -> answer +
  sources). Both return `[result, seconds, device]`. See the Space README for curl / JS / Python examples.
- **Inference Endpoint**: deploy the model repo from its page (Deploy → Inference Endpoints). `handler.py` takes
  `{"inputs": dialogue}` or `{"inputs": question, "parameters": {"task": "rag"}}`.

## Project structure
```
flan-t5-dialogue-summarizer/
├── README.md               this write-up
├── model/                  HF model repo (git submodule)
│   ├── model.py            prompt, generation, adapter merge, retriever, Predictor (single source of truth)
│   ├── handler.py          Inference Endpoint entry point
│   ├── requirements.txt    deps of model.py + handler.py
│   ├── assets/kb.json      Nova Gadgets knowledge base (41 docs)
│   └── README.md           model card
│   (after training: adapter_config.json, adapter_model.safetensors, tokenizer files, config.json,
│    metrics.json, assets/*.png)
├── space/                  HF Space (git submodule): app.py, space_utils.py, examples/, README.md
└── training/
    ├── notebook.ipynb      the controller: setup → config → data → model → train → evaluate → inference → export
    ├── src/                config, data_setup, model_builder, engine, export, utils
    └── README.md
```

## Reproduce
```bash
git clone --recurse-submodules https://github.com/shalev396/ml-lab && cd ml-lab
pip install -r requirements.txt
cd flan-t5-dialogue-summarizer/training
SMOKE_TEST=1 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp   # a few minutes, CPU
jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp --ExecutePreprocessor.timeout=-1  # full run
cd ../space && python app.py        # local Space on ../model (MODEL_DIR=../training/outputs/smoke/model for the smoke export)
```
Or open the notebook in [Colab](https://colab.research.google.com/github/shalev396/ml-lab/blob/main/flan-t5-dialogue-summarizer/training/notebook.ipynb)
with a GPU runtime. Timings are in [training/README.md](training/README.md).

## Limitations
- English, two-speaker, everyday chit-chat only (DialogSum's domain). Meetings with many speakers, and other
  languages, are out of domain.
- A 250M-parameter model makes factual mistakes in its summaries: it swaps speakers, drops facts or adds details.
  ROUGE measures word overlap, not faithfulness.
- 200 test dialogues give noisy estimates. Treat differences of a point or two between variants as ties.
- Inputs are truncated at 512 tokens, which affects about 2 % of DialogSum prompts.
- The RAG part is a demonstration: a fictional store, 41 documents, and a summarization fine-tune as the answer
  generator. The retriever always returns 3 documents, even for off-topic questions.
