---
license: mit
pipeline_tag: summarization
tags: [ml-lab, pytorch, automatic-speech-recognition, summarization, whisper, qwen2.5]
base_model: [openai/whisper-small, openai/whisper-tiny, Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen2.5-0.5B-Instruct]
ml_lab:
  title: AI Video Summarizer
  order: 10
  summary: Upload a video or audio clip and get a timestamped transcript, a short summary and 5 key points.
  framework: PyTorch (transformers)
  architecture: Whisper speech-to-text + Qwen2.5-Instruct summarizer (pretrained pipeline)
  dataset: none (pretrained models; evaluated on the JFK inaugural clip)
  space: shalev396/video-summarizer
  runtime: zerogpu
  ui_kind: custom
  pipeline_only: true
  colab: https://colab.research.google.com/github/shalev396/ml-lab/blob/main/video-summarizer/training/notebook.ipynb
  github: https://github.com/shalev396/ml-lab/tree/main/video-summarizer
---
# 🎬 AI Video Summarizer: models

This project is an **application built from pretrained models**, not a model trained here.
There are no weights in this folder and no `shalev396/video-summarizer` model repo: the Space
downloads the four public checkpoints below from the Hub and chains them.

| role | Hub id | parameters | license | used on |
|---|---|---|---|---|
| speech-to-text | [openai/whisper-small](https://huggingface.co/openai/whisper-small) | 242M | Apache-2.0 | ZeroGPU (fp16) |
| speech-to-text | [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) | 37.8M | Apache-2.0 | CPU (fp32) |
| summarizer | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | 1.54B | Apache-2.0 | ZeroGPU (fp16) |
| summarizer | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | 494M | Apache-2.0 | CPU (fp32) |

Parameter counts are measured by the training notebook (`model_builder.describe`), rounded.

## How the pipeline works
1. PyAV decodes the first audio stream of the uploaded video/audio file to 16 kHz mono float32
   (first 180 s only).
2. Whisper, via the transformers `automatic-speech-recognition` pipeline (30 s chunks,
   `return_timestamps=True`), produces the transcript and timestamped segments.
3. Qwen2.5-Instruct gets the transcript through its chat template and is asked for JSON
   `{"summary", "key_points"}`; the reply is parsed robustly (code fences, extra text, bullet fallback).
   With fewer than 5 key points, it is asked once more for the missing ones, and any gap left is
   filled with summary or transcript sentences. Only a very short input can end with fewer than 5.
   A pasted transcript skips steps 1 and 2.

## Where things live
- [`../space`](../space): the app, `/predict` API and `pipeline.py` (the single source of truth).
- [`../training`](../training): the notebook that runs and evaluates the pipeline (WER on the JFK
  clip, timings, summary length) and exports the Space examples + `results.json`.

## Limitations
- English speech was the only speech evaluated (one 11 s clip); Whisper supports other languages
  but quality there was not measured here.
- Small LLMs can drop or blur facts, and the 0.5B model on CPU often writes key points as short
  noun phrases instead of sentences. Summaries are a convenience, not a faithful record.
- Only the first 3 minutes of audio are used, so long videos are summarized from their opening.
