# 🎬 AI Video Summarizer

> Upload a video or audio clip and get a timestamped transcript, a short summary and 5 key points,
> all from open models running in the Space itself (no external APIs).

<p>
  <a href="https://huggingface.co/spaces/shalev396/video-summarizer"><img alt="Space" src="https://img.shields.io/badge/🤗%20Space-live%20demo-blue"></a>
  <a href="https://colab.research.google.com/github/shalev396/ml-lab/blob/main/video-summarizer/training/notebook.ipynb"><img alt="Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>

| | |
|---|---|
| **Task** | video/audio -> transcript (speech recognition) -> summary + key points (LLM) |
| **Framework** | PyTorch · 🤗 transformers · PyAV |
| **Architecture** | pretrained pipeline: Whisper small / tiny + Qwen2.5-Instruct 1.5B / 0.5B (see [models](model/README.md)) |
| **Dataset** | none for training. Evaluated on the 11 s JFK inaugural clip (WAV + MP4) and a sample lecture transcript |
| **Result** | WER = 0.000 on the JFK clip for both Whisper sizes, 5/5 key points and valid JSON on 3/3 inputs (see [training/](training/README.md)) |
| **Runs on** | Space: ZeroGPU (CPU basic also works, with the small models) · Notebook: CPU, GPU or Colab |

## The problem
Long recordings are slow to skim. The goal is a self-hosted tool that turns a clip into text you can read in
seconds: what was said (with timestamps), a short summary, and the main points. The original
[ProjectPro brief](https://www.projectpro.io/project-use-case/ai-video-summarization-project) used a large
hosted LLM (Mixtral-8x7B). This version uses small open models that fit on a free ZeroGPU slice and still run on a CPU.

## The data
Nothing is trained, so there is no training set. The notebook checks the pipeline on:
- `jfk.wav`: the 11 s JFK clip that Whisper's own tests use (16 kHz mono), from
  [whisper.cpp](https://github.com/ggml-org/whisper.cpp/blob/master/samples/jfk.wav);
- `sample.mp4`: the same audio inside an H.264/AAC video, to exercise the video path;
- `lecture_transcript.txt`: a short photosynthesis lecture written for this project, to exercise the
  paste-a-transcript path.

All three are also the Space's examples.

## Architecture
```
video / audio file ──PyAV──> 16 kHz mono float32 (first 180 s)
        │
        └─> Whisper (transformers ASR pipeline, 30 s chunks, timestamps) ──> transcript + segments
                                                                               │
pasted transcript ─────────────────────────────────────────────────────────────┤
                                                                               v
                          Qwen2.5-Instruct (chat template, "reply with JSON") ──> {summary, key_points}
```
| hardware | speech-to-text | summarizer | precision |
|---|---|---|---|
| ZeroGPU | [openai/whisper-small](https://huggingface.co/openai/whisper-small) (241.7M) | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (1.54B) | fp16 |
| CPU | [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) (37.8M) | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (494M) | fp32 |

All four are Apache-2.0. The LLM reply is parsed robustly (`pipeline.parse_summary`): plain JSON, JSON
inside code fences or extra text, truncated JSON, and plain bullet lines as a last resort. Whisper's
occasional degenerate loops ("IIIII…", a word repeated many times) are trimmed (`pipeline.clean_asr_text`).
PyAV's wheels bundle ffmpeg, so the Space needs no `packages.txt`.

## Training & experiments
There is no training. The [notebook](training/notebook.ipynb) evaluates the two model pairs, `cpu` and
`gpu`, stage by stage. Results of the full run (2026-09-25, desktop CPU, both variants on the CPU):

| variant | WER | RTF | ASR s | LLM s | summary words | key points | valid JSON |
|---|---|---|---|---|---|---|---|
| `cpu`: whisper-tiny + Qwen2.5-0.5B | 0.000 | 0.40 | 4.4 | 21.0 | 29.0 | 5.0 | 3/3 |
| **`gpu`: whisper-small + Qwen2.5-1.5B** (timed on CPU) | 0.000 | 0.99 | 10.8 | 53.6 | 21.3 | 5.0 | 3/3 |

The `gpu` pair is the one deployed on ZeroGPU. Its timings above are CPU timings and are much slower
than on the GPU. GPU timings were not measured here.

## Results
- The JFK clip is transcribed word-perfectly by both Whisper sizes, from the WAV and from the MP4.
- On the lecture, Qwen2.5-1.5B returns five full-sentence key points that match the text (for example
  "Glucose can be used immediately, stored as starch, or built into cell walls."). Qwen2.5-0.5B returns
  a good summary, but its key points are short noun phrases ("Carbon Dioxide Absorption").
- Every LLM reply in the run was valid JSON.

## Deployment
- **Space / API**: https://huggingface.co/spaces/shalev396/video-summarizer — `POST /gradio_api/call/predict`
  with `[media file | null, transcript text]` -> `[{language, transcript, segments, summary, key_points, models}, seconds, device]`
  (full docs in [space/README.md](space/README.md)).
- **Model repo**: none. This is an application built from pretrained models. [`model/README.md`](model/README.md)
  lists them with their licenses.
- ZeroGPU: the models are loaded on `cuda` at module level and each call runs in `@spaces.GPU(duration=120)`.
  A visitor who is out of GPU quota gets a clear error rather than a CPU fallback, because the 1.5B LLM would take minutes on a shared CPU.

## Project structure
```
video-summarizer/
├── model/README.md          models used + licenses (no weights, no Hub model repo)
├── space/                   Hugging Face Space (git submodule)
│   ├── app.py               Gradio UI + /predict
│   ├── pipeline.py          decode -> transcribe -> summarize (single source of truth)
│   ├── space_utils.py       shared ml-lab Space helpers
│   ├── requirements.txt     torch, transformers, av, numpy, huggingface_hub
│   └── examples/            jfk.wav, sample.mp4, lecture_transcript.txt
└── training/                notebook.ipynb + src/ (config, data_setup, model_builder, engine, export, utils)
```

## Reproduce
```bash
pip install -r requirements.txt                       # repo root
cd video-summarizer/training && jupyter lab notebook.ipynb     # or SMOKE_TEST=1 for a quick run
cd ../space && python app.py                          # local Space at http://127.0.0.1:7860
```

## Limitations
- Only the first 3 minutes of audio are used, and uploads are capped at 8 MB.
- The evaluation is small: one 22-word English clip for WER and one lecture for the summaries. Accuracy
  on noisy audio, accents or other languages was not measured.
- Small LLMs can drop facts or add generic statements. The summaries are a reading aid, not a record.
- Whisper's segment timestamps from the chunked pipeline are coarse: often one segment per ~30 s chunk.
