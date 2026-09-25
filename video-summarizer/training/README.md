# AI Video Summarizer: training (run + evaluate the pipeline)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shalev396/ml-lab/blob/main/video-summarizer/training/notebook.ipynb)

This is a **pipeline-only** project: nothing is trained. The notebook runs the exact pipeline the
Space serves (`../space/pipeline.py`: PyAV decode -> Whisper -> Qwen2.5-Instruct JSON summary) on
sample media, evaluates it, and exports the Space examples plus a `results.json`.

## What the notebook does
1. **Setup**: 1a finds the project files (clones the repo + the `space` submodule on a fresh runtime),
   1b installs `../../requirements.txt`, 1c picks `device` (cuda -> mps -> cpu), then imports `src` and `pipeline`.
2. **Config**: `Config()` from `src/config.py`, including the `SMOKE_TEST` switch.
3. **Data**: gets `jfk.wav`, `sample.mp4` and `lecture_transcript.txt` into `training/data/`, then decodes
   and plots the audio exactly as the Space does (`P.load_audio`).
4. **Load model**: `model_builder.build_pipeline` builds `pipeline.Pipeline` and lists the Hub ids and parameter counts.
5. **Training**: there is nothing to train. Each variant runs on every sample stage by stage
   (`engine.run_samples`: decode -> transcribe -> LLM reply -> parse), with timings.
6. **Evaluation**: WER against the known JFK words, real-time factor, LLM seconds, summary length,
   key-point count and JSON rate for each variant, a per-stage timing plot, and the lecture summaries side by side.
7. **Inference**: `P.load(device).predict(media=...)` and `predict(transcript=...)` on `../space/examples`,
   which is what `/predict` runs.
8. **Export**: writes `../space/examples/` and `outputs/results.json` + `outputs/assets/timings.png`.
   There is no model repo to upload to.

## Variants
| variant | speech-to-text | summarizer | where it is used |
|---|---|---|---|
| `cpu` | openai/whisper-tiny (37.8M) | Qwen/Qwen2.5-0.5B-Instruct (494M) | CPU hardware (fp32) |
| `gpu` | openai/whisper-small (241.7M) | Qwen/Qwen2.5-1.5B-Instruct (1.54B) | the ZeroGPU Space (fp16) |

## Results (full run, 2026-09-25, Windows desktop CPU, 6 torch threads, machine shared with other jobs)
From `outputs/results.json`. Both variants ran **on the CPU**, so the `gpu` timings are not what the
ZeroGPU Space sees. WER, RTF and ASR time are averaged over `jfk.wav` + `sample.mp4`. LLM time, summary
words and key points are averaged over all three inputs.

| variant | WER | RTF (ASR s / audio s) | ASR s | LLM s | summary words | key points | valid JSON |
|---|---|---|---|---|---|---|---|
| `cpu` | 0.000 | 0.40 | 4.4 | 21.0 | 29.0 | 5.0 | 3/3 |
| `gpu` (on CPU) | 0.000 | 0.99 | 10.8 | 53.6 | 21.3 | 5.0 | 3/3 |

Both Whisper sizes transcribe the 22-word JFK clip perfectly, so the WER test is a sanity check, not a
benchmark. The larger LLM wrote clearer key points for the lecture: whole sentences instead of the 0.5B
model's noun phrases such as "Carbon Dioxide Absorption". The notebook prints both.

## Run it
- **Colab:** click the badge (a GPU runtime is preselected), then *Run all*. On a GPU the Inference
  section uses the `gpu` pair.
- **Local:** `pip install -r ../../requirements.txt`, then open `notebook.ipynb` from this folder.
- **Headless:** `SMOKE_TEST=1 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir <tmp>`.

## Hardware and time
| run | CPU (measured, this desktop) | GPU |
|---|---|---|
| smoke (`cpu` variant only, outputs in `outputs/smoke/`) | ~4.5 min | not measured |
| full (both variants) | ~7 min, plus ~1 GB whisper-small + ~3 GB Qwen-1.5B download on the first run | not measured |

## Data
- `jfk.wav`: the 11 s JFK inaugural-address clip (16 kHz mono). Source:
  [whisper.cpp samples/jfk.wav](https://github.com/ggml-org/whisper.cpp/blob/master/samples/jfk.wav).
  Fallback: [openai/whisper tests/jfk.flac](https://github.com/openai/whisper/blob/main/tests/jfk.flac),
  decoded with PyAV and written as WAV.
- `sample.mp4`: the same audio in an H.264/AAC video. `data_setup.make_video` builds it with PyAV if it is missing.
- `lecture_transcript.txt`: a short lecture on photosynthesis written for this project (`data_setup.LECTURE_TRANSCRIPT`),
  used for the paste-a-transcript path.
- WER reference: the spoken words, "And so, my fellow Americans, ask not what your country can do for
  you, ask what you can do for your country." (`config.JFK_REFERENCE`).

## `src/` file map
| file | role |
|---|---|
| `__init__.py` | puts `../space` on `sys.path`, so `import pipeline as P` is the Space's pipeline |
| `config.py` | `@dataclass Config`: samples, variants, generation limits, smoke handling, output dirs |
| `data_setup.py` | download with fallback, PyAV video builder, lecture text, sample description |
| `model_builder.py` | `build_pipeline(cfg, device, variant)`, `describe()` (Hub ids + parameter counts) |
| `engine.py` | stage-timed runs, `wer()`, JSON-reply check, per-variant evaluation table |
| `export.py` | Space examples writer, timing plot, `results.json` |
| `utils.py` | shared ml-lab helpers, identical in every project |
