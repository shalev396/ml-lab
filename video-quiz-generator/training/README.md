# AI Quiz Generator: training (pipeline evaluation)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shalev396/ml-lab/blob/main/video-quiz-generator/training/notebook.ipynb)

This is a pipeline-only project: nothing is trained. The notebook runs the Space's own
[`space/pipeline.py`](../space/pipeline.py) (PyAV decode -> Whisper -> Qwen2.5-Instruct -> lenient JSON parsing,
strict validation, up to 2 repair turns and 1 fresh sampled retry) over a small evaluation grid, measures how often the output is a valid quiz, scores every
question with an LLM judge, and exports the results to `outputs/`.

## What the notebook does
1. **Setup**: 1a clones the repo if the files are missing (e.g. a fresh Colab runtime) and fetches `space/`
   (the code under test), 1b installs `../../requirements.txt`, 1c picks the `device` (cuda -> mps -> cpu),
   1d imports `src/` and `pipeline`.
2. **Config**: `Config()` from `src/config.py` and the `SMOKE_TEST` switch.
3. **Data**: `data_setup.load_samples()`: the JFK clip (downloaded to `data/` if missing), `sample.mp4`
   (built from it with PyAV), the Gettysburg Address text from `../space/examples/`.
4. **Load model**: `P.load(device)`, the pipeline the Space runs on this hardware; parameter counts.
5. **Training**: there is nothing to train, so this runs `engine.run_variant()` for every variant: ASR on the
   audio sample (with word error rate), then one quiz per sample and difficulty. Table + seconds per quiz.
6. **Evaluation**: `engine.judge_quizzes()` (G-Eval-lite) and `engine.summarize()`: the comparison table and plots.
7. **Inference**: `pipe.predict(media=../space/examples/sample.mp4, ...)` and the pasted-text path, exactly
   what the Space's `/predict` runs.
8. **Export**: `export.build_metrics()` + `export.export()`: `metrics.json`, every quiz with the raw model
   replies, the judgments and plots, and the results table in `../model/README.md`.

## Evaluation grid (defaults in `src/config.py`)
| | full run | smoke run (`SMOKE_TEST=1`) |
|---|---|---|
| Variants | `cpu` (whisper-tiny + Qwen2.5-0.5B), `gpu` (whisper-small + Qwen2.5-1.5B, deployed) | `cpu` only |
| Samples | JFK clip (audio), Gettysburg Address (text) | JFK clip |
| Difficulties | easy, medium, hard | mixed |
| Questions per quiz | 4 | 2 |
| Judge | Qwen2.5-1.5B-Instruct | Qwen2.5-0.5B-Instruct |
| Output | `outputs/results/` + table in `../model/README.md` | `outputs/smoke/` only |

Metrics per variant: `schema_valid_rate` (primary), `first_try_valid_rate`, `strict_json_first_try_rate`,
`mean_attempts`, `repair_rate`, `regenerate_rate`, `question_yield`,
`answer_agreement` (judge's blind pick = marked answer), `judge_relevance`, `judge_clarity` (1-5), `asr_wer`,
`mean_quiz_seconds`. Definitions are in [`src/engine.py`](src/engine.py).

## Run it
- **Colab**: click the badge (a GPU runtime is preselected), then *Run all*.
- **Locally**: `pip install -r ../../requirements.txt`, then open `notebook.ipynb` from this folder.
- **Headless**:
  ```bash
  SMOKE_TEST=1 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp   # sanity run
  SMOKE_TEST=0 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp --ExecutePreprocessor.timeout=7200
  ```

## Hardware and time
Greedy generation is the cost. Measured on a shared desktop CPU (torch 2.14, 6 threads, other jobs running):
Qwen2.5-0.5B writes a 2-4 question quiz in roughly 25-100 s, Qwen2.5-1.5B a 4-question quiz in about 190 s,
and whisper-tiny transcribes the 11 s clip in 1-2 s. The smoke run (one 2-question quiz, 2 judge calls, one video inference) takes a few minutes on
an idle CPU; the headless check on 2026-09-25, on a CPU shared with other training jobs, took 19 min. The
full run (12 quizzes + about 48 judge calls, including the 1.5B model) takes around 40-60 min on CPU (the
evaluation part took 40 min on 2026-09-25; the 0.5B variant used all 5 calls of the retry policy per quiz) and a few
minutes on a GPU (not measured here). The first run also downloads the checkpoints (about 4.5 GB in total).

## Data
- `jfk.wav`: 11 s, 16 kHz mono, from [whisper.cpp `samples/jfk.wav`](https://github.com/ggml-org/whisper.cpp/tree/master/samples);
  fallback mirror: `Xenova/transformers.js-docs` on the Hugging Face Hub. Cached in `data/` (gitignored).
- `sample.mp4`: built locally from `jfk.wav` by `data_setup.build_sample_video()` (PyAV, H.264 + AAC).
- `gettysburg.txt`: the Gettysburg Address (Bliss copy, public domain), in `../space/examples/`.

## `src/` file map
| file | role |
|---|---|
| `__init__.py` | puts `../space` on `sys.path`, so `import pipeline as P` is the Space's code |
| `config.py` | `@dataclass Config`: variants, grid, judge, smoke handling, `out_dir` |
| `data_setup.py` | JFK download with fallback, PyAV video builder, sample loader |
| `model_builder.py` | builds pipeline variants and the judge from `pipeline.py`; parameter counts |
| `engine.py` | `run_variant` (ASR + quiz grid), `judge_quizzes` (G-Eval-lite), `wer`, `summarize` |
| `export.py` | `metrics.json` builder, results/plots writer, results table in the model README |
| `utils.py` | shared ml-lab helpers, identical in every project |
