# 📝 AI Quiz Generator

> Turn a lecture, talk or meeting recording into a multiple-choice quiz you can take right away.

<p>
  <a href="https://huggingface.co/spaces/shalev396/video-quiz-generator"><img alt="Space" src="https://img.shields.io/badge/🤗%20Space-live%20demo-blue"></a>
  <a href="https://colab.research.google.com/github/shalev396/ml-lab/blob/main/video-quiz-generator/training/notebook.ipynb"><img alt="Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>

| | |
|---|---|
| **Task** | audio/video or transcript -> multiple-choice quiz (strict JSON) |
| **Framework** | PyTorch + transformers, PyAV for decoding |
| **Architecture** | pipeline of pretrained models: Whisper (ASR) -> Qwen2.5-Instruct (quiz writer) -> lenient JSON parsing + strict schema validation, with up to 2 repair turns and 1 fresh retry. ZeroGPU: whisper-small (241.7M) + Qwen2.5-1.5B-Instruct (1.54B); CPU: whisper-tiny (37.8M) + Qwen2.5-0.5B-Instruct (494M) |
| **Dataset** | evaluation only: the [JFK inaugural clip](https://github.com/ggml-org/whisper.cpp/tree/master/samples) (11 s audio) and the Gettysburg Address (public-domain text) |
| **Result** | deployed variant (whisper-small + Qwen2.5-1.5B): **schema-valid rate 1.00** (6/6 quizzes, all valid JSON on the first try, no repair needed), question yield 1.00, judge answer agreement 0.75, ASR WER 0.00 on the JFK clip. CPU variant (Qwen2.5-0.5B): schema-valid 0.00, yield 0.58. Full table in [`model/README.md`](model/README.md#experiments) |
| **Runs on** | Space: ZeroGPU · Evaluation notebook: CPU, GPU or Colab |

## The problem
Making a quiz from a recording takes a person a while: listen, pick the key facts, write questions with
believable wrong answers. The goal here is a one-click version: drop in a file (or paste a transcript),
choose how many questions and how hard, and get a quiz that software can use directly. That last part
matters: a quiz that is "almost JSON" is useless to an app, so the output has a fixed schema and is checked.

## The data
Nothing is trained, so there is no training set. The pipeline is evaluated on two inputs:
- the 11 s JFK inaugural clip ("ask not what your country can do for you..."), which goes through the
  speech recogniser; its true text is known, so the word error rate can be measured. A video version
  (`sample.mp4`) is built from it with PyAV to test the video path;
- Lincoln's Gettysburg Address as pasted text (about 1,450 characters), a longer input with more to ask about.

## Architecture
```
media file ─► PyAV decode (16 kHz mono, ≤ 180 s) ─► Whisper (30 s windows, one batch) ─┐
pasted transcript ─────────────────────────────────────────────────────────────────────┤
                                                                                       ▼
            Qwen2.5-Instruct (greedy, one worked example in the prompt) ─► JSON reply
                                                                                       ▼
   lenient parse (fences, smart quotes, trailing commas, stray quotes; salvage whole questions)
                                                                                       ▼
   validate {questions: [{question, options[4], answer_index 0-3, explanation}]} + exact count
                    │ valid ─► quiz                  │ problems ─► up to 2 repair turns:
                    │                                │  "fix these problems" or "write the N missing questions"
                    │                                │ still invalid ─► 1 fresh sampled generation (+1 repair)
                    ▼                                ▼
                 /predict  ◄──── first valid attempt, else the one with most good questions (flagged)
```
- **Decoding** with [PyAV](https://github.com/PyAV-Org/PyAV) (bundled FFmpeg), so no system `ffmpeg`.
- **ASR** with transformers `WhisperForConditionalGeneration`: whisper-small on the GPU, whisper-tiny on CPU.
- **Quiz writer**: Qwen2.5-1.5B-Instruct on the GPU, Qwen2.5-0.5B-Instruct on CPU. The prompt shows one
  worked example (a 2-question quiz on an unrelated topic) as a previous chat turn, because small models
  follow a concrete example much better than a schema description. Greedy decoding with a mild repetition
  penalty (1.1) stops loops like four identical options.
- **Validation** (`space/pipeline.py:validate_quiz`): non-empty strings, exactly 4 different options that
  are not placeholders ("Option A"), an integer `answer_index` 0-3, an explanation, and exactly the
  requested number of questions, and no repeated question.
- **Lenient parsing** (`parse_reply`): only the syntax is fixed, never the words. The outermost `{...}` is
  taken (markdown fences and text around it ignored); smart quotes used as delimiters, trailing commas,
  a missing comma between objects, raw newlines and unescaped quotation marks inside a string (the
  failure that broke the notebook's video example: `"explanation": "The speaker states that "and so..."`)
  are fixed, and missing closing brackets at the very end are added. If the whole object still does not
  parse (e.g. a bracket closed too early), every complete `{"question": ...}` object in it is salvaged on
  its own. Each attempt records how it was read: `strict`, `lenient`, `salvaged` or `unreadable`.
- **Retry policy** (`generate_quiz`), stopping at the first fully valid attempt: (1) one greedy
  generation; (2) up to **2 repair turns** in the same chat: if every question was fine but there were too
  few, the model is asked only for the missing ones (small models often stop early), otherwise it gets the
  list of problems (a JSON syntax error first); (3) if still invalid, **one fresh generation** in a new chat
  with light seeded sampling (temperature 0.7, top-p 0.9, seed 1234, so it is reproducible) and **1 repair
  turn**. At most 5 model calls. The first valid attempt is returned, else the one with the most
  well-formed questions (flagged `valid: false`); the call fails only if no attempt gave a single usable
  question. On ZeroGPU no new call starts after half of the GPU time budget.

The earlier version of this project also had an OpenAI (whisper-1 + GPT-4o via LangChain) path; it was
removed so the app runs on open models only, with no API key or per-call cost.

## Training & experiments
There is no training step. The [notebook](training/notebook.ipynb) runs the Space's own `pipeline.py` on
both hardware variants (`cpu`, `gpu`) over the grid 2 inputs x 3 difficulties (easy, medium, hard), 4
questions per quiz, and then runs a **G-Eval-lite judge** (Qwen2.5-1.5B-Instruct): for every question the
judge picks the correct option itself without seeing the marked answer, and scores relevance and clarity
1-5. Metrics: `schema_valid_rate` (primary, after the whole retry policy), `first_try_valid_rate` (valid with
no repair or retry), `strict_json_first_try_rate` (first reply was valid JSON as written),
`mean_attempts`, `repair_rate`, `regenerate_rate`, `question_yield`, `answer_agreement`, `judge_relevance`,
`judge_clarity`, `asr_wer`, `mean_quiz_seconds`. Every quiz in `training/outputs/results/quizzes.json` keeps
all raw replies, the parse mode of each attempt and the repairs that ran.

## Results
Full run of 2026-09-25 (CPU, 12 quizzes of 4 questions: 2 inputs x 3 difficulties per variant, judge
Qwen2.5-1.5B-Instruct):

| metric | `cpu` (whisper-tiny + Qwen2.5-0.5B) | **`gpu`** (whisper-small + Qwen2.5-1.5B, deployed) |
|---|---|---|
| schema_valid_rate (primary) | 0.00 (0/6) | **1.00 (6/6)** |
| first_try_valid_rate | 0.00 | 1.00 |
| strict_json_first_try_rate | 0.67 | 1.00 |
| mean_attempts (model calls per quiz) | 5.00 | 1.00 |
| repair_rate / regenerate_rate | 1.00 / 1.00 | 0.00 / 0.00 |
| question_yield | 0.58 (14/24) | 1.00 (24/24) |
| answer_agreement (judge) | 0.21 (3/14) | 0.75 (18/24) |
| asr_wer (JFK clip) | 0.00 | 0.00 |
| mean_quiz_seconds (same CPU for both) | 168.1 | 150.7 |

- The deployed 1.5B writer produced a schema-valid 4-question quiz on the first try for all 6 inputs, as
  plain valid JSON: the lenient parser and the retry turns were not needed. The judge's blind pick matched
  the marked answer for 18 of 24 questions, so roughly one marked answer in four may be wrong.
- The 0.5B CPU writer used all 5 model calls on every quiz and never reached a valid 4-question quiz: it
  stops early, repeats itself or breaks the JSON (of its 30 replies, 6 needed syntax fixes, 5 were salvaged and 2 were
  unreadable). Thanks to the salvage and retries the app still returns its well-formed questions
  (14 of 24, flagged `valid: false`) instead of failing, which is what happened before the retry policy
  on the notebook's video example.
- The judge gave every question relevance 4 and clarity 5, so those two scores carry no information
  here; answer agreement is the judge signal that discriminates.

The results table in [`model/README.md`](model/README.md#experiments) is written by
`training/src/export.py` from the full notebook run (`SMOKE_TEST=0`), together with
`training/outputs/results/` (every quiz with the raw model replies, the judgments and the plots). The
smoke run checks that the whole path works (see [`training/README.md`](training/README.md)); its
2-question numbers are not reported as results.

What was seen while building it (single runs on CPU, not a benchmark): Qwen2.5-0.5B often writes fewer
questions than asked, repeats options or gives an out-of-range `answer_index`; Qwen2.5-1.5B wrote a
schema-valid 4-question quiz on the Gettysburg text on the first try, but one of its four marked answers
was wrong (which is what the judge's answer agreement is there to catch).

## Deployment
- **Space / API**: https://huggingface.co/spaces/shalev396/video-quiz-generator — `POST /gradio_api/call/predict`,
  inputs `[media file | null, transcript, n_questions, difficulty]`, outputs `[quiz, seconds, device]`
  (full contract in [`space/README.md`](space/README.md)). The interactive quiz and grading are UI-only events.
- **Hardware**: ZeroGPU (`@spaces.GPU`, a time budget of 14 s + 2 s per question, at most 30 s). If the GPU
  call fails for any reason (no quota, timeout, CUDA error) the CPU pipeline answers in its own process
  (`space/cpu_backend.py`). On CPU hardware the same code runs the small variant.
- **No model repo / Inference Endpoint**: there are no trained weights; every model is a pretrained Hub
  checkpoint, preloaded into the Space (`preload_from_hub`). [`model/`](model/) only documents them.

## Project structure
```
video-quiz-generator/
├── README.md
├── model/README.md        the pretrained models used, pipeline details, evaluation table
├── space/                 Hugging Face Space (git submodule)
│   ├── app.py             Gradio UI + /predict (ZeroGPU)
│   ├── pipeline.py        decode -> ASR -> quiz LLM -> parse + validate + repair/retry (self-contained)
│   ├── space_utils.py     shared ml-lab Space helpers
│   ├── requirements.txt
│   └── examples/          sample.mp4, jfk.wav, gettysburg.txt
└── training/
    ├── notebook.ipynb     the controller: run + evaluate the pipeline
    ├── README.md
    └── src/               config, data_setup, model_builder, engine, export, utils
```

## Reproduce
```bash
pip install -r requirements.txt                     # from the repo root
cd video-quiz-generator/training
SMOKE_TEST=1 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp   # minutes on an idle CPU
SMOKE_TEST=0 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp --ExecutePreprocessor.timeout=7200
cd ../space && python app.py                        # local app on http://127.0.0.1:7860
```

## Limitations
- Tiny evaluation (6 quizzes per variant): it checks the output format and gives a rough idea of quality; it
  is not a benchmark.
- The judge is a 1.5B model, the same checkpoint as the deployed writer, so it can be lenient with its own
  mistakes; `answer_agreement` is a proxy for correctness, not ground truth, and its relevance/clarity
  scores were the same (4 and 5) for every question in the full run.
- Only the first 180 s of a recording and the first 6000 characters of a transcript are used.
- Whisper mis-hears names and jargon, and the CPU variant (0.5B) writes clearly weaker quizzes.
- Greedy decoding (the fresh retry samples with a fixed seed): the same input gives the same quiz on the
  same hardware.
- Lenient parsing fixes syntax only: a badly written question is dropped (or fixed by the model in a repair
  turn), never completed by the code, so a quiz can still come back with fewer questions than asked
  (`valid: false`).
