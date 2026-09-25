# AI Quiz Generator: models used

This project is an **application built from pretrained models**, not a model trained here. There are no
weights in this folder and no Hugging Face model repo: the app is the
[Space](https://huggingface.co/spaces/shalev396/video-quiz-generator) ([`../space`](../space), code in
`space/pipeline.py`), and [`../training`](../training) holds the notebook that runs and evaluates the
pipeline.

## Pretrained models

| Stage | Hub id | Params | License | Used on |
|---|---|---|---|---|
| Speech-to-text | [openai/whisper-small](https://huggingface.co/openai/whisper-small) | 241.7M | Apache-2.0 | ZeroGPU (deployed) |
| Speech-to-text | [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) | 37.8M | Apache-2.0 | CPU |
| Quiz writer | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | 1.54B | Apache-2.0 | ZeroGPU (deployed); also the evaluation judge |
| Quiz writer | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | 494M | Apache-2.0 | CPU |

Audio/video decoding uses [PyAV](https://github.com/PyAV-Org/PyAV) (its wheels bundle FFmpeg), so no
system `ffmpeg` is needed.

## Pipeline
1. **Decode**: PyAV reads the first audio stream of any audio or video file, resampled to 16 kHz mono,
   capped at 180 s.
2. **Transcribe**: transformers Whisper on 30 s windows, decoded as one batch (fp16 on GPU, fp32 on CPU).
3. **Write the quiz**: Qwen2.5-Instruct gets the transcript (up to 6000 characters), the requested number
   of questions and the difficulty, with one worked example (an unrelated topic) as a previous chat turn.
   Decoding is greedy with a mild repetition penalty (1.1).
4. **Parse leniently** (`parse_reply`, syntax only, never the words): the outermost `{...}` of the
   reply is taken (markdown fences and text around it ignored); smart quotes used as delimiters, trailing
   commas, a missing comma between objects, raw newlines and unescaped quotation marks inside strings are
   fixed, and missing closing brackets at the very end are added. If the object still does not parse,
   every complete `{"question": ...}` object in it is salvaged on its own. Each attempt is recorded as
   `strict` (valid JSON as written), `lenient`, `salvaged` or `unreadable`.
5. **Validate strictly** against
   `{"questions": [{"question": str, "options": [4 str], "answer_index": 0-3, "explanation": str}]}`:
   non-empty strings, 4 different options that are not placeholders like "Option A", an integer
   `answer_index` in range, no repeated question, and exactly the requested number of questions.
6. **Retry policy**, stopping at the first fully valid attempt:
   - up to **2 repair turns** in the same chat: when every question was fine but there were too few, the
     model is asked only for the missing ones; otherwise it gets the list of problems (a JSON syntax
     error first) and is asked for the whole corrected object;
   - if still invalid, **1 fresh generation** in a new chat with light seeded sampling (temperature 0.7,
     top-p 0.9, seed 1234), followed by **1 repair turn**. At most 5 model calls per quiz.

   The first valid attempt is returned, else the one with the most well-formed questions, flagged
   `valid: false`. The request fails only when no attempt produced a single usable question.

## Evaluation
The notebook runs both hardware variants (`cpu` = whisper-tiny + Qwen2.5-0.5B, `gpu` = whisper-small +
Qwen2.5-1.5B) on the JFK inaugural clip (11 s audio, through the ASR) and the Gettysburg Address (text),
one quiz of 4 questions per sample and difficulty (easy, medium, hard). It reports:
- **schema_valid_rate** (primary): share of quizzes that pass every check after the whole retry policy;
  **first_try_valid_rate**: share whose first reply was valid (after lenient parsing) with no repair or
  retry; **strict_json_first_try_rate**: share whose first reply was valid JSON as written;
  **mean_attempts** (model calls per quiz), **repair_rate** (share that needed a repair turn),
  **regenerate_rate** (share that needed the fresh generation); **question_yield**: well-formed questions
  delivered / requested.
- **G-Eval-lite judge** (Qwen2.5-1.5B-Instruct, evaluation only): for every question it picks the
  correct option itself without seeing the marked answer, and scores **relevance** and **clarity** 1-5.
  **answer_agreement** is the share of questions where its pick matches the generator's `answer_index`.
- **asr_wer**: word error rate of the JFK transcript against the known text.

## Experiments
Nothing is trained, so the experiments compare the two pipeline variants the Space can run:

| variant | ASR | quiz writer | where it runs |
|---|---|---|---|
| `cpu` | whisper-tiny (37.8M) | Qwen2.5-0.5B-Instruct (494M) | CPU hardware / local |
| **`gpu`** (deployed) | whisper-small (241.7M) | Qwen2.5-1.5B-Instruct (1.54B) | ZeroGPU |

Both get the same 6 quizzes (2 inputs x 3 difficulties, 4 questions each) and the same judge. The table
below is written by `training/src/export.py` from `training/outputs/results/metrics.json`; the headline
(primary) metric is the deployed `gpu` column. The `mean_quiz_seconds` of both variants were measured on
the same CPU machine, so they compare the variants with each other, not with the Space's GPU latency.

**Headline (deployed `gpu` variant, full run 2026-09-25): schema_valid_rate 1.00** (6/6 quizzes, all
valid JSON on the first try, mean 1.0 model call per quiz), question_yield 1.00, answer_agreement 0.75
(18/24 questions), asr_wer 0.00. The `cpu` variant (Qwen2.5-0.5B) reached 0.00 schema-valid (it used all
5 calls of the retry policy on every quiz) with a question yield of 0.58. The judge scored every question
relevance 4 / clarity 5, so those two rows do not discriminate.

<!-- metrics:start -->
| metric | cpu | gpu |
|---|---|---|
| schema_valid_rate | 0.000 | 1.000 |
| first_try_valid_rate | 0.000 | 1.000 |
| strict_json_first_try_rate | 0.667 | 1.000 |
| question_yield | 0.583 | 1.000 |
| answer_agreement | 0.214 | 0.750 |
| judge_relevance | 4.000 | 4.000 |
| judge_clarity | 5.000 | 5.000 |
| asr_wer | 0.000 | 0.000 |
| mean_attempts | 5.00 | 1.00 |
| repair_rate | 1.000 | 0.000 |
| regenerate_rate | 1.000 | 0.000 |
| mean_quiz_seconds | 166.3 | 153.6 |

_12 quizzes of 4 questions (jfk, gettysburg x easy, medium, hard); judge Qwen/Qwen2.5-1.5B-Instruct; run 2026-09-25 on cpu._
<!-- metrics:end -->

**Graphs.** This folder holds only this README (a pipeline project has no model repo and no `assets/`),
so the plots are not published with the card. The notebook writes them to
`training/outputs/results/assets/`: `variant_comparison.png` (schema-valid rate, first-try valid rate,
question yield and answer agreement per variant) and `judge_scores.png` (judge relevance and clarity per
difficulty and variant). `training/outputs/` is regenerated by every run and is not committed.

## Limitations
- Very small sample: 2 inputs x 3 difficulties = 6 quizzes per variant. The rates are a sanity check of
  the output format, not a benchmark of question quality.
- The judge is a 1.5B model and is the same checkpoint as the deployed generator, so it may be lenient
  with its own mistakes; answer agreement is a proxy, not ground truth. In the full run it gave every
  question relevance 4 and clarity 5, so those scores carry no signal.
- Qwen2.5-0.5B (the CPU variant) often writes too few questions, repeats options or marks the wrong
  answer; the repair turns fix the format more often than the content.
- Whisper can mis-hear names and technical terms, and only the first 180 s of a recording are used.
- Questions are only as good as the transcript: long recordings are cut to 6000 characters.
