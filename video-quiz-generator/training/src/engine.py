"""The evaluation loop: run the pipeline over the grid, judge the questions, aggregate the metrics.

Metrics per variant
- asr_wer                 word error rate of the transcript vs the reference (audio samples only)
- schema_valid_rate       share of quizzes that pass validation after the whole retry policy
                          (up to 2 repair turns, then 1 fresh sampled generation + 1 repair)
- first_try_valid_rate    share whose first reply was valid (lenient parse), with no repair or retry
- strict_json_first_try_rate  share whose first reply was valid JSON as written (no syntax fix)
- mean_attempts           model calls per quiz (generations + repair turns)
- repair_rate             share of quizzes that needed at least one repair turn
- regenerate_rate         share of quizzes that needed a fresh generation
- question_yield          delivered well-formed questions / requested questions
- judge_relevance/clarity mean 1-5 scores from the judge (G-Eval-lite)
- answer_agreement        share of questions where the judge, answering blind, picks the same option
                          as the generator's answer_index (a proxy for "the marked answer is right")
"""
from __future__ import annotations

import re
import time

import numpy as np

import pipeline as P


# --------------------------------------------------------------------------- ASR
def _words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split()


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate (lower-cased, punctuation removed): edit distance / reference length."""
    ref, hyp = _words(reference), _words(hypothesis)
    d = np.arange(len(hyp) + 1)
    for i, r in enumerate(ref, 1):
        prev, d[0] = d.copy(), i
        for j, h in enumerate(hyp, 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r != h))
    return float(d[-1] / max(1, len(ref)))


# --------------------------------------------------------------------------- generation grid
def run_variant(pipe: P.QuizPipeline, name: str, samples: list[dict], cfg, log=print) -> dict:
    """Transcribe the audio samples, then generate one quiz per (sample, difficulty)."""
    asr_rows, transcripts = [], {}
    for s in samples:
        if s["kind"] == "audio":
            t0 = time.perf_counter()
            out = pipe.transcribe(s["path"])
            row = {"variant": name, "sample": s["name"], "asr_model": pipe.asr_model_id, "text": out["text"],
                   "audio_seconds": out["audio_seconds"], "seconds": round(time.perf_counter() - t0, 2),
                   "wer": round(wer(s["reference"], out["text"]), 4) if s.get("reference") else None}
            asr_rows.append(row)
            transcripts[s["name"]] = out["text"]
            log(f"[{name}] ASR {s['name']}: wer={row['wer']} ({row['seconds']} s) -> {out['text'][:80]!r}")
        else:
            transcripts[s["name"]] = s["text"]
    quizzes = []
    for s in samples:
        for diff in cfg.difficulties:
            t0 = time.perf_counter()
            try:
                quiz = pipe.generate_quiz(transcripts[s["name"]], cfg.n_questions, diff, repair=cfg.repair)
            except ValueError as err:  # empty transcript: an invalid quiz with no model call, not a crash
                quiz = {"questions": [], "valid": False, "valid_first_try": False, "strict_json_first_try": False,
                        "attempts": 0, "repairs": [], "regenerated": False, "parse": [], "chosen_attempt": None,
                        "problems": [str(err)], "transcript": transcripts[s["name"]], "raw": None}
            row = {"variant": name, "sample": s["name"], "difficulty": diff, "llm_model": pipe.llm_model_id,
                   "n_requested": cfg.n_questions, "n_delivered": len(quiz["questions"]),
                   "valid": quiz["valid"], "valid_first_try": quiz["valid_first_try"],
                   "strict_json_first_try": quiz["strict_json_first_try"], "attempts": quiz["attempts"],
                   "repairs": quiz["repairs"], "regenerated": quiz["regenerated"], "parse": quiz["parse"],
                   "chosen_attempt": quiz["chosen_attempt"], "problems": quiz["problems"],
                   "seconds": round(time.perf_counter() - t0, 2), "transcript": quiz["transcript"],
                   "questions": quiz["questions"], "raw": quiz["raw"]}
            quizzes.append(row)
            log(f"[{name}] quiz {s['name']}/{diff}: valid={row['valid']} (first try {row['valid_first_try']}, "
                f"{row['attempts']} call(s), repairs={row['repairs']}, regenerated={row['regenerated']}, "
                f"parse={row['parse']}, {row['n_delivered']}/{row['n_requested']} questions, {row['seconds']} s)")
    return {"asr": asr_rows, "quizzes": quizzes}


# --------------------------------------------------------------------------- G-Eval-lite judge
JUDGE_SYSTEM = "You are a strict reviewer of quiz questions. You reply with ONE JSON object only."


def judge_messages(transcript: str, question: dict) -> list[dict]:
    options = "\n".join(f"{i}) {o}" for i, o in enumerate(question["options"]))
    user = (f'Transcript:\n"""\n{transcript}\n"""\n\nQuestion: {question["question"]}\n{options}\n\n'
            "Review this multiple-choice question about the transcript:\n"
            '- "answer_index": using only the transcript, the number (0-3) of the correct option.\n'
            '- "relevance": 1-5, how much the question is about the content of the transcript.\n'
            '- "clarity": 1-5, is the question clear, with exactly one defensible correct option.\n'
            'Reply with JSON only: {"answer_index": <0-3>, "relevance": <1-5>, "clarity": <1-5>}')
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def _score(v, lo: int, hi: int):
    return v if isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi else None


def judge_quizzes(judge: P.ChatLLM, quizzes: list[dict], max_new_tokens: int = 60, log=print) -> list[dict]:
    """One judge call per delivered question. Unparseable judgments are kept (scores None)."""
    rows = []
    for qz in quizzes:
        for i, q in enumerate(qz["questions"]):
            reply = judge.chat(judge_messages(qz["transcript"], q), max_new_tokens=max_new_tokens)
            try:
                data = P.extract_json(reply)
            except ValueError:
                data = {}
            pick = _score(data.get("answer_index"), 0, 3)
            rows.append({"variant": qz["variant"], "sample": qz["sample"], "difficulty": qz["difficulty"],
                         "question_index": i, "question": q["question"], "answer_index": q["answer_index"],
                         "judge_answer": pick, "agree": None if pick is None else pick == q["answer_index"],
                         "relevance": _score(data.get("relevance"), 1, 5),
                         "clarity": _score(data.get("clarity"), 1, 5), "raw": reply})
        log(f"[judge] {qz['variant']} {qz['sample']}/{qz['difficulty']}: {len(qz['questions'])} question(s) scored")
    return rows


# --------------------------------------------------------------------------- aggregation
def _mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return round(float(np.mean(xs)), 4) if xs else None


def summarize(runs: dict, judgments: list[dict] | None = None) -> dict:
    """{variant: {metric: value}} from run_variant() outputs and judge rows."""
    judgments = judgments or []
    out = {}
    for name, run in runs.items():
        qz, js = run["quizzes"], [j for j in judgments if j["variant"] == name]
        out[name] = {
            "asr_wer": _mean(r["wer"] for r in run["asr"]),
            "schema_valid_rate": _mean(q["valid"] for q in qz),
            "first_try_valid_rate": _mean(q["valid_first_try"] for q in qz),
            "strict_json_first_try_rate": _mean(q["strict_json_first_try"] for q in qz),
            "mean_attempts": _mean(q["attempts"] for q in qz),
            "repair_rate": _mean(bool(q["repairs"]) for q in qz),
            "regenerate_rate": _mean(q["regenerated"] for q in qz),
            "question_yield": round(sum(q["n_delivered"] for q in qz) / max(1, sum(q["n_requested"] for q in qz)), 4),
            "mean_quiz_seconds": _mean(q["seconds"] for q in qz),
            "judge_relevance": _mean(j["relevance"] for j in js),
            "judge_clarity": _mean(j["clarity"] for j in js),
            "answer_agreement": _mean(j["agree"] for j in js),
            "judge_parse_rate": _mean(j["agree"] is not None for j in js) if js else None,
        }
    return out
