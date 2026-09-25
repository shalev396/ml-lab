"""Run the pipeline stage by stage on the samples and measure it (no training loop here).

Metrics: WER of the transcript vs the known reference, wall time per stage, real-time factor
(ASR seconds / audio seconds), summary length, key-point count, and whether the LLM reply was
valid JSON (before the lenient fallback parser)."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pipeline as P


def normalize(text: str) -> list[str]:
    """Lower-case words, punctuation removed (the usual WER normalisation)."""
    return re.sub(r"[^\w\s']", " ", text.lower()).split()


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate = (substitutions + deletions + insertions) / reference words."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[-1] / max(1, len(ref))


def is_json_reply(raw: str) -> bool:
    text = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    try:
        data = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        return False
    return isinstance(data, dict) and "summary" in data and "key_points" in data


def run_media(pipe: P.Pipeline, path: Path, reference: str | None = None) -> dict:
    """decode -> transcribe -> summarize one media file, timing every stage."""
    t0 = time.perf_counter()
    audio, duration = P.load_audio(path, pipe.max_seconds)
    t1 = time.perf_counter()
    asr = pipe.transcribe(audio)
    t2 = time.perf_counter()
    raw = pipe.generate(asr["transcript"])
    t3 = time.perf_counter()
    summary = P.parse_summary(raw, pipe.n_key_points)
    return {"sample": path.name, "audio_s": duration, "decode_s": t1 - t0, "asr_s": t2 - t1,
            "llm_s": t3 - t2, "total_s": t3 - t0, "rtf": (t2 - t1) / duration,
            "wer": wer(reference, asr["transcript"]) if reference else None,
            "json_ok": is_json_reply(raw), "summary_words": len(summary["summary"].split()),
            "n_key_points": len(summary["key_points"]),
            "output": {**asr, **summary, "models": pipe.models}}


def run_transcript(pipe: P.Pipeline, path: Path) -> dict:
    """Pasted-transcript path: summarize only."""
    text = path.read_text(encoding="utf-8").strip()
    t0 = time.perf_counter()
    raw = pipe.generate(text)
    t1 = time.perf_counter()
    summary = P.parse_summary(raw, pipe.n_key_points)
    return {"sample": path.name, "audio_s": None, "decode_s": 0.0, "asr_s": 0.0, "llm_s": t1 - t0,
            "total_s": t1 - t0, "rtf": None, "wer": None, "json_ok": is_json_reply(raw),
            "summary_words": len(summary["summary"].split()), "n_key_points": len(summary["key_points"]),
            "output": {"language": None, "transcript": text, "segments": [], **summary,
                       "models": pipe.models}}


def run_samples(pipe: P.Pipeline, samples: dict[str, Path], reference: str) -> list[dict]:
    """Every sample through the pipeline: audio + video (with WER) and the pasted transcript."""
    rows = [run_media(pipe, samples["audio"], reference), run_media(pipe, samples["video"], reference),
            run_transcript(pipe, samples["transcript"])]
    for row in rows:
        print(f"{row['sample']:24s} total {row['total_s']:6.1f}s  json_ok={row['json_ok']}"
              + (f"  WER={row['wer']:.3f}" if row["wer"] is not None else ""))
    return rows


def evaluate(results: dict[str, list[dict]]) -> dict[str, dict]:
    """Per-variant summary over the samples (media rows for WER/RTF, all rows for the LLM)."""
    table = {}
    for variant, rows in results.items():
        media = [r for r in rows if r["wer"] is not None]
        table[variant] = {
            "asr": rows[0]["output"]["models"]["asr"], "llm": rows[0]["output"]["models"]["llm"],
            "wer": sum(r["wer"] for r in media) / len(media),
            "rtf": sum(r["rtf"] for r in media) / len(media),
            "asr_s": sum(r["asr_s"] for r in media) / len(media),
            "llm_s": sum(r["llm_s"] for r in rows) / len(rows),
            "total_s_media": sum(r["total_s"] for r in media) / len(media),
            "summary_words": sum(r["summary_words"] for r in rows) / len(rows),
            "key_points": sum(r["n_key_points"] for r in rows) / len(rows),
            "json_ok_rate": sum(r["json_ok"] for r in rows) / len(rows),
        }
    return table
