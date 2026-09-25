"""Evaluation inputs: the JFK clip (audio + a video built from it) and the Gettysburg Address text.

- `jfk.wav`: the 11 s JFK inaugural clip ("ask not what your country can do for you"), 16 kHz mono,
  as shipped with whisper.cpp. Cached in training/data/, downloaded only when missing.
- `sample.mp4`: the same audio muxed into a small solid-colour H.264/AAC video with PyAV, so the
  full video -> audio -> transcript -> quiz path is exercised.
- `gettysburg.txt`: Lincoln's Gettysburg Address (public domain, Bliss copy), a longer text-only
  input (the "paste a transcript" path). It lives in ../space/examples/.
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import requests

from . import utils

JFK_URLS = [  # the same file on two mirrors; the second is the fallback
    "https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav",
    "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav",
]
JFK_REFERENCE = ("And so, my fellow Americans, ask not what your country can do for you, "
                 "ask what you can do for your country.")
EXAMPLES_DIR = utils.SPACE_DIR / "examples"
SAMPLES = {
    "jfk": {"kind": "audio", "file": "jfk.wav", "reference": JFK_REFERENCE,
            "source": "JFK inaugural address, 1961 (11 s clip, whisper.cpp samples/jfk.wav)"},
    "gettysburg": {"kind": "text", "file": "gettysburg.txt",
                   "source": "Gettysburg Address, A. Lincoln 1863 (public-domain text)"},
}


def download_jfk(data_dir: Path = utils.DATA_DIR) -> Path:
    out = Path(data_dir) / "jfk.wav"
    if out.is_file() and out.stat().st_size > 0:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    for url in JFK_URLS:
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            out.write_bytes(r.content)
            print(f"downloaded {url} ({out.stat().st_size / 1024:.0f} KB)")
            return out
        except requests.RequestException as err:
            print(f"download failed ({url}): {err}")
    raise RuntimeError("could not download jfk.wav from any mirror; put it in training/data/ by hand")


def build_sample_video(wav: Path, out: Path | None = None, size=(320, 240), fps: int = 12) -> Path:
    """Mux `wav` into a solid-colour H.264 (or MPEG-4) + AAC mp4 with PyAV; no ffmpeg binary needed."""
    import av

    import pipeline as P

    out = Path(out) if out else Path(wav).with_name("sample.mp4")
    if out.is_file() and out.stat().st_size > 0:
        return out
    audio, _ = P.load_audio(wav, max_seconds=600)
    seconds = audio.size / P.SAMPLE_RATE
    with av.open(str(out), "w") as container:
        v = container.add_stream("libx264" if "libx264" in av.codecs_available else "mpeg4", rate=fps)
        v.width, v.height, v.pix_fmt = size[0], size[1], "yuv420p"
        a = container.add_stream("aac", rate=P.SAMPLE_RATE, layout="mono")
        image = np.zeros((size[1], size[0], 3), np.uint8)
        image[:] = (90, 24, 60)
        for _ in range(int(np.ceil(seconds * fps))):
            container.mux(v.encode(av.VideoFrame.from_ndarray(image, format="rgb24")))
        container.mux(v.encode())
        step = 1024
        for i in range(0, audio.size, step):
            frame = av.AudioFrame.from_ndarray(audio[None, i:i + step], format="flt", layout="mono")
            frame.sample_rate, frame.pts, frame.time_base = P.SAMPLE_RATE, i, Fraction(1, P.SAMPLE_RATE)
            container.mux(a.encode(frame))
        container.mux(a.encode())
    print(f"built {out.name} ({out.stat().st_size / 1024:.0f} KB, {seconds:.1f} s)")
    return out


def load_samples(names: list[str]) -> list[dict]:
    """Resolve the evaluation samples: [{name, kind, path, text, reference, source}]."""
    wav = download_jfk()
    build_sample_video(wav)
    rows = []
    for name in names:
        spec = SAMPLES[name]
        if spec["kind"] == "audio":
            rows.append({"name": name, "kind": "audio", "path": utils.DATA_DIR / spec["file"], "text": None,
                         "reference": spec["reference"], "source": spec["source"]})
        else:
            path = EXAMPLES_DIR / spec["file"]
            rows.append({"name": name, "kind": "text", "path": path,
                         "text": path.read_text(encoding="utf-8").strip(), "reference": None,
                         "source": spec["source"]})
    return rows
