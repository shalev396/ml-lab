"""Sample media for the pipeline, cached in training/data/ (downloaded or generated once).

- jfk.wav: the 11 s JFK inaugural-address clip (16 kHz mono WAV) that Whisper's own tests use.
  Primary source: whisper.cpp's samples/jfk.wav. Fallback: openai/whisper's tests/jfk.flac,
  decoded with PyAV and written as WAV.
- sample.mp4: the same audio inside a small H.264/AAC video (built here with PyAV), so the
  video -> audio path is exercised.
- lecture_transcript.txt: a short original lecture text (written for this project) for the
  pasted-transcript path, which skips speech recognition.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import requests

import pipeline as P

JFK_WAV_URLS = [
    "https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav",
    "https://raw.githubusercontent.com/ggerganov/whisper.cpp/master/samples/jfk.wav",
]
JFK_FLAC_URL = "https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac"

LECTURE_TRANSCRIPT = """\
Good morning everyone. Today we are going to talk about how plants make their food, a process \
called photosynthesis. Plants take in carbon dioxide from the air through tiny pores in their \
leaves called stomata, and they take in water through their roots. Inside the leaf cells there \
are structures called chloroplasts, which contain a green pigment named chlorophyll. Chlorophyll \
absorbs sunlight, mostly red and blue light, and reflects green light, which is why leaves look \
green to us. The energy from that light is used to split water molecules. This releases oxygen, \
which the plant lets out into the air, and it produces energy carriers that the cell uses in the \
second stage. In that second stage, often called the Calvin cycle, the plant uses the stored \
energy to turn carbon dioxide into sugar. The plant can burn this sugar for energy right away, \
store it as starch, or use it to build cellulose for its cell walls. Photosynthesis matters far \
beyond the plant itself. Almost all the oxygen we breathe comes from it, and nearly every food \
chain on Earth starts with it. Next week we will look at how temperature and the amount of \
carbon dioxide change the speed of photosynthesis, so please read chapter six before class.
"""


def _fetch(url: str, out: Path) -> bool:
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        out.write_bytes(r.content)
        print(f"downloaded {url} -> {out.name} ({out.stat().st_size / 1024:.0f} KB)")
        return True
    except Exception as err:  # noqa: BLE001 - try the next source
        print(f"download failed ({url}): {err}")
        return False


def write_wav(path: Path, audio: np.ndarray, sr: int = P.SAMPLE_RATE) -> Path:
    """float32 [-1, 1] mono -> 16-bit PCM WAV."""
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())
    return path


def download_jfk(data_dir: Path) -> Path:
    out = data_dir / "jfk.wav"
    if out.is_file() and out.stat().st_size > 0:
        return out
    for url in JFK_WAV_URLS:
        if _fetch(url, out):
            return out
    flac = data_dir / "jfk.flac"
    if _fetch(JFK_FLAC_URL, flac):
        audio, _ = P.load_audio(flac)
        return write_wav(out, audio)
    raise RuntimeError("could not download the JFK sample from any source (see messages above)")


def make_video(audio_path: Path, out: Path, size: tuple[int, int] = (320, 240), fps: int = 10) -> Path:
    """Wrap an audio file into a small video (dark frames + AAC audio) with PyAV."""
    import av

    audio, _ = P.load_audio(audio_path)
    duration = audio.size / P.SAMPLE_RATE
    codec = "libx264" if "libx264" in av.codecs_available else "mpeg4"
    with av.open(str(out), "w") as container:
        vstream = container.add_stream(codec, rate=fps)
        vstream.width, vstream.height, vstream.pix_fmt = size[0], size[1], "yuv420p"
        astream = container.add_stream("aac", rate=P.SAMPLE_RATE, layout="mono")
        frame_img = np.full((size[1], size[0], 3), (30, 30, 60), dtype=np.uint8)
        for _ in range(int(np.ceil(duration * fps))):
            for packet in vstream.encode(av.VideoFrame.from_ndarray(frame_img, format="rgb24")):
                container.mux(packet)
        for packet in vstream.encode():
            container.mux(packet)
        step = 1024
        for i in range(0, audio.size, step):
            chunk = audio[i:i + step].reshape(1, -1)
            frame = av.AudioFrame.from_ndarray(chunk, format="flt", layout="mono")
            frame.sample_rate, frame.pts = P.SAMPLE_RATE, i
            for packet in astream.encode(frame):
                container.mux(packet)
        for packet in astream.encode():
            container.mux(packet)
    print(f"built {out.name} from {audio_path.name} ({out.stat().st_size / 1024:.0f} KB)")
    return out


def ensure_samples(cfg) -> dict[str, Path]:
    """Download / build every sample into cfg.data_dir (skipped when present)."""
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    audio = download_jfk(cfg.data_dir)
    video = cfg.data_dir / cfg.video_sample
    if not video.is_file():
        make_video(audio, video)
    text = cfg.data_dir / cfg.transcript_sample
    if not text.is_file():
        text.write_text(LECTURE_TRANSCRIPT, encoding="utf-8")
    return {"audio": audio, "video": video, "transcript": text}


def describe(samples: dict[str, Path], max_seconds: float = P.MAX_SECONDS) -> list[dict]:
    """One row per sample: file, size, decoded duration (media) or word count (text)."""
    rows = []
    for kind, path in samples.items():
        row = {"sample": path.name, "kind": kind, "size_kb": round(path.stat().st_size / 1024, 1)}
        if kind == "transcript":
            row["words"] = len(path.read_text(encoding="utf-8").split())
        else:
            audio, duration = P.load_audio(path, max_seconds)
            row.update(duration_s=duration, samples=audio.size, peak=round(float(np.abs(audio).max()), 3))
        rows.append(row)
    return rows
