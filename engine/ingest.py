"""Get audio in: a YouTube link or a local audio file -> 44.1 kHz stereo WAV."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac"}


def is_youtube(source: str) -> bool:
    return bool(re.match(r"https?://(www\.|m\.|music\.)?(youtube\.com|youtu\.be)/", source))


def youtube_id(url: str) -> str:
    u = urlparse(url)
    if u.netloc.endswith("youtu.be"):
        return u.path.strip("/").split("/")[0]
    if "v" in parse_qs(u.query):
        return parse_qs(u.query)["v"][0]
    parts = [p for p in u.path.split("/") if p]
    if len(parts) >= 2 and parts[0] in ("shorts", "embed", "live"):
        return parts[1]
    raise ValueError(f"Can't find a video id in {url}")


def song_id(source: str) -> str:
    """Stable folder name for a song so cached stages can be reused."""
    if is_youtube(source):
        return "yt_" + youtube_id(source)
    p = Path(source).resolve()
    digest = hashlib.sha1(f"{p}|{p.stat().st_size}".encode()).hexdigest()[:8]
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", p.stem)[:40]
    return f"{safe}_{digest}"


def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    candidate = Path(sys.prefix) / "Library" / "bin" / "ffmpeg.exe"
    if candidate.exists():
        return str(candidate)
    raise RuntimeError("ffmpeg not found. Install it in the env: conda install -c conda-forge ffmpeg")


def download_youtube(url: str, dest: Path) -> tuple[Path, str]:
    """Download the audio track only. Returns (file, title)."""
    dest.mkdir(parents=True, exist_ok=True)
    existing = [p for p in dest.glob("download.*") if p.suffix != ".json"]
    info_path = dest / "download.info.json"
    if not existing:
        cmd = [sys.executable, "-m", "yt_dlp", "--no-playlist", "-f", "bestaudio/best",
               "--js-runtimes", "node", "--write-info-json", "--ffmpeg-location", find_ffmpeg(),
               "-o", str(dest / "download.%(ext)s"), url]
        subprocess.run(cmd, check=True)
        existing = [p for p in dest.glob("download.*") if p.suffix != ".json"]
    if not existing:
        raise RuntimeError("yt-dlp finished but no audio file was written")
    title = url
    if info_path.exists():
        title = json.loads(info_path.read_text(encoding="utf-8")).get("title", url)
    return existing[0], title


def parse_time(s: str | None) -> float | None:
    """'1:23' / '83' / '1:02:03' -> seconds."""
    if s is None:
        return None
    total = 0.0
    for part in s.split(":"):
        total = total * 60 + float(part)
    return total


def to_wav(src: Path, dst: Path, start: float | None = None, end: float | None = None) -> Path:
    cmd = [find_ffmpeg(), "-y", "-loglevel", "error", "-i", str(src)]
    if start is not None:
        cmd += ["-ss", str(start)]
    if end is not None:
        cmd += ["-to", str(end)]
    cmd += ["-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(dst)]
    subprocess.run(cmd, check=True)
    return dst


def ingest(source: str, work: Path, start: float | None, end: float | None) -> tuple[Path, str]:
    """Returns (source.wav, title). Re-converts only if the trim changed."""
    work.mkdir(parents=True, exist_ok=True)
    if is_youtube(source):
        raw, title = download_youtube(source, work / "download")
    else:
        raw = Path(source)
        if raw.suffix.lower() not in AUDIO_EXTS:
            raise ValueError(f"Unsupported file type {raw.suffix}; use one of {sorted(AUDIO_EXTS)}")
        title = raw.stem
    wav = work / "source.wav"
    trim_file = work / "trim.json"
    trim = {"start": start, "end": end}
    if not wav.exists() or not trim_file.exists() or json.loads(trim_file.read_text()) != trim:
        to_wav(raw, wav, start, end)
        trim_file.write_text(json.dumps(trim))
    return wav, title
