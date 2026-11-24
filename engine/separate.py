"""Split a full song into stems with Demucs (runs on the GPU)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .config import SIX_STEM_MODEL, VOCAL_MODEL


def model_for_stem(stem: str) -> str:
    return SIX_STEM_MODEL if stem in ("guitar", "piano") else VOCAL_MODEL


def separate(wav: Path, out_dir: Path, stem: str, device: str) -> Path:
    """Returns the path of the requested stem's WAV (cached after first run)."""
    model = model_for_stem(stem)
    target = out_dir / model / wav.stem / f"{stem}.wav"
    if target.exists():
        return target
    cmd = [sys.executable, "-m", "demucs", "-n", model, "-o", str(out_dir), "-d", device, str(wav)]
    subprocess.run(cmd, check=True)
    if not target.exists():
        have = sorted(p.stem for p in target.parent.glob("*.wav"))
        raise RuntimeError(f"Demucs model {model} produced {have}, not '{stem}'")
    return target
