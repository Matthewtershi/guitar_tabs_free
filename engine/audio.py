"""Playback with recorded acoustic-guitar samples (GeneralUser GS SoundFont).

Each guitar string gets its own channel, so a new note on a string stops the
note ringing on it (like a real guitar), while notes on other strings ring on.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np

SR = 44100
SOUNDFONT = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "guitar_tabs" / "soundfonts" / "GeneralUser-GS.sf2"
SOUNDFONT_URL = "https://github.com/mrbumpy409/GeneralUser-GS/raw/main/GeneralUser-GS.sf2"
SOUNDS = {"steel": 25, "nylon": 24}          # General MIDI programs in the SoundFont
VELOCITY = {"melody": 105, "bass": 88, "inner": 72}
LET_RING = 0.6                               # s a note may ring past its written length


def soundfont() -> Path:
    if not SOUNDFONT.exists():
        SOUNDFONT.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading guitar sounds (GeneralUser GS SoundFont, ~32 MB) ...")
        urllib.request.urlretrieve(SOUNDFONT_URL, SOUNDFONT)
    return SOUNDFONT


def render(notes, seconds_of, path: Path, sound: str = "steel", song: Path | None = None,
           song_path: Path | None = None) -> Path:
    """notes: ArrNote-like objects (start, dur, pitch, string, voice);
    seconds_of(tick) -> seconds on the video's own timeline, so the audio lines up
    with the video. Writes MP3 (or WAV by extension). With `song`, also writes
    `song_path`: the guitar mixed over the original recording, to hear the timing."""
    import soundfile as sf
    import tinysoundfont

    synth = tinysoundfont.Synth(samplerate=SR)
    sfid = synth.sfload(str(soundfont()))
    n_strings = max(n.string for n in notes) + 1
    for ch in range(n_strings):
        synth.program_select(ch, sfid, 0, SOUNDS[sound])

    events = []                                   # (time, 0=off/1=on, channel, key, velocity)
    by_string: dict[int, list] = {}
    for n in sorted(notes, key=lambda n: n.start):
        by_string.setdefault(n.string, []).append(n)
    for st, ns in by_string.items():
        for k, n in enumerate(ns):
            t0 = seconds_of(n.start)
            t1 = seconds_of(n.start + n.dur) + LET_RING
            if k + 1 < len(ns):
                t1 = min(t1, seconds_of(ns[k + 1].start) - 0.005)
            events.append((t0, 1, st, int(n.pitch), VELOCITY.get(n.voice, 90)))
            events.append((max(t0 + 0.02, t1), 0, st, int(n.pitch), 0))
    events.sort(key=lambda e: (e[0], e[1]))

    chunks, now = [], 0
    for t, kind, ch, key, vel in events:
        target = int(t * SR)
        if target > now:
            chunks.append(np.frombuffer(synth.generate(target - now), dtype=np.float32).copy())
            now = target
        if kind:
            synth.noteon(ch, key, vel)
        else:
            synth.noteoff(ch, key)
    chunks.append(np.frombuffer(synth.generate(3 * SR), dtype=np.float32).copy())   # let it die away
    out = np.concatenate(chunks).reshape(-1, 2)
    out *= 0.9 / (np.abs(out).max() + 1e-9)
    _write(path, out)
    if song is not None and song_path is not None:
        import librosa
        y, _ = librosa.load(str(song), sr=SR, mono=False)
        y = np.atleast_2d(y).T
        if y.shape[1] == 1:
            y = np.repeat(y, 2, axis=1)
        n = max(len(y), len(out))
        mix = np.zeros((n, 2), dtype=np.float32)
        mix[: len(y)] += 0.45 * y / (np.abs(y).max() + 1e-9)
        mix[: len(out)] += 0.8 * out
        _write(song_path, 0.95 * mix / (np.abs(mix).max() + 1e-9))
    return path


def _write(path: Path, data: np.ndarray) -> None:
    import soundfile as sf
    fmt = "MP3" if path.suffix.lower() == ".mp3" else None
    sf.write(str(path), data.astype(np.float32), SR, format=fmt)
