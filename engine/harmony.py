"""Chords, bass notes and key per half-bar.

The bass line anchors everything:
  - full songs: the separated bass stem is pitch-tracked note by note (pYIN)
  - piano / guitar sources: the lowest accompaniment note
A chord must fit the bass note (ideally with the bass as its root), should belong
to the song's key, and should match the pitch content of the accompaniment.
Changing chord costs a little, so chords don't flicker.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import TICKS_PER_BEAT
from .model import BeatGrid, Chord, MelNote, PolyNote

# Simple, guitar-friendly chord vocabulary
QUALITIES = {"": [0, 4, 7], "m": [0, 3, 7], "7": [0, 4, 7, 10]}
QUALITY_PENALTY = {"": 0.0, "m": 0.0, "7": 0.06}
CHANGE_COST = 0.15        # higher = fewer chord changes
MELODY_WEIGHT = 0.3       # how much the melody's own notes count toward the chord
BASS_IS_ROOT = 0.35       # chord whose root is the bass note
BASS_IN_CHORD = 0.05      # bass note is the chord's 3rd/5th (inversion)
BASS_NOT_IN_CHORD = -0.4  # chord doesn't contain the bass note at all
IN_KEY = 0.15             # chord belongs to the song's key
OUT_OF_KEY = -0.15

_CANDIDATES = [(r, q) for r in range(12) for q in QUALITIES]
NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def _template(root: int, quality: str) -> np.ndarray:
    v = np.zeros(12)
    for k, iv in enumerate(QUALITIES[quality]):
        v[(root + iv) % 12] = 1.0 if k == 0 else (0.6 if k == 3 else 0.8)
    return v / np.linalg.norm(v)


_TEMPLATES = np.array([_template(r, q) for r, q in _CANDIDATES])


def chord_tones(ch: Chord) -> list[int]:
    return [(ch.root + iv) % 12 for iv in QUALITIES.get(ch.quality, [0, 4, 7])]


def scale(tonic: int, mode: str) -> set[int]:
    steps = (0, 2, 4, 5, 7, 9, 11) if mode == "major" else (0, 2, 3, 5, 7, 8, 10)
    return {(tonic + s) % 12 for s in steps}


def diatonic(tonic: int, mode: str) -> set[tuple[int, str]]:
    if mode == "major":
        degs = [(0, ""), (2, "m"), (4, "m"), (5, ""), (7, ""), (7, "7"), (9, "m"), (2, "7"), (4, "7")]
    else:
        degs = [(0, "m"), (3, ""), (5, "m"), (7, "m"), (7, ""), (7, "7"), (8, ""), (10, "")]
    return {((tonic + d) % 12, q) for d, q in degs}


def segments(grid: BeatGrid, end_tick: int) -> list[tuple[int, int]]:
    """Half bars in 4/4 (2 beats), whole bars otherwise."""
    seg = 2 * TICKS_PER_BEAT if grid.beats_per_bar == 4 else grid.bar_ticks
    return [(s, min(s + seg, end_tick)) for s in range(0, end_tick, seg)]


def _melody_chroma(melody: list[MelNote], s: int, e: int) -> np.ndarray:
    v = np.zeros(12)
    for m in melody:
        ov = min(e, m.start + m.dur) - max(s, m.start)
        if ov > 0:
            v[m.pitch % 12] += ov / TICKS_PER_BEAT
    return v


# Krumhansl-Kessler key profiles
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def estimate_key(chroma: np.ndarray, bass: np.ndarray, melody: list[MelNote]) -> tuple[int, str]:
    h = chroma.sum(axis=0)
    h = h / (h.sum() + 1e-9)
    b = np.bincount(bass[bass >= 0], minlength=12).astype(float)
    m = np.zeros(12)
    for n in melody:
        m[n.pitch % 12] += n.dur
    h = h + b / (b.sum() + 1e-9) + m / (m.sum() + 1e-9)
    best = (0, "major", -2.0)
    for t in range(12):
        for prof, mode in ((_MAJOR, "major"), (_MINOR, "minor")):
            r = float(np.corrcoef(h, np.roll(prof, t))[0, 1])
            if r > best[2]:
                best = (t, mode, r)
    return best[0], best[1]


def _decode(chroma: np.ndarray, bass: np.ndarray, segs, key) -> list[Chord]:
    n = len(segs)
    norm = np.linalg.norm(chroma, axis=1, keepdims=True)
    unit = np.where(norm > 1e-9, chroma / np.maximum(norm, 1e-9), 0.0)
    score = unit @ _TEMPLATES.T
    score -= np.array([QUALITY_PENALTY[q] for _, q in _CANDIDATES])[None, :]
    dia = diatonic(*key)
    score += np.array([IN_KEY if c in dia else OUT_OF_KEY for c in _CANDIDATES])[None, :]
    # A tracked bass note outside the key is usually a tracking slip (an overtone of a
    # synth bass, a slide): don't let it pull the chord out of key.
    in_scale = scale(*key)
    bass = np.array([b if b >= 0 and b in in_scale else -1 for b in bass])
    tones = [set((r + iv) % 12 for iv in QUALITIES[q]) for r, q in _CANDIDATES]
    roots = np.array([r for r, _ in _CANDIDATES])
    for i in range(n):
        if bass[i] >= 0:
            b = int(bass[i])
            score[i] += np.where(roots == b, BASS_IS_ROOT,
                                 np.array([BASS_IN_CHORD if b in t else BASS_NOT_IN_CHORD for t in tones]))
    best = score[0].copy()
    back = np.zeros((n, len(_CANDIDATES)), dtype=int)
    for i in range(1, n):
        j = int(best.argmax())
        change = best[j] - CHANGE_COST
        use_change = change > best
        back[i] = np.where(use_change, j, np.arange(len(_CANDIDATES)))
        best = np.maximum(best, change) + score[i]
    path = [int(best.argmax())]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i][path[-1]]))
    path.reverse()
    chords: list[Chord] = []
    for (s, e), c, b in zip(segs, path, bass):
        root, q = _CANDIDATES[c]
        bpc = int(b) if b >= 0 and int(b) in tones[c] else root
        if chords and chords[-1].root == root and chords[-1].quality == q and chords[-1].bass == bpc:
            chords[-1].end = e
        else:
            chords.append(Chord(s, e, root, q, bpc))
    return chords


def from_notes(accomp: list[PolyNote], melody: list[MelNote], grid: BeatGrid,
               end_tick: int) -> tuple[list[Chord], tuple[int, str]]:
    """Harmony from transcribed accompaniment notes (piano / guitar sources)."""
    segs = segments(grid, end_tick)
    on = grid.time_to_tick(np.array([a.onset for a in accomp])) if accomp else np.array([])
    off = grid.time_to_tick(np.array([a.offset for a in accomp])) if accomp else np.array([])
    chroma = np.zeros((len(segs), 12))
    bass = np.full(len(segs), -1)
    for i, (s, e) in enumerate(segs):
        lowest = None
        for k, a in enumerate(accomp):
            ov = min(e, off[k]) - max(s, on[k])
            if ov <= 0:
                continue
            w = ov / TICKS_PER_BEAT * a.velocity * (1.5 if on[k] >= s - 6 else 1.0)
            chroma[i, a.pitch % 12] += w
            if on[k] < s + TICKS_PER_BEAT and a.pitch < 60 and (lowest is None or a.pitch < lowest):
                lowest = a.pitch
        if lowest is not None:
            bass[i] = lowest % 12
        chroma[i] += MELODY_WEIGHT * _melody_chroma(melody, s, e)
    key = estimate_key(chroma, bass, melody)
    return _decode(chroma, bass, segs, key), key


def bass_line(bass_wav: Path, grid: BeatGrid, segs, cache: Path | None = None) -> np.ndarray:
    """Most-played bass pitch class per segment (first beat counts double), -1 if silent."""
    import librosa

    if cache is not None and cache.exists():
        z = np.load(cache)
        t, f0, ok = z["t"], z["f0"], z["ok"]
    else:
        y, sr = librosa.load(str(bass_wav), sr=11025, mono=True)
        f0, vflag, vprob = librosa.pyin(y, fmin=35, fmax=300, sr=sr, frame_length=2048, hop_length=256)
        t = librosa.times_like(f0, sr=sr, hop_length=256)
        ok = vflag & (vprob > 0.5)
        if cache is not None:
            np.savez(cache, t=t, f0=f0, ok=ok)
    pcs = np.full(len(f0), -1)
    pcs[ok] = (np.round(librosa.hz_to_midi(f0[ok])) % 12).astype(int)
    tick = grid.time_to_tick(t)
    out = np.full(len(segs), -1)
    for i, (s, e) in enumerate(segs):
        m = ok & (tick >= s) & (tick < e)
        if m.sum() < 8:
            continue
        w = np.where(tick[m] < s + TICKS_PER_BEAT, 2.0, 1.0)
        out[i] = int(np.bincount(pcs[m], weights=w, minlength=12).argmax())
    return out


def from_audio(chord_wavs: list[Path], bass_wav: Path | None, melody: list[MelNote],
               grid: BeatGrid, end_tick: int, bass_cache: Path | None = None
               ) -> tuple[list[Chord], tuple[int, str]]:
    """Harmony from separated stems (full songs)."""
    import librosa

    sr, hop = 22050, 512
    y = None
    for w in chord_wavs:
        a, _ = librosa.load(str(w), sr=sr, mono=True)
        y = a if y is None else y[: len(a)] + a[: len(y)]
    chroma_f = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    t_f = librosa.frames_to_time(np.arange(chroma_f.shape[1]), sr=sr, hop_length=hop)
    segs = segments(grid, end_tick)
    chroma = np.zeros((len(segs), 12))
    for i, (s, e) in enumerate(segs):
        ts, te = grid.beat_to_time([s / TICKS_PER_BEAT, e / TICKS_PER_BEAT])
        m = (t_f >= ts) & (t_f < te)
        if m.any():
            v = chroma_f[:, m].mean(axis=1)
            chroma[i] = v / (v.sum() + 1e-9)
        mc = _melody_chroma(melody, s, e)
        chroma[i] += MELODY_WEIGHT * mc / (mc.sum() + 1e-9)
    bass = bass_line(bass_wav, grid, segs, bass_cache) if bass_wav is not None else np.full(len(segs), -1)
    key = estimate_key(chroma, bass, melody)
    return _decode(chroma, bass, segs, key), key
