"""Beat, downbeat, tempo, time signature and key detection."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from .model import BeatGrid


def _extend(beats: np.ndarray, t_min: float, t_max: float) -> tuple[np.ndarray, int]:
    """Continue the beat grid before the first and after the last tracked beat,
    using the local beat length, so every note has a beat position."""
    n_edge = min(8, len(beats) - 1)
    ibi_start = float(np.median(np.diff(beats[: n_edge + 1])))
    ibi_end = float(np.median(np.diff(beats[-n_edge - 1:])))
    before = []
    t = beats[0] - ibi_start
    while t > t_min - 4 * ibi_start:
        before.append(t)
        t -= ibi_start
    after = []
    t = beats[-1] + ibi_end
    while t < t_max + 8 * ibi_end:
        after.append(t)
        t += ibi_end
    return np.concatenate([before[::-1], beats, after]), len(before)


def track_beats(wav: Path, device: str, duration: float) -> BeatGrid:
    from beat_this.inference import File2Beats

    f2b = File2Beats(checkpoint_path="final0", device=device, dbn=False)
    beats, downbeats = f2b(str(wav))
    beats = np.asarray(beats, dtype=float)
    downbeats = np.asarray(downbeats, dtype=float)
    flags = []
    if len(beats) < 8:
        raise RuntimeError("Too few beats detected to build a rhythm grid "
                           f"({len(beats)}). Is there music in this audio?")

    # Time signature = most common number of beats between downbeats.
    db_idx = [int(np.argmin(np.abs(beats - d))) for d in downbeats]
    counts = [b - a for a, b in zip(db_idx, db_idx[1:]) if 1 < b - a <= 7]
    if counts:
        beats_per_bar = Counter(counts).most_common(1)[0][0]
    else:
        beats_per_bar = 4
        flags.append("no downbeats detected; assumed 4 beats per bar")
    bpm = 60.0 / float(np.median(np.diff(beats)))

    ext, n_before = _extend(beats, 0.0, duration)
    first_db = (db_idx[0] if db_idx else 0) + n_before
    # Bar 1 starts at the first downbeat; step back whole bars to cover pickup notes
    # (the extension guarantees at least a couple of bars before the music).
    bar_start = first_db
    while bar_start - beats_per_bar >= 0 and ext[bar_start] > 0:
        bar_start -= beats_per_bar
    return BeatGrid(ext, bar_start, beats_per_bar, round(bpm, 1), flags)


def fix_tempo_octave(grid: BeatGrid, notes, scale: str = "auto") -> None:
    """Beat trackers often lock onto half or double the real tempo (e.g. following
    chord changes every 2 beats). Correct it using the melody: tempos are usually
    70-170 BPM and melody notes are usually 1/2 to 1 beat apart.
    `scale`: 'auto', '1' (trust tracker), '2' (double), '0.5' (halve)."""
    if scale == "auto":
        on = grid.time_to_beat(np.array([n.onset for n in notes]))
        ioi = np.diff(on)
        ioi = ioi[(ioi > 0.05) & (ioi < 4)]
        med = float(np.median(ioi)) if len(ioi) else 1.0
        if grid.bpm < 70 or (grid.bpm < 95 and med <= 0.5):
            scale = "2"
        elif grid.bpm > 170 or (grid.bpm > 130 and med >= 2):
            scale = "0.5"
        else:
            scale = "1"
    old = grid.bpm
    if scale == "2":
        mids = (grid.beats[:-1] + grid.beats[1:]) / 2
        beats = np.empty(len(grid.beats) + len(mids))
        beats[0::2], beats[1::2] = grid.beats, mids
        grid.beats = beats
        grid.bar_start_beat *= 2
        if grid.beats_per_bar * 2 <= 4:
            grid.beats_per_bar *= 2
        grid.bpm = round(old * 2, 1)
        grid.flags.append(f"tempo doubled from the detected {old:.0f} BPM (use --beat-scale 1 to undo)")
    elif scale == "0.5":
        p = grid.bar_start_beat % 2
        grid.beats = grid.beats[p::2]
        grid.bar_start_beat = (grid.bar_start_beat - p) // 2
        grid.bpm = round(old / 2, 1)
        grid.flags.append(f"tempo halved from the detected {old:.0f} BPM (use --beat-scale 1 to undo)")


def regularize(grid: BeatGrid) -> int:
    """Repair beat-tracker glitches: a beat that came far too soon (an extra beat) is
    dropped, and a gap of about 2+ beats (missed beats) is filled evenly.
    Without this, every note around a glitch lands on the wrong part of the bar."""
    b = grid.beats
    ibi = np.diff(b)
    t_bar = b[grid.bar_start_beat]
    out = [b[0]]
    fixes = 0
    for k in range(1, len(b)):
        med = float(np.median(ibi[max(0, k - 9):min(len(ibi), k + 8)]))
        gap = b[k] - out[-1]
        if gap < 0.6 * med:
            fixes += 1
            continue
        n = int(round(gap / med))
        if n >= 2:
            fixes += 1
            step = gap / n
            out.extend(out[-1] + step * j for j in range(1, n))
        out.append(b[k])
    if fixes:
        grid.beats = np.array(out)
        grid.bar_start_beat = int(np.argmin(np.abs(grid.beats - t_bar)))
        grid.flags.append(f"repaired {fixes} beat-tracking glitches")
    return fixes


def align_bar_one(grid: BeatGrid, first_onset: float) -> None:
    """Make bar 1 the bar that contains the first melody note (skips empty intro bars)."""
    b = grid.bar_start_beat
    step = grid.beats_per_bar
    while b + step < len(grid.beats) and grid.beats[b + step] <= first_onset + 0.05:
        b += step
    grid.bar_start_beat = b


# Krumhansl-Kessler key profiles.
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def detect_key(wav: Path, notes=()) -> tuple[str, float]:
    """Returns ('G major', correlation). Uses the whole mix's harmony plus the
    melody's own notes (weighted by length), so accompaniment can't outvote the tune."""
    import librosa

    y, sr = librosa.load(str(wav), sr=22050, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).sum(axis=1)
    chroma = chroma / (chroma.sum() + 1e-9)
    mel = np.zeros(12)
    for n in notes:
        mel[n.pitch % 12] += n.offset - n.onset
    if mel.sum() > 0:
        chroma = chroma + mel / mel.sum()
    best = ("C major", -1.0)
    for tonic in range(12):
        for profile, mode in ((_MAJOR, "major"), (_MINOR, "minor")):
            r = float(np.corrcoef(chroma, np.roll(profile, tonic))[0, 1])
            if r > best[1]:
                best = (f"{NOTE_NAMES[tonic]} {mode}", r)
    return best
