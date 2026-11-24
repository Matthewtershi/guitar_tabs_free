"""Outputs per level: PDF (notation + tab), MusicXML, guitar WAV, MIDI, text tab; plus song.json."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import audio, musicxml, pdf
from .config import TICKS_PER_BEAT
from .model import Arrangement, BeatGrid, Chord, MelNote, Note

TPB = TICKS_PER_BEAT
NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def _labels(tuning: list[int]) -> list[str]:
    names = [NOTE_NAMES[p % 12] for p in tuning]
    if names[-1] == "E":
        names[-1] = "e"
    w = max(len(n) for n in names)
    return [n.ljust(w) for n in names]


def render_text(arr: Arrangement, beats_per_bar: int, header: list[str], width: int = 100) -> str:
    """Plain-text tab: one column per 8th (per 16th-triplet where a turn happens)."""
    bar = beats_per_bar * TPB
    n_bars = max(1, math.ceil(arr.end_tick / bar))
    labels = _labels(arr.tuning)
    S = len(arr.tuning)
    by_tick: dict[int, list] = {}
    for n in arr.notes:
        by_tick.setdefault(n.start, []).append(n)
    bars = []
    for b in range(n_bars):
        b0 = b * bar
        ticks = sorted(set(range(b0, b0 + bar, 12)) | {t for t in by_tick if b0 <= t < b0 + bar})
        cols = []
        for t in ticks:
            cell = {}
            for n in by_tick.get(t, []):
                txt = ("/" if n.slide_in else "") + str(n.fret)
                cell[n.string] = txt
            off = t - b0
            count = str(off // TPB + 1) if off % TPB == 0 else ("&" if off % TPB == 12 else "")
            w = max([len(v) for v in cell.values()] + [len(count), 1]) + 1
            cols.append((cell, count, w))
        rows = ["".join(c[0].get(s, "").ljust(c[2], "-") for c in cols) for s in range(S)]
        counts = "".join(c[1].ljust(c[2]) for c in cols)
        bars.append((rows, counts))
    lines = list(header) + [""]
    pad = " " * (len(labels[0]) + 1)
    b = 0
    while b < n_bars:
        system, used = [], len(pad)
        while b < n_bars and (not system or used + len(bars[b][1]) + 2 <= width):
            system.append(b)
            used += len(bars[b][1]) + 2
            b += 1
        lines.append(f"Bar {system[0] + 1}")
        for s in range(S - 1, -1, -1):
            lines.append(labels[s] + "|" + "|".join("-" + bars[i][0][s] for i in system) + "|")
        lines.append(pad + " ".join(" " + bars[i][1] for i in system))
        lines.append("")
    return "\n".join(lines)


def write_midi(path: Path, notes, seconds_of) -> None:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=25)            # Acoustic Guitar (steel)
    for n in notes:
        s, e = seconds_of(n.start), seconds_of(n.start + n.dur)
        vel = {"melody": 100, "bass": 80, "inner": 65}.get(getattr(n, "voice", "melody"), 90)
        inst.notes.append(pretty_midi.Note(velocity=vel, pitch=int(n.pitch), start=s, end=max(e, s + 0.05)))
    pm.instruments.append(inst)
    pm.write(str(path))


def write_outputs(out_dir: Path, slug: str, title: str, levels: dict[str, Arrangement],
                  grid: BeatGrid, chords: list[Chord], clean: list[MelNote], master: list[Note],
                  key: tuple[int, str], info: dict, timings: dict, song_wav: Path | None = None,
                  sound: str = "steel", kind: str = "fingerstyle arrangement") -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    bpm = round(grid.bpm)
    bpb = grid.beats_per_bar
    for old in list(out_dir.glob("*.wav")):
        try:
            old.unlink()                   # replaced by MP3s on the video's timeline
        except OSError:
            pass                           # still open in a player; harmless

    def video_time(tick):
        """Tab position -> seconds in the video (follows the performer's real beats)."""
        return float(grid.beat_to_time(tick / TPB))

    key_name = f"{NOTE_NAMES[key[0]]} {key[1]}"
    written = []
    for lv, arr in levels.items():
        base = out_dir / f"{slug}_{lv}"
        capo_txt = f"Capo {arr.capo}" if arr.capo else "No capo"
        shape = NOTE_NAMES[(key[0] - arr.capo) % 12] + ("m" if key[1] == "minor" else "")
        left = (f"{capo_txt}   ·   Tuning: {arr.tuning_name}   ·   Key: {key_name}"
                + (f" (played as {shape} shapes)" if arr.capo else ""))
        subtitle = kind.capitalize() if arr.level == "exact" else f"{arr.level.capitalize()} — {kind}"
        mx = musicxml.write_musicxml(arr, base.with_suffix(".musicxml"), title, bpb, bpm, key[0], key[1])
        pdf.render_pdf(mx, base.with_suffix(".pdf"), title, subtitle, left, bpm)
        with_song = base.with_name(base.name + "_with_song.mp3")
        audio.render(arr.notes, video_time, base.with_suffix(".mp3"), sound, song_wav, with_song)
        write_midi(base.with_suffix(".mid"), arr.notes, video_time)
        header = [title, f"{subtitle}   |   {left}   |   ~{bpm} BPM, {bpb}/4", ""]
        base.with_suffix(".txt").write_text(render_text(arr, bpb, header), encoding="utf-8")
        written += [base.with_suffix(s) for s in (".pdf", ".mp3")] + ([with_song] if song_wav else [])
        written += [base.with_suffix(s) for s in (".musicxml", ".mid", ".txt")]
    song = {
        "title": title, "key": key_name, "bpm": bpm, "beats_per_bar": bpb, "info": info,
        "chords": [c.to_dict() for c in chords],
        "clean_melody": [m.to_dict() for m in clean],
        "master_melody": [n.to_dict() for n in master],
        "levels": {k: v.to_dict() for k, v in levels.items()},
        "grid": grid.to_dict(), "timings_seconds": timings,
    }
    (out_dir / "song.json").write_text(json.dumps(song, indent=1), encoding="utf-8")
    written.append(out_dir / "song.json")
    return written
