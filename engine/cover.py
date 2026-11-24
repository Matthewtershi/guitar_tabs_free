"""Exact transcription of a guitar performance (a cover or tutorial) into tab.

Every note the guitarist plays is kept: melody, bass and chord notes, so the tab
sounds like the performance. Steps:
  1. transcribe all notes (Basic Pitch), drop overtone ghosts and noise
  2. group notes struck together into strikes (chords)
  3. place strikes on the beat grid (8ths, or 16ths where the playing needs them;
     never more than MAX_TIMING_ERROR_S from where they were played)
  4. choose string + fret for every strike so each chord is one playable hand
     shape and the hand moves as little as possible; try every capo, keep the best
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import GUITAR_FRETS, TICKS_PER_BEAT
from .model import Arrangement, ArrNote, BeatGrid, Note, PolyNote
from .transcribe import _Poly, remove_ghosts

TPB = TICKS_PER_BEAT
SIXTEENTH = TPB // 4
CHORD_WINDOW = 0.045        # s: notes struck within this of each other form one strike
MAX_TIMING_ERROR_S = 0.06   # s: a strike is never placed further than this from where it was played
SIXTEENTH_COST = 0.04       # beats of error a beat must save before it is written in 16ths
QUIET = 0.18                # notes quieter than this share of the loud notes are dropped as noise
GUITAR_RANGE = (36, 88)     # MIDI pitches a guitar can play (with drop tunings / high frets)
SPAN = 4                    # widest fret stretch allowed inside one chord shape
BASS_SPLIT = 55             # notes below G3 that are the lowest of their strike are written as bass


def transcribe_guitar(wav: Path) -> list[PolyNote]:
    cache = wav.with_name(wav.stem + "_guitar_notes.json")
    if cache.exists():
        return [PolyNote(*r) for r in json.loads(cache.read_text())]
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict

    _, _, ev = predict(str(wav), ICASSP_2022_MODEL_PATH, onset_threshold=0.5, frame_threshold=0.3,
                       minimum_note_length=50, melodia_trick=False)
    raw = [_Poly(float(a), float(b), int(p), float(v)) for a, b, p, v, _ in ev
           if GUITAR_RANGE[0] <= int(p) <= GUITAR_RANGE[1]]
    clean = remove_ghosts(raw)
    loud = np.percentile([e.vel for e in clean], 90) if clean else 1.0
    notes = [PolyNote(e.on, e.off, e.pitch, e.vel) for e in clean if e.vel >= QUIET * loud]
    notes.sort(key=lambda n: (n.onset, n.pitch))
    cache.write_text(json.dumps([[n.onset, n.offset, n.pitch, n.velocity] for n in notes]))
    return notes


def strikes_of(notes: list[PolyNote]) -> list[list[PolyNote]]:
    groups: list[list[PolyNote]] = []
    for n in notes:
        if groups and n.onset - groups[-1][0].onset <= CHORD_WINDOW:
            if all(m.pitch != n.pitch for m in groups[-1]):
                groups[-1].append(n)
        else:
            groups.append([n])
    return groups


def quantize(groups: list[list[PolyNote]], grid: BeatGrid) -> list[tuple[int, list[tuple[int, int, float]]]]:
    """-> [(tick, [(pitch, end_tick, velocity), ...]), ...] sorted by tick."""
    pos = grid.time_to_beat(np.array([g[0].onset for g in groups]))
    cap = MAX_TIMING_ERROR_S / (60.0 / grid.bpm)
    by_beat: dict[int, list[int]] = {}
    for i, p in enumerate(pos):
        if p > -0.125:
            by_beat.setdefault(int(np.floor(p + 0.125)), []).append(i)
    tick_of = {}
    for b, idx in by_beat.items():
        rel = pos[idx] - b
        best = None
        for g, pen in ((2, 0.0), (4, SIXTEENTH_COST)):
            pts = np.round(rel * g) / g
            err = np.abs(pts - rel)
            cost = err.sum() + pen + 10 * np.maximum(0, err - cap).sum() + 0.3 * (len(pts) - len(np.unique(pts)))
            if best is None or cost < best[0]:
                best = (cost, pts)
        for i, p in zip(idx, best[1]):
            tick_of[i] = max(0, int(round((b + p) * TPB)))
    merged: dict[int, dict[int, tuple[int, float]]] = {}
    for i, g in enumerate(groups):
        if i not in tick_of:
            continue
        t = tick_of[i]
        for n in g:
            end = int(round(grid.time_to_beat(n.offset) * 4)) * SIXTEENTH
            end = max(end, t + SIXTEENTH)
            cur = merged.setdefault(t, {})
            if n.pitch not in cur or cur[n.pitch][0] < end:
                cur[n.pitch] = (end, n.velocity)
    return [(t, [(p, e, v) for p, (e, v) in sorted(merged[t].items())]) for t in sorted(merged)]


def _assignments(pitches: list[int], open_: np.ndarray, maxf: int):
    """Every way to put these pitches (low->high) on strings low->high, as playable shapes.
    Returns [(cost, [(string, fret), ...], hand)] where hand = lowest fretted fret or 0."""
    out = []
    S = len(open_)

    def rec(k, s_min, acc):
        if k == len(pitches):
            fr = [f for _, f in acc if f > 0]
            span = (max(fr) - min(fr)) if fr else 0
            if span > SPAN:
                return
            cost = sum(0.05 * f for _, f in acc) + (0.6 if span == SPAN else 0.0) + 0.15 * len(fr)
            out.append((cost, list(acc), min(fr) if fr else 0))
            return
        for s in range(s_min, S):
            f = pitches[k] - int(open_[s])
            if 0 <= f <= maxf:
                rec(k + 1, s + 1, acc + [(s, f)])

    rec(0, 0, [])
    return out


def finger(strikes, tuning: list[int], capo: int, sec_per_tick: float):
    open_ = np.array(tuning) + capo
    maxf = GUITAR_FRETS - capo
    layers, dropped = [], 0
    for t, notes in strikes:
        pitches = sorted(p for p, _, _ in notes)
        opts = _assignments(pitches, open_, maxf)
        while not opts and len(pitches) > 1:
            # can't be played as one shape: drop the quietest inner note and try again
            inner = [n for n in notes if n[0] in pitches[1:-1]] or [n for n in notes if n[0] in pitches]
            q = min(inner, key=lambda n: n[2])[0]
            pitches.remove(q)
            dropped += 1
            opts = _assignments(pitches, open_, maxf)
        if not opts:
            for shift in (12, -12, 24):
                opts = _assignments([pitches[0] + shift], open_, maxf)
                if opts:
                    pitches = [pitches[0] + shift]
                    break
        ends = {p: e for p, e, _ in notes}
        layers.append((t, pitches, opts, [ends.get(p, t + SIXTEENTH) for p in pitches]))

    # dynamic programming over shapes: move the hand as little as possible
    prev_cost, backs = None, []
    for k, (t, pitches, opts, ends) in enumerate(layers):
        local = np.array([o[0] for o in opts])
        if prev_cost is None:
            cur = local
            back = np.zeros(len(opts), dtype=int)
        else:
            pt, _, popts, pends = layers[k - 1]
            dt = (t - pt) * sec_per_tick
            speed = 1.0 + 3.0 * max(0.0, 0.25 - dt) / 0.25
            trans = np.zeros((len(popts), len(opts)))
            for i, po in enumerate(popts):
                for j, o in enumerate(opts):
                    if po[2] and o[2]:
                        d = abs(po[2] - o[2])
                        trans[i, j] = (d + 0.8 * (d > 0)) * speed
                    # cutting off a previous note that should still ring
                    ringing = {s for (s, _), e in zip(po[1], pends) if e > t}
                    trans[i, j] += 0.4 * len(ringing & {s for s, _ in o[1]})
            total = prev_cost[:, None] + trans
            back = total.argmin(axis=0)
            cur = total.min(axis=0) + local
        backs.append(back)
        prev_cost = cur
    j = int(prev_cost.argmin())
    best = float(prev_cost[j])
    choice = [0] * len(layers)
    for k in range(len(layers) - 1, -1, -1):
        choice[k] = j
        j = int(backs[k][j])

    out = []
    for (t, pitches, opts, ends), c in zip(layers, choice):
        shape = opts[c][1]
        for (s, f), p, e in zip(shape, pitches, ends):
            voice = "bass" if (p == pitches[0] and len(pitches) > 1 and p < BASS_SPLIT) else "melody"
            out.append(ArrNote(t, max(SIXTEENTH, e - t), int(open_[s]) + f, voice, s, f))
    return best, out, dropped


def transcribe_cover(wav: Path, grid: BeatGrid, tuning: list[int], tuning_name: str,
                     capo: int | None = None) -> tuple[Arrangement, list[PolyNote], dict]:
    notes = transcribe_guitar(wav)
    strikes = quantize(strikes_of(notes), grid)
    end_tick = max(e for _, ns in strikes for _, e, _ in ns)
    sec_per_tick = 60.0 / grid.bpm / TPB
    results = {}
    for c in ([capo] if capo is not None else range(0, 8)):
        cost, arr_notes, dropped = finger(strikes, tuning, c, sec_per_tick)
        results[c] = (cost + (2.0 if c else 0.0) + 0.1 * c, arr_notes, dropped)
    best = min(results, key=lambda c: results[c][0])
    cost, arr_notes, dropped = results[best]
    arr = Arrangement("exact", best, list(tuning), tuning_name, arr_notes, end_tick,
                      {"unplayable_notes_left_out": dropped}, round(cost, 1),
                      {c: round(v[0], 1) for c, v in results.items()})
    info = {"notes_heard": len(notes), "strikes": len(strikes), "notes_in_tab": len(arr_notes)}
    return arr, notes, info


def first_onset_notes(notes: list[PolyNote]) -> list[Note]:
    """Note objects for the beat-grid helpers (tempo fix, bar alignment)."""
    return [Note(n.onset, n.offset, n.pitch) for n in notes]
