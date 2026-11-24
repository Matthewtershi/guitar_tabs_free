"""Master melody (exact, real time) -> clean melody on the 8th-note grid.

This is where "simple to read" happens:
  - decorations (grace notes, runs, scoops) are dropped; runs of 2+ fast notes are
    kept on the side as a "turn" that Hard can play as a 16th-note triplet
  - every remaining note snaps to the nearest 8th; if two land on the same 8th,
    the longer one wins
  - small gaps are filled (legato); only real breaks of a beat or more become rests
  - the whole melody moves by octaves into the guitar's melody range
"""
from __future__ import annotations

import numpy as np

from .config import EIGHTH, MELODY_TARGET_PITCH, TICKS_PER_BEAT
from .model import BeatGrid, MelNote, Note

DECOR_MAX_BEATS = 0.22     # shorter than this, next to a note 2x longer = decoration
ADJACENT_BEATS = 0.15      # "next to" = within this gap
MIN_REST_BEATS = 1.0       # gaps shorter than this are filled by holding the previous note
RELEASE_BIAS_BEATS = 0.15  # notes are usually released a bit early
SIXTEENTH_COST = 0.06      # beats of timing error a beat must save before it's written in 16ths
OCTAVE_SLIP_MAX_BEATS = 0.5  # only notes shorter than this can be treated as octave slips
STUTTER_TICKS = 12         # same note re-struck within an 8th: hold it (let it ring) instead
MAX_TIMING_ERROR_S = 0.08  # no note may be placed further than this from where it was sung
ATTACK_WINDOW = 0.07       # s; move a note's start onto a real attack this close


def refine_onsets(master: list[Note], attacks: np.ndarray) -> int:
    """Pitch tracking finds a note's start from its pitch, which can lag or lead the
    actual attack; snap each start to the nearest real attack (onset) close by."""
    moved = 0
    if attacks is None or not len(attacks):
        return 0
    for n in master:
        i = int(np.searchsorted(attacks, n.onset))
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(attacks) and abs(attacks[j] - n.onset) <= ATTACK_WINDOW:
                if best is None or abs(attacks[j] - n.onset) < abs(best - n.onset):
                    best = attacks[j]
        if best is not None and best < n.offset - 0.03:
            n.onset = float(best)
            moved += 1
    return moved


def clean_melody(master: list[Note], grid: BeatGrid) -> tuple[list[MelNote], dict]:
    if not master:
        return [], {}
    on = grid.time_to_beat(np.array([n.onset for n in master]))
    off = grid.time_to_beat(np.array([n.offset for n in master]))
    ioi = np.append(np.diff(on), np.inf)
    length = np.minimum(off - on, ioi)

    # 1. decorations
    decor = np.zeros(len(master), dtype=bool)
    for i in range(len(master)):
        if length[i] >= DECOR_MAX_BEATS:
            continue
        near_prev = i > 0 and on[i] - off[i - 1] < ADJACENT_BEATS and length[i - 1] >= 2 * length[i]
        near_next = (i + 1 < len(master) and on[i + 1] - off[i] < ADJACENT_BEATS
                     and length[i + 1] >= 2 * length[i])
        decor[i] = near_prev or near_next

    # 2. snap main notes to the grid, beat by beat: 8ths unless the singing really
    #    sits on 16ths there. If two notes land on one spot, the longer wins.
    main_idx = [int(i) for i in np.where(~decor)[0] if on[i] > -0.125]
    beat_of = {}
    for i in main_idx:
        beat_of.setdefault(int(np.floor(on[i] + 0.125)), []).append(i)
    tick_of: dict[int, int] = {}
    sixteenth_beats = 0
    beat_sec = 60.0 / grid.bpm
    cap = MAX_TIMING_ERROR_S / beat_sec            # the timing cap, in beats
    for b, idx in beat_of.items():
        pos = on[idx] - b
        best = None
        for g, pen in ((2, 0.0), (4, SIXTEENTH_COST)):
            pts = np.round(pos * g) / g
            err = np.abs(pts - pos)
            cost = float(err.sum()) + pen + 0.5 * (len(pts) - len(np.unique(pts)))
            cost += 10.0 * float(np.maximum(0.0, err - cap).sum())     # breaking the cap is not an option
            if best is None or cost < best[0]:
                best = (cost, g, pts)
        sixteenth_beats += best[1] == 4
        for i, p in zip(idx, best[2]):
            t = int(round((b + p) * TICKS_PER_BEAT))
            j = tick_of.get(t)
            if j is None or length[i] > length[j]:
                tick_of[t] = i
    mains = sorted(tick_of.items())
    collisions = len(main_idx) - len(mains)
    SIXTEENTH = TICKS_PER_BEAT // 4

    # 3. durations: legato over small gaps, rests only for real breaks
    out: list[MelNote] = []
    for k, (start, i) in enumerate(mains):
        end = int(round((off[i] + RELEASE_BIAS_BEATS) * 4)) * SIXTEENTH
        end = max(end, start + SIXTEENTH)
        if k + 1 < len(mains):
            nxt = mains[k + 1][0]
            if nxt - end < MIN_REST_BEATS * TICKS_PER_BEAT:
                end = nxt
            end = min(end, nxt)
        # turn: 2+ decorations right before this note (Hard plays them as a triplet)
        pre = []
        j = i - 1
        while j >= 0 and decor[j] and on[i] - off[j] < 0.5:
            pre.insert(0, master[j].pitch)
            j -= 1
        turn = []
        if len(pre) >= 3:
            turn = pre[-3:]
        elif len(pre) == 2 and out:
            turn = [out[-1].pitch] + pre
        out.append(MelNote(start, end - start, master[i].pitch, master[i].glide_in, turn, src=int(i)))

    # 4. octave slips: a SHORT note far from its neighbours that fits right in an octave
    #    away (pitch trackers jump octaves for a moment; a held note far away is a real leap)
    pitches = np.array([m.pitch for m in out])
    octave_fixed = 0
    for k, m in enumerate(out):
        nb = np.concatenate([pitches[max(0, k - 4):k], pitches[k + 1:k + 5]])
        if len(nb) < 2 or length[m.src] >= OCTAVE_SLIP_MAX_BEATS:
            continue
        local = float(np.median(nb))
        if abs(m.pitch - local) > 9:
            for d in (12, -12, 24, -24):
                if abs(m.pitch + d - local) <= 5:
                    m.pitch += d
                    m.turn = [p + d for p in m.turn]
                    m.changed += f"octave{d:+d} "
                    pitches[k] = m.pitch
                    octave_fixed += 1
                    break

    # 5. stutters: the same pitch re-struck a 16th after it started is held instead
    merged: list[MelNote] = []
    stutters = 0
    for m in out:
        p = merged[-1] if merged else None
        if (p and p.pitch == m.pitch and p.start + p.dur == m.start and m.start - p.start <= STUTTER_TICKS
                and not m.turn):
            p.dur += m.dur
            stutters += 1
        else:
            merged.append(m)
    out = merged

    # 6. octave placement for the whole melody
    med = float(np.median([m.pitch for m in out]))
    shift = 12 * int(round((MELODY_TARGET_PITCH - med) / 12))
    for m in out:
        m.pitch += shift
        m.turn = [p + shift for p in m.turn]
    info = {"decorations_dropped": int(decor.sum()), "same_spot_dropped": collisions,
            "beats_written_in_16ths": int(sixteenth_beats), "beats_with_melody": len(beat_of),
            "octave_slips_fixed": octave_fixed, "stutters_held": stutters, "melody_octave_shift": shift,
            "turns": sum(1 for m in out if m.turn)}
    return out, info


# (An "out-of-key correction" used to live here. Checked against the singer it moved
#  notes that were really sung, so it was removed: never change a melody pitch to fit
#  a theory of the key.)
