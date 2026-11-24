"""Clean melody + chords -> fingerstyle arrangement for one level.

1. make_slices: decide WHAT to play at each moment (the level's pattern)
     Easy   : melody + one bass note at the start of each bar
     Medium : melody + bass on the strong beats + a chord note on every beat
              that has no melody note
     Hard   : steady 8ths: melody, bass on the strong beats, and a chord note
              on every 8th with no melody note (like the example tab);
              the recording's ornaments as 16th-triplet turns
2. finger: decide WHERE to play it. Dynamic programming over
     (hand position, melody string); at each moment the bass and chord notes
     are chosen to fit the same hand shape, below the melody string.
     Run for every capo; keep the cheapest.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import numpy as np

from .config import EIGHTH, GUITAR_FRETS, HAND_SPAN, TICKS_PER_BEAT, LevelSettings
from .harmony import chord_tones
from .model import Arrangement, ArrNote, Chord, MelNote

INF = 1e9
FOLD_COST = 25.0          # moving a single melody note an octave (only if unavoidable)
SOFT_LIMIT_COST = 4.0     # going above the level's fret limit when a lower option exists
HELD_CUT_COST = 1.0       # the ringing melody note has to be released early
TURN_DROP_COST = 2.5      # a Hard ornament that couldn't be fingered on one string


@dataclass
class Slice:
    tick: int
    mel: int | None = None            # melody pitch starting here
    mel_dur: int = 0
    mel_glide: bool = False
    turn: list = field(default_factory=list)   # 3 pitches -> 16th-triplet turn
    held: int | None = None           # melody pitch still ringing from before
    bass_pcs: list = field(default_factory=list)
    bass_dur: int = 0
    inner_pcs: list = field(default_factory=list)
    inner_dur: int = 0


def _chord_at(chords: list[Chord], starts: list[int], tick: int) -> Chord:
    i = max(0, bisect.bisect_right(starts, tick) - 1)
    return chords[i]


def make_slices(melody: list[MelNote], chords: list[Chord], beats_per_bar: int,
                end_tick: int, level: LevelSettings) -> list[Slice]:
    bar = beats_per_bar * TICKS_PER_BEAT
    mel = [MelNote(m.start, m.dur, m.pitch, m.glide_in, list(m.turn)) for m in melody]
    starts = [c.start for c in chords]

    # Hard: put each ornament turn in the 8th just before its note, if that 8th can be freed
    turn_at: dict[int, list] = {}
    if level.ornaments:
        onsets = {m.start for m in mel}
        for k, m in enumerate(mel):
            slot = m.start - EIGHTH
            if not m.turn or slot < 0 or any(slot <= o < m.start for o in onsets):
                continue
            prev = mel[k - 1] if k > 0 else None
            if prev and prev.start < slot < prev.start + prev.dur:
                if slot - prev.start < EIGHTH:
                    continue
                prev.dur = slot - prev.start
            elif prev and prev.start + prev.dur > slot:
                continue
            turn_at[slot] = m.turn

    mel_at = {m.start: m for m in mel}
    pattern = level.pattern
    bass_ticks, fill_ticks = set(), set()
    for b0 in range(0, end_tick, bar):
        bass_ticks.add(b0)                 # bass at each bar start, and (below) at chord changes;
        if pattern == "easy":              # never re-struck mid-bar on the same chord: it rings
            continue
        step = TICKS_PER_BEAT if pattern == "medium" else EIGHTH
        fill_ticks.update(range(b0, min(b0 + bar, end_tick), step))
    if pattern != "easy":
        bass_ticks.update(c.start for c in chords if c.start < end_tick)
    bass_ticks = sorted(t for t in bass_ticks if t < end_tick)

    ticks = sorted(set(mel_at) | set(turn_at) | set(bass_ticks) | fill_ticks)
    slices = []
    sounding = sorted(mel, key=lambda m: m.start)
    mi = 0
    last_fill = None                      # (chord start, pitch class) of the previous fill note
    for t in ticks:
        s = Slice(t)
        ch = _chord_at(chords, starts, t)
        m = mel_at.get(t)
        if t in turn_at:
            s.turn = turn_at[t]
            s.mel, s.mel_dur = s.turn[0], EIGHTH
        elif m is not None:
            s.mel, s.mel_dur, s.mel_glide = m.pitch, m.dur, m.glide_in
        else:
            while mi < len(sounding) and sounding[mi].start + sounding[mi].dur <= t:
                mi += 1
            if mi < len(sounding) and sounding[mi].start < t:
                s.held = sounding[mi].pitch
        if t in bass_ticks:
            s.bass_pcs = [ch.bass] + ([ch.root] if ch.root != ch.bass else [])
            nxt = next((b for b in bass_ticks if b > t), end_tick)
            s.bass_dur = nxt - t
        if t in fill_ticks and s.mel is None:
            dur = TICKS_PER_BEAT if pattern == "medium" else EIGHTH
            # melody notes that will sound together with this chord note
            near = [s.held] if s.held is not None else []
            near += [m.pitch for m in mel if t < m.start < t + dur]
            tones = chord_tones(ch)
            pref = [tones[1], tones[2], tones[0]] + tones[3:]          # 3rd, 5th, root, 7th

            def clashes(pc):
                return any((p - pc) % 12 in (0, 1, 6, 11) for p in near)

            s.inner_pcs = [pc for pc in pref if pc != ch.bass and not clashes(pc)]
            if not s.inner_pcs:
                s.inner_pcs = [pc for pc in pref if not clashes(pc)]
            # arpeggiate: don't re-strike the chord note that is still ringing from the last fill
            if last_fill and last_fill[0] == ch.start and last_fill[1] in s.inner_pcs and len(s.inner_pcs) > 1:
                s.inner_pcs.remove(last_fill[1])
                s.inner_pcs.append(last_fill[1])
            if s.inner_pcs:
                last_fill = (ch.start, s.inner_pcs[0])
            # let it ring until the chord changes (a new note on its string still stops it)
            s.inner_dur = max(dur, min(ch.end, end_tick) - t)
        if s.mel is not None or s.bass_pcs or s.inner_pcs:
            slices.append(s)
        # (a fill slot whose every chord tone would clash with the melody stays empty)
    return slices


# ---------------------------------------------------------------------------
# fingering
# ---------------------------------------------------------------------------
def _speed(dt: float) -> float:
    return 1.0 + 3.0 * max(0.0, 0.3 - dt) / 0.3


def _accomp_options(s: Slice, open_: np.ndarray, maxf: int, soft: int, capo: int,
                    level: LevelSettings):
    """All ways to play this slice's bass + chord note. Returns arrays describing each
    option: cost, highest string used, highest pitch, lowest/highest fretted fret,
    and the notes [(string, fret, pitch, voice), ...]."""
    def positions(pcs, strings, lo_pitch, hi_pitch):
        out = []
        for k, pc in enumerate(pcs):
            for st in strings:
                for f in range(0, min(maxf, soft) + 1):
                    p = int(open_[st]) + f
                    if p % 12 == pc and lo_pitch <= p <= hi_pitch:
                        out.append((st, f, p, k))
        return out

    bass = positions(s.bass_pcs, range(min(4, len(open_))), 0, 127) if s.bass_pcs else []
    inner = positions(s.inner_pcs, range(1, len(open_)), 50, 127) if s.inner_pcs else []

    def cost(f, rank, voice):
        base = 0.0 if f == 0 else (0.3 if voice == "bass" else 0.2) + level.height_weight * (f + capo)
        return base + (0.5 if voice == "bass" else 0.5) * rank

    opts = []
    b_opts = [None] + bass if s.bass_pcs else [None]
    i_opts = [None] + inner if s.inner_pcs else [None]
    for b in b_opts:
        for i in i_opts:
            if b and i and (i[0] <= b[0] or i[2] <= b[2]):
                continue
            c = 0.0
            notes = []
            if s.bass_pcs:
                if b is None:
                    c += level.bass_drop_cost
                else:
                    c += cost(b[1], b[3], "bass") + 0.03 * (b[2] - open_[0])
                    notes.append((b[0], b[1], b[2], "bass"))
            if s.inner_pcs:
                if i is None:
                    c += level.inner_drop_cost
                else:
                    c += cost(i[1], i[3], "inner")
                    notes.append((i[0], i[1], i[2], "inner"))
            fretted = [n[1] for n in notes if n[1] > 0]
            opts.append((c, max([n[0] for n in notes], default=-1), max([n[2] for n in notes], default=-1),
                         min(fretted, default=999), max(fretted, default=-1), notes))
    arr = {k: np.array([o[j] for o in opts]) for j, k in enumerate(("cost", "smax", "pmax", "fmin", "fmax"))}
    arr["notes"] = [o[5] for o in opts]
    return arr


def finger(slices: list[Slice], tuning: list[int], capo: int, level: LevelSettings,
           bpm: float) -> tuple[float, list[ArrNote], dict]:
    S = len(tuning)
    open_ = np.array(tuning) + capo
    maxf = GUITAR_FRETS - capo
    soft = level.soft_max_fret
    H = max(1, maxf - HAND_SPAN)
    hs = np.arange(1, H + 1)[:, None]                 # (H,1)
    strings = np.arange(S)[None, :]                   # (1,S)
    sec_per_tick = 60.0 / bpm / TICKS_PER_BEAT

    costs, backs, picks, layers = [], [], [], []
    prev_cost = None
    prev_tick = None
    for s in slices:
        mel_cost = np.zeros((H, S))
        fold = 0
        mel_fret = None
        turn_ok = None
        if s.mel is not None:
            p = s.mel
            for shift in (0, -12, 12, -24, 24):
                fr = p + shift - open_
                if ((fr >= 0) & (fr <= maxf)).any():
                    fold, p = shift, p + shift
                    break
            fr = p - open_                                       # (S,)
            ok = (fr >= 0) & (fr <= maxf)
            in_soft = ok & (fr <= soft)
            win = (fr[None, :] == 0) | ((hs <= fr[None, :]) & (fr[None, :] <= hs + HAND_SPAN))
            mel_cost = np.where(ok[None, :] & win, 0.0, INF)
            mel_cost = mel_cost + level.height_weight * np.where(fr > 0, fr + capo, 0)[None, :]
            mel_cost = mel_cost + level.thin_string_cost * strings
            if in_soft.any():
                mel_cost = mel_cost + SOFT_LIMIT_COST * (fr > soft)[None, :]
            if fold:
                mel_cost = mel_cost + FOLD_COST
            mel_fret = fr
            if s.turn:
                tf = np.array([[q + fold - o for o in open_] for q in s.turn])   # (3,S)
                t_ok = ((tf >= 0) & (tf <= maxf)).all(axis=0)
                t_win = np.ones((H, S), dtype=bool)
                for row in tf:
                    t_win &= (row[None, :] == 0) | ((hs <= row[None, :]) & (row[None, :] <= hs + HAND_SPAN))
                turn_ok = t_ok[None, :] & t_win
                mel_cost = mel_cost + np.where(turn_ok, 0.0, TURN_DROP_COST)
            mel_pitch = p
        elif s.held is not None:
            fr = s.held - open_
            outside = (fr[None, :] > 0) & ((hs > fr[None, :]) | (fr[None, :] > hs + HAND_SPAN))
            mel_cost = HELD_CUT_COST * outside.astype(float)
            mel_pitch = s.held
        else:
            mel_pitch = None

        acc = _accomp_options(s, open_, maxf, soft, capo, level)
        limit = strings if mel_pitch is not None else np.full((1, S), S)          # accomp below melody string
        v_s = (acc["smax"][None, None, :] < limit[:, :, None])                     # (1,S,P)
        if mel_pitch is not None:
            v_s = v_s & (acc["pmax"][None, None, :] < mel_pitch)
        v_h = (acc["fmin"][None, None, :] >= hs[:, :, None]) & (acc["fmax"][None, None, :] <= hs[:, :, None] + HAND_SPAN)
        tot = np.where(v_s & v_h, acc["cost"][None, None, :], INF)                # (H,S,P)
        acc_pick = tot.argmin(axis=2)
        acc_cost = tot.min(axis=2)
        local = mel_cost + acc_cost

        if prev_cost is None:
            cur = local
            back = np.zeros((H, S), dtype=np.int64)
        else:
            sp = _speed((s.tick - prev_tick) * sec_per_tick)
            dh = np.abs(hs - hs.T)                                                 # (H,H)
            t_h = level.shift_weight * (dh + 0.8 * (dh > 0)) * sp
            ds = np.abs(np.arange(S)[:, None] - np.arange(S)[None, :])
            if s.mel is None:
                t_s = np.where(ds == 0, 0.0, INF)        # no new melody note: melody string stays
            else:
                t_s = 0.05 * ds
            total = (prev_cost[:, :, None, None] + t_h[:, None, :, None] + t_s[None, :, None, :])
            flat = total.reshape(H * S, H, S)
            back = flat.argmin(axis=0)
            cur = flat.min(axis=0) + local
        costs.append(cur)
        backs.append(back)
        picks.append((acc_pick, acc, mel_fret, fold, turn_ok, mel_pitch if s.mel is not None else None))
        prev_cost, prev_tick = cur, s.tick

    if not slices:
        return 0.0, [], {}
    idx = int(costs[-1].argmin())
    total_cost = float(costs[-1].reshape(-1)[idx])
    states = [None] * len(slices)
    h_i, s_i = divmod(idx, S)
    for k in range(len(slices) - 1, -1, -1):
        states[k] = (h_i, s_i)
        if k > 0:
            h_i, s_i = divmod(int(backs[k][h_i, s_i]), S)

    notes: list[ArrNote] = []
    dropped = {"bass": 0, "chord_notes": 0, "turns": 0, "octave_moved": 0}
    last_mel = None
    for s, (h_i, s_i), (acc_pick, acc, mel_fret, fold, turn_ok, mel_pitch) in zip(slices, states, picks):
        if s.mel is not None:
            f = int(mel_fret[s_i])
            if fold:
                dropped["octave_moved"] += 1
            if s.turn and turn_ok is not None and turn_ok[h_i, s_i]:
                for k, q in enumerate(s.turn):
                    qf = q + fold - int(open_[s_i])
                    notes.append(ArrNote(s.tick + 4 * k, 4, q + fold + capo * 0, "melody", s_i, qf,
                                         triplet=k + 1, octave_moved=fold))
                last_mel = notes[-1]
            else:
                if s.turn:
                    dropped["turns"] += 1
                slide = (level.slides and s.mel_glide and last_mel is not None and last_mel.string == s_i
                         and last_mel.fret > 0 and f > 0 and last_mel.fret != f
                         and last_mel.start + last_mel.dur == s.tick)
                n = ArrNote(s.tick, s.mel_dur, mel_pitch, "melody", s_i, f, slide_in=slide, octave_moved=fold)
                notes.append(n)
                last_mel = n
        chosen = acc["notes"][int(acc_pick[h_i, s_i])]
        voices = {v for *_, v in chosen}
        if s.bass_pcs and "bass" not in voices:
            dropped["bass"] += 1
        if s.inner_pcs and "inner" not in voices:
            dropped["chord_notes"] += 1
        for st, f, p, v in chosen:
            notes.append(ArrNote(s.tick, s.bass_dur if v == "bass" else s.inner_dur, p, v, st, f))
    # pitches above are shape pitches + capo already (open_ includes capo) -> sounding
    return total_cost, notes, dropped


def arrange_level(melody: list[MelNote], chords: list[Chord], beats_per_bar: int, end_tick: int,
                  level_key: str, level: LevelSettings, tuning: list[int], tuning_name: str,
                  bpm: float, capo: int | None = None) -> Arrangement:
    slices = make_slices(melody, chords, beats_per_bar, end_tick, level)
    capos = [capo] if capo is not None else range(0, level.max_capo + 1)
    results = {}
    for c in capos:
        cost, notes, dropped = finger(slices, tuning, c, level, bpm)
        cost += level.capo_cost * c + (level.capo_fixed_cost if c else 0.0)
        results[c] = (cost, notes, dropped)
    best = min(results, key=lambda c: results[c][0])
    cost, notes, dropped = results[best]
    notes, dropped["let_ring"] = let_ring(notes)
    notes.sort(key=lambda n: (n.start, n.string))
    return Arrangement(level_key, best, list(tuning), tuning_name, notes, end_tick, dropped,
                       round(cost, 2), {c: round(v[0], 2) for c, v in results.items()})


def let_ring(notes: list[ArrNote]) -> tuple[list[ArrNote], int]:
    """A bass or chord note struck again on the next 8th, on the same string, at the same
    pitch, is left out: the first strike just keeps ringing (as a player would do)."""
    by_string: dict[int, list[ArrNote]] = {}
    for n in sorted(notes, key=lambda n: n.start):
        by_string.setdefault(n.string, []).append(n)
    drop = set()
    for ns in by_string.values():
        keep = None
        for n in ns:
            if (keep is not None and n.voice != "melody" and keep.voice != "melody" and n.pitch == keep.pitch
                    and n.start - keep.start <= EIGHTH and n.start - (keep.start + keep.dur) <= EIGHTH):
                keep.dur = max(keep.dur, n.start + n.dur - keep.start)
                drop.add(id(n))
                continue
            keep = n
    return [n for n in notes if id(n) not in drop], len(drop)


def parse_tuning(spec: str) -> tuple[list[int], str]:
    """'standard' / 'dropd' / 'D2,A2,D3,G3,B3,E4' -> (MIDI pitches low..high, label)."""
    import librosa

    from .config import TUNING_LABELS, TUNINGS

    key = spec.lower().replace(" ", "").replace("_", "")
    if key in TUNINGS:
        names, label = TUNINGS[key], TUNING_LABELS[key]
    else:
        names, label = [s.strip() for s in spec.split(",")], "Custom"
    pitches = [int(librosa.note_to_midi(n)) for n in names]
    if pitches != sorted(pitches):
        raise ValueError("List tuning strings from lowest (thickest) to highest")
    return pitches, f"{label} ({' '.join(n.rstrip('0123456789') for n in names)})"
