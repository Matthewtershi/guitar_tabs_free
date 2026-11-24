"""Arrangement -> MusicXML: standard notation (melody stems up, bass + chord notes
stems down) above a TAB staff, like the example tab.

Notation is written in the capo "shape" key (what your fingers play, as if there
were no capo), the way the example writes it; the header says "Capo N".
The file also opens in MuseScore / Guitar Pro / TuxGuitar for editing.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

from .config import TICKS_PER_BEAT
from .model import Arrangement, ArrNote

TPB = TICKS_PER_BEAT
_FIFTHS_MAJOR = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 1: -5, 8: -4, 3: -3, 10: -2, 5: -1}
_SHARP = [("C", 0), ("C", 1), ("D", 0), ("D", 1), ("E", 0), ("F", 0), ("F", 1), ("G", 0), ("G", 1),
          ("A", 0), ("A", 1), ("B", 0)]
_FLAT = [("C", 0), ("D", -1), ("D", 0), ("E", -1), ("E", 0), ("F", 0), ("G", -1), ("G", 0), ("A", -1),
         ("A", 0), ("B", -1), ("B", 0)]
_SHARP_ORDER = "FCGDAEB"
_TYPES = {96: ("whole", 0), 72: ("half", 1), 48: ("half", 0), 36: ("quarter", 1), 24: ("quarter", 0),
          18: ("eighth", 1), 12: ("eighth", 0), 9: ("16th", 1), 6: ("16th", 0), 3: ("32nd", 0)}
_VALUES = [96, 72, 48, 36, 24, 18, 12, 9, 6, 3]


def key_fifths(tonic: int, mode: str) -> int:
    major_tonic = tonic if mode == "major" else (tonic + 3) % 12
    return _FIFTHS_MAJOR[major_tonic % 12]


def _spell(pitch: int, fifths: int):
    step, alter = (_SHARP if fifths >= 0 else _FLAT)[pitch % 12]
    return step, alter, pitch // 12 - 1


def _key_alters(fifths: int) -> dict:
    alters = {s: 0 for s in "CDEFGAB"}
    if fifths > 0:
        for s in _SHARP_ORDER[:fifths]:
            alters[s] = 1
    elif fifths < 0:
        for s in _SHARP_ORDER[::-1][:-fifths]:
            alters[s] = -1
    return alters


def _pieces(s: int, e: int) -> list[tuple[int, int]]:
    """Split [s, e) (ticks inside one bar) into readable note values:
    values of a beat or longer start on a beat, shorter ones don't cross a beat."""
    out = []
    while s < e:
        for v in _VALUES:
            if s + v > e:
                continue
            if v >= 48 and s % 48:
                continue
            if v >= TPB and s % TPB:
                continue
            if v < TPB and (s % TPB) + v > TPB:
                continue
            break
        else:
            v = e - s
        out.append((s, v))
        s += v
    return out


class _Event:
    def __init__(self, start, dur, notes, tie_start=False, tie_stop=False, triplet=0):
        self.start, self.dur, self.notes = start, dur, notes      # notes: list[ArrNote] (empty = rest)
        self.tie_start, self.tie_stop, self.triplet = tie_start, tie_stop, triplet


def _voice_events(notes: list[ArrNote], end_tick: int, bar: int) -> list[_Event]:
    """Notes of one voice -> non-overlapping events (chords where notes start together),
    each lasting until the next event, split at barlines, with rests in gaps."""
    by_start = defaultdict(list)
    for n in notes:
        by_start[n.start].append(n)
    starts = sorted(by_start)
    raw = []
    for k, t in enumerate(starts):
        group = by_start[t]
        nxt = starts[k + 1] if k + 1 < len(starts) else end_tick
        dur = min(max(n.dur for n in group), nxt - t)
        raw.append((t, dur, group))
    events: list[_Event] = []
    cursor = 0
    for t, dur, group in raw:
        if t > cursor:
            events += _split(cursor, t - cursor, [], bar)
        trip = group[0].triplet
        if trip:
            events.append(_Event(t, dur, group, triplet=trip))
        else:
            events += _split(t, dur, group, bar)
        cursor = t + dur
    if cursor < end_tick:
        events += _split(cursor, end_tick - cursor, [], bar)
    return events


def _split(t: int, dur: int, group, bar: int) -> list[_Event]:
    out = []
    s, e = t, t + dur
    while s < e:
        b0 = (s // bar) * bar
        seg_e = min(e, b0 + bar)
        for ps, pv in _pieces(s - b0, seg_e - b0):
            out.append(_Event(b0 + ps, pv, group))
        s = seg_e
    if group and len(out) > 1:
        for i, ev in enumerate(out):
            ev.tie_start = i < len(out) - 1
            ev.tie_stop = i > 0
    return out


def _beams(events: list[_Event]) -> dict:
    """Beam 8ths/16ths together within each beat. Returns {id(event): [beam1, beam2]}."""
    out = {}
    group = []

    def flush():
        if len(group) >= 2:
            for i, ev in enumerate(group):
                b1 = "begin" if i == 0 else ("end" if i == len(group) - 1 else "continue")
                out[id(ev)] = [b1, None]
            # second beam over runs of 16ths
            run = []
            for ev in group + [None]:
                short = ev is not None and _type_of(ev)[0] == "16th"
                if short:
                    run.append(ev)
                else:
                    if len(run) >= 2:
                        for i, r in enumerate(run):
                            out[id(r)][1] = "begin" if i == 0 else ("end" if i == len(run) - 1 else "continue")
                    run = []
        group.clear()

    for ev in events:
        beamable = ev.notes and _type_of(ev)[0] in ("eighth", "16th", "32nd")
        if beamable and group and ev.start // TPB == group[-1].start // TPB:
            group.append(ev)
        else:
            flush()
            if beamable:
                group.append(ev)
    flush()
    return out


def _type_of(ev: _Event):
    if ev.triplet:
        return "16th", 0
    return _TYPES.get(ev.dur, ("eighth", 0))


def write_musicxml(arr: Arrangement, path: Path, title: str, beats_per_bar: int, bpm: float,
                   key_tonic: int, key_mode: str) -> Path:
    bar = beats_per_bar * TPB
    end_tick = ((arr.end_tick + bar - 1) // bar) * bar
    capo = arr.capo
    fifths = key_fifths((key_tonic - capo) % 12, key_mode)
    n_str = len(arr.tuning)
    mel = [n for n in arr.notes if n.voice == "melody"]
    acc = [n for n in arr.notes if n.voice != "melody"]
    v1 = _voice_events(mel, end_tick, bar)
    v2 = _voice_events(acc, end_tick, bar)
    beams1, beams2 = _beams(v1), _beams(v2)

    # accidentals: per bar, notes in time order across both voices
    accidental = {}
    key_alt = _key_alters(fifths)
    for b0 in range(0, end_tick, bar):
        state = {}
        evs = sorted([e for e in v1 + v2 if b0 <= e.start < b0 + bar and e.notes], key=lambda e: e.start)
        for ev in evs:
            for n in ev.notes:
                step, alter, octv = _spell(n.pitch - capo, fifths)
                cur = state.get((step, octv), key_alt[step])
                if ev.tie_stop:
                    state[(step, octv)] = alter
                    continue
                if alter != cur:
                    accidental[(id(ev), id(n))] = {1: "sharp", -1: "flat", 0: "natural"}[alter]
                state[(step, octv)] = alter

    x = ['<?xml version="1.0" encoding="UTF-8"?>',
         '<score-partwise version="3.1">',
         f"<work><work-title>{escape(title)}</work-title></work>",
         '<part-list><score-part id="P1"><part-name print-object="no">Guitar</part-name>',
         '<score-instrument id="P1-I1"><instrument-name>Acoustic Guitar (steel)</instrument-name></score-instrument>',
         '<midi-instrument id="P1-I1"><midi-program>26</midi-program></midi-instrument>',
         "</score-part></part-list>", '<part id="P1">']

    def note_xml(ev: _Event, n: ArrNote | None, voice: int, staff: int, chord: bool, beams, stem):
        parts = ["<note>"]
        if n is not None and staff == 2 and ev.tie_stop:
            parts = ['<note print-object="no">']
        if chord:
            parts.append("<chord/>")
        if n is None:
            parts.append("<rest/>")
        else:
            step, alter, octv = _spell(n.pitch - capo, fifths)
            alt = f"<alter>{alter}</alter>" if alter else ""
            parts.append(f"<pitch><step>{step}</step>{alt}<octave>{octv}</octave></pitch>")
        parts.append(f"<duration>{ev.dur}</duration>")
        if n is not None and ev.tie_start:
            parts.append('<tie type="start"/>')
        if n is not None and ev.tie_stop:
            parts.append('<tie type="stop"/>')
        parts.append(f"<voice>{voice}</voice>")
        typ, dots = _type_of(ev)
        parts.append(f"<type>{typ}</type>" + "<dot/>" * dots)
        if n is not None and staff == 1 and (id(ev), id(n)) in accidental:
            parts.append(f"<accidental>{accidental[(id(ev), id(n))]}</accidental>")
        if ev.triplet:
            parts.append("<time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes>"
                         "</time-modification>")
        if n is not None and staff == 1:
            parts.append(f"<stem>{stem}</stem>")
        parts.append(f"<staff>{staff}</staff>")
        if staff == 1 and not chord and id(ev) in beams:
            b1, b2 = beams[id(ev)]
            parts.append(f'<beam number="1">{b1}</beam>')
            if b2:
                parts.append(f'<beam number="2">{b2}</beam>')
        nots = []
        if n is not None and ev.tie_start:
            nots.append('<tied type="start"/>')
        if n is not None and ev.tie_stop:
            nots.append('<tied type="stop"/>')
        if ev.triplet == 1 and not chord:
            nots.append('<tuplet type="start" bracket="yes"/>' if staff == 1 else "")
            nots.append('<slur type="start"/>')
        if ev.triplet == 3 and not chord:
            nots.append('<tuplet type="stop"/>' if staff == 1 else "")
            nots.append('<slur type="stop"/>')
        if n is not None and staff == 2:
            nots.append(f"<technical><string>{n_str - n.string}</string><fret>{n.fret}</fret></technical>")
        if any(nots):
            parts.append("<notations>" + "".join(nots) + "</notations>")
        parts.append("</note>")
        return "".join(parts)

    def voice_xml(events, b0, voice, staff, beams, stem):
        out = []
        for ev in events:
            if not (b0 <= ev.start < b0 + bar):
                continue
            if not ev.notes:
                out.append(note_xml(ev, None, voice, staff, False, beams, stem))
                continue
            for k, n in enumerate(sorted(ev.notes, key=lambda n: n.pitch)):
                out.append(note_xml(ev, n, voice, staff, k > 0, beams, stem))
        return out

    tuning_xml = "".join(
        f'<staff-tuning line="{i + 1}"><tuning-step>{_spell(p, 0)[0]}</tuning-step>'
        + (f"<tuning-alter>{_spell(p, 0)[1]}</tuning-alter>" if _spell(p, 0)[1] else "")
        + f"<tuning-octave>{_spell(p, 0)[2]}</tuning-octave></staff-tuning>"
        for i, p in enumerate(arr.tuning))
    for m, b0 in enumerate(range(0, end_tick, bar)):
        x.append(f'<measure number="{m + 1}">')
        if m == 0:
            x.append(f"<attributes><divisions>{TPB}</divisions><key><fifths>{fifths}</fifths></key>"
                     f"<time><beats>{beats_per_bar}</beats><beat-type>4</beat-type></time><staves>2</staves>"
                     '<clef number="1"><sign>G</sign><line>2</line><clef-octave-change>-1</clef-octave-change></clef>'
                     '<clef number="2"><sign>TAB</sign><line>5</line></clef>'
                     f'<staff-details number="2"><staff-lines>{n_str}</staff-lines>{tuning_xml}</staff-details>'
                     "</attributes>")
            x.append(f'<sound tempo="{round(bpm)}"/>')
        x += voice_xml(v1, b0, 1, 1, beams1, "up")
        x.append(f"<backup><duration>{bar}</duration></backup>")
        x += voice_xml(v2, b0, 2, 1, beams2, "down")
        x.append(f"<backup><duration>{bar}</duration></backup>")
        x += voice_xml(v1, b0, 5, 2, {}, "up")
        x.append(f"<backup><duration>{bar}</duration></backup>")
        x += voice_xml(v2, b0, 6, 2, {}, "down")
        x.append("</measure>")
    x += ["</part>", "</score-partwise>"]
    path.write_text("\n".join(x), encoding="utf-8")
    return path
