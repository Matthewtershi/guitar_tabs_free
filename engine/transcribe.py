"""Audio -> master melody (list[Note]).

Three paths, chosen by what the audio is:
  - vocals / any single-note line : torchcrepe pitch contour -> notes.
        The contour is kept so slides, bends and vibrato can be read off it.
  - piano                         : ByteDance piano transcription -> highest-note line
  - guitar / other                : Spotify Basic Pitch -> highest-note line
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .model import Note, PolyNote

# ---- tunable detection thresholds (to be tuned on your test songs) ----
VOICED_PERIODICITY = 0.25     # crepe confidence (smoothed over ~70 ms) needed to count as sung
CONFIDENCE_SMOOTH = 7         # frames; crepe confidence flickers frame to frame on separated stems
VOICED_DB = -38.0             # frames quieter than this (dB below the loud parts) are silence
BRIDGE_GAP_FRAMES = 15        # bridge dropouts up to 150 ms if pitch and loudness carry on across
EXTEND_FRAMES = 15            # look back/forward up to 150 ms for where a note really starts/ends
WOBBLE_MAX_FRAMES = 15        # 1-semitone excursions shorter than 150 ms inside a note = vibrato
SEMITONE_SWITCH_COST = 1.5    # higher = fewer pitch changes (ignores vibrato wobble)
MIN_NOTE_FRAMES = 5           # 50 ms; shorter pitch segments are glides, not notes
REPEAT_SPLIT_MIN = 0.08       # s; min note length on each side of a repeated-note split
REPEAT_DIP_DB = 5.0           # a repeated note needs a loudness dip this deep (a consonant/breath)
VIBRATO_RATE = (4.0, 9.0)     # Hz
VIBRATO_DEPTH = (0.2, 1.3)    # semitones (peak)
BEND_START_DEV = 0.8          # semitones below the note at its start to count as a bend/scoop


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Start/end (exclusive) of each run of True."""
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _close_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    out = mask.copy()
    runs = _runs(~mask)
    for a, b in runs:
        if a > 0 and b < len(mask) and b - a <= max_gap:
            out[a:b] = True
    return out


def _contour_techniques(note: Note, dev: np.ndarray, fps: float) -> None:
    """Read vibrato and bends off a note's pitch deviation (semitones vs. its pitch)."""
    if len(dev) < 3:
        return
    dur = len(dev) / fps
    # Bend / scoop: starts clearly flat and rises to the note.
    head = dev[: max(3, int(0.15 * fps))]
    if dur >= 0.2 and np.mean(head[:3]) <= -BEND_START_DEV:
        reach = np.where(np.abs(dev) < 0.3)[0]
        if len(reach) and reach[0] / fps <= 0.25:
            note.bend_from = int(np.clip(round(-float(np.mean(head[:3]))), 1, 2))
    # Vibrato: steady oscillation around the note.
    if dur >= 0.3:
        body = dev[int(0.05 * fps):]
        win = max(3, int(0.15 * fps))
        trend = np.convolve(body, np.ones(win) / win, mode="same")
        wob = body - trend
        crossings = np.sum(np.diff(np.sign(wob)) != 0)
        rate = crossings / (2 * (len(body) / fps))
        depth = float(np.sqrt(2) * np.std(wob))
        if VIBRATO_RATE[0] <= rate <= VIBRATO_RATE[1] and VIBRATO_DEPTH[0] <= depth <= VIBRATO_DEPTH[1]:
            note.vibrato = True


# ---------------------------------------------------------------------------
# single-note line (vocals)
# ---------------------------------------------------------------------------
def _viterbi_semitones(m: np.ndarray, switch_cost: float) -> np.ndarray:
    """Piecewise-constant semitone path through a pitch contour.
    Staying is free, switching costs `switch_cost`, so vibrato doesn't create notes."""
    lo, hi = int(np.floor(m.min())) - 1, int(np.ceil(m.max())) + 1
    states = np.arange(lo, hi + 1)
    emis = np.minimum((m[:, None] - states[None, :]) ** 2, 2.25)
    T, S = emis.shape
    cost = emis[0].copy()
    back = np.zeros((T, S), dtype=np.int32)
    idx = np.arange(S)
    for t in range(1, T):
        bp = int(cost.argmin())
        switch = cost[bp] + switch_cost
        stay = cost <= switch
        back[t] = np.where(stay, idx, bp)
        cost = np.where(stay, cost, switch) + emis[t]
    path = np.empty(T, dtype=np.int32)
    path[-1] = int(cost.argmin())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return states[path]


@dataclass
class _Seg:
    start: int
    end: int
    pitch: int
    glide_in: bool = False
    bend_from: int = 0
    legato: bool = True


def _merge_short(segs: list[_Seg], min_len: int) -> list[_Seg]:
    segs = list(segs)
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, s in enumerate(segs):
            if s.end - s.start >= min_len:
                continue
            prev = segs[i - 1] if i > 0 else None
            nxt = segs[i + 1] if i + 1 < len(segs) else None
            if prev and nxt:
                lo, hi = sorted((prev.pitch, nxt.pitch))
                if lo <= s.pitch <= hi and prev.pitch != nxt.pitch:
                    nxt.glide_in = True          # passing through on the way: a slide
                target = prev if abs(prev.pitch - s.pitch) < abs(nxt.pitch - s.pitch) else nxt
            else:
                target = prev or nxt
                if target is nxt and 1 <= nxt.pitch - s.pitch <= 2:
                    nxt.bend_from = nxt.pitch - s.pitch   # scoop up into the first note
            if target is prev:
                prev.end = s.end
            else:
                nxt.start = s.start
            segs.pop(i)
            changed = True
            break
    # Wobble: a brief one-semitone excursion between two pieces of the same pitch
    # is vibrato or a scoop, not a separate note.
    changed = True
    while changed and len(segs) > 2:
        changed = False
        for i in range(1, len(segs) - 1):
            p, s, nx = segs[i - 1], segs[i], segs[i + 1]
            if (p.pitch == nx.pitch and abs(s.pitch - p.pitch) == 1
                    and s.end - s.start < WOBBLE_MAX_FRAMES):
                p.end = nx.end
                segs[i - 1:i + 2] = [p]
                changed = True
                break
    # neighbours that ended up on the same pitch join back together
    out: list[_Seg] = []
    for s in segs:
        if out and out[-1].pitch == s.pitch and out[-1].end == s.start:
            out[-1].end = s.end
        else:
            out.append(s)
    return out


def transcribe_monophonic(wav: Path, device: str) -> tuple[list[Note], dict]:
    import librosa
    import torch
    import torchcrepe
    from scipy.signal import medfilt

    sr, hop = 16000, 160
    fps = sr / hop
    y, _ = librosa.load(str(wav), sr=sr, mono=True)
    x = torch.from_numpy(y).float()[None]
    f0, per = torchcrepe.predict(x, sr, hop, fmin=65.0, fmax=1400.0, model="full",
                                 decoder=torchcrepe.decode.viterbi, return_periodicity=True,
                                 batch_size=512, device=device, pad=True)
    f0 = f0[0].cpu().numpy()
    per = medfilt(per[0].cpu().numpy(), CONFIDENCE_SMOOTH)
    n = len(f0)

    def loudness(frame_length):
        r = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop, center=True)[0]
        r = np.pad(r, (0, max(0, n - len(r))))[:n]
        d = 20 * np.log10(r + 1e-8)
        return d - np.percentile(d, 99)

    db = loudness(1024)
    db_fast = loudness(400)        # 25 ms window: sees consonant dips between syllables
    voiced = (per > VOICED_PERIODICITY) & (db > VOICED_DB)
    midi = 69 + 12 * np.log2(np.maximum(f0, 1e-3) / 440.0)

    # Recording tuning (e.g. A=432): circular mean of the cents offset.
    info = {"tuning_offset_semitones": 0.0}
    if voiced.any():
        ang = 2 * np.pi * (midi[voiced] - np.round(midi[voiced]))
        offset = float(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) / (2 * np.pi))
        midi = midi - offset
        info["tuning_offset_semitones"] = round(offset, 3)

    voiced = _close_gaps(voiced, 3)
    voiced = _bridge_dropouts(voiced, midi, db)
    voiced = _extend_regions(voiced, midi, db)
    split_min = int(REPEAT_SPLIT_MIN * fps)

    notes: list[Note] = []
    for a, b in _runs(voiced):
        if b - a < MIN_NOTE_FRAMES + 1:
            continue
        path = _viterbi_semitones(midi[a:b], SEMITONE_SWITCH_COST)
        segs = [_Seg(a + s, a + e, int(path[s])) for s, e in _runs_of_values(path)]
        segs = _merge_short(segs, MIN_NOTE_FRAMES)
        segs[0].legato = False
        # Repeated notes on one pitch: split where the voice dips in loudness
        # (a consonant or breath). Drum bleed adds energy, so it can't fake a dip.
        final: list[_Seg] = []
        for s in segs:
            cuts = _loudness_dips(db_fast, s.start + split_min, s.end - split_min)
            pieces, cur = [], s.start
            for c in cuts:
                if c - cur >= split_min:
                    pieces.append((cur, c))
                    cur = c
            pieces.append((cur, s.end))
            for k, (ps, pe) in enumerate(pieces):
                final.append(_Seg(ps, pe, s.pitch,
                                  glide_in=s.glide_in if k == 0 else False,
                                  bend_from=s.bend_from if k == 0 else 0,
                                  legato=s.legato if k == 0 else False))
        for i, s in enumerate(final):
            dev = midi[s.start:s.end] - s.pitch
            note = Note(onset=s.start / fps, offset=s.end / fps, pitch=s.pitch,
                        velocity=float(np.clip(1 + db[s.start:s.end].mean() / 40, 0.1, 1.0)),
                        legato=s.legato, glide_in=s.glide_in, bend_from=s.bend_from)
            if note.legato and i > 0 and not note.glide_in:
                # long approach from the previous pitch = slide
                prev_pitch = final[i - 1].pitch
                toward = np.sign(prev_pitch - s.pitch)
                approach = 0
                while approach < len(dev) and dev[approach] * toward > 0.4:
                    approach += 1
                note.glide_in = approach >= 4
            _contour_techniques(note, dev, fps)
            if note.legato:
                note.bend_from = 0   # a bend into a legato note reads as a slide instead
            notes.append(note)
    return notes, [], info


def _bridge_dropouts(voiced: np.ndarray, midi: np.ndarray, db: np.ndarray) -> np.ndarray:
    """Fill short unvoiced gaps where the same pitch continues on both sides and the
    loudness doesn't drop: the pitch tracker lost confidence, the singer didn't stop."""
    out = voiced.copy()
    for a, b in _runs(~voiced):
        if a == 0 or b >= len(voiced) or b - a > BRIDGE_GAP_FRAMES:
            continue
        left = np.median(midi[max(0, a - 5):a])
        right = np.median(midi[b:b + 5])
        sides = min(np.median(db[max(0, a - 5):a]), np.median(db[b:b + 5]))
        if abs(left - right) <= 1.0 and db[a:b].min() > sides - 6.0:
            out[a:b] = True
    return out


def _extend_regions(voiced: np.ndarray, midi: np.ndarray, db: np.ndarray) -> np.ndarray:
    """Pitch confidence often lags the real note start (e.g. when a chord hits at the
    same moment). Walk each sung region outward while the raw pitch still matches
    and the voice is still audible."""
    out = voiced.copy()
    for a, b in _runs(voiced):
        if b - a < 3:
            continue
        p0, p1 = np.median(midi[a:a + 5]), np.median(midi[max(a, b - 5):b])
        j = a - 1
        while j >= max(0, a - EXTEND_FRAMES) and not out[j] and abs(midi[j] - p0) < 0.8 and db[j] > VOICED_DB:
            out[j] = True
            j -= 1
        j = b
        while j < min(len(out), b + EXTEND_FRAMES) and not out[j] and abs(midi[j] - p1) < 0.8 and db[j] > VOICED_DB:
            out[j] = True
            j += 1
    return out


def _loudness_dips(db: np.ndarray, lo: int, hi: int) -> list[int]:
    """Frames in [lo, hi] that are local loudness minima at least REPEAT_DIP_DB below
    the louder of the two sides (within ~100 ms)."""
    cuts = []
    for i in range(max(lo, 1), min(hi, len(db) - 1) + 1):
        if db[i] > db[i - 1] or db[i] > db[i + 1]:
            continue
        left = db[max(0, i - 10):i].max()
        right = db[i + 1:i + 11].max()
        if min(left, right) - db[i] >= REPEAT_DIP_DB:
            cuts.append(i)
    return cuts


def _runs_of_values(path: np.ndarray) -> list[tuple[int, int]]:
    change = np.where(np.diff(path) != 0)[0] + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(path)]])
    return list(zip(starts.tolist(), ends.tolist()))


# ---------------------------------------------------------------------------
# polyphonic -> highest-note line
# ---------------------------------------------------------------------------
@dataclass
class _Poly:
    on: float
    off: float
    pitch: int
    vel: float
    contour: np.ndarray | None = None   # semitone deviation, at `fps`
    fps: float = 0.0
    used: bool = False                  # became a melody note


CHORD_WINDOW = 0.035     # s; notes starting this close together count as one chord
LEGATO_OVERLAP = 0.12    # s; a higher note ringing less than this into the next is legato overlap
MIN_MELODY_NOTE = 0.04   # s


GHOST_MIN_DUR = 0.03          # s; detections shorter than this are attack noise
GHOST_MAX_DUR = 0.12          # s; overtone ghosts are always shorter than this
GHOST_INTERVALS = (12, 19, 24, 28, 31, 36)   # overtones: octave, 12th, 2 octaves, ...


def remove_ghosts(events: list[_Poly]) -> list[_Poly]:
    """Drop overtone 'ghost' notes: very short, or an overtone interval above a note
    that starts at the same moment and rings much longer. These would otherwise
    win the highest-note rule and replace the real melody note."""
    ev = [e for e in events if e.off - e.on >= GHOST_MIN_DUR]
    ev.sort(key=lambda e: e.on)
    out = []
    for i, e in enumerate(ev):
        ghost = False
        j = i - 1
        while j >= 0 and e.on - ev[j].on <= 0.06:
            j -= 1
        for o in ev[j + 1:]:
            if o.on - e.on > 0.06:
                break
            if (o is not e and (e.pitch - o.pitch) in GHOST_INTERVALS
                    and (e.off - e.on) < GHOST_MAX_DUR
                    and (e.off - e.on) < 0.5 * (o.off - o.on)):
                ghost = True
                break
        if not ghost:
            out.append(e)
    return out


def split_melody(events: list[_Poly]) -> tuple[list[Note], list[PolyNote]]:
    """Highest-note melody + everything else as accompaniment (for bass and chords)."""
    clean = remove_ghosts(events)
    melody = skyline(clean)
    accomp = [PolyNote(e.on, e.off, e.pitch, e.vel) for e in clean if not e.used]
    return melody, accomp


def skyline(events: list[_Poly]) -> list[Note]:
    """Highest note wins: keep a note only if nothing higher starts with it or is still ringing."""
    ev = sorted(events, key=lambda e: (e.on, -e.pitch))
    kept: list[_Poly] = []
    for i, e in enumerate(ev):
        j, higher_in_chord = i - 1, False
        while j >= 0 and e.on - ev[j].on <= CHORD_WINDOW:
            higher_in_chord |= ev[j].pitch > e.pitch
            j -= 1
        j = i + 1
        while j < len(ev) and ev[j].on - e.on <= CHORD_WINDOW:
            higher_in_chord |= ev[j].pitch > e.pitch
            j += 1
        if higher_in_chord:
            continue
        if kept and kept[-1].pitch > e.pitch and kept[-1].off > e.on + LEGATO_OVERLAP:
            continue
        if kept and kept[-1].off > e.on:
            kept[-1].off = e.on
        e.used = True
        kept.append(_Poly(e.on, e.off, e.pitch, e.vel, e.contour, e.fps))
    notes = []
    for k in kept:
        if k.off - k.on < MIN_MELODY_NOTE:
            continue
        n = Note(onset=k.on, offset=k.off, pitch=k.pitch, velocity=float(k.vel))
        if k.contour is not None and len(k.contour):
            keep = int((k.off - k.on) * k.fps) + 1
            _contour_techniques(n, k.contour[:keep], k.fps)
        notes.append(n)
    return notes


def transcribe_piano(wav: Path, device: str) -> tuple[list[Note], dict]:
    raw = wav.with_name(wav.stem + "_piano_raw.json")
    if raw.exists():
        rows = json.loads(raw.read_text())
    else:
        import librosa
        from piano_transcription_inference import PianoTranscription, sample_rate

        audio, _ = librosa.load(str(wav), sr=sample_rate, mono=True)
        pt = PianoTranscription(device=device, checkpoint_path=str(piano_checkpoint()))
        result = pt.transcribe(audio, str(wav.with_name(wav.stem + "_piano_all_notes.mid")))
        rows = [[float(e["onset_time"]), float(e["offset_time"]), int(e["midi_note"]),
                 float(e["velocity"]) / 127.0] for e in result["est_note_events"]]
        raw.write_text(json.dumps(rows))
    events = [_Poly(*r) for r in rows]
    melody, accomp = split_melody(events)
    return melody, accomp, {"polyphonic_notes": len(events)}


def piano_checkpoint() -> Path:
    """The package downloads with `wget`, which Windows lacks; fetch it ourselves."""
    import urllib.request

    path = Path.home() / "piano_transcription_inference_data" / "note_F1=0.9677_pedal_F1=0.9186.pth"
    if not path.exists() or path.stat().st_size < 1.6e8:
        path.parent.mkdir(parents=True, exist_ok=True)
        url = ("https://zenodo.org/record/4034264/files/"
               "CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1")
        print("Downloading piano transcription model (~170 MB) ...")
        urllib.request.urlretrieve(url, path)
    return path


def transcribe_poly(wav: Path) -> tuple[list[Note], dict]:
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict

    _, _, note_events = predict(str(wav), ICASSP_2022_MODEL_PATH, onset_threshold=0.5,
                                frame_threshold=0.3, minimum_note_length=80,
                                melodia_trick=True, multiple_pitch_bends=False)
    fps = 22050 / 256   # Basic Pitch frame rate; bends are in 1/3 semitones
    events = []
    for on, off, pitch, amp, bends in note_events:
        contour = np.asarray(bends, dtype=float) / 3.0 if bends is not None else None
        events.append(_Poly(float(on), float(off), int(pitch), float(amp), contour, fps))
    melody, accomp = split_melody(events)
    return melody, accomp, {"polyphonic_notes": len(events)}
