"""Make test audio with a KNOWN melody (Ode to Joy, public domain) so the
pipeline can be scored without needing real songs.

Writes to the given folder:
  synth_piano.wav     piano melody + quieter left-hand chords (like a piano tutorial)
  synth_song.wav      sung-like melody (vibrato, legato glides) + chords + drums
  reference.mid       the true melody notes

Usage: python tools/make_synthetic.py <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

SR = 44100
BPM = 100.0
BEAT = 60.0 / BPM

# (MIDI pitch or None for rest, length in beats). Includes 8ths, a dotted
# rhythm, rests, and one fast grace note (0.125 beat) as an ornament.
MELODY = [
    (64, 1), (64, 1), (65, 1), (67, 1),
    (67, 1), (65, 1), (64, 1), (62, 1),
    (60, 1), (60, 1), (62, 1), (64, 1),
    (64, 1.5), (62, 0.5), (62, 1), (None, 1),
    (64, 1), (64, 1), (65, 1), (67, 1),
    (67, 0.5), (69, 0.5), (67, 0.5), (65, 0.5), (64, 1), (62, 1),
    (60, 1), (60, 1), (62, 1), (64, 1),
    (62, 1.5), (60, 0.5), (60, 2),
    (None, 2), (66, 0.125), (67, 1.875),
    (72, 2), (None, 2),
]
CHORDS = ["C", "G", "C", "G", "C", "G", "C", "G", "C", "C"]   # one per bar
CHORD_NOTES = {"C": [48, 52, 55], "G": [43, 47, 50]}
LEAD_IN = 2 * BEAT   # half a bar of drums/silence before the melody


def hz(p):
    return 440.0 * 2 ** ((p - 69) / 12)


def piano_tone(p, dur, vel=0.8):
    n = int((dur + 0.4) * SR)
    t = np.arange(n) / SR
    f = hz(p)
    sig = sum(np.sin(2 * np.pi * k * f * np.sqrt(1 + 0.0004 * k * k) * t) * np.exp(-t * (0.9 + 0.7 * k)) / k
              for k in range(1, 9) if k * f < SR / 2)
    env = np.ones(n)
    r0 = int(dur * SR)
    env[r0:] = np.exp(-np.arange(n - r0) / (0.06 * SR))
    env[: int(0.004 * SR)] = np.linspace(0, 1, int(0.004 * SR))
    return vel * sig * env


def voice(notes, total):
    """Continuous sung-like line: sawtooth through a soft low-pass, glides between
    joined notes, vibrato on long notes, small breath gaps before rests."""
    out = np.zeros(total)
    phase = 0.0
    for i, (on, off, p) in enumerate(notes):
        a, b = int(on * SR), int(off * SR)
        t = np.arange(b - a) / SR
        pitch = np.full(b - a, float(p))
        if i > 0 and abs(notes[i - 1][1] - on) < 1e-6:         # legato from previous: glide
            g = min(len(t), int(0.05 * SR))
            pitch[:g] = np.linspace(notes[i - 1][2], p, g)
        if off - on > 0.5:                                      # vibrato on long notes
            pitch += 0.3 * np.sin(2 * np.pi * 5.5 * t) * np.clip((t - 0.2) / 0.2, 0, 1)
        f = hz(pitch)
        ph = phase + 2 * np.pi * np.cumsum(f) / SR
        phase = ph[-1] if len(ph) else phase
        saw = sum(np.sin(k * ph) / k for k in range(1, 12))
        env = np.ones(b - a)
        fade = min(len(env), int(0.02 * SR))
        env[:fade] = np.linspace(0.3, 1, fade)
        env[-fade:] = np.linspace(1, 0.2, fade)
        out[a:b] += 0.5 * saw * env
    # gentle low-pass
    k = np.exp(-np.arange(40) / 8.0)
    return np.convolve(out, k / k.sum(), mode="same")


def drums(total, n_beats):
    out = np.zeros(total)
    rng = np.random.default_rng(0)
    for i in range(n_beats * 2):
        a = int(i * BEAT / 2 * SR)
        if a >= total:
            break
        hat = rng.standard_normal(int(0.03 * SR)) * np.exp(-np.arange(int(0.03 * SR)) / (0.006 * SR))
        out[a:a + len(hat)] += 0.12 * hat[: total - a]
        if i % 2 == 0:
            beat = i // 2
            n = int(0.15 * SR)
            t = np.arange(n) / SR
            if beat % 2 == 0:
                hit = np.sin(2 * np.pi * (50 + 80 * np.exp(-t * 30)) * t) * np.exp(-t * 18)
            else:
                hit = rng.standard_normal(n) * np.exp(-t * 25) * 0.6
            out[a:a + n] += 0.5 * hit[: total - a]
    return out


def main(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    notes, t = [], LEAD_IN
    for p, beats in MELODY:
        dur = beats * BEAT
        if p is not None:
            notes.append((t, t + dur, p))
        t += dur
    total_beats = int(round((t - LEAD_IN) / BEAT)) + 2
    total = int((t + 2.0) * SR)

    pm = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    inst = pretty_midi.Instrument(0)
    for on, off, p in notes:
        inst.notes.append(pretty_midi.Note(90, p, on, off))
    pm.instruments.append(inst)
    pm.write(str(out_dir / "reference.mid"))

    chords = np.zeros(total)
    for bar, name in enumerate(CHORDS):
        on = LEAD_IN + bar * 4 * BEAT
        for half in range(2):
            for p in CHORD_NOTES[name]:
                a = int((on + half * 2 * BEAT) * SR)
                tone = piano_tone(p, 1.8 * BEAT, 0.35)
                chords[a:a + len(tone)] += tone[: max(0, total - a)]

    mel_piano = np.zeros(total)
    for on, off, p in notes:
        a = int(on * SR)
        tone = piano_tone(p, (off - on) * 0.92)
        mel_piano[a:a + len(tone)] += tone[: max(0, total - a)]

    piano = mel_piano + chords
    sf.write(str(out_dir / "synth_piano.wav"), 0.8 * piano / np.abs(piano).max(), SR)

    song = voice(notes, total) + 0.6 * chords + drums(total, total_beats + 2)
    sf.write(str(out_dir / "synth_song.wav"), 0.8 * song / np.abs(song).max(), SR)
    print(f"wrote synth_piano.wav, synth_song.wav, reference.mid to {out_dir}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "work/synthetic"))
