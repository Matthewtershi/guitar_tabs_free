"""Check an arrangement against the recording it came from.

For each half-bar it compares:
  - bass notes in the tab   vs  the strongest pitch class in the separated bass stem
  - chord used for fills    vs  the full mix's pitch content (chroma)
  - melody notes            vs  the pitch classes actually sounding in the full mix
and lists clashes (bass or chord note a semitone/tritone from the melody) and
melody notes that leap far away from both neighbours.

Usage: python tools/diagnose.py <tabs/song folder> <work/song folder> [level]
"""
import json
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.harmony import QUALITIES, _CANDIDATES, _TEMPLATES  # noqa: E402
from engine.model import BeatGrid  # noqa: E402

TPB = 24
NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def chroma_per_segment(y, sr, grid, segs, fmin=None, n_oct=7):
    hop = 512
    if fmin is None:
        c = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    else:
        cq = np.abs(librosa.cqt(y, sr=sr, hop_length=hop, fmin=fmin, n_bins=12 * n_oct, bins_per_octave=12))
        start_pc = int(round(librosa.hz_to_midi(fmin))) % 12
        c = np.zeros((12, cq.shape[1]))
        for k in range(cq.shape[0]):
            c[(start_pc + k) % 12] += cq[k]
    t = librosa.frames_to_time(np.arange(c.shape[1]), sr=sr, hop_length=hop)
    out = []
    for s, e in segs:
        ts, te = grid.beat_to_time([s / TPB, e / TPB])
        m = (t >= ts) & (t < te)
        v = c[:, m].mean(axis=1) if m.any() else np.zeros(12)
        out.append(v / (v.sum() + 1e-9))
    return np.array(out)


def best_chord(v):
    u = v / (np.linalg.norm(v) + 1e-9)
    i = int((_TEMPLATES @ u).argmax())
    r, q = _CANDIDATES[i]
    return r, q


def main():
    tabs, work = Path(sys.argv[1]), Path(sys.argv[2])
    level = sys.argv[3] if len(sys.argv) > 3 else "medium"
    song = json.loads((tabs / "song.json").read_text())
    grid = BeatGrid.from_dict(song["grid"])
    # the CLI aligns bar 1 at run time; recover the same alignment from the clean melody
    master0 = song["master_melody"][0]["onset"]
    from engine.beats import align_bar_one, fix_tempo_octave
    from engine.model import Note
    fix_tempo_octave(grid, [Note.from_dict(n) for n in song["master_melody"]], "auto") if False else None
    align_bar_one(grid, master0)
    arr = song["levels"][level]
    notes = arr["notes"]
    end = arr["end_tick"]
    seg_len = 2 * TPB
    segs = [(s, min(s + seg_len, end)) for s in range(0, end, seg_len)]
    sr = 22050
    stems = sorted((work / "stems").glob("*/*/"))
    stem_dir = next(d for d in stems if (d / "bass.wav").exists())
    mix, _ = librosa.load(str(work / "source.wav"), sr=sr, mono=True)
    drums, _ = librosa.load(str(stem_dir / "drums.wav"), sr=sr, mono=True)
    bass, _ = librosa.load(str(stem_dir / "bass.wav"), sr=sr, mono=True)
    n = min(len(mix), len(drums))
    nodrums = mix[:n] - drums[:n]
    mix_c = chroma_per_segment(nodrums, sr, grid, segs)
    from engine.harmony import bass_line
    real_bass = bass_line(stem_dir / "bass.wav", grid, segs, stem_dir / "bass_pyin.npz")

    chords = song["chords"]

    def chord_at(t):
        for c in chords:
            if c["start"] <= t < c["end"]:
                return c
        return chords[-1]

    rows, bass_ok, bass_n, chord_ok, mel_ok, mel_n, clashes, leaps = [], 0, 0, 0, 0, 0, [], []
    mel = sorted([x for x in notes if x["voice"] == "melody"], key=lambda x: x["start"])
    chord_has_bass, chord_has_n = 0, 0
    for i, (s, e) in enumerate(segs):
        top_bass = [int(real_bass[i])] if real_bass[i] >= 0 else []
        tab_bass = [x for x in notes if x["voice"] == "bass" and s <= x["start"] < e]
        if top_bass:
            for b in tab_bass:
                bass_n += 1
                bass_ok += int(b["pitch"] % 12 in top_bass)
        mr, mq = best_chord(mix_c[i])
        used = chord_at(s)
        same = used["root"] == mr
        chord_ok += int(same)
        if top_bass:
            chord_has_n += 1
            tones = {(used["root"] + iv) % 12 for iv in QUALITIES.get(used["quality"], [0, 4, 7])}
            chord_has_bass += int(top_bass[0] in tones)
        for m in [x for x in mel if s <= x["start"] < e]:
            mel_n += 1
            strong = list(np.argsort(mix_c[i])[::-1][:5])
            mel_ok += int(m["pitch"] % 12 in strong)
        rows.append((s // (4 * TPB) + 1, (s % (4 * TPB)) // TPB + 1, NAMES[used["root"]] + used["quality"],
                     NAMES[mr] + mq, [NAMES[x["pitch"] % 12] for x in tab_bass], [NAMES[p] for p in top_bass]))
    # (rows: bar, beat, chord used, mix-only guess, tab bass, real bass)
    # clashes: accompaniment note a minor 2nd / major 7th / tritone against the melody sounding then
    for a in [x for x in notes if x["voice"] != "melody"]:
        for m in mel:
            if m["start"] <= a["start"] < m["start"] + m["dur"]:
                iv = (m["pitch"] - a["pitch"]) % 12
                if iv in (1, 11, 6):
                    clashes.append((a["start"] // (4 * TPB) + 1, a["voice"], NAMES[a["pitch"] % 12],
                                    NAMES[m["pitch"] % 12]))
    for k in range(1, len(mel) - 1):
        d1 = abs(mel[k]["pitch"] - mel[k - 1]["pitch"])
        d2 = abs(mel[k]["pitch"] - mel[k + 1]["pitch"])
        if d1 >= 9 and d2 >= 9:
            leaps.append((mel[k]["start"] // (4 * TPB) + 1, mel[k - 1]["pitch"], mel[k]["pitch"], mel[k + 1]["pitch"]))

    print(f"level {level}: {len(segs)} half-bars")
    print(f"tab bass notes = the real bass note (pitch-tracked):   {bass_ok}/{bass_n} = {bass_ok / max(1, bass_n):.0%}")
    print(f"chords containing the real bass note:                   {chord_has_bass}/{chord_has_n} = "
          f"{chord_has_bass / max(1, chord_has_n):.0%}")
    print(f"chord root matching the full mix (no drums):            {chord_ok}/{len(segs)} = {chord_ok / len(segs):.0%}")
    print(f"melody notes among the mix's 5 strongest pitch classes: {mel_ok}/{mel_n} = {mel_ok / max(1, mel_n):.0%}")
    print(f"clashing accompaniment notes (b2 / maj7 / tritone vs melody): {len(clashes)}")
    print(f"melody notes leaping 9+ semitones away from both neighbours: {len(leaps)}  {leaps[:8]}")
    print("\nbar beat | chord used | mix says | tab bass | bass stem top-2")
    for r in rows[:40]:
        print(f"{r[0]:>3} {r[1]:>4} | {r[2]:<10} | {r[3]:<8} | {','.join(r[4]):<8} | {','.join(r[5])}")


if __name__ == "__main__":
    main()
