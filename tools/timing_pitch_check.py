"""Find every melody note whose TIMING (>= 100 ms) or PITCH (>= 1 semitone) is off
compared with the singer, and say why.

Usage: python tools/timing_pitch_check.py <tabs/song folder> <work/song folder>
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.beats import align_bar_one  # noqa: E402
from engine.model import BeatGrid, Note  # noqa: E402

TPB = 24


def crepe_contour(wav, cache):
    if cache.exists():
        z = np.load(cache)
        return z["t"], z["midi"], z["ok"]
    import librosa
    import torch
    import torchcrepe
    y, _ = librosa.load(str(wav), sr=16000, mono=True)
    f0, per = torchcrepe.predict(torch.from_numpy(y)[None], 16000, 160, 65, 1400, "full",
                                 return_periodicity=True, batch_size=512,
                                 device="cuda" if torch.cuda.is_available() else "cpu")
    f0, per = f0[0].cpu().numpy(), per[0].cpu().numpy()
    t = np.arange(len(f0)) * 0.01
    midi = 69 + 12 * np.log2(np.maximum(f0, 1) / 440)
    ok = per > 0.4
    np.savez(cache, t=t, midi=midi, ok=ok)
    return t, midi, ok


def main():
    tabs, work = Path(sys.argv[1]), Path(sys.argv[2])
    s = json.loads((tabs / "song.json").read_text())
    master = [Note.from_dict(n) for n in s["master_melody"]]
    grid = BeatGrid.from_dict(s["grid"])
    align_bar_one(grid, master[0].onset)
    shift = s["info"].get("melody_octave_shift", 0)
    stem = next((work / "stems" / "htdemucs_ft").glob("*/vocals.wav"))   # the stem the pipeline used
    t, midi, ok = crepe_contour(stem, stem.parent / "crepe_check.npz")

    beats = grid.beats
    ibi = np.diff(beats)
    med = np.median(ibi[(beats[:-1] > 0) & (beats[:-1] < t[-1])])
    odd = np.where((np.abs(ibi / med - 1) > 0.3) & (beats[:-1] > 0) & (beats[:-1] < t[-1]))[0]
    print(f"beat length median {med * 1000:.0f} ms; irregular beats (>30% off): {len(odd)}  "
          f"e.g. {[round(float(ibi[i]) * 1000) for i in odd[:10]]}")

    m_on = np.array([n.onset for n in master])
    late, wrong = [], []
    for m in s["clean_melody"]:
        tt = float(grid.beat_to_time(m["start"] / TPB))
        bar = m["start"] // (grid.beats_per_bar * TPB) + 1
        same = [n for n in master if abs(n.onset - tt) < 0.3 and n.pitch % 12 == m["pitch"] % 12]
        if same:
            src = min(same, key=lambda n: abs(n.onset - tt))
            err = (tt - src.onset) * 1000
            if abs(err) >= 100:
                b = int(np.searchsorted(beats, src.onset)) - 1
                ib = ibi[b] * 1000 if 0 <= b < len(ibi) else 0
                nearby = int(np.sum(np.abs(m_on - src.onset) < 0.13))
                late.append((bar, round(err), round(ib), nearby, round(src.offset - src.onset, 2)))
        # pitch vs what was sung during the note's first 250 ms
        dur_s = float(grid.beat_to_time((m["start"] + m["dur"]) / TPB)) - tt
        win = (t >= tt) & (t < tt + min(0.25, max(0.06, dur_s))) & ok
        if win.sum() >= 3:
            sung = float(np.median(midi[win])) + shift
            d = (m["pitch"] - sung + 6) % 12 - 6
            if abs(d) >= 0.75:
                wrong.append((bar, m["pitch"], round(sung, 1), round(d, 1)))
    n = len(s["clean_melody"])
    print(f"\nnotes >= 100 ms off: {len(late)}/{n}")
    print("  (bar, error ms, beat length ms there, master notes within 130 ms, sung length s)")
    for r in late[:25]:
        print("  ", r)
    print(f"\nnotes >= 1 semitone off the singer: {len(wrong)}/{n}")
    print("  (bar, tab pitch, sung pitch, difference)")
    for r in wrong[:25]:
        print("  ", r)


if __name__ == "__main__":
    main()
