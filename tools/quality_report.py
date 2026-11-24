"""Quality report for a finished song: timing, pitch, and re-struck notes.

  timing  : each tab melody note vs the moment the singer started that same note
  pitch   : each tab melody note vs the singer's pitch during that note
  restrike: a note struck again on the same string while the same pitch is still
            ringing there (the "same note back to back" problem), per level and voice

Usage: python tools/quality_report.py <tabs/song folder> <work/song folder>
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.config import VOCAL_MODEL  # noqa: E402
from engine.model import BeatGrid, Note  # noqa: E402
from tools.timing_pitch_check import crepe_contour  # noqa: E402

TPB = 24


def main():
    tabs, work = Path(sys.argv[1]), Path(sys.argv[2])
    s = json.loads((tabs / "song.json").read_text())
    master = [Note.from_dict(n) for n in s["master_melody"]]
    grid = BeatGrid.from_dict(s["grid"])
    shift = s["info"].get("melody_octave_shift", 0)
    stem = next((work / "stems" / VOCAL_MODEL).glob("*/vocals.wav"))   # the stem the pipeline used
    t, midi, ok = crepe_contour(stem, stem.parent / "crepe_check.npz")

    errs, wrong_pc, octave = [], [], 0
    for m in s["clean_melody"]:
        src = master[m["src"]]
        errs.append(abs(float(grid.beat_to_time(m["start"] / TPB)) - src.onset) * 1000)
        win = (t >= src.onset) & (t < src.offset) & ok
        if win.sum() >= 3:
            d = m["pitch"] - (float(np.median(midi[win])) + shift)
            pc_d = (d + 6) % 12 - 6
            if abs(pc_d) >= 1.0:
                wrong_pc.append((m["start"] // (grid.beats_per_bar * TPB) + 1, round(pc_d, 1)))
            elif abs(d) >= 6:
                octave += 1
    errs = np.array(errs)
    n = len(errs)
    print(f"TIMING  ({n} melody notes vs when they were sung): median {np.median(errs):.0f} ms, "
          f"95% within {np.percentile(errs, 95):.0f} ms, worst {errs.max():.0f} ms, >=100 ms: {int((errs >= 100).sum())}")
    print(f"PITCH   wrong note (>=1 semitone, ignoring octave): {len(wrong_pc)}  {wrong_pc[:10]}")
    print(f"        moved an octave (smoothing): {octave}")
    print(f"        sung notes left out (decorations / two on one spot): "
          f"{len(master) - n} of {len(master)}")

    bpm_sec = lambda tick: float(grid.beat_to_time(tick / TPB))  # noqa: E731
    for lv, arr in s["levels"].items():
        by_string = {}
        for x in sorted(arr["notes"], key=lambda x: x["start"]):
            by_string.setdefault(x["string"], []).append(x)
        c = Counter()
        for ns in by_string.values():
            for a, b in zip(ns, ns[1:]):
                if a["pitch"] == b["pitch"] and bpm_sec(b["start"]) - bpm_sec(a["start"]) < 0.6:
                    c[b["voice"]] += 1
        total = len(arr["notes"])
        print(f"RESTRIKE {lv:<6}: same note struck again on the same string within 0.6 s: "
              f"{sum(c.values())} of {total} notes  {dict(c)}")


if __name__ == "__main__":
    main()
