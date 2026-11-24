"""How close does our exact tab come to the guitarist's actual performance?

  tab vs performance  : notes in the tab (at their real video times) vs the notes heard
                        in the cover; correct = same pitch, starting within 50 ms
  audio vs performance: our rendered guitar audio, transcribed the same way as the
                        cover, vs the cover (what you would hear side by side)

Usage: python tools/sound_check.py <tabs/song folder> <work/song folder>
"""
import json
import sys
from pathlib import Path

import mir_eval
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.model import BeatGrid  # noqa: E402


def f1(ref, est, tol=0.05, pc=False):
    ri = np.array([[a, max(b, a + 0.02)] for a, b, _ in ref])
    ei = np.array([[a, max(b, a + 0.02)] for a, b, _ in est])
    rp = np.array([p for *_, p in ref], dtype=float)
    ep = np.array([p for *_, p in est], dtype=float)
    if pc:
        rp, ep = 60 + rp % 12, 60 + ep % 12
    hz = lambda m: 440.0 * 2 ** ((m - 69) / 12)  # noqa: E731
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(ri, hz(rp), ei, hz(ep),
                                                                    onset_tolerance=tol, offset_ratio=None)
    return p, r, f


def main():
    tabs, work = Path(sys.argv[1]), Path(sys.argv[2])
    song = json.loads((tabs / "song.json").read_text())
    grid = BeatGrid.from_dict(song["grid"])
    heard = [tuple(n[:2]) + (n[2],) for n in json.loads((work / "source_guitar_notes.json").read_text())]
    lv = next(iter(song["levels"]))
    tab = [(float(grid.beat_to_time(n["start"] / 24)), float(grid.beat_to_time((n["start"] + n["dur"]) / 24)),
            n["pitch"]) for n in song["levels"][lv]["notes"]]
    p, r, f = f1(heard, tab)
    print(f"TAB vs PERFORMANCE   notes in the tab that the guitarist played: {p:.0%} | "
          f"guitarist's notes found in the tab: {r:.0%} | F1 {f:.0%}")
    _, _, f2 = f1(heard, tab, tol=0.1)
    print(f"                     (with 100 ms tolerance: F1 {f2:.0%})")

    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict
    mp3 = next(p for p in tabs.glob(f"*_{lv}.mp3"))
    _, _, ev = predict(str(mp3), ICASSP_2022_MODEL_PATH, onset_threshold=0.5, frame_threshold=0.3,
                       minimum_note_length=50, melodia_trick=False)
    ours = [(float(a), float(b), int(pp)) for a, b, pp, _, _ in ev]
    p, r, f = f1(heard, ours)
    print(f"AUDIO vs PERFORMANCE our rendered audio vs the cover (both transcribed): "
          f"precision {p:.0%} recall {r:.0%} F1 {f:.0%}")


if __name__ == "__main__":
    main()
