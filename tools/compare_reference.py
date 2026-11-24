"""Compare a human fingerstyle cover (reference) with the original song and with our tab.

  1. transcribe the cover (Basic Pitch; it's solo guitar so this is reliable)
  2. align cover <-> original song (chroma DTW, trying all 12 transpositions)
  3. report how the cover treats the song (follows the vocal? plays what in the
     instrumental parts? register, density, bass) and how our arrangement compares.

Usage: python tools/compare_reference.py <work/ref id> <work/original id> <tabs/original folder> [level]
"""
import json
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.config import VOCAL_MODEL  # noqa: E402
from engine.model import BeatGrid  # noqa: E402

SR, HOP = 22050, 2048
NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def transcribe(ref_dir: Path):
    cache = ref_dir / "ref_notes.json"
    if cache.exists():
        return json.loads(cache.read_text())
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict
    _, _, ev = predict(str(ref_dir / "source.wav"), ICASSP_2022_MODEL_PATH, onset_threshold=0.5,
                       frame_threshold=0.3, minimum_note_length=60)
    notes = sorted([[float(a), float(b), int(p), float(v)] for a, b, p, v, _ in ev])
    cache.write_text(json.dumps(notes))
    return notes


def chroma(path, sr=SR):
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    c = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    return c / (np.linalg.norm(c, axis=0, keepdims=True) + 1e-9)


def stats(notes_sec, beat_sec, label):
    on = np.array([n[0] for n in notes_sec])
    p = np.array([n[2] for n in notes_sec])
    span = on.max() - on.min()
    groups = np.round(on / 0.03)
    simult = np.bincount(np.unique(groups, return_inverse=True)[1]).mean()
    # melody = highest note of each onset group, bass = lowest
    tops, lows = {}, {}
    for o, q in zip(groups, p):
        tops[o] = max(tops.get(o, 0), q)
        lows[o] = min(lows.get(o, 999), q)
    top = np.array(list(tops.values()))
    low = np.array(list(lows.values()))
    def nm(m):
        m = int(m)
        return f"{NAMES[m % 12]}{m // 12 - 1}"
    print(f"  {label:<26} notes/beat {len(on) / (span / beat_sec):4.1f} | notes per strike {simult:3.1f} | "
          f"top note median {nm(np.median(top))} range {nm(np.percentile(top, 5))}-{nm(np.percentile(top, 95))} | "
          f"bass median {nm(np.median(low))}")


def main():
    ref, orig, tabs = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    level = sys.argv[4] if len(sys.argv) > 4 else "hard"
    notes = transcribe(ref)
    song = json.loads((tabs / "song.json").read_text())
    grid = BeatGrid.from_dict(song["grid"])
    beat_sec = 60 / grid.bpm

    # --- alignment cover -> original (try all transpositions)
    cr = chroma(ref / "source.wav")
    stem_dir = next((orig / "stems" / VOCAL_MODEL).glob("*/"))
    y_mix, _ = librosa.load(str(orig / "source.wav"), sr=SR, mono=True)
    y_dr, _ = librosa.load(str(stem_dir / "drums.wav"), sr=SR, mono=True)
    n = min(len(y_mix), len(y_dr))
    co = librosa.feature.chroma_cqt(y=y_mix[:n] - y_dr[:n], sr=SR, hop_length=HOP)
    co = co / (np.linalg.norm(co, axis=0, keepdims=True) + 1e-9)
    best = None
    for k in range(12):
        D, wp = librosa.sequence.dtw(X=np.roll(cr, k, axis=0), Y=co, metric="cosine", subseq=True)
        cost = D[-1, wp[0, 1]] / len(wp)
        if best is None or cost < best[0]:
            best = (cost, k, wp[::-1])
    cost, shift, wp = best
    print(f"cover is transposed by {shift:+d} semitones vs the original (alignment cost {cost:.3f}; lower = better)")
    t_ref = wp[:, 0] * HOP / SR
    t_org = wp[:, 1] * HOP / SR

    def to_orig(t):
        return float(np.interp(t, t_ref, t_org))

    # --- the original singer's pitch over time
    z = np.load(stem_dir / "crepe_check.npz") if (stem_dir / "crepe_check.npz").exists() else None
    # --- how the cover treats the song: melody notes (top of each strike) vs the vocal
    groups = {}
    for a, b, p, v in notes:
        g = round(a / 0.03)
        if g not in groups or p > groups[g][2]:
            groups[g] = (a, b, p)
    top = sorted(groups.values())
    vocal_follow, inst, inst_n, sung_n = 0, 0, 0, 0
    ours_mel = [n_ for n_ in song["levels"][level]["notes"] if n_["voice"] == "melody"]
    ours_t = np.array([float(grid.beat_to_time(n_["start"] / 24)) for n_ in ours_mel])
    ours_p = np.array([n_["pitch"] for n_ in ours_mel])
    ours_match, ours_n = 0, 0
    for a, b, p in top:
        to = to_orig(a)
        pc_cover = (p + shift) % 12
        if z is not None:
            w = (z["t"] >= to - 0.05) & (z["t"] < to + 0.15) & z["ok"]
            if w.sum() >= 3:
                sung_n += 1
                vocal_follow += int(round(float(np.median(z["midi"][w]))) % 12 == pc_cover)
            else:
                inst_n += 1
        # our melody note sounding at that moment of the original
        i = int(np.searchsorted(ours_t, to + 0.08)) - 1
        if 0 <= i < len(ours_t) and to - ours_t[i] < 1.0:
            ours_n += 1
            ours_match += int(ours_p[i] % 12 == pc_cover)
    # --- the fair test: for each note the singer sings, is it played at that moment?
    master = song["master_melody"]
    org_to_ref = lambda t: float(np.interp(t, t_org, t_ref))  # noqa: E731
    cov_on = np.array([n_[0] for n_ in notes])
    cov_p = np.array([n_[2] for n_ in notes])
    covered = [t_org.min() + 1, t_org.max() - 1]
    has_cover, has_ours, total, octs = 0, 0, 0, []
    all_ours = song["levels"][level]["notes"]
    all_t = np.array([float(grid.beat_to_time(x["start"] / 24)) for x in all_ours])
    all_p = np.array([x["pitch"] for x in all_ours])
    for mn in master:
        if not (covered[0] <= mn["onset"] <= covered[1]) or mn["offset"] - mn["onset"] < 0.12:
            continue
        total += 1
        tr = org_to_ref(mn["onset"])
        w = (cov_on >= tr - 0.12) & (cov_on <= tr + 0.18) & ((cov_p + shift) % 12 == mn["pitch"] % 12)
        if w.any():
            has_cover += 1
            octs.append(int(np.round((cov_p[w].max() + shift - mn["pitch"]) / 12)))
        w2 = (all_t >= mn["onset"] - 0.12) & (all_t <= mn["onset"] + 0.18) & (all_p % 12 == mn["pitch"] % 12)
        has_ours += int(w2.any())
    from collections import Counter
    print(f"SUNG NOTES (>=120 ms) that are played at that moment:  cover {has_cover}/{total} = "
          f"{has_cover / max(1, total):.0%}   ours ({level}) {has_ours}/{total} = {has_ours / max(1, total):.0%}")
    print(f"  octave the cover plays the melody in, vs the singer: {dict(Counter(octs).most_common())}  "
          f"(ours: {song['info'].get('melody_octave_shift', 0) // 12:+d})")
    print(f"cover melody notes during singing that match the sung note: {vocal_follow}/{sung_n} = "
          f"{vocal_follow / max(1, sung_n):.0%}")
    print(f"cover melody notes where nobody sings (instrumental hooks): {inst_n} of {len(top)} "
          f"({inst_n / max(1, len(top)):.0%})")
    print(f"our {level} melody agrees with the cover's melody note: {ours_match}/{ours_n} = "
          f"{ours_match / max(1, ours_n):.0%}")

    # --- style numbers
    print("\nstyle:")
    stats(notes, beat_sec, "cover")
    for lv in ("easy", "medium", "hard"):
        arr = song["levels"][lv]["notes"]
        ns = [[float(grid.beat_to_time(x["start"] / 24)), 0, x["pitch"], 1] for x in arr]
        stats(ns, beat_sec, f"ours {lv} (capo {song['levels'][lv]['capo']})")
    on = np.array([n_[0] for n_ in notes])
    ib = np.diff(np.unique(np.round(on / 0.03))) * 0.03
    print(f"  cover: share of strikes a 16th or less apart: {np.mean(ib < beat_sec / 4 + 0.02):.0%}, "
          f"an 8th apart: {np.mean((ib >= beat_sec / 4 + 0.02) & (ib < beat_sec / 2 + 0.04)):.0%}")


if __name__ == "__main__":
    main()
