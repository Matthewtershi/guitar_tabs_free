"""Score a transcription against a known-correct melody (both as MIDI).

Reports note precision / recall / F1: a detected note counts as correct when
its pitch matches (within 50 cents) and it starts within 50 ms (or the given
tolerance) of a reference note. Note lengths are ignored.

Usage: python tools/score.py reference.mid estimate.mid [--tolerance 0.05]
"""
import argparse

import mir_eval
import numpy as np
import pretty_midi


def load(path):
    pm = pretty_midi.PrettyMIDI(path)
    notes = sorted((n.start, n.end, n.pitch) for inst in pm.instruments for n in inst.notes)
    iv = np.array([[s, max(e, s + 0.01)] for s, e, _ in notes])
    hz = np.array([440.0 * 2 ** ((p - 69) / 12) for _, _, p in notes])
    return iv, hz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("estimate")
    ap.add_argument("--tolerance", type=float, default=0.05)
    a = ap.parse_args()
    ri, rp = load(a.reference)
    ei, ep = load(a.estimate)
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ri, rp, ei, ep, onset_tolerance=a.tolerance, offset_ratio=None)
    print(f"reference notes {len(rp)}  estimated notes {len(ep)}")
    print(f"precision {p:.2f}  recall {r:.2f}  F1 {f:.2f}  (onset tolerance {a.tolerance * 1000:.0f} ms)")
    # Pitch-class only (ignores octave mistakes) to separate octave errors from wrong notes
    p2, r2, f2, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ri, 440.0 * 2 ** ((np.round(12 * np.log2(rp / 440)) % 12) / 12),
        ei, 440.0 * 2 ** ((np.round(12 * np.log2(ep / 440)) % 12) / 12),
        onset_tolerance=a.tolerance, offset_ratio=None)
    print(f"ignoring octave: precision {p2:.2f}  recall {r2:.2f}  F1 {f2:.2f}")


if __name__ == "__main__":
    main()
