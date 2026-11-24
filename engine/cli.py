"""YouTube link or audio file -> Easy / Medium / Hard fingerstyle tabs (PDF + audio).

Run from the guitar_tabs folder inside the `guitartabs` env, e.g.:
  python -m engine.cli "https://www.youtube.com/watch?v=..." --source full --stem vocals
  python -m engine.cli "C:/music/song.mp3" --source piano --start 0:12 --end 2:40
  python -m engine.cli song.m4a --source guitar --tuning dropd

Finished tabs go to tabs/<song title>/. Slow stages (download, separation,
transcription, beats) are cached in work/<song>/, so re-running is quick.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from . import arrange, beats, harmony, ingest, melody, render, separate, transcribe
from .config import LEVELS, STEMS, TICKS_PER_BEAT
from .model import BeatGrid, Note, PolyNote

ROOT = Path(__file__).resolve().parent.parent


@contextmanager
def timed(timings: dict, name: str):
    t0 = time.perf_counter()
    print(f"[{name}] ...", flush=True)
    yield
    timings[name] = round(time.perf_counter() - t0, 1)
    print(f"[{name}] done in {timings[name]} s", flush=True)


def on_gpu_or_cpu(fn, device: str, *args):
    """Run fn(*args, device); if the GPU is out of memory (e.g. Ollama holds it), retry on CPU."""
    try:
        return fn(*args, device)
    except (RuntimeError, subprocess.CalledProcessError) as e:
        oom = "out of memory" in str(e).lower() or isinstance(e, subprocess.CalledProcessError)
        if device != "cuda" or not oom:
            raise
        import torch
        torch.cuda.empty_cache()
        print("  GPU ran out of memory (is another program using it, e.g. Ollama?). "
              "Retrying on CPU - this is slower.", flush=True)
        return fn(*args, "cpu")


def pick_method(source: str, stem: str | None) -> str:
    kind = stem or source
    if kind == "vocals":
        return "monophonic"
    if kind == "piano":
        return "piano"
    return "poly"


def run_cover(args, wav, play_wav, title, work, duration, device, tuning, tuning_label, timings) -> int:
    """Exact transcription of a guitar cover / tutorial: every note the guitarist plays."""
    import numpy as np

    from . import cover
    with timed(timings, "transcribe"):
        notes = cover.transcribe_guitar(play_wav)
    if not notes:
        print("No guitar notes found.")
        return 1
    trim_tag = json.loads((work / "trim.json").read_text())
    beat_cache = work / "beats.json"
    if beat_cache.exists() and not args.retranscribe and json.loads(beat_cache.read_text())["trim"] == trim_tag:
        grid = BeatGrid.from_dict(json.loads(beat_cache.read_text())["grid"])
    else:
        with timed(timings, "beats"):
            grid = on_gpu_or_cpu(lambda w, d, dev: beats.track_beats(w, dev, d), device, wav, duration)
        beat_cache.write_text(json.dumps({"trim": trim_tag, "grid": grid.to_dict()}))
    beats.fix_tempo_octave(grid, cover.first_onset_notes(notes), args.beat_scale)
    glitches = beats.regularize(grid)
    beats.align_bar_one(grid, notes[0].onset)
    with timed(timings, "tab"):
        arr, notes, info = cover.transcribe_cover(play_wav, grid, tuning, tuning_label, args.capo)
    info["beat_glitches_repaired"] = glitches
    hist = np.zeros(12)
    for n in notes:
        hist[n.pitch % 12] += n.offset - n.onset
    key = harmony.estimate_key(hist[None, :], np.array([-1]), [])
    out_dir = Path(args.out) / safe_name(title)
    slug = safe_name(title).replace(" ", "_")[:40]
    with timed(timings, "render"):
        written = render.write_outputs(out_dir, slug, title, {"exact": arr}, grid, [], [], [], key, info,
                                       timings, wav, args.sound, kind="exact transcription")
    print()
    print(title)
    print(f"  key {render.NOTE_NAMES[key[0]]} {key[1]} | ~{grid.bpm:.0f} BPM | {grid.beats_per_bar}/4 | capo {arr.capo}")
    print(f"  {info}")
    print(f"  timings (s): {timings}")
    print(f"  outputs in {out_dir}:")
    for p in written:
        print(f"    {p.name}")
    return 0


def safe_name(s: str) -> str:
    s = re.sub(r"[^\w\s\-]+", "", s).strip()
    return re.sub(r"\s+", " ", s)[:80] or "song"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="YouTube link, or an mp3/wav/m4a/flac file")
    ap.add_argument("--source", required=True, choices=["full", "piano", "guitar"],
                    help="full = original song with band; piano / guitar = tutorial or cover")
    ap.add_argument("--stem", choices=[s for s in STEMS if s != "drums"],
                    help="which separated part carries the melody (required for --source full)")
    ap.add_argument("--tuning", default="standard",
                    help="standard | dropd | custom list low->high, e.g. D2,G2,D3,G3,B3,D4")
    ap.add_argument("--capo", type=int, help="force this capo for every level (default: best per level)")
    ap.add_argument("--start", help="trim: start time, e.g. 0:15")
    ap.add_argument("--end", help="trim: end time, e.g. 3:40")
    ap.add_argument("--beat-scale", default="auto", choices=["auto", "1", "2", "0.5"],
                    help="fix a tempo detected at half (2) or double (0.5) speed")
    ap.add_argument("--sound", default="steel", choices=["steel", "nylon"],
                    help="guitar sound for the audio files")
    ap.add_argument("--work", default=str(ROOT / "work"), help="cache folder")
    ap.add_argument("--out", default=str(ROOT / "tabs"), help="where finished tabs go")
    ap.add_argument("--retranscribe", action="store_true", help="redo melody picking and beats")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    args = ap.parse_args(argv)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.source == "full" and not args.stem:
        ap.error("--source full needs --stem: which part has the melody? vocals | guitar | piano | other | bass")

    tuning, tuning_label = arrange.parse_tuning(args.tuning)
    timings: dict = {}
    work = Path(args.work) / ingest.song_id(args.input)
    print(f"Working folder: {work}   device: {device}")

    with timed(timings, "ingest"):
        wav, title = ingest.ingest(args.input, work, ingest.parse_time(args.start), ingest.parse_time(args.end))
    import soundfile as sf
    duration = sf.info(str(wav)).duration

    melody_wav = wav
    if args.stem:
        with timed(timings, "separate"):
            melody_wav = on_gpu_or_cpu(separate.separate, device, wav, work / "stems", args.stem)

    if args.source == "guitar":
        return run_cover(args, wav, melody_wav, title, work, duration, device, tuning, tuning_label, timings)

    method = pick_method(args.source, args.stem)
    trim_tag = json.loads((work / "trim.json").read_text())
    cache = work / f"melody_{method}_{args.stem or 'mix'}.json"
    data = json.loads(cache.read_text()) if cache.exists() else None
    if data and not args.retranscribe and data.get("trim") == trim_tag and "accomp" in data:
        master = [Note.from_dict(d) for d in data["notes"]]
        accomp = [PolyNote(*a) for a in data["accomp"]]
        info = data["info"]
        print(f"[transcribe] using cached notes ({len(master)} melody, {len(accomp)} accompaniment)")
    else:
        with timed(timings, "transcribe"):
            if method == "monophonic":
                master, accomp, info = on_gpu_or_cpu(transcribe.transcribe_monophonic, device, melody_wav)
            elif method == "piano":
                master, accomp, info = on_gpu_or_cpu(transcribe.transcribe_piano, device, melody_wav)
            else:
                master, accomp, info = transcribe.transcribe_poly(melody_wav)
        cache.write_text(json.dumps({"trim": trim_tag, "info": info, "notes": [n.to_dict() for n in master],
                                     "accomp": [[a.onset, a.offset, a.pitch, a.velocity] for a in accomp]}))
    if not master:
        print("No melody notes found. Check the stem / source choice.")
        return 1

    beat_cache = work / "beats.json"
    if beat_cache.exists() and not args.retranscribe and json.loads(beat_cache.read_text())["trim"] == trim_tag:
        grid = BeatGrid.from_dict(json.loads(beat_cache.read_text())["grid"])
    else:
        with timed(timings, "beats"):
            grid = on_gpu_or_cpu(lambda w, d, dev: beats.track_beats(w, dev, d), device, wav, duration)
        beat_cache.write_text(json.dumps({"trim": trim_tag, "grid": grid.to_dict()}))
    beats.fix_tempo_octave(grid, master, args.beat_scale)
    info["beat_glitches_repaired"] = beats.regularize(grid)
    beats.align_bar_one(grid, master[0].onset)

    with timed(timings, "arrange"):
        if method == "monophonic":
            import librosa
            y, sr = librosa.load(str(melody_wav), sr=22050, mono=True)
            attacks = librosa.onset.onset_detect(y=y, sr=sr, hop_length=256, units="time", backtrack=True)
            info["note_starts_moved_to_attacks"] = melody.refine_onsets(master, attacks)
        clean, cinfo = melody.clean_melody(master, grid)
        info.update(cinfo)
        bar = grid.bar_ticks
        end_tick = -(-max(m.start + m.dur for m in clean) // bar) * bar
        if accomp:
            chords, key = harmony.from_notes(accomp, clean, grid, end_tick)
            info["harmony_from"] = "accompaniment notes in the recording"
        else:
            stems = sorted(p for p in melody_wav.parent.glob("*.wav") if p.stem not in (args.stem, "drums", "bass"))
            bass_wav = melody_wav.parent / "bass.wav"
            chords, key = harmony.from_audio(stems or [wav], bass_wav if bass_wav.exists() else None, clean,
                                             grid, end_tick, melody_wav.parent / "bass_pyin.npz")
            info["harmony_from"] = "separated stems: " + ", ".join(p.stem for p in stems) + (
                " + bass line" if bass_wav.exists() else "")
        levels = {k: arrange.arrange_level(clean, chords, grid.beats_per_bar, end_tick, k, lvl, tuning,
                                           tuning_label, grid.bpm, args.capo)
                  for k, lvl in LEVELS.items()}

    out_dir = Path(args.out) / safe_name(title)
    slug = safe_name(title).replace(" ", "_")[:40]
    with timed(timings, "render"):
        written = render.write_outputs(out_dir, slug, title, levels, grid, chords, clean, master, key, info,
                                       timings, wav, args.sound)

    print()
    print(title)
    print(f"  key {render.NOTE_NAMES[key[0]]} {key[1]} | ~{grid.bpm:.0f} BPM | {grid.beats_per_bar}/4 | "
          f"{len(clean)} clean melody notes (from {len(master)} detected) | {len(chords)} chord segments")
    print(f"  harmony from: {info['harmony_from']}")
    for k, a in levels.items():
        print(f"  {a.level:<6} capo {a.capo}  notes {len(a.notes):4d}  couldn't fit: {a.dropped}")
    print(f"  timings (s): {timings}")
    print(f"  outputs in {out_dir}:")
    for p in written:
        print(f"    {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
