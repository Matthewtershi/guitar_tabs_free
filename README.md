# Guitar Tabs

Turns a song (YouTube link or audio file) into **easy-to-read fingerstyle guitar tabs** at three
levels, Easy, Medium and Hard. Each is a PDF with notation + tab in the style of a typical
fingerstyle tab, plus a guitar-sounding audio file to hear it. The design is in [SPEC.md](SPEC.md).

## Use

Open a terminal in this folder, then run `conda activate guitartabs`.

**Main mode: exact transcription of a guitar cover or tutorial.** Every note the guitarist plays goes into the tab:

```powershell
python -m engine.cli "https://www.youtube.com/watch?v=yUka8hbTivY" --source guitar
python tools/sound_check.py "tabs/<song>" work/yt_<id>     # how close the tab is to the performance
```

**Secondary mode: arranging from an original song** (fingerstyle Easy / Medium / Hard):

```powershell
# Original song: say which separated part has the melody
python -m engine.cli "https://www.youtube.com/watch?v=VIDEO" --source full --stem vocals

# Piano tutorial (harmony comes from the left hand)
python -m engine.cli "https://youtu.be/VIDEO" --source piano --start 0:20

# Guitar cover in Drop D
python -m engine.cli "C:\music\cover.mp3" --source guitar --tuning dropd
```

| Option | Meaning |
|---|---|
| `--source full\|piano\|guitar` | Original song with a band, or a piano/guitar tutorial or cover |
| `--stem vocals\|guitar\|piano\|other\|bass` | Which separated part carries the melody (required for `full`) |
| `--tuning` | `standard`, `dropd`, or a custom list, lowest string first |
| `--capo N` | Force one capo for every level (default: best capo per level) |
| `--start` / `--end` | Trim the audio, e.g. `0:15` |
| `--beat-scale 2\|0.5` | Fix a tempo detected at half or double speed |
| `--retranscribe` | Redo melody picking and beats |

## Output (in `tabs/<song title>/`)

| File | What it is |
|---|---|
| `*_easy.pdf`, `*_medium.pdf`, `*_hard.pdf` | **The tabs**: title, capo, tempo; notation (melody stems up, bass and chord notes stems down) above the tab |
| `*.mp3` | The arrangement played with recorded acoustic-guitar samples, **on the video's own timeline**: it starts when the video starts and follows its tempo. `--sound nylon` switches from steel-string to nylon-string. The GeneralUser GS SoundFont is downloaded automatically to `%LOCALAPPDATA%\guitar_tabs\soundfonts` |
| `*_with_song.mp3` | The same guitar mixed over the original recording, to hear that every note lines up |
| `*.musicxml` | Opens in MuseScore, Guitar Pro or TuxGuitar for editing |
| `*.mid`, `*.txt` | MIDI and a plain-text tab |
| `song.json` | Everything: chords, clean melody, all three arrangements |

Levels:
- **Easy**: melody plus one bass note per bar, open position.
- **Medium**: adds bass on the strong beats and chord notes on free beats.
- **Hard**: steady 8th-note pattern, plus the recording's ornaments as triplet turns.

The slow stages (download, stem separation, transcription, beats) are cached in `work/<song>/`.

## How it works

1. Separate stems.
2. Detect the melody notes.
3. **Clean the melody**: 8th-note grid, decorations dropped, moved into the guitar's range.
4. **Detect chords and bass** per half-bar, from the accompaniment in the recording.
5. **Arrange**: build the level's pattern, then finger melody, bass and chord notes together so each moment is one playable hand shape. Every capo is tried and the cheapest kept.
6. Engrave with Verovio.

The settings to tune are in [engine/config.py](engine/config.py), [engine/melody.py](engine/melody.py) and [engine/harmony.py](engine/harmony.py).

To check a song's quality:

```powershell
python -m tools.quality_report "tabs/<song>" work/<song id>
```

It reports:
- **timing:** every tab melody note vs when it was sung
- **pitch:** every tab melody note vs the singer's pitch
- **re-strikes:** the same note struck again on the next 8th

To check a result against its recording:

```powershell
python tools/diagnose.py "tabs/<song>" work/<song id> medium
```

It scores:
- the tab's bass notes against the pitch-tracked bass line
- the chords against the full mix
- notes that clash with the melody

This matters because wrong chords make the bass and chord notes sound out of place together.

## Setup (already done on this PC)

```powershell
conda create -n guitartabs python=3.11
conda install -n guitartabs -c conda-forge ffmpeg
conda activate guitartabs
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install "setuptools<81" torch==2.5.1 torchaudio==2.5.1 demucs==4.0.1 "yt-dlp[default]" torchcrepe librosa soundfile pretty_midi mido basic-pitch piano_transcription_inference pywebview verovio svglib reportlab pymupdf "https://github.com/CPJKU/beat_this/archive/main.zip" --extra-index-url https://download.pytorch.org/whl/cu124
```

YouTube downloads use Node.js (already installed) as yt-dlp's JavaScript runtime.
If Ollama is holding the GPU, the engine falls back to the CPU, which is slower.
