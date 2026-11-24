# Guitar Tabs: Song-to-Fingerstyle-Tab Desktop App

Turns a song into an **easy-to-read fingerstyle guitar arrangement** (melody + bass + chord notes) at three
difficulty levels, from a YouTube link or an audio file. Personal tool: one user, one Windows PC.

**North star** (the user's own words, 2026-11-24): *"Produce a tab like the one I uploaded (EasyTab's
Moonlike Smile fingerstyle tab). Simple to read, sounds great, not tracing every sound down to the
millisecond."* This is an **arrangement** tool, not an exact-transcription tool.

Everything below was decided by the user, except sections marked **Proposed** (technical choices to
check in the prototype) or listed under **Open questions**.

> Revision 2 (2026-11-24): the first version tabbed a single-note melody traced exactly from the
> recording. That's replaced by fingerstyle arrangements with simplified, clean rhythm.
>
> **Revision 3 (2026-11-24). These decisions take priority where the rest of this spec differs.**
> - **Main job: transcribe covers exactly.**
>   - Input: a guitar cover or tutorial video.
>   - Output: a tab of every note the guitarist plays (melody, bass, chords), so it sounds exactly like that performance.
>   - Model: songscription.ai, which the user called "really good". Its example is a transcription of an acoustic guitar cover.
>   - Arranging from an original song stays as a secondary mode.
> - **Product: a web app on the user's PC**, like songscription in function (own name and design, nothing copied):
>   - paste a link or upload a file
>   - notation + tab view
>   - playback with speed control
>   - editing
>   - export to PDF / MIDI / MusicXML / Guitar Pro
>
>   This replaces the pywebview desktop shell.
> - **Instruments: guitar + piano** (piano sheet music with fingering).
> - **Target:** "the guitar song to sound exactly the same if not better than the original".
>   Reference covers: youtube.com/watch?v=ZrUKg5_4B1k, youtube.com/watch?v=yUka8hbTivY.

---

## 1. Inputs

| Input | Supported |
|---|---|
| Single YouTube link | Yes. Audio is downloaded with yt-dlp for personal use |
| Audio files: mp3 / wav / m4a / flac | Yes |
| Video files, mic recording, playlists | No |

Kinds of audio the app must handle:
- **Original full song**: the user picks which separated stem carries the melody.
- **Piano tutorial.**
- **Guitar tutorial / cover.**

**Audio range:** the app detects the music automatically and skips talking and silence. The user can adjust the result.

## 2. What the tab contains

**Fingerstyle at every level:** melody on top, bass notes on the low strings, chord notes in between.

### Levels (thin to full)

| | Easy | Medium | Hard |
|---|---|---|---|
| Texture | Melody + **one bass note per bar** | Melody + bass + **chord notes on the main beats** | **Steady 8th-note pattern like the example** (melody, bass, alternating chord notes) |
| Ornaments | Dropped | Dropped | **The recording's ornaments, written as quick 16th-note triplet turns with slurs** (like the example) |
| Frets | About 0–5 (soft limit: higher only when a note needs it) | Up to about 12 | Full neck; fingering chosen for tone over ease |

Calibrated for an **intermediate** player. Notes in sequence must be reasonable to play: chord shapes within one hand span, sensible position shifts.

### Simplification (all allowed)
- **Straighten rhythm**: snap to a clean grid and ignore the performer's pushes and drags.
- **Drop decorations**: runs, grace notes, scoops and wobbles become one clean note.
- **Make repeats identical**: a returning verse or chorus uses one clean version.
- **Change key for guitar**: shapes in a guitar-friendly key, plus a **capo so it still sounds at the original pitch** (like the example's "Capo 1").

### Rhythm
- **Timing must match the recording** (user, 2026-11-24: "make sure the timing of the notes matches up directly with the notes of the video"). This takes priority over rhythmic simplicity.
- Each beat is written in 8ths unless the performance really sits on 16ths there. Ornaments are 16th-note triplets (Hard only).
- Audio renders on the video's own timeline, so it lines up with the video.
- **Hard cap:** no melody note is placed more than 80 ms from where it was sung.

### Quality bar (user: "make or break", 2026-11-24)
- **Melody pitches are never changed to fit the key or chords.** Only short octave slips are corrected.
- **Notes ring.**
  - The same note is never struck again on the very next 8th: accompaniment repeats are merged, and melody repeats within an 8th are held.
  - Chord fills cycle through different chord notes.
  - The bass is re-struck only at bar starts and chord changes.
  - Repeats a beat apart are fine, as in the example.

### Harmony source
- **From the recording when it's clear**: a piano tutorial's left hand, a cover's accompaniment, the song's bass line.
- **Otherwise from detected chords**, with a standard pattern written under the melody.

### Melody
- From a polyphonic source (piano, guitar): the **highest note wins**.
- From a full song: the stem the user picks.

### Capo and tuning
- Recommended capo **per level**.
- Guess of the original capo and tuning, shown with a confidence level (Phase 4).
- **Any tuning**, with Standard and **Drop D** as the main presets.

## 3. Song information shown
- tempo and time signature
- sections (Intro / Verse / Chorus)
- lyrics under the notes (auto-transcribed)
- key and chord names
- repeats written once with "x3"

## 4. Output and app
- **PDF, one per level**, laid out like the example:
  - title, "Capo N", tempo mark
  - standard notation (melody stems up, accompaniment stems down) above the tab, with bar numbers
  - standard notation is written in the **capo shape key**, like the example
- **Playback with an acoustic guitar sound.** Controls: volume, speed slider, loop a section.
- **Desktop app** (pywebview + alphaTab) with switchable views: tab + rhythm stems, tab + notation, plain text.
- **Editor:** the user fixes the master melody and harmony, and the levels regenerate from it.
- **Song library.**

## 5. Technical constraints
- All processing runs locally and for free on the user's PC (GTX 1660 Ti, 6 GB). Up to 15+ minutes per song is acceptable.
- An API key may be provided later if local processing is too slow.
- **Stack:** Python engine; pywebview + alphaTab for the app; **LilyPond** for PDF engraving (portable, in `%LOCALAPPDATA%\guitar_tabs\lilypond`).
- The user reads **Python only**.

## 6. Pipeline (Proposed)

| Stage | Proposed method |
|---|---|
| Ingest | yt-dlp → ffmpeg → WAV |
| Stems | Demucs: `htdemucs_ft` (vocals, bass, other, drums); `htdemucs_6s` (guitar, piano) |
| Notes | Vocals: torchcrepe contour → notes. Piano: ByteDance piano transcription. Guitar / other: Basic Pitch. Polyphonic sources split into **melody** (highest note) and **accompaniment** (everything else) |
| Beats | Beat This!, with a half/double-tempo correction from the melody |
| Clean melody | Snap to the 8th grid; keep one main note per 8th; detect ornament runs (Hard → triplet turn); move the whole melody by octave into the guitar's melody range |
| Harmony | Per half-bar: chord (major, minor or dominant 7th) and bass note. **The bass line anchors the choice**: a pitch-tracked bass stem (pYIN) for full songs, the lowest accompaniment note for piano/guitar sources. The chord must contain the bass note, should belong to the key, and should match the accompaniment's pitch content. Smoothed so chords don't flicker. Tracked bass notes outside the key are ignored as tracking slips |
| Melody check | Short melody notes outside the key are moved to the nearest note in the key (pitch-tracking slips); notes a beat or longer are kept |
| Key & capo | Key from harmony + melody. Notation in the capo shape key |
| Arrangement | Pattern per level (Easy / Medium / Hard) → voices. Fingering by dynamic programming over hand position + melody string, choosing bass and chord-note strings and frets that fit one hand shape, for every capo 0–7 → keep the cheapest per level |
| Render | LilyPond → PDF (notation + tab); MIDI; WAV with a plucked-string guitar synth; plain-text tab |

## 7. Build order
1. **Phase 0: Environment.** Done.
2. **Phase 1: Prototype CLI.** Now fingerstyle arrangements + PDF + guitar-sound playback. Evaluate on the user's test songs.
3. **Phase 2: App shell.** pywebview + alphaTab views, playback controls, library, automatic music detection.
4. **Phase 3: Editor.**
5. **Phase 4: Extras.**
   - lyrics, sections and repeats ("make repeats identical")
   - chord names on the tab
   - original capo/tuning guess

## 8. Open questions
- **Unplayable notes:** below the lowest open string or above the highest fret. The prototype moves the whole melody by octave so it fits; single stray notes are moved an octave and flagged.
- **The user's guitar:** acoustic or electric, number of frets, cutaway. The prototype assumes 20 frets.
- **Fret limits with a capo:** counted from the capo (as tab is written) or from the nut? The prototype counts from the capo.
- **Test-song references** for scoring.
- **Cloud fallback provider**, if one is needed.
