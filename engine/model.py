"""Data shapes passed between pipeline stages.

master melody (list[Note], real time)  ->  clean melody (list[MelNote], beat grid)
harmony (list[Chord])                  ->  arrangement per level (Arrangement)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .config import TICKS_PER_BEAT


@dataclass
class Note:
    """One melody note as heard in the recording (real time, unquantized)."""
    onset: float                  # seconds
    offset: float                 # seconds
    pitch: int                    # MIDI note number (sounding)
    velocity: float = 0.8
    legato: bool = False          # joined to the previous note with no new attack
    glide_in: bool = False        # pitch slid in from the previous note
    bend_from: int = 0
    vibrato: bool = False
    flags: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return Note(**d)


@dataclass
class BeatGrid:
    """Beat times that follow the performer's tempo drift."""
    beats: np.ndarray             # seconds, one per beat, extended past both ends
    bar_start_beat: int           # index into `beats` where bar 1 starts
    beats_per_bar: int
    bpm: float
    flags: list = field(default_factory=list)

    def time_to_beat(self, t):
        """Seconds -> beats counted from the start of bar 1 (float)."""
        idx = np.interp(t, self.beats, np.arange(len(self.beats)))
        return idx - self.bar_start_beat

    def beat_to_time(self, b):
        return np.interp(np.asarray(b) + self.bar_start_beat,
                         np.arange(len(self.beats)), self.beats)

    def time_to_tick(self, t):
        return self.time_to_beat(t) * TICKS_PER_BEAT

    @property
    def bar_ticks(self):
        return self.beats_per_bar * TICKS_PER_BEAT

    def to_dict(self):
        return {"beats": self.beats.tolist(), "bar_start_beat": self.bar_start_beat,
                "beats_per_bar": self.beats_per_bar, "bpm": self.bpm, "flags": self.flags}

    @staticmethod
    def from_dict(d):
        return BeatGrid(np.array(d["beats"]), d["bar_start_beat"], d["beats_per_bar"],
                        d["bpm"], d.get("flags", []))


@dataclass
class PolyNote:
    """A note from a polyphonic transcription (accompaniment)."""
    onset: float
    offset: float
    pitch: int
    velocity: float = 0.8


@dataclass
class MelNote:
    """Clean melody note on the 8th-note grid."""
    start: int                    # ticks from bar 1
    dur: int                      # ticks
    pitch: int                    # sounding MIDI pitch
    glide_in: bool = False
    turn: list = field(default_factory=list)   # Hard: 3 pitches played as a 16th triplet in place of this 8th
    src: int = -1                 # index of the master note it came from
    changed: str = ""             # why the pitch differs from what was detected (if it does)

    def to_dict(self):
        return asdict(self)


@dataclass
class Chord:
    start: int                    # ticks
    end: int
    root: int                     # pitch class 0-11
    quality: str                  # "", "m", "7", "m7", "maj7", "sus4", "dim"
    bass: int                     # pitch class of the bass note

    @property
    def name(self):
        names = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
        n = names[self.root] + self.quality
        return n if self.bass == self.root else f"{n}/{names[self.bass]}"

    def to_dict(self):
        d = asdict(self)
        d["name"] = self.name
        return d


@dataclass
class ArrNote:
    """One note of a level's arrangement."""
    start: int                    # ticks from bar 1
    dur: int                      # ticks (how long it rings)
    pitch: int                    # sounding MIDI pitch
    voice: str                    # "melody" | "bass" | "inner"
    string: int                   # 0 = lowest (thickest) string
    fret: int                     # counted from the capo
    triplet: int = 0              # 1..3 = position inside a 16th-triplet turn
    slide_in: bool = False        # slide from the previous melody note
    octave_moved: int = 0

    def to_dict(self):
        return asdict(self)


@dataclass
class Arrangement:
    level: str
    capo: int
    tuning: list                  # open-string MIDI pitches (no capo), low to high
    tuning_name: str
    notes: list                   # list[ArrNote]
    end_tick: int
    dropped: dict = field(default_factory=dict)   # what couldn't be fingered
    cost: float = 0.0
    capo_costs: dict = field(default_factory=dict)

    def to_dict(self):
        d = asdict(self)
        d["notes"] = [n.to_dict() for n in self.notes]
        return d
