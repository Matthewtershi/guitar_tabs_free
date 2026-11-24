"""Every tunable setting in one place.

Values marked PLACEHOLDER are open questions in SPEC.md section 8: they are
temporary, flagged, and meant to be replaced once you decide.
"""
from dataclasses import dataclass

# Rhythm resolution: 24 ticks per beat. 8th = 12, 16th-triplet = 4.
TICKS_PER_BEAT = 24
EIGHTH = 12

# Open-string pitches, lowest (thickest) string first.
TUNINGS = {
    "standard": ["E2", "A2", "D3", "G3", "B3", "E4"],
    "dropd": ["D2", "A2", "D3", "G3", "B3", "E4"],
}
TUNING_LABELS = {"standard": "Standard", "dropd": "Drop D"}

# PLACEHOLDER (SPEC open question "your guitar"): frets on your guitar.
GUITAR_FRETS = 20
# Fretting-hand window: index finger at fret h reaches h..h+3.
HAND_SPAN = 3
# PLACEHOLDER (SPEC open question): fret limits are counted from the capo.

# Melody placement: the whole melody is moved by octaves so its middle note
# sits near this sounding pitch (G4), leaving room for bass and chord notes below.
MELODY_TARGET_PITCH = 67


@dataclass(frozen=True)
class LevelSettings:
    name: str
    pattern: str                 # "easy" | "medium" | "hard" (see arrange.make_slices)
    soft_max_fret: int           # frets above this only when a note can't be played lower
    ornaments: bool              # write the recording's ornaments as 16th-triplet turns
    slides: bool                 # write slides where the melody glides between notes
    shift_weight: float          # cost of moving the hand along the neck
    height_weight: float         # cost per fret of playing higher up the neck
    thin_string_cost: float = 0.0   # Hard "tone over ease": prefer the melody on thicker strings
    bass_drop_cost: float = 3.0     # leaving out a bass note that couldn't be fingered
    inner_drop_cost: float = 1.2    # leaving out a chord note that couldn't be fingered
    capo_fixed_cost: float = 3.0    # recommend a capo only when it clearly helps
    capo_cost: float = 0.15         # per capo fret
    max_capo: int = 7


LEVELS = {
    "easy": LevelSettings(name="Easy", pattern="easy", soft_max_fret=5, ornaments=False,
                          slides=False, shift_weight=1.4, height_weight=0.25),
    "medium": LevelSettings(name="Medium", pattern="medium", soft_max_fret=12, ornaments=False,
                            slides=True, shift_weight=1.0, height_weight=0.08),
    "hard": LevelSettings(name="Hard", pattern="hard", soft_max_fret=GUITAR_FRETS, ornaments=True,
                          slides=True, shift_weight=0.6, height_weight=0.02, thin_string_cost=0.05),
}

# Demucs models: htdemucs_ft gives the cleanest vocals (+ bass, other, drums);
# htdemucs_6s is the only one with separate guitar and piano stems.
VOCAL_MODEL = "htdemucs_ft"
SIX_STEM_MODEL = "htdemucs_6s"
STEMS = ["vocals", "guitar", "piano", "bass", "other", "drums"]
