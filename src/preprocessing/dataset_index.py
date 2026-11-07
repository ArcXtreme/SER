"""
Stage 2 - Dataset indexing and unified six-class label mapping.

PURPOSE
-------
Turn two differently-organised corpora into ONE table (pandas DataFrame) with one row
per audio file. Every later stage (splitting, feature extraction, training) reads only
this table, so dataset-specific quirks are handled in exactly one place.

WHY IT MATTERS FOR THE RESEARCH
-------------------------------
* Speaker-independent evaluation needs a reliable speaker ID per file -> parsed here.
* RAVDESS and CREMA-D have DIFFERENT emotion sets. We map both onto the 6 shared
  classes agreed by the team. Files with emotions outside that set (RAVDESS "calm",
  "surprised") are KEPT in the index with included=False, so the exclusion is visible
  and countable instead of silently disappearing.

FILENAME CONVENTIONS (from the dataset documentation)
-----------------------------------------------------
RAVDESS  03-01-06-01-02-01-12.wav = 7 numeric fields:
    modality(03=audio-only) - vocal channel(01=speech, 02=song) - emotion -
    intensity(01 normal, 02 strong) - statement(01 "Kids are talking by the door",
    02 "Dogs are sitting by the door") - repetition(01/02) - actor(01..24; odd=male, even=female)
    emotion: 01 neutral, 02 calm, 03 happy, 04 sad, 05 angry, 06 fearful, 07 disgust, 08 surprised

CREMA-D  1001_DFA_ANG_XX.wav = actorID _ sentence code _ emotion _ level
    emotion: ANG, DIS, FEA, HAP, NEU, SAD
    level:   LO, MD, HI (only sentence IEO has graded levels), XX = unspecified
    Actor demographics (age, sex, race, ethnicity) come from VideoDemographics.csv.

OUTPUT (one row per file)
-------------------------
dataset, rel_path, speaker, gender, emotion_orig, label, label_id, included,
intensity, sentence, repetition, age (CREMA-D only)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---- The unified label space (order defines the integer class id used everywhere) ----
CLASSES: list[str] = ["neutral", "happy", "sad", "angry", "fearful", "disgust"]
LABEL2ID: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}

# Dataset-specific code -> unified label (None = not part of our 6-class task)
RAVDESS_EMOTION = {"01": "neutral", "02": None,  # calm: excluded (overlaps neutral, absent in CREMA-D)
                   "03": "happy", "04": "sad", "05": "angry", "06": "fearful",
                   "07": "disgust", "08": None}  # surprised: excluded (absent in CREMA-D)
RAVDESS_EMOTION_NAME = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad",
                        "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}

CREMAD_EMOTION = {"NEU": "neutral", "HAP": "happy", "SAD": "sad", "ANG": "angry",
                  "FEA": "fearful", "DIS": "disgust"}


def parse_ravdess(root: Path) -> pd.DataFrame:
    """
    Index all RAVDESS speech files under `root`.

    Input : root directory containing Actor_01 ... Actor_24 folders.
    Output: DataFrame (one row per .wav) with the columns listed in the module docstring.
    Raises: ValueError on any filename that does not follow the documented convention,
            so malformed/unexpected files are never silently mis-labelled.
    """
    rows = []
    for wav in sorted(root.rglob("*.wav")):
        parts = wav.stem.split("-")
        if len(parts) != 7:
            raise ValueError(f"Unexpected RAVDESS filename: {wav.name}")
        modality, channel, emo, intensity, statement, rep, actor = parts
        if modality != "03" or channel != "01":
            raise ValueError(f"Not an audio-only speech file: {wav.name}")
        label = RAVDESS_EMOTION[emo]
        rows.append(dict(
            dataset="RAVDESS",
            rel_path=wav.relative_to(root.parent).as_posix(),  # e.g. RAVDESS/Actor_01/03-...wav (portable "/")
            speaker=f"RAV_{actor}",                          # globally unique speaker ID
            gender="M" if int(actor) % 2 == 1 else "F",
            emotion_orig=RAVDESS_EMOTION_NAME[emo],
            label=label,
            label_id=LABEL2ID[label] if label else -1,
            included=label is not None,
            intensity={"01": "normal", "02": "strong"}[intensity],
            sentence={"01": "kids", "02": "dogs"}[statement],
            repetition=int(rep),
            age=float("nan"),
        ))
    return pd.DataFrame(rows)


def parse_cremad(root: Path) -> pd.DataFrame:
    """
    Index all CREMA-D WAV files under root/AudioWAV and attach actor demographics.

    Input : CREMA-D root containing AudioWAV/ and VideoDemographics.csv.
    Output: DataFrame with the same columns as parse_ravdess().
    """
    demo = pd.read_csv(root / "VideoDemographics.csv")
    demo = demo.set_index(demo["ActorID"].astype(str))
    rows = []
    for wav in sorted((root / "AudioWAV").glob("*.wav")):
        parts = wav.stem.split("_")
        if len(parts) != 4:
            raise ValueError(f"Unexpected CREMA-D filename: {wav.name}")
        actor, sentence, emo, level = parts
        label = CREMAD_EMOTION[emo]                     # KeyError -> unknown code, fail loudly
        sex = demo.loc[actor, "Sex"] if actor in demo.index else None
        rows.append(dict(
            dataset="CREMA-D",
            rel_path=wav.relative_to(root.parent).as_posix(),
            speaker=f"CRE_{actor}",
            gender={"Male": "M", "Female": "F"}.get(sex, "?"),
            emotion_orig=emo,
            label=label,
            label_id=LABEL2ID[label],
            included=True,
            intensity=level,
            sentence=sentence,
            repetition=1,
            age=float(demo.loc[actor, "Age"]) if actor in demo.index else float("nan"),
        ))
    return pd.DataFrame(rows)


PARSERS = {"RAVDESS": parse_ravdess, "CREMA-D": parse_cremad}


def build_index(raw_root: Path, datasets=("RAVDESS", "CREMA-D")) -> pd.DataFrame:
    """Concatenate the requested corpora into the single project-wide index."""
    df = pd.concat([PARSERS[d](raw_root / d) for d in datasets], ignore_index=True)
    # Sanity: speaker IDs are dataset-prefixed, so no ID can collide across corpora.
    assert df.groupby("speaker")["dataset"].nunique().max() == 1
    return df
