"""
Tests for Stage 2 (indexing + integrity checks), using tiny synthetic files so they
run in seconds and do not need the real datasets.

Run:  C:\\SER_data\\.venv\\Scripts\\python.exe -m pytest tests -q
"""
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from src.preprocessing.audio_check import check_file
from src.preprocessing.dataset_index import LABEL2ID, parse_cremad, parse_ravdess

SR = 16_000


def _tone(seconds=1.0, amp=0.3):
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype("float32")


# ----------------------------------------------------------------- RAVDESS parsing
def test_ravdess_label_mapping_gender_and_exclusion(tmp_path):
    root = tmp_path / "RAVDESS" / "Actor_07"
    root.mkdir(parents=True)
    # emotion 05 = angry (actor 07 -> odd -> male); emotion 02 = calm (excluded)
    for name in ["03-01-05-02-01-02-07.wav", "03-01-02-01-01-01-07.wav"]:
        sf.write(root / name, _tone(), SR)
    df = parse_ravdess(tmp_path / "RAVDESS").set_index("emotion_orig")
    assert df.loc["angry", "label_id"] == LABEL2ID["angry"]
    assert df.loc["angry", "gender"] == "M"
    assert df.loc["angry", "intensity"] == "strong"
    assert df.loc["angry", "speaker"] == "RAV_07"
    assert not df.loc["calm", "included"] and df.loc["calm", "label_id"] == -1


def test_ravdess_rejects_malformed_and_song_files(tmp_path):
    root = tmp_path / "RAVDESS" / "Actor_01"
    root.mkdir(parents=True)
    sf.write(root / "03-02-05-01-01-01-01.wav", _tone(), SR)      # vocal channel 02 = song
    with pytest.raises(ValueError):
        parse_ravdess(tmp_path / "RAVDESS")


# ----------------------------------------------------------------- CREMA-D parsing
def test_cremad_parsing_with_demographics(tmp_path):
    root = tmp_path / "CREMA-D"
    (root / "AudioWAV").mkdir(parents=True)
    pd.DataFrame({"ActorID": [1001, 1002], "Age": [51, 21], "Sex": ["Male", "Female"],
                  "Race": ["x", "y"], "Ethnicity": ["a", "b"]}).to_csv(root / "VideoDemographics.csv", index=False)
    sf.write(root / "AudioWAV" / "1002_IEO_FEA_HI.wav", _tone(), SR)
    row = parse_cremad(root).iloc[0]
    assert row.speaker == "CRE_1002" and row.gender == "F" and row.age == 21
    assert row.label == "fearful" and row.intensity == "HI"


# ----------------------------------------------------------------- integrity checks
@pytest.mark.parametrize("signal,expected", [
    (_tone(1.0), "ok"),
    (np.zeros(SR, dtype="float32"), "silent"),
    (_tone(0.2), "too_short"),
])
def test_check_file_status(tmp_path, signal, expected):
    p = tmp_path / "x.wav"
    sf.write(p, signal, SR)
    assert check_file(p)["status"] == expected


def test_check_file_unreadable(tmp_path):
    p = tmp_path / "broken.wav"
    p.write_bytes(b"RIFF....not really a wav")
    out = check_file(p)
    assert out["status"] == "unreadable" and out["error"]
