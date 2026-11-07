"""
Central definition of every filesystem location used by the project.

Why this module exists
----------------------
Source code lives in the (OneDrive-synced) project folder, but audio files,
cached features and checkpoints are several GB and must live OUTSIDE OneDrive.
Putting every path in one place means:
  * no other module hard-codes "C:\\SER_data";
  * a team member with a different layout only changes ONE environment variable.

Usage
-----
    from src.utils.paths import PATHS
    PATHS.raw / "RAVDESS"

Override the data root by setting the environment variable SER_DATA, e.g. in
PowerShell:   $env:SER_DATA = "D:\\my_ser_data"
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Project root = two levels above this file (src/utils/paths.py -> project/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Large-data root (outside OneDrive). Default agreed with the team: C:\SER_data
DATA_ROOT = Path(os.environ.get("SER_DATA", r"C:\SER_data"))


@dataclass(frozen=True)
class _Paths:
    # ---- large, NOT in the repository ----
    data_root: Path = DATA_ROOT
    raw: Path = DATA_ROOT / "raw"                  # original downloaded audio
    features: Path = DATA_ROOT / "features"        # cached MFCC / prosody / HuBERT arrays
    cache: Path = DATA_ROOT / "cache"              # HuggingFace model cache, temp files
    checkpoints: Path = DATA_ROOT / "checkpoints"  # trained model weights

    # ---- small, IN the repository (needed to reproduce / grade the work) ----
    project: Path = PROJECT_ROOT
    configs: Path = PROJECT_ROOT / "configs"
    metadata: Path = PROJECT_ROOT / "data" / "metadata"  # file index CSVs
    splits: Path = PROJECT_ROOT / "data" / "splits"      # speaker-fold definitions
    results: Path = PROJECT_ROOT / "results"              # metrics JSON / CSV
    plots: Path = PROJECT_ROOT / "plots"
    experiments: Path = PROJECT_ROOT / "experiments"      # experiment log


PATHS = _Paths()

# Make HuggingFace download HuBERT into C:\SER_data\cache instead of the user
# profile, keeping all large files in one known place.
os.environ.setdefault("HF_HOME", str(PATHS.cache / "huggingface"))
