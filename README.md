# Speech Emotion Recognition: Does Cross-Attention Fusion Help?

Course project (team of 2). A controlled study of self-supervised (HuBERT) and hand-crafted
(MFCC, prosody) features for **speaker-independent** speech emotion recognition, comparing
concatenation-based fusion with bidirectional cross-attention fusion.

Reference paper (used as a research reference, not reproduced):
R. Zhao et al., *Leveraging Cross-Attention Transformer and Multi-Feature Fusion for
Cross-Linguistic Speech Emotion Recognition*, arXiv:2501.10408, 2025.

## Setup (Windows, Python 3.12)

```powershell
python -m venv C:\SER_data\.venv
C:\SER_data\.venv\Scripts\python.exe -m pip install -r requirements.txt
C:\SER_data\.venv\Scripts\python.exe scripts\check_env.py        # environment check
C:\SER_data\.venv\Scripts\python.exe scripts\check_hubert.py     # HuBERT on GPU check
```

Large files (datasets, features, checkpoints, venv) live in `C:\SER_data` - outside OneDrive.
To use another location set `$env:SER_DATA` (see `src/utils/paths.py`).

## Data

```powershell
C:\SER_data\.venv\Scripts\python.exe scripts\download_datasets.py --workers 24
C:\SER_data\.venv\Scripts\python.exe scripts\stage2_prepare_data.py
```

| Dataset | Source | Licence |
|---|---|---|
| RAVDESS (speech) | Zenodo record 1188976 (MD5-verified) | CC BY-NC-SA 4.0 |
| CREMA-D | github.com/CheyneyComputerScience/CREMA-D | ODbL |

Six shared classes: neutral, happy, sad, angry, fearful, disgust.

## Layout

```
configs/        experiment configurations
data/metadata/  file index + verified dataset statistics (small, versioned)
data/splits/    speaker-independent fold definitions
src/            preprocessing/ features/ models/ training/ evaluation/ utils/
scripts/        runnable entry points for each stage
tests/          unit tests (pytest)
experiments/    LOG.md - central experiment log
results/        metrics (JSON/CSV) of executed runs
plots/          figures
reports/        project report and slides
```

## Tests

```powershell
C:\SER_data\.venv\Scripts\python.exe -m pytest
```
