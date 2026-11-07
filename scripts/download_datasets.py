"""
Stage 2a - download RAVDESS and CREMA-D from their OFFICIAL sources.

Run:   C:\\SER_data\\.venv\\Scripts\\python.exe scripts/download_datasets.py [--only ravdess|cremad]

Sources (both public, licensed for non-commercial research/teaching):
  * RAVDESS speech audio  - Zenodo record 1188976, file Audio_Speech_Actors_01-24.zip
                            (Livingstone & Russo, 2018; licence CC BY-NC-SA 4.0).
                            Integrity: MD5 published by Zenodo is checked after download.
  * CREMA-D audio (WAV)   - GitHub repo CheyneyComputerScience/CREMA-D, folder AudioWAV/
                            (Cao et al., 2014; licence ODbL). Files are stored with Git LFS,
                            so each WAV is fetched from media.githubusercontent.com.
                            We also fetch VideoDemographics.csv (actor age/sex/ethnicity).

Design notes
  * Idempotent: re-running skips files that already exist, so an interrupted download
    can simply be restarted.
  * Every failed file is reported and written to a text file - nothing is silently skipped.
  * Zip extraction refuses paths that would escape the target directory ("zip slip").
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.paths import PATHS  # noqa: E402

UA = {"User-Agent": "SER-course-project-downloader/1.0"}

RAVDESS_URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
RAVDESS_MD5 = "bc696df654c87fed845eb13823edef8a"   # from https://zenodo.org/api/records/1188976

CREMAD_TREE_API = "https://api.github.com/repos/CheyneyComputerScience/CREMA-D/git/trees/master?recursive=1"
CREMAD_MEDIA = "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/"
CREMAD_RAW = "https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master/"
CREMAD_EXTRA = ["VideoDemographics.csv", "SentenceFilenames.csv", "LICENSE.txt", "README.md"]


# --------------------------------------------------------------------------- helpers
def _get(url: str, dest: Path, retries: int = 3, timeout: int = 60) -> None:
    """Download `url` to `dest` atomically (via a .part file), with retries."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.replace(dest)
            return
        except Exception:
            if attempt == retries:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(2 * attempt)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _safe_extract(zf: zipfile.ZipFile, target: Path) -> None:
    """Extract, refusing any member whose resolved path leaves `target`."""
    target = target.resolve()
    for m in zf.infolist():
        out = (target / m.filename).resolve()
        if not str(out).startswith(str(target)):
            raise RuntimeError(f"Unsafe path in zip: {m.filename}")
    zf.extractall(target)


# --------------------------------------------------------------------------- RAVDESS
def download_ravdess() -> None:
    out_dir = PATHS.raw / "RAVDESS"
    zip_path = PATHS.raw / "_downloads" / "Audio_Speech_Actors_01-24.zip"
    if out_dir.exists() and len(list(out_dir.rglob("*.wav"))) >= 1440:
        print(f"[RAVDESS] already extracted at {out_dir}")
        return
    if not (zip_path.exists() and _md5(zip_path) == RAVDESS_MD5):
        print("[RAVDESS] downloading 208 MB from Zenodo ...")
        _get(RAVDESS_URL, zip_path, timeout=300)
    digest = _md5(zip_path)
    if digest != RAVDESS_MD5:
        raise RuntimeError(f"[RAVDESS] MD5 mismatch: got {digest}, expected {RAVDESS_MD5}")
    print("[RAVDESS] MD5 verified; extracting ...")
    with zipfile.ZipFile(zip_path) as zf:
        _safe_extract(zf, out_dir)
    print(f"[RAVDESS] {len(list(out_dir.rglob('*.wav')))} wav files in {out_dir}")


# --------------------------------------------------------------------------- CREMA-D
def download_cremad(workers: int = 8) -> None:
    out_dir = PATHS.raw / "CREMA-D"
    wav_dir = out_dir / "AudioWAV"

    # 1) list the WAV files from the repository tree (single API call)
    req = urllib.request.Request(CREMAD_TREE_API, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        tree = json.load(r)
    if tree.get("truncated"):
        raise RuntimeError("GitHub tree listing truncated - cannot trust the file list")
    names = sorted(Path(t["path"]).name for t in tree["tree"]
                   if t["type"] == "blob" and t["path"].startswith("AudioWAV/")
                   and t["path"].endswith(".wav"))
    print(f"[CREMA-D] repository lists {len(names)} WAV files")

    # 2) small metadata files
    for name in CREMAD_EXTRA:
        dest = out_dir / name
        if not dest.exists():
            _get(CREMAD_RAW + name, dest)

    # 3) WAVs, in parallel, skipping ones already present
    todo = [n for n in names if not (wav_dir / n).exists() or (wav_dir / n).stat().st_size == 0]
    print(f"[CREMA-D] {len(names) - len(todo)} present, {len(todo)} to download")
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_get, CREMAD_MEDIA + "AudioWAV/" + n, wav_dir / n): n for n in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                fut.result()
            except Exception as e:
                failed.append(f"{futs[fut]}\t{e!r}")
            if i % 500 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} done, {len(failed)} failed")

    fail_log = out_dir / "download_failures.txt"
    if failed:
        fail_log.write_text("\n".join(failed), encoding="utf-8")
        print(f"[CREMA-D] WARNING: {len(failed)} files failed - see {fail_log}. Re-run to retry.")
    else:
        fail_log.unlink(missing_ok=True)
        print(f"[CREMA-D] complete: {len(list(wav_dir.glob('*.wav')))} WAV files in {wav_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["ravdess", "cremad"])
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel CREMA-D downloads (GitHub LFS is slow per request, ~20 s/file)")
    args = ap.parse_args()
    if args.only in (None, "ravdess"):
        download_ravdess()
    if args.only in (None, "cremad"):
        download_cremad(workers=args.workers)
