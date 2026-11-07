"""
G-study integrity manifest (report section 8.6 c / 8.8 S3). SHA-256 of every EXISTING file the G-study must not change:
  results/** (everything except results/G/), C:\\SER_data\\checkpoints\\** , configs/*.json (except g_study.json),
  the feature caches used (C:\\SER_data\\features\\<DS>\\<cache_id>\\*).
  make    -> results/G/manifest_before_g.json (refuses to overwrite)
  verify  -> exit 0 if every file is unchanged and present, 1 otherwise;  --quick skips the feature caches
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.paths import PATHS  # noqa: E402

MANIFEST = PATHS.results / "G" / "manifest_before_g.json"
CACHE_ID = "v1_efe9081789"


def files(quick=False):
    out = [p for p in PATHS.results.rglob("*") if p.is_file() and "G" not in p.relative_to(PATHS.results).parts[:1]]
    out += [p for p in PATHS.checkpoints.rglob("*") if p.is_file() and p.relative_to(PATHS.checkpoints).parts[0] != "G"]
    out += [p for p in PATHS.configs.glob("*.json") if p.name != "g_study.json"]
    if not quick:
        for ds in ("RAVDESS", "CREMA-D"):
            out += [p for p in (Path(r"C:\SER_data\features") / ds / CACHE_ID).iterdir() if p.is_file()]
    return sorted(set(out))


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["make", "verify"])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.mode == "make":
        if MANIFEST.exists():
            raise SystemExit(f"{MANIFEST} exists - not overwritten")
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        m = {str(p): sha(p) for p in files()}
        MANIFEST.write_text(json.dumps(m, indent=0))
        print(f"manifest: {len(m)} files -> {MANIFEST}")
        return 0
    m = json.loads(MANIFEST.read_text())
    keep = {str(p) for p in files(a.quick)} if a.quick else None
    bad = [p for p, h in m.items() if (keep is None or p in keep or not Path(p).exists()) and (not Path(p).exists() or sha(Path(p)) != h)]
    checked = len(m) if keep is None else len([p for p in m if p in keep])
    new = [str(p) for p in files(a.quick) if str(p) not in m]
    print(f"G manifest verify{' (quick)' if a.quick else ''}: checked {checked} files; changed/missing {len(bad)} {bad[:10]}; "
          f"new files in protected locations {len(new)} {new[:10]}")
    ok = not bad and not new
    print("PROTECTED FILES UNCHANGED" if ok else "PROTECTED FILES CHANGED - STOP")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
