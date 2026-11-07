"""
SHA-256 protection of completed work around the CREMA-D restart (read-only except the snapshot it creates).

  make    -> results/E4/snapshot_before_cremad_restart.json  (refuses to overwrite an existing snapshot)
             covers: E4 RAVDESS fold results (all 6 folds) + E4 RAVDESS checkpoints + E4 CREMA-D fold-0 pilot files
             (32 S + selection.json) + E4 pilot reports + configs/e4_cross_attention.json
  verify  -> (a) every file of that snapshot unchanged, (b) the 1,432-file E1-E3 snapshot unchanged,
             (c) no new file in the RAVDESS E4 result / checkpoint folders
  Exit code 0 = all unchanged, 1 = something changed or is missing.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.paths import PATHS  # noqa: E402

E4 = PATHS.results / "E4"
SNAP = E4 / "snapshot_before_cremad_restart.json"
SNAP_E1_E3 = E4 / "snapshot_before_e4_pilot.json"


def protected() -> list[Path]:
    # completed RAVDESS runs = the fold folders (summary/control files written later at the dataset level are allowed)
    f = [p for p in (E4 / "RAVDESS").glob("fold*/*") if p.is_file()]
    f += [p for p in (PATHS.checkpoints / "E4" / "RAVDESS").glob("*.pt")]
    f += [p for p in (E4 / "CREMA-D" / "fold0").glob("*.json")]
    f += [E4 / "pilot_RAVDESS.json", E4 / "pilot_CREMA-D.json", PATHS.configs / "e4_cross_attention.json"]
    return sorted(f)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    mode = sys.argv[1]
    if mode == "make":
        if SNAP.exists():
            raise SystemExit(f"{SNAP} already exists - not overwritten")
        snap = {str(p): sha(p) for p in protected()}
        SNAP.write_text(json.dumps(snap, indent=0))
        print(f"snapshot: {len(snap)} files -> {SNAP}")
        return 0
    bad = []
    snap = json.loads(SNAP.read_text())
    bad += [p for p, h in snap.items() if not Path(p).exists() or sha(Path(p)) != h]
    new = [str(p) for p in protected() if str(p) not in snap and "CREMA-D" not in str(p)]
    old = json.loads(SNAP_E1_E3.read_text())
    bad13 = [p for p, h in old.items() if not Path(p).exists() or sha(Path(p)) != h]
    print(f"E4 completed work: {len(snap)} files, changed/missing {len(bad)} {bad[:10]}; new RAVDESS files {len(new)} {new[:5]}")
    print(f"E1-E3: {len(old)} files, changed/missing {len(bad13)} {bad13[:10]}")
    ok = not bad and not new and not bad13
    print("PROTECTED FILES UNCHANGED" if ok else "PROTECTED FILES CHANGED - STOP")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
