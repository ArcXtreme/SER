"""
Stage 1 - environment verification.

Run:   C:\\SER_data\\.venv\\Scripts\\python.exe scripts/check_env.py

What it checks and why
----------------------
1. Python version      - all pinned packages were chosen for Python 3.12.
2. Library versions    - recorded so results are reproducible.
3. CUDA availability   - HuBERT feature extraction is ~10-50x faster on GPU.
4. GPU memory          - we only have 4 GB; we confirm a real fp16 matmul works
                         and report how much memory it used.
5. Data directories    - large files must live outside OneDrive.
"""
import importlib
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `import src`
from src.utils.paths import PATHS  # noqa: E402

PACKAGES = ["numpy", "scipy", "pandas", "sklearn", "matplotlib",
            "soundfile", "librosa", "torch", "transformers", "pytest", "yaml"]


def main() -> int:
    ok = True
    print(f"Python      : {platform.python_version()}  ({sys.executable})")
    if sys.version_info[:2] != (3, 12):
        print("  WARNING: environment was designed for Python 3.12")

    print("\nPackages:")
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            print(f"  {name:<13} {getattr(mod, '__version__', '?')}")
        except Exception as e:  # report every failure, do not stop at the first
            ok = False
            print(f"  {name:<13} MISSING / BROKEN -> {e!r}")

    print("\nGPU:")
    try:
        import torch
        print(f"  torch built for CUDA : {torch.version.cuda}")
        print(f"  cuda available       : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"  device               : {props.name}")
            print(f"  total memory         : {props.total_memory / 2**30:.2f} GiB")
            # Smoke test: a half-precision matrix multiply on the GPU.
            a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
            b = a @ a.T
            torch.cuda.synchronize()
            print(f"  fp16 matmul OK, peak alloc {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB,"
                  f" finite={bool(torch.isfinite(b).all())}")
        else:
            ok = False
    except Exception as e:
        ok = False
        print(f"  GPU check failed: {e!r}")

    print("\nData directories (must be outside OneDrive):")
    for p in [PATHS.data_root, PATHS.raw, PATHS.features, PATHS.cache, PATHS.checkpoints]:
        flag = "OK " if p.exists() else "MISSING"
        if "onedrive" in str(p).lower():
            flag, ok = "INSIDE ONEDRIVE!", False
        print(f"  [{flag}] {p}")

    print("\nRESULT:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
