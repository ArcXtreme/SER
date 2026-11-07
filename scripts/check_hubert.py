"""
Stage 1 - verify HuBERT-base loads with the installed transformers version and fits in 4 GB.

Downloads facebook/hubert-base-ls960 (~360 MB) into C:\\SER_data\\cache\\huggingface.
Runs a forward pass on 7 s of random audio (the longest clip length we plan to keep)
and reports: number of hidden-state tensors (expect 13 = CNN-feature projection + 12
Transformer layers), their shape, frame rate, and peak GPU memory.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.paths import PATHS  # noqa: E402,F401  (sets HF_HOME before transformers import)

import torch  # noqa: E402
from transformers import HubertModel  # noqa: E402

MODEL = "facebook/hubert-base-ls960"

model = HubertModel.from_pretrained(MODEL).eval().half().cuda()
n_params = sum(p.numel() for p in model.parameters())
print(f"loaded {MODEL}: {n_params / 1e6:.1f} M parameters")

sr, seconds = 16_000, 7.0
wav = torch.randn(1, int(sr * seconds), device="cuda", dtype=torch.float16) * 0.1
torch.cuda.reset_peak_memory_stats()
with torch.inference_mode():
    t0 = time.perf_counter()
    out = model(wav, output_hidden_states=True)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
hs = out.hidden_states
print(f"hidden_states: {len(hs)} tensors, each shape {tuple(hs[0].shape)} (batch, frames, dim)")
print(f"frames per second: {hs[0].shape[1] / seconds:.2f}  (expected ~50 -> 20 ms hop)")
print(f"all finite: {all(bool(torch.isfinite(h).all()) for h in hs)}")
print(f"forward time (first call, incl. warm-up): {dt * 1000:.0f} ms")
print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB of 4096 MiB")
