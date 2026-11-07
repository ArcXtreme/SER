"""
Stage 4 - Padding, masks and aligned cropping for variable-length sequences.

WHY MASKS?
  A mini-batch tensor must be rectangular (B, T_max, D), so shorter recordings are padded
  with zeros. Those padded frames are NOT speech: any pooling (mean/std over time) and any
  attention must ignore them, otherwise a short clip's features get diluted by zeros and
  attention can "look at" padding. The mask is True for real frames, False for padding.

WHY A SHARED CROP?
  If recordings longer than a limit are cut, the SAME [start, end) frame window must be cut
  from HuBERT, MFCC and prosody - cropping each stream independently would destroy the
  frame-to-frame alignment that cross-attention fusion relies on.
  (The reference paper instead repeats short clips up to 7 s; we pad + mask, because
  repetition would duplicate content and bias utterance statistics.)
"""
from __future__ import annotations

import numpy as np
import torch


def pad_sequences(seqs: list[np.ndarray | torch.Tensor], pad_value: float = 0.0
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Input : list of B arrays, each (T_i, *feature_dims)
    Output: padded (B, T_max, *feature_dims) float32, mask (B, T_max) bool (True = real frame)
    """
    # np.array (copy) rather than asarray: memory-mapped cache arrays are read-only
    seqs = [torch.from_numpy(np.array(s, dtype=np.float32)) if not torch.is_tensor(s) else s.float()
            for s in seqs]
    lengths = torch.tensor([len(s) for s in seqs])
    B, T = len(seqs), int(lengths.max())
    out = torch.full((B, T, *seqs[0].shape[1:]), pad_value, dtype=torch.float32)
    for b, s in enumerate(seqs):
        out[b, : len(s)] = s
    mask = torch.arange(T)[None, :] < lengths[:, None]
    return out, mask


def crop_window(n: int, max_frames: int | None, mode: str = "center",
                rng: np.random.Generator | None = None) -> tuple[int, int]:
    """
    [start, end) frame window of length <= max_frames for a recording with n frames.
    mode="center" (deterministic: validation/test) or "random" (training augmentation).
    Apply the returned window to EVERY stream of the recording.
    """
    if max_frames is None or n <= max_frames:
        return 0, n
    if mode == "center":
        start = (n - max_frames) // 2
    elif mode == "random":
        start = int((rng or np.random.default_rng()).integers(0, n - max_frames + 1))
    else:
        raise ValueError(mode)
    return start, start + max_frames


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over time of real frames only. x (B, T, D), mask (B, T) -> (B, D)."""
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(1) / m.sum(1).clamp_min(1.0)


def masked_std(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Population std over real frames only. (B, T, D) -> (B, D)."""
    mu = masked_mean(x, mask)
    m = mask.unsqueeze(-1).to(x.dtype)
    var = (((x - mu.unsqueeze(1)) ** 2) * m).sum(1) / m.sum(1).clamp_min(1.0)
    return var.clamp_min(0).sqrt()
