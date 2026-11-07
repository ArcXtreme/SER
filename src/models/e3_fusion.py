"""
E3 - frame-level CONCATENATION fusion of HuBERT, MFCC and prosody (+ self-attention control).

INPUTS (one recording, all on the shared 20 ms grid of the Stage 4 cache, T frames)
  hubert   (T, 768)  ONE hidden state; the layer is chosen per fold from that fold's own E2
                     phase-A validation results (never from an average over folds)
  mfcc     (T, 39)   13 MFCC + delta + delta-delta
  prosody  (T, 4)    [f0_st, voiced, periodicity, energy_db]
  mask     (T,)      True = real frame (batches are padded to the longest recording)
  Every stream is standardised with TRAINING-fold statistics (voiced, periodicity passed through).

ARCHITECTURE (d = d_model)
  1. stream projection   p_s(t) = Dropout(LN(W_s x_s(t) + b_s))          s in streams, (T, d) each
  2. concatenation       c(t)   = [p_h(t) ; p_m(t) ; p_p(t)]              (T, k*d), k = #streams
  3. fusion layer        z(t)   = Dropout(GELU(W_f c(t) + b_f))           (T, d)
     [control only] z  <- LN(Block_n(...Block_1(z)))   n self-attention blocks over the FUSED
                     frames (src/models/attention.py), padding masked as keys
  4. masked statistics pooling  u = [mean_t z(t) ; std_t z(t)]  over REAL frames only   (2d)
     [DisVoice ablation only]  u <- [u ; GELU(W_v v)], v = 103 (or 206) utterance features
  5. head                logits = W_2 Dropout(GELU(W_1 u))                  (6)

  Concatenation lets the fusion layer form any LINEAR combination of the three streams at the
  SAME frame t, followed by one non-linearity; it cannot let one stream select WHICH frames of
  another stream matter. That ability is what E4's cross-attention adds; the self-attention
  control adds attention capacity WITHOUT stream-specific queries, so E4 vs control isolates
  "attention across feature types" from "more attention / more parameters".

PARAMETER MATCHING (control vs E4)
  e4_reference_fusion_params() counts the fusion module planned for E4 (bidirectional
  MFCC<->prosody cross-attention, fuse, bidirectional HuBERT<->fused, fuse) built from the SAME
  AttentionBlock class; matched_sa_blocks() picks the number of self-attention blocks whose
  total parameter count is closest to it (asserted within 5 % in the tests). If E4's final
  design changes, the control must be re-matched (documented in the report).
"""
from __future__ import annotations

import torch
from torch import nn

from src.models.attention import AttentionBlock

STREAM_DIMS = {"hubert": 768, "mfcc": 39, "prosody": 4}
STREAM_ORDER = ("hubert", "mfcc", "prosody")
SUBSETS = {                                     # ablation name -> streams
    "H": ("hubert",), "M": ("mfcc",), "P": ("prosody",),
    "HM": ("hubert", "mfcc"), "HP": ("hubert", "prosody"), "MP": ("mfcc", "prosody"),
    "HMP": ("hubert", "mfcc", "prosody"),
}


def masked_mean_std(z: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """(B, T, d), mask (B, T) -> (B, 2d) = [mean ; sqrt(var + eps)] over real frames only.
    eps keeps the gradient of sqrt finite if a dimension is constant over time."""
    m = mask.unsqueeze(-1).to(z.dtype)
    n = m.sum(1).clamp_min(1.0)
    mu = (z * m).sum(1) / n
    var = (((z - mu.unsqueeze(1)) ** 2) * m).sum(1) / n
    return torch.cat([mu, torch.sqrt(var + eps)], dim=-1)


class StreamProjection(nn.Module):
    def __init__(self, d_in: int, d_model: int, dropout: float):
        super().__init__()
        self.lin, self.ln, self.drop = nn.Linear(d_in, d_model), nn.LayerNorm(d_model), nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.ln(self.lin(x)))


class E3Fusion(nn.Module):
    def __init__(self, streams=STREAM_ORDER, d_model: int = 128, dropout: float = 0.2, head_hidden: int = 256,
                 n_sa_blocks: int = 0, n_heads: int = 4, d_ffn: int = 256, disvoice_dim: int = 0, n_classes: int = 6):
        super().__init__()
        self.streams = tuple(s for s in STREAM_ORDER if s in streams)      # canonical order
        if not self.streams or len(self.streams) != len(tuple(streams)):
            raise ValueError(f"invalid streams {streams}")
        self.proj = nn.ModuleDict({s: StreamProjection(STREAM_DIMS[s], d_model, dropout) for s in self.streams})
        self.fuse = nn.Sequential(nn.Linear(len(self.streams) * d_model, d_model), nn.GELU(), nn.Dropout(dropout))
        self.blocks = nn.ModuleList([AttentionBlock(d_model, n_heads, d_ffn, dropout) for _ in range(n_sa_blocks)])
        self.final_ln = nn.LayerNorm(d_model) if n_sa_blocks else None
        self.dv = (nn.Sequential(nn.Linear(disvoice_dim, d_model), nn.GELU(), nn.Dropout(dropout))
                   if disvoice_dim else None)
        d_u = 2 * d_model + (d_model if disvoice_dim else 0)
        self.head = nn.Sequential(nn.Linear(d_u, head_hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(head_hidden, n_classes))

    def frame_features(self, batch: dict) -> torch.Tensor:
        z = self.fuse(torch.cat([self.proj[s](batch[s]) for s in self.streams], dim=-1))
        for blk in self.blocks:
            z = blk(z, c_mask=batch["mask"])
        return self.final_ln(z) if self.final_ln is not None else z

    def forward(self, batch: dict) -> torch.Tensor:
        u = masked_mean_std(self.frame_features(batch), batch["mask"])
        if self.dv is not None:
            u = torch.cat([u, self.dv(batch["disvoice"])], dim=-1)
        return self.head(u)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def e4_reference_fusion_params(d_model=128, n_heads=4, d_ffn=256) -> int:
    """Parameters of the E4 fusion module as planned (4 cross-attention blocks + 2 fuse layers
    + final LayerNorm). Projections, pooling and head are identical to E3 and not counted."""
    blk = n_params(AttentionBlock(d_model, n_heads, d_ffn, 0.0, cross=True))
    fuse = n_params(nn.Linear(2 * d_model, d_model))
    return 4 * blk + 2 * fuse + n_params(nn.LayerNorm(d_model))


def e3_fusion_params(model: E3Fusion) -> int:
    return n_params(model.fuse) + n_params(model.blocks) + (n_params(model.final_ln) if model.final_ln else 0)


def matched_sa_blocks(d_model=128, n_heads=4, d_ffn=256, max_blocks=8) -> int:
    target = e4_reference_fusion_params(d_model, n_heads, d_ffn)
    diffs = {n: abs(e3_fusion_params(E3Fusion(STREAM_ORDER, d_model, 0.0, n_sa_blocks=n, n_heads=n_heads,
                                              d_ffn=d_ffn)) - target) for n in range(1, max_blocks + 1)}
    return min(diffs, key=lambda n: (diffs[n], n))
