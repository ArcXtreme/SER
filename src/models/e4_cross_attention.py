"""
E4 - bidirectional CROSS-ATTENTION fusion of HuBERT, MFCC and prosody (+ its ablation structures).

INPUTS: identical to E3 (src/models/e3_fusion.py): per-fold single HuBERT layer (T, 768),
MFCC (T, 39), prosody (T, 4), mask (T,) True = real frame; all on the shared 20 ms grid and
standardised with TRAINING-fold statistics.

ARCHITECTURE "HMP" (the E4 model; d = d_model = 128)
  1. stream projection  p_s = Dropout(LN(W_s x_s + b_s))                s in {H, M, P}   (same as E3)
  2. stage 1 (MFCC <-> prosody)
       m' = X(p_M <- p_P)        MFCC frames query prosody frames
       p' = X(p_P <- p_M)        prosody frames query MFCC frames
  3. fuse 1             a  = Dropout(GELU(W_a [m' ; p'] + b_a))          "acoustic" stream, (T, d)
  4. stage 2 (HuBERT <-> acoustic)
       h' = X(p_H <- a),  a' = X(a <- p_H)
  5. fuse 2             z  = LN(Dropout(GELU(W_z [h' ; a'] + b_z)))
  6. masked mean + std pooling over real frames (2d), 7. MLP head - both identical to E3.
  X(x <- c) is src/models/attention.AttentionBlock(cross=True), the SAME code as the E3
  self-attention control:  x <- x + Drop(MHA(LN_q x, LN_c c, key-mask of c));
                           x <- x + Drop(FFN(LN_f x)).

ABLATION STRUCTURES (same blocks, fewer of them; trained with E4's selected hyper-parameters)
  "HM"       one stage H <-> M                  (prosody removed)
  "HP"       one stage H <-> P                  (MFCC removed)
  "H_MPcat"  stage 1 replaced by concatenation: a = Dropout(GELU(W_a [p_M ; p_P])), then stage 2

METHODOLOGICAL LIMITATION (part of the approved design; kept unchanged)
  There is NO positional encoding (as in the self-attention control) and pooling is order-free.
  - The attention weights depend on frame CONTENT only: a query frame cannot prefer the context
    frame at the same time step or nearby ones, and the output of a cross-attention sub-layer
    does not depend on the order of the context frames (tested).
  - The shared 20 ms grid is used only by the residual path and the fuse layers, which
    concatenate the two outputs frame by frame ([h'(t) ; a'(t)]), as E3's concatenation does.
  - The whole model is invariant to a JOINT permutation of the time frames of all streams
    (tested): temporal order and dynamics are not modelled, as in E3.
"""
from __future__ import annotations

import torch
from torch import nn

from src.models.attention import AttentionBlock
from src.models.e3_fusion import STREAM_DIMS, StreamProjection, masked_mean_std, n_params

STRUCTURES = {                                  # structure -> streams used
    "HMP": ("hubert", "mfcc", "prosody"),
    "HM": ("hubert", "mfcc"),
    "HP": ("hubert", "prosody"),
    "H_MPcat": ("hubert", "mfcc", "prosody"),
}


class E4CrossAttention(nn.Module):
    def __init__(self, structure: str = "HMP", d_model: int = 128, dropout: float = 0.2, head_hidden: int = 256,
                 n_heads: int = 4, d_ffn: int = 256, n_classes: int = 6):
        super().__init__()
        if structure not in STRUCTURES:
            raise ValueError(f"unknown E4 structure {structure}")
        self.structure = structure
        self.streams = STRUCTURES[structure]
        d = d_model
        self.proj = nn.ModuleDict({s: StreamProjection(STREAM_DIMS[s], d, dropout) for s in self.streams})
        X = lambda: AttentionBlock(d, n_heads, d_ffn, dropout, cross=True)  # noqa: E731
        fuse = lambda: nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(dropout))  # noqa: E731
        self.m_from_p = X() if structure == "HMP" else None
        self.p_from_m = X() if structure == "HMP" else None
        self.fuse_mp = fuse() if structure in ("HMP", "H_MPcat") else None
        self.h_from_a, self.a_from_h = X(), X()
        self.fuse_out = fuse()
        self.final_ln = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(2 * d, head_hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(head_hidden, n_classes))

    def acoustic(self, p: dict, mask: torch.Tensor) -> torch.Tensor:
        """The context stream that HuBERT exchanges information with in stage 2."""
        if self.structure == "HMP":
            m2 = self.m_from_p(p["mfcc"], p["prosody"], mask)
            p2 = self.p_from_m(p["prosody"], p["mfcc"], mask)
            return self.fuse_mp(torch.cat([m2, p2], dim=-1))
        if self.structure == "H_MPcat":
            return self.fuse_mp(torch.cat([p["mfcc"], p["prosody"]], dim=-1))
        return p["mfcc"] if self.structure == "HM" else p["prosody"]

    def frame_features(self, batch: dict) -> torch.Tensor:
        mask = batch["mask"]                    # one mask for all streams (shared frame grid)
        p = {s: self.proj[s](batch[s]) for s in self.streams}
        a = self.acoustic(p, mask)
        h2 = self.h_from_a(p["hubert"], a, mask)
        a2 = self.a_from_h(a, p["hubert"], mask)
        return self.final_ln(self.fuse_out(torch.cat([h2, a2], dim=-1)))

    def forward(self, batch: dict) -> torch.Tensor:
        return self.head(masked_mean_std(self.frame_features(batch), batch["mask"]))


def e4_fusion_params(model: E4CrossAttention) -> int:
    """Everything except the stream projections and the head (comparable to e3_fusion_params)."""
    return n_params(model) - n_params(model.proj) - n_params(model.head)
