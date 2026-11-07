"""
G-study (report section 8) - HuBERT-only frame-level models used to decompose the E2-vs-frame-model gap.

All frame models are built from the SAME E3 class (E3Fusion with the single stream "hubert"), so
  build_frame_model(d=128, ln=True)  ==  the E4 HuBERT-only model `concat_H` (Z), weight for weight,
and every variant differs from Z in exactly one switch:
  d = 768          no 128-d bottleneck (projection and fusion layer are 768 x 768; pooled 1,536 = P0's input)
  ln = False       the per-frame LayerNorm after the projection is replaced by the identity. LayerNorm
                   initialisation consumes no random numbers, so the linear layers start from the SAME
                   weights as in Z for a given seed (tested).

B0 (bridge): no per-frame layers at all.
  u = [mean_t x_t ; std_t x_t]  over real frames of the standardised frames x_t (population std, no eps),
  u' = (u - mu_train) / sd_train  with statistics of the TRAINING recordings' pooled vectors,
  logits = HubertHead(1536 -> h -> 6)  (the E2 head class).
  Frame standardisation is affine per dimension, so after the pooled standardisation u' equals E2's
  standardised pooled input up to fp16 rounding of the frame cache (checked: report section 8.6 c).

Z-W (learned 13-layer weighting at frame level; CREMA-D only):
  input x_t^l per layer, already standardised per layer and dimension with training-frame statistics
  (src/training/streamed_layers.py); x_t = sum_l softmax(a)_l x_t^l, a in R^13 initialised to 0
  (uniform), then the Z model. Created BEFORE the core so that the core's initial weights equal Z's
  for the same seed (the zero-initialised logits consume no random numbers; tested).
"""
from __future__ import annotations

import torch
from torch import nn

from src.models.e2_hubert import HubertHead
from src.models.e3_fusion import E3Fusion, masked_mean_std


def build_frame_model(d: int = 128, ln: bool = True, dropout: float = 0.2, head_hidden: int = 256) -> E3Fusion:
    """E3Fusion(("hubert",)) exactly as E3/E4 call it (n_sa_blocks 0, n_heads 4, d_ffn 256, no DisVoice)."""
    model = E3Fusion(("hubert",), d, dropout, head_hidden, 0, 4, 256, disvoice_dim=0)
    if not ln:
        model.proj["hubert"].ln = nn.Identity()
    return model


class PooledFrameHead(nn.Module):
    """B0: pooled statistics of standardised frames -> pooled standardisation (buffers) -> E2 head."""

    def __init__(self, pooled_mean, pooled_std, hidden: int = 256, dropout: float = 0.2, d_frame: int = 768):
        super().__init__()
        self.register_buffer("mu", torch.as_tensor(pooled_mean, dtype=torch.float32))
        self.register_buffer("sd", torch.as_tensor(pooled_std, dtype=torch.float32))
        self.head = HubertHead(2 * d_frame, hidden, dropout)

    @staticmethod
    def pooled(batch: dict) -> torch.Tensor:
        return masked_mean_std(batch["hubert"], batch["mask"], eps=0.0)

    def standardised_input(self, batch: dict) -> torch.Tensor:
        return (self.pooled(batch) - self.mu) / self.sd

    def forward(self, batch: dict) -> torch.Tensor:
        return self.head(self.standardised_input(batch))


class LayerMix(nn.Module):
    """x (B, T, 13, D) -> sum_l softmax(a)_l x[..., l, :]; a initialised to 0 (uniform weights)."""

    def __init__(self, n_layers: int = 13):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_layers))

    def forward(self, x):
        return torch.einsum("l,btld->btd", torch.softmax(self.logits, 0), x)


class LayerWeightedFrameModel(nn.Module):
    """Z-W: learned layer weighting in front of the Z model."""

    def __init__(self, d: int = 128, ln: bool = True, dropout: float = 0.2, head_hidden: int = 256, n_layers: int = 13):
        super().__init__()
        self.mix = LayerMix(n_layers)                      # created first: consumes no random numbers
        self.core = build_frame_model(d, ln, dropout, head_hidden)

    def forward(self, batch: dict) -> torch.Tensor:
        return self.core({"hubert": self.mix(batch["hubert13"]), "mask": batch["mask"]})

    def layer_weights(self):
        return torch.softmax(self.logits_detached(), 0).tolist()

    def logits_detached(self):
        return self.mix.logits.detach().cpu()
