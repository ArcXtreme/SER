"""
E2 - Frozen HuBERT representations + lightweight classification head.

WHAT IS FROZEN, WHAT IS TRAINED
  HuBERT-base was run once in Stage 4 (no gradient, fp32) and its 13 hidden states are cached.
  E2 trains ONLY the small head below; HuBERT's 94.4 M parameters are never updated.

TEMPORAL POOLING (variable length T -> fixed vector), per hidden state l:
  mean      : mu_l     = (1/T) sum_t h_{t,l}                         768 dims
  mean_std  : [mu_l ; sigma_l],  sigma_l = sqrt((1/T) sum_t (h_{t,l} - mu_l)^2)   1536 dims
  Only real frames are used (masked_mean / masked_std in src/features/batching.py for padded
  batches). The Stage 4 cache already stores [mu_l ; sigma_l] computed per UNPADDED recording
  in fp32; scripts/stage5_e2_inspect.py verified it equals masked pooling of padded batches
  (max rel. diff 1e-4 = fp16 rounding of the frame cache) and that padding values have no effect.

REPRESENTATIONS
  layer l (0..12)          one hidden state                               D = 768 / 1536
  concat(1, 9)             [x_1 ; x_9] (the reference paper's layers)      D = 1536 / 3072
  weighted(all)            sum_l softmax(a)_l * x_l, a in R^13 learned     D = 768 / 1536
                           (SUPERB-style layer weighting; each layer standardised separately)

HEAD (per input vector x in R^D, already standardised with training-fold mean/std)
  z      = GELU(W1 x + b1)           W1: h x D
  z      = Dropout_p(z)
  logits = W2 z + b2                 W2: 6 x h
  loss   = class-weighted (or unweighted) cross-entropy
           L = - sum_i w_{y_i} log softmax(logits_i)_{y_i} / sum_i w_{y_i}
  Trainable parameters: h(D+1) + 6(h+1)  (+13 for weighted(all)).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

N_LAYERS = 13
POOLINGS = ("mean", "mean_std")
REPRESENTATIONS = tuple(f"layer{l}" for l in range(N_LAYERS)) + ("concat_1_9", "weighted_all")


def pooled_input(pooled: np.ndarray, rows: np.ndarray, rep: str, pooling: str) -> np.ndarray:
    """
    pooled: cache array (n, 13, 1536) = [mean | std] per hidden state.
    Returns (len(rows), D) for single/concat representations, (len(rows), 13, D) for weighted_all.
    """
    p = np.asarray(pooled[rows], dtype=np.float32)
    p = p if pooling == "mean_std" else p[..., :768]
    if rep.startswith("layer"):
        return p[:, int(rep[5:])]
    if rep == "concat_1_9":
        return np.concatenate([p[:, 1], p[:, 9]], axis=1)
    if rep == "weighted_all":
        return p
    raise ValueError(rep)


class LayerWeighting(nn.Module):
    """x (B, 13, D) -> sum_l softmax(a)_l x_l. a initialised to 0 = uniform average."""

    def __init__(self, n_layers: int = N_LAYERS):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_layers))

    def forward(self, x):
        return torch.einsum("l,bld->bd", torch.softmax(self.logits, 0), x)


class HubertHead(nn.Module):
    def __init__(self, d_in: int, hidden: int, dropout: float, n_classes: int = 6, weighted_layers: bool = False):
        super().__init__()
        self.layer_weighting = LayerWeighting() if weighted_layers else None
        self.mlp = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x):
        if self.layer_weighting is not None:
            x = self.layer_weighting(x)
        return self.mlp(x)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
