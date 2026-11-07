"""
Explicit multi-head attention (Q/K/V written out, no nn.MultiheadAttention) and a residual
Transformer block. Used by the E3 self-attention CONTROL now and by E4 cross-attention later,
so both share exactly the same attention code and parameter counts.

SCALED DOT-PRODUCT ATTENTION (one head h, query sequence X_q (Tq x d), context X_c (Tk x d))
  Q_h = X_q W_Q^h      K_h = X_c W_K^h      V_h = X_c W_V^h          W^h : d x d_k,  d_k = d / H
  S_h = Q_h K_h^T / sqrt(d_k)                                        (Tq x Tk) similarity scores
  S_h[:, j] = -inf  for every padded context frame j                 (key padding mask)
  A_h = softmax_row(S_h)                                             each row sums to 1 over REAL frames
  O_h = A_h V_h                                                      (Tq x d_k)
  MHA(X_q, X_c) = [O_1 ; ... ; O_H] W_O                              W_O : d x d
  Self-attention: X_c = X_q.  Cross-attention: X_c = another stream (E4).

  The 1/sqrt(d_k) factor keeps the variance of the scores ~1 when Q and K entries have unit
  variance, so the softmax does not saturate (vanishing gradients) as d_k grows.
  Numerical stability: the scores are computed in float32 and masked with the most negative
  finite float (not -inf), so even a row whose keys were ALL masked gives a finite (uniform)
  distribution instead of NaN; torch.softmax subtracts the row maximum internally.

RESIDUAL BLOCK (pre-LayerNorm; stable without learning-rate warm-up on small data)
  X = X + Dropout(MHA(LN_q(X), LN_c(C), mask_C))        LN_c = LN_q for self-attention
  X = X + Dropout(FFN(LN_f(X)))                         FFN(z) = W_2 Dropout(GELU(W_1 z))
  Output frames at padded query positions are computed but never used: pooling is masked.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def _split(self, x):                                   # (B, T, d) -> (B, H, T, d_k)
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, x_q: torch.Tensor, x_c: torch.Tensor, c_mask: torch.Tensor | None = None,
                return_weights: bool = False):
        """
        x_q (B, Tq, d) queries; x_c (B, Tk, d) context (keys/values);
        c_mask (B, Tk) bool, True = real context frame.  Returns (B, Tq, d) [, weights (B, H, Tq, Tk)].
        """
        q, k, v = self._split(self.w_q(x_q)), self._split(self.w_k(x_c)), self._split(self.w_v(x_c))
        scores = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(self.d_k)      # (B, H, Tq, Tk)
        if c_mask is not None:
            scores = scores.masked_fill(~c_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        out = self.attn_drop(weights).to(v.dtype) @ v                                 # (B, H, Tq, d_k)
        out = out.transpose(1, 2).reshape(x_q.shape[0], x_q.shape[1], self.d_model)
        out = self.w_o(out)
        return (out, weights) if return_weights else out


class AttentionBlock(nn.Module):
    """Pre-LN residual block: attention sub-layer + position-wise FFN sub-layer."""

    def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float, cross: bool = False):
        super().__init__()
        self.cross = cross
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_c = nn.LayerNorm(d_model) if cross else None
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln_f = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ffn, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None, c_mask: torch.Tensor | None = None):
        q = self.ln_q(x)
        ctx = q if not self.cross else self.ln_c(c)
        x = x + self.drop(self.attn(q, ctx, c_mask))
        return x + self.drop(self.ffn(self.ln_f(x)))
