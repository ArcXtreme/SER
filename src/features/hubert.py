"""
Stage 4 - Frozen HuBERT-base feature extraction.

WHAT HuBERT IS (and what the 13 "hidden states" are)
-----------------------------------------------------
HuBERT (Hsu et al., 2021) is a self-supervised speech model. During pre-training on
960 h of unlabelled LibriSpeech it learned to predict cluster IDs (first of MFCC features,
then of its own features) of MASKED frames from the surrounding context - similar to BERT's
masked-word prediction. It never saw emotion labels; we test whether its representations
nonetheless encode emotion.

Architecture of hubert-base (verified in transformers 5.22.1, modeling_hubert.py):
  waveform (N,)                                          16 kHz, z-normalised
   -> FEATURE ENCODER: 7 conv layers (kernels 10,3,3,3,3,2,2; strides 5,2,2,2,2,2,2;
      GroupNorm in layer 1, GELU)           -> (T, 512)   T = 1 + (N - 400) // 320
   -> FEATURE PROJECTION: LayerNorm(512) + Linear(512 -> 768)            -> (T, 768)
   -> ENCODER INPUT: x + PositionalConvEmbedding(x) (conv, kernel 128, 16 groups),
      then LayerNorm                                                    -> (T, 768)
         = hidden_states[0]   ("layer 0": acoustic, before any self-attention)
   -> 12 x Transformer layer (post-LN):
         h = LayerNorm(h + MultiHeadSelfAttention(h))      12 heads x 64 dims
         h = LayerNorm(h + FFN(h))                          768 -> 3072 -> 768, GELU
         = hidden_states[i], i = 1..12  (output of layer i)
  hidden_states[12] == last_hidden_state.

The reference paper uses "the 1st and 9th transformer layers". In this indexing that is
hidden_states[1] and hidden_states[9]; hidden_states[0] is the Transformer INPUT. We store
all 13 so either interpretation (and every other layer) can be tested.

EXTRACTION POLICY
  * Batch size 1, NO padding: hubert-base uses GroupNorm in its first conv layer, which
    normalises over the whole time axis, so zero-padding would change the features of the
    real frames. One utterance at a time gives exact, padding-free features.
  * Input normalisation: (x - mean) / sqrt(var + 1e-7), as Wav2Vec2FeatureExtractor does
    for this checkpoint (preprocessor_config.json: do_normalize = true).
  * model.eval() + torch.inference_mode(): dropout off, no gradients, weights frozen.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import HubertModel

from src.features.config import HubertConfig, n_frames


def zero_mean_unit_var(y: np.ndarray) -> np.ndarray:
    """Exactly Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm for a single utterance."""
    return ((y - y.mean()) / np.sqrt(y.var() + 1e-7)).astype(np.float32)


class HubertExtractor:
    def __init__(self, cfg: HubertConfig, device: str | None = None):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if cfg.inference_dtype == "float16" else torch.float32
        self.model = HubertModel.from_pretrained(cfg.model_id, revision=cfg.revision)
        self.model = self.model.eval().to(self.device, dtype=self.dtype)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, y: np.ndarray) -> torch.Tensor:
        """
        Input : y (N,) float32, 16 kHz.
        Output: float32 tensor (13, T, 768) on the extractor's device, T = 1 + (N-400)//320.
        """
        x = torch.from_numpy(zero_mean_unit_var(y))[None].to(self.device, dtype=self.dtype)
        hs = self.model(x, output_hidden_states=True).hidden_states       # 13 x (1, T, 768)
        out = torch.stack([h[0] for h in hs]).float()                      # (13, T, 768)
        expected = n_frames(len(y))
        if out.shape != (self.cfg.n_hidden_states, expected, 768):
            raise RuntimeError(f"unexpected HuBERT output {tuple(out.shape)}, expected T={expected}")
        return out

    def cache_snapshots(self) -> list[dict]:
        """
        Which exact files were available for this model in the local HuggingFace cache
        (commit hash, refs, file names) - recorded in the feature-cache metadata.
        transformers 5 resolved the weights from the converted safetensors revision
        (refs/pr/2) while config/preprocessor come from the pinned 'main' commit.
        """
        from huggingface_hub import scan_cache_dir
        out = []
        for repo in scan_cache_dir().repos:
            if repo.repo_id == self.cfg.model_id:
                for rev in repo.revisions:
                    out.append({"commit": rev.commit_hash, "refs": sorted(rev.refs),
                                "files": sorted(f.file_name for f in rev.files)})
        return out

    @staticmethod
    def pooled_stats(hs: torch.Tensor) -> np.ndarray:
        """Utterance-level mean and std over time per layer: (13, T, 768) -> (13, 1536) fp32."""
        return torch.cat([hs.mean(1), hs.std(1, unbiased=False)], dim=1).cpu().numpy()

    @torch.inference_mode()
    def verify_hidden_state_semantics(self, y: np.ndarray) -> dict:
        """
        Recompute the pipeline manually from the model's sub-modules and compare with
        hidden_states, proving what each index means (rather than trusting documentation).
        """
        m = self.model
        x = torch.from_numpy(zero_mean_unit_var(y))[None].to(self.device, dtype=self.dtype)
        hs = m(x, output_hidden_states=True).hidden_states
        cnn = m.feature_extractor(x).transpose(1, 2)                       # (1, T, 512)
        proj = m.feature_projection(cnn)                                   # (1, T, 768)
        enc_in = m.encoder.layer_norm(proj + m.encoder.pos_conv_embed(proj))
        err = {"cnn_shape": list(cnn.shape), "hs0_vs_encoder_input": float((hs[0] - enc_in).abs().max())}
        h = enc_in
        for i, layer in enumerate(m.encoder.layers, start=1):
            h = layer(h)
            h = h[0] if isinstance(h, tuple) else h
            err[f"hs{i}_vs_layer{i}_output"] = float((hs[i] - h).abs().max())
        return err
