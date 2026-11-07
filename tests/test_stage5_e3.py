"""
Stage 5 / E3 tests: explicit attention (dimensions, masks, numerical stability, gradient flow,
variable length), E3 fusion model (streams, padding invariance, parameter matching), frame-level
data (alignment, train-only standardisation, bucketing), trainer reproducibility, per-fold layer
selection and test isolation during selection.
"""
import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from src.models.attention import AttentionBlock, MultiHeadAttention
from src.models.e3_fusion import (SUBSETS, E3Fusion, e3_fusion_params, e4_reference_fusion_params, masked_mean_std,
                                  matched_sa_blocks, n_params)
from src.training.data import FoldRows, without_test
from src.training.seq_trainer import SeqTrainConfig, predict_rows, train_sequence_classifier
from src.training.sequence_data import SequenceData
from src.training.trainer import device

ROOT = Path(__file__).resolve().parents[1]
torch.manual_seed(0)


def _driver():
    spec = importlib.util.spec_from_file_location("e3drv", ROOT / "scripts" / "stage5_e3_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _padded(lengths, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    T = max(lengths)
    x = torch.randn(len(lengths), T, d, generator=g)
    mask = torch.arange(T)[None] < torch.tensor(lengths)[:, None]
    return x, mask


# ================================================================== attention
def test_mha_dimensions_and_weights_are_distributions_over_real_frames():
    mha = MultiHeadAttention(16, 4).eval()
    xq, _ = _padded([5, 3], seed=1)
    xc, cm = _padded([7, 4], seed=2)
    out, w = mha(xq, xc, cm, return_weights=True)
    assert out.shape == (2, 5, 16) and w.shape == (2, 4, 5, 7)
    torch.testing.assert_close(w.sum(-1), torch.ones(2, 4, 5))
    assert (w[1, :, :, 4:] == 0).all()                         # padded keys get exactly zero weight


def test_mha_matches_reference_scaled_dot_product_attention():
    mha = MultiHeadAttention(16, 4).eval()
    xq, _ = _padded([6, 6], seed=3)
    xc, cm = _padded([9, 5], seed=4)
    B, H, dk = 2, 4, 4
    q = mha.w_q(xq).view(B, 6, H, dk).transpose(1, 2)
    k = mha.w_k(xc).view(B, 9, H, dk).transpose(1, 2)
    v = mha.w_v(xc).view(B, 9, H, dk).transpose(1, 2)
    # explicit formula softmax(QK^T / sqrt(d_k)) V, written independently of the module
    s = q @ k.transpose(-2, -1) / math.sqrt(dk)
    s = s.masked_fill(~cm[:, None, None, :], float("-inf"))
    ref = mha.w_o((torch.softmax(s, -1) @ v).transpose(1, 2).reshape(B, 6, 16))
    torch.testing.assert_close(mha(xq, xc, cm), ref, atol=1e-5, rtol=1e-5)
    ref2 = F.scaled_dot_product_attention(q, k, v, attn_mask=cm[:, None, None, :])
    torch.testing.assert_close(mha(xq, xc, cm), mha.w_o(ref2.transpose(1, 2).reshape(B, 6, 16)), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("cross", [False, True])
def test_attention_block_ignores_padding_content_and_amount(cross):
    blk = AttentionBlock(16, 4, 32, 0.0, cross=cross).eval()
    x, m = _padded([8, 5, 3], seed=5)
    c, cm = (_padded([6, 9, 2], seed=6) if cross else (None, m))
    ref = blk(x, c, cm)
    # garbage in padded positions of the keys/values must not change any real query output
    if cross:
        c2 = c.clone(); c2[~cm] = 1e4
        out = blk(x, c2, cm)
    else:
        x2 = x.clone(); x2[~m] = 1e4
        out = blk(x2, None, m)
    torch.testing.assert_close(out[m], ref[m], atol=1e-5, rtol=1e-5)
    # one sequence alone (no padding) == the same sequence inside the padded batch
    if cross:
        alone = blk(x[1:2, :5], c[1:2, :9], cm[1:2, :9])        # row 1 of c has 9 real frames
    else:
        alone = blk(x[1:2, :5], None, m[1:2, :5])
    torch.testing.assert_close(alone[0], ref[1, :5], atol=1e-5, rtol=1e-5)


def test_fully_masked_context_and_extreme_inputs_stay_finite():
    mha = MultiHeadAttention(16, 4)
    x, _ = _padded([4, 4])
    cm = torch.tensor([[True] * 4, [False] * 4])                 # second row: no valid key
    out = mha(x, x, cm)
    assert torch.isfinite(out).all()
    blk = AttentionBlock(16, 4, 32, 0.1)
    big = (x * 1e4).requires_grad_(True)
    y = blk(big, None, torch.ones(2, 4, dtype=torch.bool))
    y.sum().backward()
    assert torch.isfinite(y).all() and torch.isfinite(big.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_attention_is_finite_under_fp16_autocast():
    blk = AttentionBlock(128, 4, 256, 0.1).cuda()
    x = torch.randn(4, 250, 128, device="cuda") * 30
    m = torch.ones(4, 250, dtype=torch.bool, device="cuda"); m[1, 100:] = False
    with torch.autocast("cuda", dtype=torch.float16):
        y = blk(x, None, m)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("cross", [False, True])
def test_gradient_reaches_every_attention_parameter(cross):
    blk = AttentionBlock(16, 4, 32, 0.0, cross=cross)
    x, m = _padded([6, 4])
    c, cm = _padded([5, 3], seed=9) if cross else (None, m)
    blk(x, c, cm)[m].pow(2).sum().backward()
    for name, p in blk.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name


# ================================================================== E3 model
def _batch(lengths, seed=0, dv=0):
    g = torch.Generator().manual_seed(seed)
    T = max(lengths)
    mask = torch.arange(T)[None] < torch.tensor(lengths)[:, None]
    b = {s: torch.randn(len(lengths), T, d, generator=g) * mask[..., None] for s, d in
         (("hubert", 768), ("mfcc", 39), ("prosody", 4))}
    b["mask"] = mask
    if dv:
        b["disvoice"] = torch.randn(len(lengths), dv, generator=g)
    return b


def test_masked_mean_std_pooling():
    x, m = _padded([5, 2, 7], d=8)
    u = masked_mean_std(x, m, eps=0.0)
    for b, L in enumerate([5, 2, 7]):
        torch.testing.assert_close(u[b, :8], x[b, :L].mean(0))
        torch.testing.assert_close(u[b, 8:], x[b, :L].std(0, unbiased=False))


@pytest.mark.parametrize("n_sa,dv", [(0, 0), (4, 0), (0, 103)])
def test_e3_logits_do_not_depend_on_padding(n_sa, dv):
    model = E3Fusion(n_sa_blocks=n_sa, disvoice_dim=dv).eval()
    b = _batch([40, 17, 25], dv=dv)
    ref = model(b)
    assert ref.shape == (3, 6)
    alone = {k: (v[1:2, :17] if v.dim() > 2 or k == "mask" else v[1:2]) for k, v in b.items()}
    torch.testing.assert_close(model(alone)[0], ref[1], atol=1e-5, rtol=1e-5)
    b2 = {k: v.clone() for k, v in b.items()}
    for s in ("hubert", "mfcc", "prosody"):
        b2[s][~b2["mask"]] = 1e3
    torch.testing.assert_close(model(b2), ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("subset", list(SUBSETS))
def test_stream_subsets_use_only_their_streams(subset):
    model = E3Fusion(SUBSETS[subset]).eval()
    b = _batch([12, 9])
    ref = model(b)
    for s in ("hubert", "mfcc", "prosody"):
        b2 = dict(b); b2[s] = torch.randn_like(b[s])
        changed = not torch.allclose(model(b2), ref)
        assert changed == (s in SUBSETS[subset]), (subset, s)


def test_e3_gradient_reaches_every_parameter():
    model = E3Fusion(n_sa_blocks=2, disvoice_dim=103)
    b = _batch([20, 11], dv=103)
    F.cross_entropy(model(b), torch.tensor([1, 4])).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name


def test_parameter_counts_and_e4_matching():
    d = 128
    m = E3Fusion()
    proj = (768 * d + d + 2 * d) + (39 * d + d + 2 * d) + (4 * d + d + 2 * d)
    fuse = 3 * d * d + d
    head = 2 * d * 256 + 256 + 256 * 6 + 6
    assert n_params(m) == proj + fuse + head and e3_fusion_params(m) == fuse
    n = matched_sa_blocks()
    sa = E3Fusion(n_sa_blocks=n)
    e4_total = n_params(m) - fuse + e4_reference_fusion_params()
    assert n == 4
    assert abs(n_params(sa) - e4_total) / e4_total < 0.05, (n_params(sa), e4_total)


# ================================================================== frame-level data
@dataclass
class FakeStore:
    manifest: pd.DataFrame
    arrays: dict

    def array(self, name):
        return self.arrays[name]


def _fake_store(n=60, seed=0, signal=0.0):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(8, 30, n)
    y = np.arange(n) % 6
    off = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    T = int(lengths.sum())
    man = pd.DataFrame({"frame_offset": off, "n_frames": lengths, "label_id": y})
    frame_row = np.repeat(np.arange(n), lengths)
    t = np.concatenate([np.arange(L) for L in lengths])
    H = rng.standard_normal((T, 13, 768)).astype(np.float16)
    H[:, :, 0] = frame_row[:, None]                                      # encodes (row, frame), exact in fp16
    H[:, :, 1] = t[:, None]
    M = rng.standard_normal((T, 39)).astype(np.float32)
    M[:, 0] = frame_row * 1000 + t
    M[:, 1:7] += signal * np.eye(6)[y[frame_row]]                        # learnable class signal
    P = rng.standard_normal((T, 4)).astype(np.float32)
    P[:, 0] = frame_row * 1000 + t
    P[:, 1] = rng.integers(0, 2, T); P[:, 2] = rng.uniform(0, 1, T)
    dv = rng.standard_normal((n, 103)).astype(np.float32)
    arrays = {"hubert": H, "mfcc": M, "prosody_frames": P, "prosody103_wide": dv,
              "prosody103_wide_imputed": rng.uniform(size=(n, 103)) < 0.1}
    return FakeStore(man, arrays), y


def _sd(store, layer=3, dv=False):
    H = np.ascontiguousarray(store.arrays["hubert"][:, layer, :])
    return SequenceData(store, layer, torch.device("cpu"), hubert=H, need_disvoice=dv)


def test_batches_align_all_streams_frame_by_frame_and_zero_padding():
    store, _ = _fake_store()
    data = _sd(store).fit(np.arange(40))
    rows = np.array([45, 3, 17, 59])
    b = data.batch(rows)
    for j, r in enumerate(rows):
        L = int(store.manifest.n_frames[r]); a = int(store.manifest.frame_offset[r])
        assert int(b["mask"][j].sum()) == L
        for s, raw in (("hubert", store.arrays["hubert"][a:a + L, 3]), ("mfcc", store.arrays["mfcc"][a:a + L]),
                       ("prosody", store.arrays["prosody_frames"][a:a + L])):
            np.testing.assert_array_equal(b[s][j, :L].numpy(), data.std[s].transform(raw))
            assert (b[s][j, L:] == 0).all()
        # the encoded (row, frame) identity is the same in every stream -> frame alignment
        raw_id = {s: b[s][j, :L, 0].numpy() * data.std[s].std_[0].astype(np.float32) + data.std[s].mean_[0].astype(np.float32)
                  for s in ("mfcc", "prosody")}
        np.testing.assert_allclose(raw_id["mfcc"], r * 1000 + np.arange(L), rtol=1e-5)
        np.testing.assert_allclose(raw_id["prosody"], r * 1000 + np.arange(L), rtol=1e-5)
        sh = data.std["hubert"]
        h_id = b["hubert"][j, :L, :2].numpy() * sh.std_[:2].astype(np.float32) + sh.mean_[:2].astype(np.float32)
        np.testing.assert_allclose(h_id[:, 0], r, atol=1e-3)
        np.testing.assert_allclose(h_id[:, 1], np.arange(L), atol=1e-3)


def test_standardisation_uses_training_rows_only_and_passes_through_voicing():
    store, _ = _fake_store()
    tr = np.arange(40)
    data = _sd(store).fit(tr)
    frames = np.concatenate([data.frames_np("mfcc", r) for r in tr]).astype(np.float64)
    np.testing.assert_allclose(data.std["mfcc"].mean_, frames.mean(0))
    assert data.std["prosody"].mean_[1] == 0 and data.std["prosody"].std_[1] == 1
    assert data.std["prosody"].mean_[2] == 0 and data.std["prosody"].std_[2] == 1
    # changing non-training frames does not change the statistics
    store2, _ = _fake_store()
    for k in ("mfcc", "prosody_frames"):
        a = int(store2.manifest.frame_offset[40])
        store2.arrays[k][a:] = 1e6
    data2 = _sd(store2).fit(tr)
    for s in data.std:
        np.testing.assert_array_equal(data.std[s].mean_, data2.std[s].mean_)
    with pytest.raises(RuntimeError):
        _sd(store).batch([0])                                   # not fitted


def test_disvoice_preprocessor_is_fitted_on_training_rows():
    store, _ = _fake_store()
    data = _sd(store, dv=True).fit(np.arange(40), disvoice_indicators=True)
    assert data.dv.shape == (60, 206)
    X = store.arrays["prosody103_wide"]
    imp = store.arrays["prosody103_wide_imputed"]
    med = np.nanmedian(np.where(imp[:40], np.nan, X[:40].astype(np.float64)), axis=0)
    np.testing.assert_allclose(data.dv_pre.fill_, np.where(np.isfinite(med), med, 0))


def test_length_bucketed_batches_cover_each_row_once_and_are_seeded():
    store, _ = _fake_store(n=200)
    data = _sd(store).fit(np.arange(150))
    rows = np.arange(150)
    b1 = data.train_batches(rows, 8, torch.Generator().manual_seed(0), bucket=4)
    b2 = data.train_batches(rows, 8, torch.Generator().manual_seed(0), bucket=4)
    b3 = data.train_batches(rows, 8, torch.Generator().manual_seed(1), bucket=4)
    assert sorted(np.concatenate(b1).tolist()) == rows.tolist()
    assert all(np.array_equal(x, y) for x, y in zip(b1, b2))
    assert not all(np.array_equal(x, y) for x, y in zip(b1, b3))
    # bucketing reduces padding vs random batches of the same size
    pad = lambda bs: sum(len(b) * data.lengths_np[b].max() - data.lengths_np[b].sum() for b in bs)
    rnd = np.random.default_rng(0).permutation(rows)
    assert pad(b1) < pad([rnd[i:i + 8] for i in range(0, 150, 8)])


def test_predict_rows_returns_predictions_in_requested_order():
    store, _ = _fake_store()
    data = _sd(store).fit(np.arange(40))
    model = E3Fusion(("mfcc",)).eval()
    rows = np.array([50, 41, 59, 44, 47])
    p, lg = predict_rows(model, data, rows, batch_size=2, return_logits=True)
    for j, r in enumerate(rows):
        single = model(data.batch([r])).detach().numpy()[0]
        np.testing.assert_allclose(lg[j], single, atol=1e-5)


# ================================================================== training
def _fold(n=120):
    perm = np.random.default_rng(1).permutation(n)
    rows = {"train": np.sort(perm[:72]), "val": np.sort(perm[72:96]), "test": np.sort(perm[96:])}
    y = np.arange(n) % 6
    return FoldRows("SYN", 0, rows, {p: y[r] for p, r in rows.items()}, {p: [p] for p in rows}), y


def _train(data, fr, seed, streams=("mfcc", "prosody"), epochs=30):
    return train_sequence_classifier(lambda: E3Fusion(streams, 32, 0.1, 32).to(data.dev), data,
                                     (fr.rows["train"], fr.labels["train"]), (fr.rows["val"], fr.labels["val"]),
                                     SeqTrainConfig(lr=3e-3, seed=seed, max_epochs=epochs, patience=8, batch_size=16))


def test_sequence_training_is_reproducible_learns_and_keeps_best_epoch():
    store, _ = _fake_store(n=120, signal=1.0)
    fr, _ = _fold()
    data = _sd(store).fit(fr.rows["train"])
    r1, r2, r3 = _train(data, fr, 0), _train(data, fr, 0), _train(data, fr, 1)
    for k in r1["state_dict"]:
        assert torch.equal(r1["state_dict"][k], r2["state_dict"][k])
    assert [h["val_macro_f1"] for h in r1["history"]] == [h["val_macro_f1"] for h in r2["history"]]
    assert any(not torch.equal(r1["state_dict"][k], r3["state_dict"][k]) for k in r1["state_dict"])
    assert r1["best_val_macro_f1"] == max(h["val_macro_f1"] for h in r1["history"]) > 0.5     # signal is learnable
    from sklearn.metrics import f1_score
    p = predict_rows(r1["model"], data, fr.rows["val"])
    assert f1_score(fr.labels["val"], p, average="macro", labels=list(range(6)), zero_division=0) == pytest.approx(r1["best_val_macro_f1"])


def test_trainer_refuses_statistics_fitted_on_other_rows():
    store, _ = _fake_store(n=120)
    fr, _ = _fold()
    data = _sd(store).fit(np.concatenate([fr.rows["train"], fr.rows["val"]]))
    with pytest.raises(RuntimeError):
        _train(data, fr, 0, epochs=1)


# ================================================================== driver: layer selection, isolation
def test_layer_selection_uses_single_layers_max_over_poolings_ties_lower():
    drv = _driver()
    A = [{"rep": f"layer{l}", "pooling": p, "val_macro_f1": 0.5 + 0.01 * (l == 7) + 0.01 * (l == 4) * (p == "mean_std")}
         for l in range(13) for p in ("mean", "mean_std")]
    A += [{"rep": "weighted_all", "pooling": "mean", "val_macro_f1": 0.99}, {"rep": "concat_1_9", "pooling": "mean", "val_macro_f1": 0.98}]
    assert drv.select_layer(A)["layer"] == 4                      # layer 4 ties layer 7 -> lower; weighted ignored
    with pytest.raises(ValueError):
        drv.select_layer(A[:10])


def test_variant_parsing():
    drv = _driver()
    assert drv.parse_variant("concat_HMP")["n_sa_blocks"] == 0
    assert drv.parse_variant("sa_HMP")["n_sa_blocks"] == drv.CFG.n_sa_blocks == 4
    assert drv.parse_variant("concat_HP")["streams"] == ("hubert", "prosody")
    assert drv.parse_variant("concat_HMP_dvind")["disvoice_indicators"] is True
    assert drv.hp_source("concat_MP") == "concat_HMP" and drv.hp_source("sa_HMP") == "sa_HMP"
    with pytest.raises(ValueError):
        drv.parse_variant("concat_X")


def test_selection_runs_cannot_see_test_recordings(monkeypatch):
    drv = _driver()
    store, _ = _fake_store(n=120, signal=1.0)
    fr, _ = _fold()
    monkeypatch.setattr(drv, "device", lambda: torch.device("cpu"))
    monkeypatch.setattr(drv, "CFG", drv.E3Config(max_epochs=3, patience=2, d_model=16, head_hidden=16, d_ffn=32))
    sel = without_test(fr)
    with pytest.raises(KeyError):
        _ = sel.rows["test"]
    clean = drv.run_candidate(_sd(store), sel, "concat_HMP", {"dropout": 0.1, "lr": 1e-3}, 0)
    poisoned, _ = _fake_store(n=120, signal=1.0)
    for r in fr.rows["test"]:                                     # any use of a test frame would produce NaN
        a, L = int(poisoned.manifest.frame_offset[r]), int(poisoned.manifest.n_frames[r])
        poisoned.arrays["hubert"][a:a + L] = np.nan
        poisoned.arrays["mfcc"][a:a + L] = np.nan
        poisoned.arrays["prosody_frames"][a:a + L] = np.nan
    dirty = drv.run_candidate(_sd(poisoned), sel, "concat_HMP", {"dropout": 0.1, "lr": 1e-3}, 0)
    assert clean["val_macro_f1"] == dirty["val_macro_f1"]
    assert [h["train_loss"] for h in clean["history"]] == [h["train_loss"] for h in dirty["history"]]
