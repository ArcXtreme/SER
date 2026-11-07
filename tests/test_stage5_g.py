"""
G-study tests (report section 8.6 c): E2 trainer default path unchanged; "no clipping" in the frame trainer is exact;
frame variants are E3Fusion(("hubert",)) with exactly one switch changed (identical initial weights); parameter
counts; B0 pooled input; Z-W nesting; derived features (UM, pooled UM, per-layer sums) vs direct recomputation;
streamed loader == in-memory reference, prefetch == synchronous, errors surface; test frames (NaN-poisoned) never
enter a Z-W selection run; driver grid, file sharing, scope and refusals.
"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.features.g_derived import compute_derived, verify_derived
from src.models.e3_fusion import E3Fusion, masked_mean_std, n_params
from src.models.g_frame import LayerWeightedFrameModel, PooledFrameHead, build_frame_model
from src.training.data import without_test
from src.training.preprocessing import FrameStandardizer
from src.training.seq_trainer import SeqTrainConfig, train_sequence_classifier
from src.training.streamed_layers import StreamedLayers
from src.training.trainer import TrainConfig, train_classifier

from tests.test_stage5_e3 import _batch, _fold

ROOT = Path(__file__).resolve().parents[1]


def _driver(name):
    s = importlib.util.spec_from_file_location(name.replace(".py", ""), ROOT / "scripts" / name)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


# ================================================================== trainers
def test_e2_trainer_default_path_never_clips(monkeypatch):
    X = torch.randn(60, 8); y = np.arange(60) % 6
    fn = lambda: torch.nn.Sequential(torch.nn.Linear(8, 6))  # noqa: E731
    calls = {"n": 0}
    real = torch.nn.utils.clip_grad_norm_

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    a = train_classifier(fn, (X[:40], y[:40]), (X[40:], y[40:]), TrainConfig(max_epochs=3))
    assert calls["n"] == 0
    b = train_classifier(fn, (X[:40], y[:40]), (X[40:], y[40:]), TrainConfig(max_epochs=3, grad_clip=None))
    assert all(torch.equal(a["state_dict"][k], b["state_dict"][k]) for k in a["state_dict"])
    train_classifier(fn, (X[:40], y[:40]), (X[40:], y[40:]), TrainConfig(max_epochs=1, grad_clip=1.0))
    assert calls["n"] > 0


def test_infinite_clip_norm_leaves_gradients_bit_identical():
    m = build_frame_model(16, True, 0.1, 16)
    b = _batch([9, 5]); b["hubert"] = b["hubert"][..., :768]
    F.cross_entropy(m(b), torch.tensor([1, 3])).backward()
    before = [p.grad.clone() for p in m.parameters()]
    torch.nn.utils.clip_grad_norm_(m.parameters(), float("inf"), error_if_nonfinite=True)
    assert all(torch.equal(g0, p.grad) for g0, p in zip(before, m.parameters()))


# ================================================================== models
def test_frame_variants_change_exactly_one_switch_and_param_counts(monkeypatch):
    e4 = _driver("stage5_e4_run.py")
    monkeypatch.setattr(e4, "device", lambda: torch.device("cpu"))
    torch.manual_seed(0); z_e4 = e4.build_model(e4.parse_variant("concat_H"), {"dropout": 0.2, "lr": 1e-3})
    torch.manual_seed(0); z = build_frame_model(128, True, 0.2, 256)
    assert z_e4.state_dict().keys() == z.state_dict().keys()
    assert all(torch.equal(z_e4.state_dict()[k], z.state_dict()[k]) for k in z.state_dict())
    torch.manual_seed(0); znoln = build_frame_model(128, False, 0.2, 256)
    sz, sn = z.state_dict(), znoln.state_dict()
    assert set(sz) - set(sn) == {"proj.hubert.ln.weight", "proj.hubert.ln.bias"} and all(torch.equal(sz[k], sn[k]) for k in sn)
    torch.manual_seed(0); zw = LayerWeightedFrameModel(128, True, 0.2, 256)
    assert all(torch.equal(zw.core.state_dict()[k], sz[k]) for k in sz)          # same initial weights as Z
    counts = {"Z": n_params(z), "ZnoLN": n_params(znoln), "Z768": n_params(build_frame_model(768, True)),
              "Z768noLN": n_params(build_frame_model(768, False)), "ZW": n_params(zw),
              "B0": n_params(PooledFrameHead(np.zeros(1536), np.ones(1536), 256, 0.2))}
    assert counts == {"Z": 182_534, "ZnoLN": 182_278, "Z768": 1_577_734, "Z768noLN": 1_576_198, "ZW": 182_547, "B0": 395_014}


def test_b0_uses_standardised_pooled_statistics_and_ignores_padding():
    b = _batch([12, 7, 9]); b["hubert"] = b["hubert"][..., :768]
    u = masked_mean_std(b["hubert"], b["mask"], eps=0.0).numpy().astype(np.float64)
    mu, sd = u.mean(0), u.std(0) + 1.0
    m = PooledFrameHead(mu, sd, 32, 0.0).eval()
    torch.testing.assert_close(m.standardised_input(b).double(), torch.as_tensor((u - mu) / sd), rtol=1e-5, atol=1e-5)
    b2 = {k: v.clone() for k, v in b.items()}; b2["hubert"][~b2["mask"]] = 1e3
    torch.testing.assert_close(m(b2), m(b))
    for r, L in enumerate([12, 7, 9]):                                            # population std, no eps
        x = b["hubert"][r, :L].double()
        torch.testing.assert_close(torch.as_tensor(u[r, 768:]), x.std(0, unbiased=False), rtol=1e-5, atol=1e-6)


def test_layer_weighting_nests_z_when_weights_are_one_hot():
    torch.manual_seed(0)
    zw = LayerWeightedFrameModel(16, True, 0.0, 16).eval()
    B, T = 2, 7
    x13 = torch.randn(B, T, 13, 768); mask = torch.ones(B, T, dtype=torch.bool); mask[1, 5:] = False
    with torch.no_grad():
        zw.mix.logits.copy_(torch.full((13,), -1e4)); zw.mix.logits[6] = 1e4
    out = zw({"hubert13": x13, "mask": mask})
    ref = zw.core({"hubert": x13[:, :, 6], "mask": mask})
    torch.testing.assert_close(out, ref)
    with torch.no_grad():
        zw.mix.logits.zero_()                                                     # init = uniform average
    torch.testing.assert_close(zw({"hubert13": x13, "mask": mask}), zw.core({"hubert": x13.mean(2), "mask": mask}), rtol=1e-5, atol=1e-5)


# ================================================================== derived features + streaming
def _synthetic_cache(tmp_path, n=120, seed=0, poison_rows=()):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(3, 9, n)
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    X = (rng.standard_normal((int(lengths.sum()), 13, 768)) * 2 + 0.5).astype(np.float16)
    y = np.arange(n) % 6
    X[:, 3, :6] += (3 * np.eye(6)[y[np.repeat(np.arange(n), lengths)]]).astype(np.float16)   # learnable signal in layer 3
    for r in poison_rows:
        X[offsets[r]: offsets[r] + lengths[r]] = np.nan
    p = tmp_path / f"hubert_{seed}_{len(poison_rows)}.npy"
    np.save(p, X)
    return p, X, offsets, lengths, y


def test_derived_features_equal_direct_recomputation(tmp_path):
    p, X, off, L, _ = _synthetic_cache(tmp_path, n=40)
    D = compute_derived(p, off, L, with_layer_sums=True, chunk=37)               # chunks split recordings
    avg = X.astype(np.float32).mean(1)
    assert np.array_equal(D["um_frames"], avg.astype(np.float16))
    for r in range(40):
        a = avg[off[r]: off[r] + L[r]].astype(np.float64)
        np.testing.assert_allclose(D["um_pooled"][r, 0], np.concatenate([a.mean(0), a.std(0)]), rtol=1e-6, atol=1e-6)
        x = X[off[r]: off[r] + L[r]].astype(np.float64)
        np.testing.assert_allclose(D["layer_sum"][r], x.sum(0), rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(D["layer_sumsq"][r], (x * x).sum(0), rtol=1e-12, atol=1e-9)
    assert verify_derived(p, off, L, D, range(40))["pass"]
    with pytest.raises(ValueError):
        compute_derived(p, off[::-1].copy(), L, False)


def _streamed(tmp_path, prefetch=True, **kw):
    p, X, off, L, y = _synthetic_cache(tmp_path, **kw)
    D = compute_derived(p, off, L, with_layer_sums=True)
    return StreamedLayers(p, off, L, D["layer_sum"], D["layer_sumsq"], torch.device("cpu"), prefetch=prefetch), X, off, L, y


def test_streamed_batches_equal_in_memory_reference_and_training_statistics(tmp_path):
    s, X, off, L, _ = _streamed(tmp_path)
    train = np.arange(0, 120, 2)
    s.fit(train)
    for l in (0, 6, 12):
        z = FrameStandardizer().fit([X[off[r]: off[r] + L[r], l].astype(np.float64) for r in train])
        np.testing.assert_allclose(s.mean_[l], z.mean_, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(s.std_[l], z.std_, rtol=1e-9, atol=1e-12)
    rows = np.array([5, 17, 3, 44])
    b = s.batch(rows)
    for j, r in enumerate(rows):
        ref = (X[off[r]: off[r] + L[r]].astype(np.float32) - s.mean_.astype(np.float32)) / s.std_.astype(np.float32)
        np.testing.assert_allclose(b["hubert13"][j, :L[r]].numpy(), ref, rtol=1e-6, atol=1e-6)
        assert float(b["hubert13"][j, L[r]:].abs().sum()) == 0.0 and int(b["mask"][j].sum()) == L[r]


def test_prefetch_equals_synchronous_reading_and_sequence_batching(tmp_path):
    from tests.test_stage5_e3 import _fake_store, _sd
    a, *_ = _streamed(tmp_path, prefetch=True)
    b, *_ = _streamed(tmp_path, prefetch=False)
    train = np.arange(0, 120, 2)
    a.fit(train); b.fit(train)
    la = a.train_batches(train, 16, torch.Generator().manual_seed(3))
    lb = b.train_batches(train, 16, torch.Generator().manual_seed(3))
    assert all(np.array_equal(x, y) for x, y in zip(la, lb))
    for rows in la:
        assert torch.equal(a.batch(rows)["hubert13"], b.batch(rows)["hubert13"])
    assert a.io["prefetched"] == len(la) and a.io["sync_reads"] == 0
    store, _ = _fake_store(n=120)                                                # same batching algorithm as SequenceData
    sd = _sd(store)
    sd.lengths_np = a.lengths_np
    lc = sd.train_batches(train, 16, torch.Generator().manual_seed(3))
    assert all(np.array_equal(x, y) for x, y in zip(la, lc))
    ev = a.eval_batches(np.arange(60, 120), 32)
    a.batch(ev[1])                                                               # out of order -> synchronous fallback
    assert a.io["sync_reads"] == 1
    a.close(); b.close()


def test_prefetch_errors_surface(tmp_path):
    s, *_ = _streamed(tmp_path)
    s.fit(np.arange(60))
    s.path = tmp_path / "missing.npy"
    batches = s.eval_batches(np.arange(10), 4)
    with pytest.raises(OSError):
        s.batch(batches[0])


def test_zw_selection_never_touches_test_recordings(tmp_path, monkeypatch):
    fr, _ = _fold()
    clean, *_ = _streamed(tmp_path)
    poisoned, *_ = _streamed(tmp_path, poison_rows=tuple(fr.rows["test"]))
    sel = without_test(fr)
    cfg = SeqTrainConfig(lr=3e-3, seed=0, max_epochs=3, patience=2, batch_size=16)
    run = lambda data: train_sequence_classifier(lambda: LayerWeightedFrameModel(16, True, 0.1, 16), data.fit(sel.rows["train"]),  # noqa: E731
                                                 (sel.rows["train"], sel.labels["train"]), (sel.rows["val"], sel.labels["val"]), cfg)
    a, b = run(clean), run(poisoned)
    assert [h["train_loss"] for h in a["history"]] == [h["train_loss"] for h in b["history"]]
    assert a["best_val_macro_f1"] == b["best_val_macro_f1"] and np.isfinite(a["history"][-1]["train_loss"])


# ================================================================== driver
def test_driver_grid_sharing_scope_and_refusals(monkeypatch, tmp_path):
    g = _driver("stage5_g_run.py")
    g8, b32 = g.candidates("G8"), g.candidates("B32")
    assert len(g8) == 8 and len(b32) == 32 and b32[:8] == g8
    assert g8[0] == {"hidden": 256, "class_weighted": True, "dropout": 0.2, "lr": 1e-3}
    assert {(c["hidden"], c["class_weighted"]) for c in b32} == {(256, True), (256, False), (512, True), (512, False)}
    assert g.ARMS["P0B32"]["key"] == "P0" and g.ARMS["ZB32"]["key"] == "Z"
    assert g.CFG.max_epochs == 150 and g.CFG.grad_clip == 1.0 and g.CFG.zw_datasets == ("CREMA-D",)
    assert sum(len(v) for v in g.GROUPS.values()) == 14 and set(g.ARMS) == {a for v in g.GROUPS.values() for a in v}
    with pytest.raises(SystemExit, match="CREMA-D"):
        g.pilot("RAVDESS", "zw")
    monkeypatch.setattr("sys.argv", ["x", "--dataset", "RAVDESS", "--group", "pooled"])
    with pytest.raises(SystemExit, match="phase D"):
        g.main()
    monkeypatch.setattr(g, "PATHS", SimpleNamespace(configs=tmp_path))
    (tmp_path / "g_study.json").write_text('{"grid_lr": [1]}')
    monkeypatch.setattr("sys.argv", ["x", "--pilot", "--dataset", "RAVDESS", "--group", "pooled"])
    with pytest.raises(SystemExit, match="differs"):
        g.main()


def test_same_run_detects_any_difference():
    g = _driver("stage5_g_run.py")
    r = {"val_macro_f1": 0.5, "best_epoch": 3, "epochs_run": 14, "val": {"confusion": [[1]]},
         "history": [{"val_macro_f1": 0.4, "train_loss": 1.0}, {"val_macro_f1": 0.5, "train_loss": 0.9}]}
    assert g.same_run(r, json.loads(json.dumps(r)))["exact"]
    for mut in (lambda x: x.update(best_epoch=4), lambda x: x["history"][0].update(train_loss=1.0000001),
                lambda x: x["val"].update(confusion=[[2]])):
        x = json.loads(json.dumps(r)); mut(x)
        assert not g.same_run(x, r)["exact"]
