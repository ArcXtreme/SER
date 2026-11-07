"""
Stage 5 / E4 tests: cross-attention fusion model (parameter counts and control matching, padding
and key masking, streams used per structure, gradient flow, joint time-permutation invariance =
the documented "no positional encoding" property), driver (grid, epochs cap, variants, identical
construction of the E3 comparators, raising the epoch cap does not change early-stopped runs,
selection never sees test recordings).
"""
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.models.e3_fusion import E3Fusion, e3_fusion_params, e4_reference_fusion_params, matched_sa_blocks, n_params
from src.models.e4_cross_attention import STRUCTURES, E4CrossAttention, e4_fusion_params
from src.training.data import without_test
from src.training.seq_trainer import SeqTrainConfig, train_sequence_classifier

from tests.test_stage5_e3 import _batch, _fake_store, _fold, _sd

ROOT = Path(__file__).resolve().parents[1]


def _driver(name="stage5_e4_run.py"):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ================================================================== model
def test_parameter_counts_match_the_planned_design_and_the_control():
    e4 = E4CrossAttention()
    concat = E3Fusion()
    assert e4_fusion_params(e4) == e4_reference_fusion_params() == 596_992
    assert n_params(e4) == n_params(concat) - e3_fusion_params(concat) + e4_reference_fusion_params() == 769_286
    sa = E3Fusion(n_sa_blocks=matched_sa_blocks())
    assert n_params(sa) == 751_750 and abs(n_params(sa) / n_params(e4) - 1) < 0.05
    assert {s: n_params(E4CrossAttention(s)) for s in ("HM", "HP", "H_MPcat")} == \
        {"HM": 470_022, "HP": 465_542, "H_MPcat": 503_814}


@pytest.mark.parametrize("structure", list(STRUCTURES))
def test_logits_do_not_depend_on_padding_amount_or_content(structure):
    model = E4CrossAttention(structure).eval()
    b = _batch([40, 17, 25])
    ref = model(b)
    assert ref.shape == (3, 6) and torch.isfinite(ref).all()
    alone = {k: (v[1:2, :17] if v.dim() > 2 or k == "mask" else v[1:2]) for k, v in b.items()}
    torch.testing.assert_close(model(alone)[0], ref[1], atol=1e-5, rtol=1e-5)
    b2 = {k: v.clone() for k, v in b.items()}
    for s in ("hubert", "mfcc", "prosody"):
        b2[s][~b2["mask"]] = 1e3                      # padded keys are masked, padded queries are not pooled
    torch.testing.assert_close(model(b2), ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("structure", list(STRUCTURES))
def test_each_structure_uses_exactly_its_streams(structure):
    model = E4CrossAttention(structure).eval()
    b = _batch([12, 9])
    ref = model(b)
    for s in ("hubert", "mfcc", "prosody"):
        b2 = dict(b); b2[s] = torch.randn_like(b[s]) * b["mask"][..., None]
        assert (not torch.allclose(model(b2), ref)) == (s in STRUCTURES[structure]), (structure, s)


@pytest.mark.parametrize("structure", list(STRUCTURES))
def test_gradient_reaches_every_parameter(structure):
    model = E4CrossAttention(structure)
    F.cross_entropy(model(_batch([20, 11])), torch.tensor([1, 4])).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name


def test_no_positional_encoding_joint_time_permutation_invariance():
    """Documented limitation: cross-attention is content-based; frame positions are not encoded."""
    model = E4CrossAttention().eval()
    b = _batch([30, 30])
    perm = torch.randperm(30, generator=torch.Generator().manual_seed(3))
    bp = {k: v[:, perm] for k, v in b.items()}
    torch.testing.assert_close(model(bp), model(b), atol=1e-5, rtol=1e-5)
    # The cross-attention sub-layer itself is content-based: its output does not depend on the
    # ORDER of the context frames (no position information reaches the attention weights).
    x, c, mask = torch.randn(2, 30, 128), torch.randn(2, 30, 128), torch.ones(2, 30, dtype=torch.bool)
    blk = model.h_from_a
    torch.testing.assert_close(blk(x, c[:, perm], mask), blk(x, c, mask), atol=1e-5, rtol=1e-5)
    # ... but the fuse layers concatenate the two outputs frame by frame ([h'(t) ; a'(t)]), so the
    # shared frame grid IS used there: misaligning the acoustic streams w.r.t. HuBERT changes the output.
    bm = dict(b); bm["mfcc"] = b["mfcc"][:, perm]; bm["prosody"] = b["prosody"][:, perm]
    assert not torch.allclose(model(bm), model(b), atol=1e-4)


# ================================================================== driver
def test_driver_grid_cap_and_variants():
    drv = _driver()
    assert drv.CFG.grid_lr == (1e-3, 3e-4, 1e-4, 3e-5) and drv.CFG.grid_dropout == (0.2, 0.4)
    assert drv.CFG.max_epochs == 150 and drv.CFG.patience == 10 and drv.CFG.final_seeds == (0, 1, 2)
    c = drv.candidates()
    assert len(c) == 8 and c[0] == {"dropout": 0.2, "lr": 1e-3} and c[3] == {"dropout": 0.2, "lr": 3e-5}
    # E3's candidate order is a sub-sequence of the new order (same tie-breaking among E3 cells)
    e3_order = [(0.2, 1e-3), (0.2, 3e-4), (0.4, 1e-3), (0.4, 3e-4)]
    assert [(h["dropout"], h["lr"]) for h in c if h["lr"] in (1e-3, 3e-4)] == e3_order
    assert drv.CFG.tuned == ("xattn_HMP", "sa_HMP", "concat_HMP", "concat_H")
    assert drv.parse_variant("xattn_H_MPcat")["structure"] == "H_MPcat"
    assert drv.parse_variant("sa_HMP")["n_sa_blocks"] == 4 and drv.parse_variant("concat_H")["streams"] == ("hubert",)
    assert drv.hp_source("xattn_HP") == "xattn_HMP" and drv.hp_source("concat_H") == "concat_H"
    for bad in ("xattn_X", "concat_MP", "sa_H"):
        with pytest.raises(ValueError):
            drv.parse_variant(bad)
    order = drv.search_order()
    assert len(order) == 32 and len({(v, h["dropout"], h["lr"]) for v, h in order}) == 32
    assert all(v in ("concat_HMP", "sa_HMP") and h["lr"] in (1e-3, 3e-4) for v, h in order[:8])


def test_e3_comparators_are_built_exactly_as_in_e3(monkeypatch):
    e4d, e3d = _driver(), _driver("stage5_e3_run.py")
    for d in (e4d, e3d):
        monkeypatch.setattr(d, "device", lambda: torch.device("cpu"))
    hp = {"dropout": 0.4, "lr": 3e-4}
    for v in ("concat_HMP", "sa_HMP", "concat_H"):
        torch.manual_seed(0); a = e4d.build_model(e4d.parse_variant(v), hp)
        torch.manual_seed(0); b = e3d.build_model(e3d.parse_variant(v), hp)
        sa, sb = a.state_dict(), b.state_dict()
        assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa), v


def test_raising_the_epoch_cap_does_not_change_an_early_stopped_run():
    store, _ = _fake_store(n=120, signal=1.0)
    fr, _ = _fold()
    data = _sd(store).fit(fr.rows["train"])
    run = lambda cap: train_sequence_classifier(  # noqa: E731
        lambda: E3Fusion(("mfcc", "prosody"), 16, 0.1, 16), data, (fr.rows["train"], fr.labels["train"]),
        (fr.rows["val"], fr.labels["val"]), SeqTrainConfig(lr=3e-3, seed=0, max_epochs=cap, patience=3, batch_size=16))
    a, b = run(60), run(150)
    assert a["epochs_run"] < 60, "test needs a run that stops early"
    assert a["history"] == [dict(h, seconds=a["history"][i]["seconds"]) for i, h in enumerate(b["history"])]
    assert all(torch.equal(a["state_dict"][k], b["state_dict"][k]) for k in a["state_dict"])


def test_selection_runs_cannot_see_test_recordings(monkeypatch):
    drv = _driver()
    store, _ = _fake_store(n=120, signal=1.0)
    fr, _ = _fold()
    monkeypatch.setattr(drv, "device", lambda: torch.device("cpu"))
    monkeypatch.setattr(drv, "CFG", drv.E4Config(max_epochs=3, patience=2, d_model=16, head_hidden=16, d_ffn=32))
    sel = without_test(fr)
    hp = {"dropout": 0.1, "lr": 1e-3}
    clean = drv.run_candidate(_sd(store), sel, "xattn_HMP", hp, 0)
    poisoned, _ = _fake_store(n=120, signal=1.0)
    for r in fr.rows["test"]:                     # any use of a test frame would produce NaN
        a, L = int(poisoned.manifest.frame_offset[r]), int(poisoned.manifest.n_frames[r])
        for k in ("hubert", "mfcc", "prosody_frames"):
            poisoned.arrays[k][a:a + L] = np.nan
    dirty = drv.run_candidate(_sd(poisoned), sel, "xattn_HMP", hp, 0)
    assert clean["val_macro_f1"] == dirty["val_macro_f1"]
    assert [h["train_loss"] for h in clean["history"]] == [h["train_loss"] for h in dirty["history"]]
    assert clean["hit_max_epochs"] in (True, False) and clean["n_params_fusion"] > 0


def test_config_file_must_match_the_protocol(monkeypatch, tmp_path):
    drv = _driver()
    (tmp_path / "e4_cross_attention.json").write_text('{"grid_lr": [1]}')
    monkeypatch.setattr(drv, "PATHS", SimpleNamespace(configs=tmp_path, metadata=ROOT / "metadata"))
    monkeypatch.setattr("sys.argv", ["stage5_e4_run.py", "--datasets", "RAVDESS"])
    with pytest.raises(SystemExit, match="differs"):
        drv.main()


def test_run_hash_of_the_protocol_is_unchanged_so_pilot_runs_are_reused():
    """The fold-0 pilot S files were written with this exact E4Config; any change would retrain them."""
    drv = _driver()
    saved = json.loads((ROOT / "configs" / "e4_cross_attention.json").read_text())
    assert saved == json.loads(json.dumps(asdict(drv.CFG)))
    for ds, h in (("RAVDESS", "e584bddc7755"), ("CREMA-D", "6c711ebab212")):
        p = ROOT / "results" / "E4" / f"pilot_{ds}.json"
        if p.exists():
            pil = json.loads(p.read_text())
            assert pil["run_hash"] == h
            store = SimpleNamespace(meta={"cache_id": "v1_efe9081789"})
            assert drv.run_hash(ds, store, pil["layer_info"]["layer"]) == h


# ================================================================== phase D, audit, controls (synthetic end to end)
def _tiny_cfg(drv):
    return drv.E4Config(max_epochs=3, patience=2, d_model=16, head_hidden=16, d_ffn=32, grid_lr=(1e-3, 3e-4),
                        grid_dropout=(0.1,), final_seeds=(0, 1))


@pytest.fixture
def synthetic_run(monkeypatch, tmp_path):
    aud = _driver("stage5_e4_audit.py")
    drv = aud.drv                                   # the audit checks with the same driver instance
    monkeypatch.setattr(drv, "device", lambda: torch.device("cpu"))
    monkeypatch.setattr(drv, "CFG", _tiny_cfg(drv))
    monkeypatch.setattr(drv, "PATHS", SimpleNamespace(results=tmp_path / "results", checkpoints=tmp_path / "ck",
                                                      project=tmp_path))
    monkeypatch.setattr(drv, "fold_layer", lambda ds, f: {"layer": 3, "source": "synthetic", "source_sha256": "x"})
    monkeypatch.setattr(drv, "run_hash", lambda ds, store, layer: "h_syn")
    store, _ = _fake_store(n=120, signal=1.0)
    fr, _ = _fold()
    data = _sd(store)
    dc = SimpleNamespace(get=lambda layer: data)
    calls = {"n": 0}
    real = drv.train_sequence_classifier

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)
    monkeypatch.setattr(drv, "train_sequence_classifier", counting)
    drv.run_fold("SYN", store, dc, fr, check_e3=False)
    return SimpleNamespace(aud=aud, drv=drv, store=store, fr=fr, data=data, dc=dc, calls=calls,
                           d=tmp_path / "results" / "E4" / "SYN" / "fold0")


def test_phase_d_writes_frozen_selection_first_and_reloaded_checkpoints_reproduce(synthetic_run):
    s = synthetic_run
    n_s = len(s.drv.CFG.tuned) * 2
    n_d = len(s.drv.CFG.tuned + s.drv.CFG.ablations) * 2
    assert s.calls["n"] == n_s + n_d                              # S once per candidate, D once per (variant, seed)
    assert len(list(s.d.glob("S_*.json"))) == n_s and len(list(s.d.glob("D_*.json"))) == n_d
    sel_t = (s.d / "selection.json").stat().st_mtime
    assert all(sel_t <= p.stat().st_mtime for p in s.d.glob("D_*.json"))
    D = json.loads((s.d / "D_xattn_HP_seed1.json").read_text())
    assert D["hp"] == json.loads((s.d / "selection.json").read_text())["selected"]["xattn_HMP"]
    assert D["test_rows"] == s.fr.rows["test"].tolist() and len(D["test_pred"]) == len(s.fr.rows["test"])
    # second call: everything reused, nothing retrained, no file rewritten
    mt = {p: p.stat().st_mtime_ns for p in s.d.glob("*.json")}
    s.drv.run_fold("SYN", s.store, s.dc, s.fr, check_e3=False)
    assert s.calls["n"] == n_s + n_d and mt == {p: p.stat().st_mtime_ns for p in s.d.glob("*.json")}


def test_audit_passes_on_a_clean_run_and_detects_tampering(synthetic_run):
    s = synthetic_run
    ok, det = s.aud.audit_fold(0, s.d, s.fr, s.data, s.store, 3, None, cfg=s.drv.CFG)
    assert all(ok.values()), {k: v for k, v in ok.items() if not v}
    assert det["std_max_abs_diff"] and max(det["std_max_abs_diff"].values()) == 0.0
    p = s.d / "D_concat_H_seed0.json"
    r = json.loads(p.read_text()); r["test_pred"][0] = (r["test_pred"][0] + 1) % 6; p.write_text(json.dumps(r))
    ok2, det2 = s.aud.audit_fold(0, s.d, s.fr, s.data, s.store, 3, None, cfg=s.drv.CFG)
    assert not ok2["repredict"] and det2["repredict_mismatch"] == [[0, "concat_H", 0]]
    assert not s.aud.audit_fold(0, s.d, s.fr, s.data, s.store, 5, None, cfg=s.drv.CFG)[0]["layer"]


def test_e3_discrepancy_stops_before_any_test_prediction(monkeypatch, tmp_path):
    drv = _driver()
    monkeypatch.setattr(drv, "device", lambda: torch.device("cpu"))
    monkeypatch.setattr(drv, "CFG", _tiny_cfg(drv))
    monkeypatch.setattr(drv, "PATHS", SimpleNamespace(results=tmp_path, checkpoints=tmp_path / "ck", project=tmp_path))
    monkeypatch.setattr(drv, "fold_layer", lambda ds, f: {"layer": 3})
    monkeypatch.setattr(drv, "run_hash", lambda ds, store, layer: "h")
    monkeypatch.setattr(drv, "e3_reproduction", lambda ds, fold, S: [{"variant": "sa_HMP", "hp": {}, "exact": False}] * 4)
    store, _ = _fake_store(n=120, signal=1.0)
    fr, _ = _fold()
    data = _sd(store)
    with pytest.raises(RuntimeError, match="E3 reproduction discrepancy"):
        drv.run_fold("SYN", store, SimpleNamespace(get=lambda l: data), fr)
    assert not list((tmp_path / "E4" / "SYN" / "fold0").glob("D_*.json")) and not (tmp_path / "ck").exists()


def test_controls_run_on_saved_checkpoints(synthetic_run):
    s = synthetic_run
    ctl = _driver("stage5_e4_controls.py")
    ctl.drv = s.drv
    sel = json.loads((s.d / "selection.json").read_text())
    perm = ctl.permutation_fold(s.data, s.fr, sel, 0)
    assert perm["share_train_labels_unchanged"] < 0.5 and 0 <= perm["test_macro_f1"] <= 1
    rows = ctl.swap_fold(s.data, s.fr, s.d, 0)
    assert len(rows) == 2 * len(s.drv.CFG.final_seeds)
    for r in rows:
        assert r["test"]["macro_f1_intact"] == json.loads((s.d / f"D_{r['variant']}_seed{r['seed']}.json").read_text())["test"]["macro_f1"]


# ================================================================== swap helpers, statistics
def test_derangement_has_no_fixed_points_and_is_seeded():
    drv = _driver()
    p = drv.derangement(50, 0)
    assert sorted(p) == list(range(50)) and not np.any(p == np.arange(50))
    assert np.array_equal(p, drv.derangement(50, 0)) and not np.array_equal(p, drv.derangement(50, 1))


def test_swapped_batch_identity_reproduces_batch_and_stretches_donors():
    drv = _driver()
    store, _ = _fake_store(n=40)
    rows = np.arange(40)
    data = _sd(store).fit(rows[:30])
    b, bs = data.batch(rows[:6]), drv.swapped_batch(data, rows[:6], rows[:6])
    for k in ("hubert", "mfcc", "prosody", "mask"):
        assert torch.equal(b[k], bs[k]), k
    donors = rows[[6, 7, 8, 9, 10, 11]]
    sw = drv.swapped_batch(data, rows[:6], donors)
    assert torch.equal(sw["hubert"], b["hubert"]) and torch.equal(sw["mask"], b["mask"])
    raw = lambda r, t: (store.arrays["mfcc"][int(store.manifest.frame_offset[r]) + t, 0])  # noqa: E731  encodes row*1000+t
    for j, (r, dn) in enumerate(zip(rows[:6], donors)):
        L, Ld = int(store.manifest.n_frames[r]), int(store.manifest.n_frames[dn])
        z = data.std["mfcc"]
        got = sw["mfcc"][j, :L, 0].numpy() * z.std_[0] + z.mean_[0]
        want = [raw(dn, min(t * Ld // L, Ld - 1)) for t in range(L)]
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-2)
        assert float(sw["mfcc"][j, L:].abs().sum()) == 0.0


def test_holm_and_decision_rule():
    drv = _driver()
    h = drv.holm({"a": 0.03125, "b": 0.0625, "c": 0.5})
    assert h == {"a": pytest.approx(0.09375), "b": pytest.approx(0.125), "c": pytest.approx(0.5)}

    def S(mean, nb, n, sd):
        p = {name: {"mean": mean, "n_better": nb, "n": n, "sign_flip_p": 0.1} for name in drv.KEY_COMPARISONS}
        return {"paired_test_macro_f1": p, "holm_key_comparisons_macro_f1": {k: 0.3 for k in p},
                "variants": {"xattn_HMP": {"seed_std_macro_f1_mean_within_fold": sd}}}
    yes = drv.decision({"RAVDESS": S(0.02, 5, 6, 0.01), "CREMA-D": S(0.02, 4, 5, 0.01)})
    assert all(v["verdict"] == "E4 improves" for v in yes.values())
    for bad in ({"RAVDESS": S(0.02, 4, 6, 0.01), "CREMA-D": S(0.02, 4, 5, 0.01)},     # too few RAVDESS folds
                {"RAVDESS": S(0.02, 6, 6, 0.03), "CREMA-D": S(0.02, 5, 5, 0.01)},     # below seed noise
                {"RAVDESS": S(0.02, 6, 6, 0.01), "CREMA-D": S(-0.01, 4, 5, 0.001)}):  # negative on one dataset
        assert all(v["verdict"] == "no consistent improvement" for v in drv.decision(bad).values())
