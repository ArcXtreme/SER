"""
Stage 5 / E2 tests: layer extraction, masked pooling, padding invariance, head dimensions,
leakage, reproducible training, test isolation during selection, checkpoint save/load.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import f1_score

from src.features.batching import masked_mean, masked_std, pad_sequences
from src.models.e2_hubert import HubertHead, LayerWeighting, n_params, pooled_input
from src.training.data import FoldRows, without_test
from src.training.trainer import TrainConfig, class_weights, device, predict, train_classifier

RNG = np.random.default_rng(0)
ROOT = Path(__file__).resolve().parents[1]


def _driver():
    spec = importlib.util.spec_from_file_location("e2drv", ROOT / "scripts" / "stage5_e2_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic(n=180, signal=1.5):
    """pooled-cache-like array (n, 13, 1536) where layer 9 carries the label."""
    y = np.repeat(np.arange(6), n // 6)
    pooled = RNG.standard_normal((n, 13, 1536)).astype(np.float32)
    pooled[:, 9, :6] += signal * np.eye(6)[y]
    perm = RNG.permutation(n)
    rows = {"train": perm[:100], "val": perm[100:140], "test": perm[140:]}
    fr = FoldRows("SYN", 0, rows, {p: y[r] for p, r in rows.items()}, {p: [p] for p in rows})
    return pooled, y, fr


# ------------------------------------------------------------------ layer extraction
def test_pooled_input_extracts_requested_layer_and_pooling():
    pooled = np.zeros((4, 13, 1536), np.float32)
    for l in range(13):
        pooled[:, l, :768], pooled[:, l, 768:] = l, 100 + l           # mean part = l, std part = 100 + l
    rows = np.array([2, 0])
    for l in (1, 9, 12):
        x = pooled_input(pooled, rows, f"layer{l}", "mean_std")
        assert x.shape == (2, 1536) and (x[:, :768] == l).all() and (x[:, 768:] == 100 + l).all()
        xm = pooled_input(pooled, rows, f"layer{l}", "mean")
        assert xm.shape == (2, 768) and (xm == l).all()
    c = pooled_input(pooled, rows, "concat_1_9", "mean")
    assert c.shape == (2, 1536) and (c[:, :768] == 1).all() and (c[:, 768:] == 9).all()
    assert pooled_input(pooled, rows, "weighted_all", "mean_std").shape == (2, 13, 1536)


# ------------------------------------------------------------------ masked pooling
def test_masked_mean_std_match_unpadded_statistics_and_ignore_padding():
    seqs = [RNG.standard_normal((n, 768)).astype(np.float32) for n in (5, 61, 17, 2)]
    x, mask = pad_sequences(seqs)
    mu, sd = masked_mean(x, mask), masked_std(x, mask)
    for b, s in enumerate(seqs):
        np.testing.assert_allclose(mu[b].numpy(), s.mean(0), atol=1e-5)
        np.testing.assert_allclose(sd[b].numpy(), s.std(0), atol=1e-4)      # population std (ddof 0)
    x2 = x.clone(); x2[~mask] = 1e4                                          # garbage in padding
    assert torch.equal(masked_mean(x2, mask), mu)
    torch.testing.assert_close(masked_std(x2, mask), sd)
    # padding amount does not matter: extra all-padding columns change nothing
    x3 = torch.cat([x, torch.full((4, 50, 768), -7.0)], 1)
    m3 = torch.cat([mask, torch.zeros(4, 50, dtype=torch.bool)], 1)
    torch.testing.assert_close(masked_mean(x3, m3), mu)


# ------------------------------------------------------------------ model dimensions
@pytest.mark.parametrize("weighted", [False, True])
def test_head_outputs_six_classes_and_parameter_count(weighted):
    d, h = 1536, 256
    m = HubertHead(d, h, 0.2, weighted_layers=weighted)
    x = torch.randn(7, 13, d) if weighted else torch.randn(7, d)
    assert m(x).shape == (7, 6)
    assert n_params(m) == h * (d + 1) + 6 * (h + 1) + (13 if weighted else 0)


def test_layer_weighting_starts_as_uniform_average():
    x = torch.randn(3, 13, 8)
    torch.testing.assert_close(LayerWeighting()(x), x.mean(1))


def test_class_weights_from_training_labels():
    y = np.array([0] * 10 + [1] * 20 + [2] * 20 + [3] * 20 + [4] * 20 + [5] * 20)
    w = class_weights(y)
    assert w[0] == pytest.approx(110 / (6 * 10)) and w[1] == pytest.approx(110 / (6 * 20))
    with pytest.raises(ValueError):
        class_weights(np.zeros(10, int))


# ------------------------------------------------------------------ training
def _train(seed, pooled, fr):
    X = torch.from_numpy(pooled[:, 9]).to(device())
    r = train_classifier(lambda: HubertHead(1536, 64, 0.2).to(device()), (X[fr.rows["train"]], fr.labels["train"]),
                         (X[fr.rows["val"]], fr.labels["val"]), TrainConfig(seed=seed, max_epochs=15, patience=5))
    return r["model"], r, X


def test_training_is_reproducible_and_early_stopping_keeps_best_epoch():
    pooled, _, fr = _synthetic()
    m1, r1, X = _train(0, pooled, fr)
    m2, r2, _ = _train(0, pooled, fr)
    for k in r1["state_dict"]:
        assert torch.equal(r1["state_dict"][k], r2["state_dict"][k])
    assert r1["history"] == r2["history"]
    _, r3, _ = _train(1, pooled, fr)
    assert any(not torch.equal(r1["state_dict"][k], r3["state_dict"][k]) for k in r1["state_dict"])
    # returned model = best epoch: its val F1 equals the recorded best
    f1 = f1_score(fr.labels["val"], predict(m1, X[fr.rows["val"]]), average="macro", labels=list(range(6)), zero_division=0)
    assert f1 == pytest.approx(r1["best_val_macro_f1"])
    assert r1["best_val_macro_f1"] == max(h["val_macro_f1"] for h in r1["history"])


# ------------------------------------------------------------------ isolation + leakage
def test_selection_code_cannot_see_test_partition():
    pooled, _, fr = _synthetic()
    sel = without_test(fr)
    with pytest.raises(KeyError):
        _ = sel.rows["test"]
    drv = _driver()
    cand = {"rep": "layer9", "pooling": "mean_std", "hidden": 64, "dropout": 0.2, "lr": 1e-3, "class_weighted": True}
    clean = drv.run_candidate(pooled, sel, cand, seed=0)
    poisoned = pooled.copy()
    poisoned[fr.rows["test"]] = np.nan                         # any use of test rows would give NaN
    dirty = drv.run_candidate(poisoned, sel, cand, seed=0)
    assert clean["val_macro_f1"] == dirty["val_macro_f1"] and clean["history"] == dirty["history"]


@pytest.mark.parametrize("ds,k", [("RAVDESS", 6), ("CREMA-D", 5)])
def test_no_speaker_leakage_in_any_fold(ds, k):
    from src.features.store import CacheError, FeatureStore
    from src.training.data import fold_rows
    from src.utils.paths import PATHS
    index = pd.read_csv(PATHS.metadata / "index.csv")
    try:
        store = FeatureStore.open(ds, index, require=())
    except CacheError:
        pytest.skip(f"{ds} cache not available")
    for f in range(k):
        fr = fold_rows(store, index, ds, f)
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            assert not set(fr.speakers[a]) & set(fr.speakers[b])
            assert not set(fr.rows[a]) & set(fr.rows[b])


# ------------------------------------------------------------------ checkpoints
def test_checkpoint_save_and_load_reproduces_predictions(tmp_path):
    drv = _driver()
    pooled, _, fr = _synthetic()
    cand = {"rep": "weighted_all", "pooling": "mean", "hidden": 32, "dropout": 0.2, "lr": 1e-3, "class_weighted": True}
    (Xtr, Xva), std, shape = drv.make_inputs(pooled, fr.rows["train"], [fr.rows["train"], fr.rows["val"]], cand["rep"], cand["pooling"])
    model = train_classifier(lambda: drv.build_model(shape, cand["rep"], cand["hidden"], cand["dropout"]),
                             (Xtr, fr.labels["train"]), (Xva, fr.labels["val"]), TrainConfig(max_epochs=3))["model"]
    p = tmp_path / "ck.pt"
    drv.save_checkpoint(p, model, std, cand, shape, {"dataset": "SYN", "fold": 0, "seed": 0})
    model2, std2, ck = drv.load_checkpoint(p)
    assert ck["candidate"] == cand and ck["dataset"] == "SYN"
    np.testing.assert_array_equal(std2.mean_, std.mean_)
    assert np.array_equal(predict(model2, Xva), predict(model, Xva))
