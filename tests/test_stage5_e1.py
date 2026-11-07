"""
Stage 5 tests: train-only preprocessing, metrics, E1 aggregation and test-set isolation.
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import compute_metrics, summarise
from src.models.e1_classical import E1Config, aggregate, select_on_validation
from src.training.preprocessing import DisVoicePreprocessor, FrameStandardizer, signed_log

RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ aggregation
def test_aggregation_statistics_and_dimensions():
    x = RNG.standard_normal((50, 39))
    m, ms, fu = aggregate(x, "mean"), aggregate(x, "mean_std"), aggregate(x, "functionals")
    assert m.shape == (39,) and ms.shape == (78,) and fu.shape == (195,)
    np.testing.assert_allclose(ms[:39], x.mean(0), rtol=1e-5)
    np.testing.assert_allclose(ms[39:], x.std(0), rtol=1e-5)
    np.testing.assert_allclose(fu[78:117], np.percentile(x, 10, axis=0), rtol=1e-5)
    np.testing.assert_allclose(fu[156:], np.percentile(x, 90, axis=0), rtol=1e-5)


def test_aggregation_is_length_invariant_for_repeated_content():
    x = RNG.standard_normal((40, 39))
    np.testing.assert_allclose(aggregate(x, "mean_std"), aggregate(np.tile(x, (3, 1)), "mean_std"), rtol=1e-5)


# ------------------------------------------------------------------ test isolation
def _toy_corpus(n=240, d=6):
    y = np.repeat(np.arange(6), n // 6)
    X = {a: (RNG.standard_normal((n, d)) + y[:, None] * 0.8).astype(np.float32) for a in ("mean", "mean_std", "functionals")}
    perm = RNG.permutation(n)
    return X, y, perm[:120], perm[120:180], perm[180:]


@pytest.mark.parametrize("model", ["svm", "rf"])
def test_selection_never_touches_test_rows(model):
    """Test rows are set to NaN: any use of them by the scaler or classifier would raise."""
    X, y, tr, va, te = _toy_corpus()
    cfg = E1Config(svm_C=(1.0, 10.0), svm_gamma_mult=(1.0,), rf_n_estimators=20)
    clean = select_on_validation(model, X, y[tr], y[va], tr, va, cfg, seed=0)
    for a in X:
        X[a][te] = np.nan
    poisoned = select_on_validation(model, X, y[tr], y[va], tr, va, cfg, seed=0)
    assert poisoned["selected"]["aggregation"] == clean["selected"]["aggregation"]
    assert poisoned["selected"]["hp"] == clean["selected"]["hp"]
    assert poisoned["selected"]["val_macro_f1"] == clean["selected"]["val_macro_f1"]


def test_selection_ties_break_to_first_candidate_deterministically():
    X, y, tr, va, _ = _toy_corpus()
    cfg = E1Config(svm_C=(1.0, 1.0), svm_gamma_mult=(1.0,), aggregations=("mean",))
    sel = select_on_validation("svm", X, y[tr], y[va], tr, va, cfg, seed=0)
    assert sel["selected"] is sel["table"][0]


# ------------------------------------------------------------------ metrics
def test_metrics_definitions():
    y = np.array([0, 0, 0, 0, 1, 1, 2, 2, 3, 4, 5, 5])
    p = np.array([0, 0, 0, 1, 1, 1, 2, 0, 3, 4, 5, 2])
    m = compute_metrics(y, p)
    recalls = [3 / 4, 1, 1 / 2, 1, 1, 1 / 2]
    assert m["UA"] == pytest.approx(np.mean(recalls)) == pytest.approx(m["macro_recall"])
    assert m["accuracy"] == pytest.approx(9 / 12)
    assert np.array(m["confusion"]).sum() == 12 and np.trace(np.array(m["confusion"])) == 9
    assert m["per_class"]["neutral"]["precision"] == pytest.approx(3 / 4)


def test_summary_uses_sample_std_over_folds():
    s = summarise([{"macro_f1": 0.5}, {"macro_f1": 0.7}], keys=("macro_f1",))
    assert s["macro_f1"]["mean"] == pytest.approx(0.6)
    assert s["macro_f1"]["std"] == pytest.approx(np.std([0.5, 0.7], ddof=1))


# ------------------------------------------------------------------ preprocessing
def test_frame_standardizer_uses_training_frames_only():
    train = [RNG.normal(5, 2, (n, 3)) for n in (30, 50, 20)]
    fs = FrameStandardizer(passthrough=(2,)).fit(train)
    allf = np.concatenate(train)
    np.testing.assert_allclose(fs.mean_[:2], allf[:, :2].mean(0))
    assert fs.mean_[2] == 0 and fs.std_[2] == 1                       # passthrough column
    before = (fs.mean_.copy(), fs.std_.copy())
    z = fs.transform(RNG.normal(100, 50, (10, 3)))                     # wildly different "test" data
    np.testing.assert_array_equal(before[0], fs.mean_)                 # transform never refits
    assert z.dtype == np.float32
    z_train = fs.transform(allf)
    np.testing.assert_allclose(z_train[:, :2].mean(0), 0, atol=1e-5)
    np.testing.assert_allclose(z_train[:, :2].std(0), 1, atol=1e-4)


def test_unfitted_transformers_refuse():
    with pytest.raises(RuntimeError):
        FrameStandardizer().transform(np.zeros((2, 2)))
    with pytest.raises(RuntimeError):
        DisVoicePreprocessor().transform(np.zeros((2, 103)), np.zeros((2, 103), bool))


def test_disvoice_preprocessor_median_fill_signed_log_clip_and_indicators():
    n = 400
    X = RNG.standard_normal((n, 103))
    X[:, 0] = RNG.lognormal(3, 1.5, n)            # heavily right-skewed, positive
    X[:, 1] = -RNG.lognormal(3, 1.5, n)           # heavily skewed, negative (plain log impossible)
    imp = np.zeros_like(X, dtype=bool)
    imp[:50, 2] = True
    X[:50, 2] = 0.0                               # cache convention: imputed -> 0
    pre = DisVoicePreprocessor(skew_threshold=2.0, clip=5.0).fit(X, imp)
    assert 0 in pre.log_cols_ and 1 in pre.log_cols_ and 3 not in pre.log_cols_
    assert pre.fill_[2] == pytest.approx(np.median(X[50:, 2]))       # median of OBSERVED values
    Z = pre.transform(X, imp)
    assert Z.shape == (n, 103) and np.abs(Z).max() <= 5.0
    np.testing.assert_allclose(Z[:50, 2], ((pre.fill_[2] - pre.mean_[2]) / pre.std_[2]), rtol=1e-5)
    extreme = X[:5].copy(); extreme[:, 3] = 1e9
    assert np.abs(pre.transform(extreme, imp[:5])).max() <= 5.0          # clipped, params unchanged
    assert signed_log(np.array([-np.e + 1, 0, np.e - 1])) == pytest.approx([-1, 0, 1])
    with_ind = DisVoicePreprocessor(add_indicators=True).fit(X, imp)
    Zi = with_ind.transform(X, imp)
    assert Zi.shape == (n, 206) and Zi[:50, 103 + 2].all() and not Zi[50:, 103 + 2].any()


# ------------------------------------------------------------------ real splits -> cache rows
def test_fold_rows_are_disjoint_and_cover_ravdess():
    from src.features.store import CacheError, FeatureStore
    from src.training.data import fold_rows
    from src.utils.paths import PATHS
    index = pd.read_csv(PATHS.metadata / "index.csv")
    try:
        store = FeatureStore.open("RAVDESS", index, require=())
    except CacheError:
        pytest.skip("RAVDESS feature cache not available")
    tested = []
    for k in range(6):
        fr = fold_rows(store, index, "RAVDESS", k)
        assert not set(fr.rows["train"]) & set(fr.rows["test"])
        assert not set(fr.speakers["val"]) & set(fr.speakers["test"])
        assert len(fr.rows["train"]) + len(fr.rows["val"]) + len(fr.rows["test"]) == len(store.manifest)
        tested.extend(fr.rows["test"])
    assert sorted(tested) == list(range(len(store.manifest)))        # every recording tested once
