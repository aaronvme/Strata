from std.math import nan, inf, sqrt
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_false,
    assert_almost_equal,
)

from strata.core.matrix import Matrix
from strata.core.dataset import Dataset
from strata.base.estimator import (
    fit as fit_ds,
    predict as predict_ds,
)
from strata.base.pipeline import PipelineRegressor
from strata.preprocessing.scaler import StandardScaler
from strata.ensemble.forest import RandomForestRegressor
from strata.exceptions.errors import (
    NotFittedError,
    DimensionMismatchError,
    InvalidParameterError,
)


def _step_1d_dataset() -> (
    Tuple[Matrix[DType.float64], List[Scalar[DType.float64]]]
):
    var X = Matrix[DType.float64](10, 1, 0)
    var y = List[Scalar[DType.float64]](capacity=10)
    for i in range(5):
        X[i, 0] = Float64(i)
        y.append(0.0)
    for i in range(5, 10):
        X[i, 0] = Float64(i + 10)
        y.append(10.0)
    return (X^, y^)


def test_rf_regressor_step_1d() raises:
    var data = _step_1d_dataset()
    var rf = RandomForestRegressor[DType.float64](
        n_estimators=10, random_state=42
    )
    rf.fit(data[0], data[1])

    assert_true(rf.is_fitted)
    assert_equal(rf.n_features_in_, 1)
    assert_equal(len(rf.estimators_), 10)

    var preds = rf.predict(data[0])
    assert_equal(len(preds), 10)
    for i in range(5):
        assert_true(preds[i] < 3.0)
    for i in range(5, 10):
        assert_true(preds[i] > 7.0)


def test_rf_regressor_linear_continuous() raises:
    var X = Matrix[DType.float64](20, 1, 0)
    var y = List[Scalar[DType.float64]](capacity=20)
    for i in range(20):
        X[i, 0] = Float64(i)
        y.append(Float64(i * 2))

    var rf = RandomForestRegressor[DType.float64](
        n_estimators=15, max_depth=5, random_state=42
    )
    rf.fit(X, y)

    var preds = rf.predict(X)
    for i in range(20):
        assert_almost_equal(preds[i], y[i], atol=3.0)


def test_rf_regressor_multivariate_features() raises:
    var X = Matrix[DType.float64](12, 3, 0)
    var y = List[Scalar[DType.float64]](capacity=12)
    for i in range(12):
        X[i, 0] = Float64(i)
        X[i, 1] = Float64(i % 3)
        X[i, 2] = Float64(i * 2)
        y.append(Float64(i * 3))

    var rf = RandomForestRegressor[DType.float64](
        n_estimators=10, random_state=42
    )
    rf.fit(X, y)

    var preds = rf.predict(X)
    assert_equal(len(preds), 12)


def test_rf_regressor_bootstrap_false() raises:
    var data = _step_1d_dataset()
    var rf = RandomForestRegressor[DType.float64](
        n_estimators=5, bootstrap=False, random_state=42
    )
    rf.fit(data[0], data[1])

    assert_true(rf.is_fitted)
    var preds = rf.predict(data[0])
    for i in range(5):
        assert_almost_equal(preds[i], 0.0, atol=1e-4)
    for i in range(5, 10):
        assert_almost_equal(preds[i], 10.0, atol=1e-4)


def test_rf_regressor_max_samples_ratio() raises:
    var data = _step_1d_dataset()
    var rf = RandomForestRegressor[DType.float64](
        n_estimators=10,
        bootstrap=True,
        max_samples_ratio=0.8,
        random_state=42,
    )
    rf.fit(data[0], data[1])
    assert_true(rf.is_fitted)

    var preds = rf.predict(data[0])
    assert_equal(len(preds), 10)


def test_rf_regressor_max_samples_count() raises:
    var data = _step_1d_dataset()
    var rf = RandomForestRegressor[DType.float64](
        n_estimators=10,
        bootstrap=True,
        max_samples_count=7,
        random_state=42,
    )
    rf.fit(data[0], data[1])
    assert_true(rf.is_fitted)

    var preds = rf.predict(data[0])
    assert_equal(len(preds), 10)


def test_rf_regressor_oob_score_enabled() raises:
    var X = Matrix[DType.float64](30, 2, 0)
    var y = List[Scalar[DType.float64]](capacity=30)
    for i in range(30):
        X[i, 0] = Float64(i)
        X[i, 1] = Float64(i % 2)
        y.append(Float64(i * 2))

    var rf = RandomForestRegressor[DType.float64](
        n_estimators=30,
        bootstrap=True,
        oob_score=True,
        random_state=42,
    )
    rf.fit(X, y)

    assert_true(rf.is_fitted)
    assert_true(rf.oob_score_ > 0.5)


def test_rf_regressor_oob_score_disabled_raises() raises:
    var data = _step_1d_dataset()
    var rf = RandomForestRegressor[DType.float64](
        n_estimators=5, oob_score=False
    )
    rf.fit(data[0], data[1])

    var caught = False
    try:
        _ = rf.get_oob_score()
    except:
        caught = True
    assert_true(caught)


def test_rf_regressor_feature_importances_sum_to_one() raises:
    var X = Matrix[DType.float64](20, 3, 0)
    var y = List[Scalar[DType.float64]](capacity=20)
    for i in range(20):
        X[i, 0] = Float64(i)
        X[i, 1] = 0.0
        X[i, 2] = Float64(i % 2)
        y.append(Float64(i * 5))

    var rf = RandomForestRegressor[DType.float64](
        n_estimators=15, random_state=42
    )
    rf.fit(X, y)

    var importances = rf.get_feature_importances()
    assert_equal(len(importances), 3)

    var sum_imp: Float64 = 0.0
    for j in range(3):
        sum_imp += importances[j]
    assert_almost_equal(sum_imp, 1.0, atol=1e-4)
    # Feature 0 explains y and must have largest importance
    assert_true(importances[0] > importances[1])


def test_rf_regressor_criterion_squared_error_and_friedman_mse() raises:
    var data = _step_1d_dataset()
    var rf_mse = RandomForestRegressor[DType.float64](
        n_estimators=5, criterion="squared_error", random_state=42
    )
    var rf_friedman = RandomForestRegressor[DType.float64](
        n_estimators=5, criterion="friedman_mse", random_state=42
    )

    rf_mse.fit(data[0], data[1])
    rf_friedman.fit(data[0], data[1])

    var p_mse = rf_mse.predict(data[0])
    var p_friedman = rf_friedman.predict(data[0])
    for i in range(10):
        assert_almost_equal(p_mse[i], p_friedman[i], atol=1.0)


def test_rf_regressor_criterion_absolute_error() raises:
    var data = _step_1d_dataset()
    var rf_mae = RandomForestRegressor[DType.float64](
        n_estimators=5, criterion="absolute_error", random_state=42
    )
    rf_mae.fit(data[0], data[1])

    var preds = rf_mae.predict(data[0])
    for i in range(5):
        assert_true(preds[i] < 3.0)
    for i in range(5, 10):
        assert_true(preds[i] > 7.0)


def test_rf_regressor_max_depth_stump() raises:
    var data = _step_1d_dataset()
    var rf = RandomForestRegressor[DType.float64](
        n_estimators=5, max_depth=1, random_state=42
    )
    rf.fit(data[0], data[1])

    for b in range(len(rf.estimators_)):
        assert_true(rf.estimators_[b].get_depth() <= 1)


def test_rf_regressor_min_samples_split_and_leaf() raises:
    var data = _step_1d_dataset()
    var rf = RandomForestRegressor[DType.float64](
        n_estimators=5,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
    )
    rf.fit(data[0], data[1])
    assert_true(rf.is_fitted)


def test_rf_regressor_min_impurity_decrease() raises:
    var data = _step_1d_dataset()
    var rf = RandomForestRegressor[DType.float64](
        n_estimators=5, min_impurity_decrease=1000.0, random_state=42
    )
    rf.fit(data[0], data[1])

    # With huge min_impurity_decrease, all trees are single root leaf nodes
    for b in range(len(rf.estimators_)):
        assert_equal(rf.estimators_[b].get_depth(), 0)


def test_rf_regressor_max_features_sqrt_and_log2() raises:
    var X = Matrix[DType.float64](16, 9, 0)
    var y = List[Scalar[DType.float64]](capacity=16)
    for i in range(16):
        for j in range(9):
            X[i, j] = Float64(i + j)
        y.append(Float64(i * 2))

    var rf_sqrt = RandomForestRegressor[DType.float64](
        n_estimators=5, max_features="sqrt", random_state=42
    )
    var rf_log2 = RandomForestRegressor[DType.float64](
        n_estimators=5, max_features="log2", random_state=42
    )
    rf_sqrt.fit(X, y)
    rf_log2.fit(X, y)

    assert_true(rf_sqrt.is_fitted)
    assert_true(rf_log2.is_fitted)


def test_rf_regressor_single_sample() raises:
    var X = Matrix[DType.float64](1, 2, 5.0)
    var y: List[Scalar[DType.float64]] = [42.0]

    var rf = RandomForestRegressor[DType.float64](n_estimators=5)
    rf.fit(X, y)

    var preds = rf.predict(X)
    assert_equal(len(preds), 1)
    assert_almost_equal(preds[0], 42.0, atol=1e-5)


def test_rf_regressor_identical_samples() raises:
    var X = Matrix[DType.float64](8, 2, 3.0)
    var y = List[Scalar[DType.float64]](capacity=8)
    for _ in range(8):
        y.append(7.0)

    var rf = RandomForestRegressor[DType.float64](n_estimators=5)
    rf.fit(X, y)

    var preds = rf.predict(X)
    for i in range(8):
        assert_almost_equal(preds[i], 7.0, atol=1e-5)


def test_rf_regressor_constant_features_varying_target() raises:
    var X = Matrix[DType.float64](6, 2, 1.0)
    var y: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    # Without bootstrap, mean of identical features is exact target mean 3.5
    var rf = RandomForestRegressor[DType.float64](
        n_estimators=5, bootstrap=False
    )
    rf.fit(X, y)

    var preds = rf.predict(X)
    for i in range(6):
        assert_almost_equal(preds[i], 3.5, atol=1e-4)

    # With bootstrap, all identical feature rows produce the exact same prediction
    var rf_boot = RandomForestRegressor[DType.float64](
        n_estimators=10, random_state=42
    )
    rf_boot.fit(X, y)
    var boot_preds = rf_boot.predict(X)
    for i in range(1, 6):
        assert_almost_equal(boot_preds[i], boot_preds[0], atol=1e-5)


def test_rf_regressor_constant_target_varying_features() raises:
    var X = Matrix[DType.float64](6, 2, 0)
    var y = List[Scalar[DType.float64]](capacity=6)
    for i in range(6):
        X[i, 0] = Float64(i)
        X[i, 1] = Float64(i * 2)
        y.append(100.0)

    var rf = RandomForestRegressor[DType.float64](n_estimators=5)
    rf.fit(X, y)

    var preds = rf.predict(X)
    for i in range(6):
        assert_almost_equal(preds[i], 100.0, atol=1e-5)


def test_rf_regressor_high_dimensional_features() raises:
    var X = Matrix[DType.float64](15, 60, 0)
    var y = List[Scalar[DType.float64]](capacity=15)
    for i in range(15):
        for j in range(60):
            X[i, j] = Float64(i * j) * 0.1
        y.append(Float64(i * 2))

    var rf = RandomForestRegressor[DType.float64](
        n_estimators=5, max_features="sqrt", random_state=42
    )
    rf.fit(X, y)

    var preds = rf.predict(X)
    assert_equal(len(preds), 15)


def test_rf_regressor_random_state_determinism() raises:
    var data = _step_1d_dataset()
    var rf1 = RandomForestRegressor[DType.float64](
        n_estimators=10, random_state=999
    )
    var rf2 = RandomForestRegressor[DType.float64](
        n_estimators=10, random_state=999
    )
    rf1.fit(data[0], data[1])
    rf2.fit(data[0], data[1])

    var p1 = rf1.predict(data[0])
    var p2 = rf2.predict(data[0])
    for i in range(10):
        assert_almost_equal(p1[i], p2[i], atol=1e-8)


def test_rf_regressor_copy_constructor_isolation() raises:
    var data = _step_1d_dataset()
    var rf1 = RandomForestRegressor[DType.float64](
        n_estimators=5, random_state=42
    )
    rf1.fit(data[0], data[1])

    var rf2 = RandomForestRegressor[DType.float64](copy=rf1)
    assert_true(rf2.is_fitted)
    assert_equal(rf2.n_features_in_, rf1.n_features_in_)

    var p1 = rf1.predict(data[0])
    var p2 = rf2.predict(data[0])
    for i in range(10):
        assert_almost_equal(p1[i], p2[i], atol=1e-8)


def test_rf_regressor_refit_resets_state() raises:
    var data = _step_1d_dataset()
    var rf = RandomForestRegressor[DType.float64](
        n_estimators=5, random_state=42
    )
    rf.fit(data[0], data[1])
    rf.fit(data[0], data[1])
    assert_equal(len(rf.estimators_), 5)


def test_rf_regressor_unfitted_predict_raises() raises:
    var rf = RandomForestRegressor[DType.float64]()
    var X = Matrix[DType.float64](3, 2, 1.0)
    var caught = False
    try:
        _ = rf.predict(X)
    except:
        caught = True
    assert_true(caught)


def test_rf_regressor_dimension_mismatch_raises() raises:
    var data = _step_1d_dataset()
    var rf = RandomForestRegressor[DType.float64](n_estimators=5)
    rf.fit(data[0], data[1])

    var X_bad = Matrix[DType.float64](4, 3, 1.0)
    var caught = False
    try:
        _ = rf.predict(X_bad)
    except:
        caught = True
    assert_true(caught)


def test_rf_regressor_invalid_hyperparameters_raises() raises:
    var caught_n_est = False
    try:
        _ = RandomForestRegressor[DType.float64](n_estimators=0)
    except:
        caught_n_est = True
    assert_true(caught_n_est)

    var caught_max_samples = False
    try:
        _ = RandomForestRegressor[DType.float64](max_samples_ratio=1.5)
    except:
        caught_max_samples = True
    assert_true(caught_max_samples)


def test_rf_regressor_nan_and_inf_rejected() raises:
    var X = Matrix[DType.float64](4, 2, 1.0)
    X[0, 0] = nan[DType.float64]()
    var y: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0, 4.0]

    var rf = RandomForestRegressor[DType.float64](n_estimators=5)
    var caught = False
    try:
        rf.fit(X, y)
    except:
        caught = True
    assert_true(caught)


def test_rf_regressor_float32_native() raises:
    var X = Matrix[DType.float32](10, 1, 0)
    var y = List[Scalar[DType.float32]](capacity=10)
    for i in range(10):
        X[i, 0] = Float32(i)
        y.append(Float32(i * 2))

    var rf = RandomForestRegressor[DType.float32](
        n_estimators=5, random_state=42
    )
    rf.fit(X, y)
    assert_true(rf.is_fitted)

    var preds = rf.predict(X)
    assert_equal(len(preds), 10)


def test_rf_regressor_dataset_integration() raises:
    var data = _step_1d_dataset()
    var ds = Dataset[DType.float64, DType.float64](
        data[0].copy(), data[1].copy()
    )

    var rf = RandomForestRegressor[DType.float64](
        n_estimators=5, random_state=42
    )
    fit_ds(rf, ds)
    assert_true(rf.is_fitted)

    var preds = predict_ds(rf, ds)
    assert_equal(len(preds), 10)


def test_rf_regressor_pipeline_integration() raises:
    var data = _step_1d_dataset()
    var scaler = StandardScaler[DType.float64]()
    var rf = RandomForestRegressor[DType.float64](
        n_estimators=5, random_state=42
    )

    var pipe = PipelineRegressor[
        StandardScaler[DType.float64],
        RandomForestRegressor[DType.float64],
        DType.float64,
    ](scaler^, rf^)

    pipe.fit(data[0], data[1])
    assert_true(pipe.regressor.is_fitted)

    var preds = pipe.predict(data[0])
    assert_equal(len(preds), 10)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
