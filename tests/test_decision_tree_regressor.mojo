from std.math import nan, inf, sin, abs
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_false,
    assert_almost_equal,
)

from strata.core.matrix import Matrix
from strata.core.dataset import Dataset
from strata.base.estimator import fit as fit_ds, predict as predict_ds
from strata.base.pipeline import PipelineRegressor
from strata.preprocessing.scaler import StandardScaler
from strata.tree.regressor import DecisionTreeRegressor
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
    for i in range(10):
        X[i, 0] = Float64(i)
        y.append(Float64(0.0 if i < 5 else 10.0))
    return (X^, y^)


def test_decision_tree_regressor_step_function_1d() raises:
    var data = _step_1d_dataset()
    var reg = DecisionTreeRegressor[DType.float64](max_depth=1)
    reg.fit(data[0], data[1])

    assert_true(reg.is_fitted)
    assert_equal(reg.n_features_in_, 1)
    assert_equal(reg.tree_.max_depth(), 1)
    assert_equal(reg.tree_.node_count(), 3)

    var preds = reg.predict(data[0])
    assert_equal(len(preds), 10)
    for i in range(5):
        assert_almost_equal(preds[i], 0.0, atol=1e-5)
    for i in range(5, 10):
        assert_almost_equal(preds[i], 10.0, atol=1e-5)


def test_decision_tree_regressor_linear_approximation() raises:
    var X = Matrix[DType.float64](16, 1, 0)
    var y = List[Scalar[DType.float64]](capacity=16)
    for i in range(16):
        X[i, 0] = Float64(i)
        y.append(Float64(i * 2))

    var reg = DecisionTreeRegressor[DType.float64](max_depth=4)
    reg.fit(X, y)

    var preds = reg.predict(X)
    assert_equal(len(preds), 16)
    for i in range(16):
        assert_almost_equal(preds[i], y[i], atol=1e-4)


def test_decision_tree_regressor_multivariate_targets() raises:
    var X = Matrix[DType.float64](12, 2, 0)
    var y = List[Scalar[DType.float64]](capacity=12)
    for i in range(12):
        var x0 = Float64(i)
        var x1 = Float64(i % 3)
        X[i, 0] = x0
        X[i, 1] = x1
        y.append(3.0 * x0 - 2.0 * x1)

    var reg = DecisionTreeRegressor[DType.float64](max_depth=4)
    reg.fit(X, y)

    var preds = reg.predict(X)
    assert_equal(len(preds), 12)
    for i in range(12):
        assert_almost_equal(preds[i], y[i], atol=1e-3)


def test_decision_tree_regressor_criteria_squared_error() raises:
    var data = _step_1d_dataset()
    var reg = DecisionTreeRegressor[DType.float64](criterion="squared_error")
    reg.fit(data[0], data[1])

    var preds = reg.predict(data[0])
    for i in range(5):
        assert_almost_equal(preds[i], 0.0, atol=1e-5)
    for i in range(5, 10):
        assert_almost_equal(preds[i], 10.0, atol=1e-5)


def test_decision_tree_regressor_criteria_friedman_mse() raises:
    var data = _step_1d_dataset()
    var reg = DecisionTreeRegressor[DType.float64](criterion="friedman_mse")
    reg.fit(data[0], data[1])

    var preds = reg.predict(data[0])
    for i in range(5):
        assert_almost_equal(preds[i], 0.0, atol=1e-5)
    for i in range(5, 10):
        assert_almost_equal(preds[i], 10.0, atol=1e-5)


def test_decision_tree_regressor_criteria_absolute_error() raises:
    var data = _step_1d_dataset()
    var reg = DecisionTreeRegressor[DType.float64](criterion="absolute_error")
    reg.fit(data[0], data[1])

    var preds = reg.predict(data[0])
    for i in range(5):
        assert_almost_equal(preds[i], 0.0, atol=1e-5)
    for i in range(5, 10):
        assert_almost_equal(preds[i], 10.0, atol=1e-5)


def test_decision_tree_regressor_max_depth_stump() raises:
    var data = _step_1d_dataset()
    var reg = DecisionTreeRegressor[DType.float64](max_depth=1)
    reg.fit(data[0], data[1])

    assert_equal(reg.tree_.max_depth(), 1)
    assert_equal(reg.tree_.node_count(), 3)


def test_decision_tree_regressor_max_depth_limits_tree_depth() raises:
    var X = Matrix[DType.float64](20, 1, 0)
    var y = List[Scalar[DType.float64]](capacity=20)
    for i in range(20):
        X[i, 0] = Float64(i)
        y.append(Float64(i * i))

    var reg = DecisionTreeRegressor[DType.float64](max_depth=2)
    reg.fit(X, y)

    assert_true(reg.tree_.max_depth() <= 2)


def test_decision_tree_regressor_min_samples_split() raises:
    var data = _step_1d_dataset()
    # min_samples_split=15 > 10 => no split allowed
    var reg = DecisionTreeRegressor[DType.float64](min_samples_split=15)
    reg.fit(data[0], data[1])

    assert_equal(reg.tree_.node_count(), 1)  # Only root node


def test_decision_tree_regressor_min_samples_leaf() raises:
    var X = Matrix[DType.float64](12, 1, 0)
    var y = List[Scalar[DType.float64]](capacity=12)
    for i in range(12):
        X[i, 0] = Float64(i)
        y.append(Float64(0.0 if i < 2 else 10.0))

    var reg = DecisionTreeRegressor[DType.float64](min_samples_leaf=4)
    reg.fit(X, y)

    for n in range(reg.tree_.node_count()):
        if reg.tree_.nodes[n].is_leaf:
            assert_true(reg.tree_.nodes[n].n_node_samples >= 4)


def test_decision_tree_regressor_min_impurity_decrease() raises:
    var data = _step_1d_dataset()
    var reg = DecisionTreeRegressor[DType.float64](
        min_impurity_decrease=10000.0
    )
    reg.fit(data[0], data[1])

    assert_equal(reg.tree_.node_count(), 1)


def test_decision_tree_regressor_max_features_sqrt() raises:
    var X = Matrix[DType.float64](20, 16, 0)
    var y = List[Scalar[DType.float64]](capacity=20)
    for i in range(20):
        for j in range(16):
            X[i, j] = Float64(i + j)
        y.append(Float64(i * 3))

    var reg = DecisionTreeRegressor[DType.float64](
        max_features="sqrt", random_state=42
    )
    reg.fit(X, y)
    assert_true(reg.is_fitted)

    var preds = reg.predict(X)
    assert_equal(len(preds), 20)


def test_decision_tree_regressor_max_features_log2() raises:
    var X = Matrix[DType.float64](20, 16, 0)
    var y = List[Scalar[DType.float64]](capacity=20)
    for i in range(20):
        for j in range(16):
            X[i, j] = Float64(i + j)
        y.append(Float64(i * 3))

    var reg = DecisionTreeRegressor[DType.float64](
        max_features="log2", random_state=42
    )
    reg.fit(X, y)
    assert_true(reg.is_fitted)


def test_decision_tree_regressor_max_features_explicit_int_and_float() raises:
    var X = Matrix[DType.float64](20, 10, 0)
    var y = List[Scalar[DType.float64]](capacity=20)
    for i in range(20):
        X[i, 0] = Float64(i)
        y.append(Float64(i * 2))

    var reg_int = DecisionTreeRegressor[DType.float64](
        max_features_count=4, random_state=42
    )
    reg_int.fit(X, y)
    assert_true(reg_int.is_fitted)

    var reg_float = DecisionTreeRegressor[DType.float64](
        max_features_ratio=0.5, random_state=42
    )
    reg_float.fit(X, y)
    assert_true(reg_float.is_fitted)


def test_decision_tree_regressor_splitter_random() raises:
    var data = _step_1d_dataset()
    var reg = DecisionTreeRegressor[DType.float64](
        splitter="random", random_state=42
    )
    reg.fit(data[0], data[1])
    assert_true(reg.is_fitted)

    var preds = reg.predict(data[0])
    assert_equal(len(preds), 10)


def test_decision_tree_regressor_single_sample() raises:
    var X = Matrix[DType.float64](1, 2, 5.0)
    var y = List[Scalar[DType.float64]]()
    y.append(42.0)

    var reg = DecisionTreeRegressor[DType.float64]()
    reg.fit(X, y)

    assert_true(reg.is_fitted)
    assert_equal(reg.tree_.node_count(), 1)

    var preds = reg.predict(X)
    assert_equal(len(preds), 1)
    assert_almost_equal(preds[0], 42.0, atol=1e-5)


def test_decision_tree_regressor_identical_samples() raises:
    var X = Matrix[DType.float64](6, 2, 1.0)
    var y = List[Scalar[DType.float64]](capacity=6)
    for _ in range(6):
        y.append(10.0)

    var reg = DecisionTreeRegressor[DType.float64]()
    reg.fit(X, y)

    assert_true(reg.is_fitted)
    assert_equal(reg.tree_.node_count(), 1)


def test_decision_tree_regressor_constant_features_varying_target() raises:
    # Identical features but varying targets => leaf predicts mean of y
    var X = Matrix[DType.float64](4, 2, 1.0)
    var y: List[Scalar[DType.float64]] = [2.0, 4.0, 6.0, 8.0]

    var reg = DecisionTreeRegressor[DType.float64]()
    reg.fit(X, y)

    assert_true(reg.is_fitted)
    assert_equal(reg.tree_.node_count(), 1)

    var preds = reg.predict(X)
    for i in range(4):
        assert_almost_equal(
            preds[i], 5.0, atol=1e-5
        )  # Mean is (2+4+6+8)/4 = 5.0


def test_decision_tree_regressor_constant_target_varying_features() raises:
    var X = Matrix[DType.float64](8, 2, 0)
    var y = List[Scalar[DType.float64]](capacity=8)
    for i in range(8):
        X[i, 0] = Float64(i)
        X[i, 1] = Float64(i * 10)
        y.append(7.77)

    var reg = DecisionTreeRegressor[DType.float64]()
    reg.fit(X, y)

    assert_true(reg.is_fitted)
    assert_equal(reg.tree_.node_count(), 1)  # Impurity is 0 => no split needed

    var preds = reg.predict(X)
    for i in range(8):
        assert_almost_equal(preds[i], 7.77, atol=1e-5)


def test_decision_tree_regressor_sine_wave_nonlinear() raises:
    var X = Matrix[DType.float64](20, 1, 0)
    var y = List[Scalar[DType.float64]](capacity=20)
    for i in range(20):
        var val = Float64(i) * 0.314159
        X[i, 0] = val
        y.append(sin(val))

    var reg = DecisionTreeRegressor[DType.float64](max_depth=5)
    reg.fit(X, y)

    var preds = reg.predict(X)
    assert_equal(len(preds), 20)
    for i in range(20):
        assert_true(abs(preds[i] - y[i]) < 0.25)


def test_decision_tree_regressor_high_dimensional_features() raises:
    var X = Matrix[DType.float64](20, 30, 0)
    var y = List[Scalar[DType.float64]](capacity=20)
    for i in range(20):
        for j in range(30):
            X[i, j] = Float64(i * j) * 0.1
        y.append(Float64(i + 1))

    var reg = DecisionTreeRegressor[DType.float64](max_depth=3)
    reg.fit(X, y)

    var preds = reg.predict(X)
    assert_equal(len(preds), 20)


def test_decision_tree_regressor_copy_constructor_isolation() raises:
    var data = _step_1d_dataset()
    var reg1 = DecisionTreeRegressor[DType.float64](max_depth=2)
    reg1.fit(data[0], data[1])

    var reg2 = reg1.copy()
    assert_true(reg2.is_fitted)
    assert_equal(reg2.tree_.node_count(), reg1.tree_.node_count())

    var preds1 = reg1.predict(data[0])
    var preds2 = reg2.predict(data[0])
    for i in range(10):
        assert_almost_equal(preds1[i], preds2[i], atol=1e-8)


def test_decision_tree_regressor_refit_resets_state() raises:
    var data = _step_1d_dataset()
    var reg = DecisionTreeRegressor[DType.float64]()
    reg.fit(data[0], data[1])
    assert_equal(reg.n_features_in_, 1)

    # Refit on 3 features
    var X3 = Matrix[DType.float64](8, 3, 1.0)
    var y3: List[Scalar[DType.float64]] = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
    ]
    reg.fit(X3, y3)

    assert_equal(reg.n_features_in_, 3)


def test_decision_tree_regressor_unfitted_predict_raises() raises:
    var reg = DecisionTreeRegressor[DType.float64]()
    var X = Matrix[DType.float64](4, 2, 1.0)

    var caught = False
    try:
        var _ = reg.predict(X)
    except:
        caught = True
    assert_true(caught)


def test_decision_tree_regressor_dimension_mismatch_raises() raises:
    var data = _step_1d_dataset()
    var reg = DecisionTreeRegressor[DType.float64]()
    reg.fit(data[0], data[1])

    var X_bad = Matrix[DType.float64](4, 2, 1.0)
    var caught = False
    try:
        var _ = reg.predict(X_bad)
    except:
        caught = True
    assert_true(caught)


def test_decision_tree_regressor_invalid_parameters_raises() raises:
    var caught_crit = False
    try:
        var _ = DecisionTreeRegressor[DType.float64](criterion="unknown")
    except:
        caught_crit = True
    assert_true(caught_crit)

    var caught_split = False
    try:
        var _ = DecisionTreeRegressor[DType.float64](min_samples_split=1)
    except:
        caught_split = True
    assert_true(caught_split)

    var caught_leaf = False
    try:
        var _ = DecisionTreeRegressor[DType.float64](min_samples_leaf=0)
    except:
        caught_leaf = True
    assert_true(caught_leaf)

    var caught_depth = False
    try:
        var _ = DecisionTreeRegressor[DType.float64](max_depth=0)
    except:
        caught_depth = True
    assert_true(caught_depth)


def test_decision_tree_regressor_nan_and_inf_rejected() raises:
    var X = Matrix[DType.float64](4, 2, 1.0)
    X[0, 0] = inf[DType.float64]()
    var y: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0, 4.0]

    var reg = DecisionTreeRegressor[DType.float64]()
    var caught = False
    try:
        reg.fit(X, y)
    except:
        caught = True
    assert_true(caught)


def test_decision_tree_regressor_float32_native() raises:
    var X = Matrix[DType.float32](8, 2, 0)
    var y = List[Scalar[DType.float32]](capacity=8)
    for i in range(8):
        X[i, 0] = Float32(i)
        X[i, 1] = Float32(i * 2)
        y.append(Float32(i * 10))

    var reg = DecisionTreeRegressor[DType.float32]()
    reg.fit(X, y)
    assert_true(reg.is_fitted)

    var preds = reg.predict(X)
    assert_equal(len(preds), 8)


def test_decision_tree_regressor_dataset_integration() raises:
    var data = _step_1d_dataset()
    var ds = Dataset[DType.float64, DType.float64](
        data[0].copy(), data[1].copy()
    )

    var reg = DecisionTreeRegressor[DType.float64]()
    fit_ds(reg, ds)
    assert_true(reg.is_fitted)

    var preds = predict_ds(reg, ds)
    assert_equal(len(preds), 10)


def test_decision_tree_regressor_pipeline_integration() raises:
    var data = _step_1d_dataset()
    var scaler = StandardScaler[DType.float64]()
    var tree = DecisionTreeRegressor[DType.float64](max_depth=3)

    var pipe = PipelineRegressor[
        StandardScaler[DType.float64],
        DecisionTreeRegressor[DType.float64],
        DType.float64,
    ](scaler^, tree^)

    pipe.fit(data[0], data[1])
    assert_true(pipe.regressor.is_fitted)

    var preds = pipe.predict(data[0])
    assert_equal(len(preds), 10)


def test_decision_tree_regressor_deep_tree_exact_memorization() raises:
    var X = Matrix[DType.float64](10, 1, 0)
    var y = List[Scalar[DType.float64]](capacity=10)
    for i in range(10):
        X[i, 0] = Float64(i)
        y.append(Float64(i * 7 + 3))

    var reg = DecisionTreeRegressor[DType.float64](max_depth=-1)
    reg.fit(X, y)

    var preds = reg.predict(X)
    for i in range(10):
        assert_almost_equal(preds[i], y[i], atol=1e-5)


def test_decision_tree_regressor_get_depth_and_get_n_leaves_properties() raises:
    var data = _step_1d_dataset()
    var reg = DecisionTreeRegressor[DType.float64](max_depth=1)
    reg.fit(data[0], data[1])

    assert_equal(reg.get_depth(), 1)
    assert_equal(reg.get_n_leaves(), 2)


def test_decision_tree_regressor_unfitted_get_depth_and_n_leaves_raises() raises:
    var reg = DecisionTreeRegressor[DType.float64]()
    var caught_depth = False
    try:
        _ = reg.get_depth()
    except:
        caught_depth = True
    assert_true(caught_depth)

    var caught_leaves = False
    try:
        _ = reg.get_n_leaves()
    except:
        caught_leaves = True
    assert_true(caught_leaves)


def test_decision_tree_regressor_piecewise_constant_staircase_5_steps() raises:
    var X = Matrix[DType.float64](25, 1, 0)
    var y = List[Scalar[DType.float64]](capacity=25)
    for i in range(25):
        X[i, 0] = Float64(i)
        var step_level = Float64((i // 5) * 10)
        y.append(step_level)

    var reg = DecisionTreeRegressor[DType.float64](max_depth=4)
    reg.fit(X, y)

    var preds = reg.predict(X)
    assert_equal(len(preds), 25)
    for i in range(25):
        assert_almost_equal(preds[i], y[i], atol=1e-4)


def test_decision_tree_regressor_friedman_mse_vs_squared_error_identical_stump() raises:
    var data = _step_1d_dataset()
    var reg_mse = DecisionTreeRegressor[DType.float64](
        criterion="squared_error", max_depth=1
    )
    var reg_friedman = DecisionTreeRegressor[DType.float64](
        criterion="friedman_mse", max_depth=1
    )
    reg_mse.fit(data[0], data[1])
    reg_friedman.fit(data[0], data[1])

    var preds_mse = reg_mse.predict(data[0])
    var preds_friedman = reg_friedman.predict(data[0])
    for i in range(10):
        assert_almost_equal(preds_mse[i], preds_friedman[i], atol=1e-5)


def test_decision_tree_regressor_mae_median_outlier_resistance() raises:
    # 4 samples at 0.0, 4 samples at 10.0, 1 outlier at 1000.0
    var X = Matrix[DType.float64](9, 1, 0)
    var y = List[Scalar[DType.float64]](capacity=9)
    for i in range(4):
        X[i, 0] = Float64(i)
        y.append(0.0)
    for i in range(4, 8):
        X[i, 0] = Float64(i)
        y.append(10.0)
    X[8, 0] = 8.0
    y.append(1000.0)

    # min_samples_leaf=4 forces split at t=3.5 (4 left, 5 right)
    var reg_mae = DecisionTreeRegressor[DType.float64](
        criterion="absolute_error", max_depth=1, min_samples_leaf=4
    )
    reg_mae.fit(X, y)

    var preds = reg_mae.predict(X)
    for i in range(4):
        assert_almost_equal(preds[i], 0.0, atol=1e-4)
    # Right partition [10, 10, 10, 10, 1000] median is 10.0
    for i in range(4, 9):
        assert_almost_equal(preds[i], 10.0, atol=1e-4)


def test_decision_tree_regressor_random_state_determinism() raises:
    var data = _step_1d_dataset()
    var reg1 = DecisionTreeRegressor[DType.float64](
        splitter="random", random_state=999
    )
    var reg2 = DecisionTreeRegressor[DType.float64](
        splitter="random", random_state=999
    )
    reg1.fit(data[0], data[1])
    reg2.fit(data[0], data[1])

    var p1 = reg1.predict(data[0])
    var p2 = reg2.predict(data[0])
    for i in range(10):
        assert_almost_equal(p1[i], p2[i], atol=1e-7)


def test_decision_tree_regressor_large_fat_matrix_100_features() raises:
    var X = Matrix[DType.float64](15, 100, 0)
    var y = List[Scalar[DType.float64]](capacity=15)
    for i in range(15):
        for j in range(100):
            X[i, j] = Float64(i * 3 + j)
        y.append(Float64(i * 5))

    var reg = DecisionTreeRegressor[DType.float64](
        max_depth=3, max_features="sqrt", random_state=42
    )
    reg.fit(X, y)
    assert_true(reg.is_fitted)

    var preds = reg.predict(X)
    assert_equal(len(preds), 15)


def test_decision_tree_regressor_min_samples_split_and_leaf_interaction() raises:
    var X = Matrix[DType.float64](10, 1, 0)
    var y = List[Scalar[DType.float64]](capacity=10)
    for i in range(10):
        X[i, 0] = Float64(i)
        y.append(Float64(i))

    var reg = DecisionTreeRegressor[DType.float64](
        min_samples_split=4, min_samples_leaf=2
    )
    reg.fit(X, y)

    for n in range(reg.tree_.node_count()):
        if reg.tree_.nodes[n].is_leaf:
            assert_true(reg.tree_.nodes[n].n_node_samples >= 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
