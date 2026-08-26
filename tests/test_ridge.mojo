from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_almost_equal,
    assert_raises,
)
from strata import (
    Matrix,
    Dataset,
    Ridge,
    StandardScaler,
    PipelineRegressor,
    DimensionMismatchError,
    InvalidParameterError,
    NotFittedError,
)
from strata.base import fit, predict


def test_ridge_1d_fit_intercept() raises:
    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = 0.0
    X[1, 0] = 1.0
    X[2, 0] = 2.0
    X[3, 0] = 3.0

    # y = 2.0 * x + 1.0
    var y: List[Scalar[DType.float64]] = [1.0, 3.0, 5.0, 7.0]

    var solvers: List[String] = ["auto", "cholesky", "svd", "solve"]
    for idx in range(len(solvers)):
        var solver_name = solvers[idx]

        # 1. alpha = 0.0 recovers exact OLS
        var model_ols = Ridge(alpha=0.0, fit_intercept=True, solver=solver_name)
        model_ols.fit(X, y)
        assert_almost_equal(model_ols.intercept_, 1.0, rtol=1e-4)
        assert_almost_equal(model_ols.coef_[0], 2.0, rtol=1e-4)

        # 2. alpha = 1.0 shrinks coefficient towards zero
        var model_ridge = Ridge(
            alpha=1.0, fit_intercept=True, solver=solver_name
        )
        model_ridge.fit(X, y)
        # Centered sum of squares is 5.0. With alpha=1.0, slope = 2.0 * (5.0 / (5.0 + 1.0)) = 1.6667
        assert_almost_equal(model_ridge.coef_[0], 1.6666666667, rtol=1e-4)
        assert_almost_equal(model_ridge.intercept_, 1.5, rtol=1e-4)


def test_ridge_no_intercept() raises:
    # y = 3.0*x1 + 4.0*x2 (passes through origin)
    var X = Matrix[DType.float64](3, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 2.0
    X[1, 0] = 2.0
    X[1, 1] = 1.0
    X[2, 0] = 3.0
    X[2, 1] = 0.0

    var y: List[Scalar[DType.float64]] = [11.0, 10.0, 9.0]

    var solvers: List[String] = ["auto", "cholesky", "svd", "solve"]
    for idx in range(len(solvers)):
        var solver_name = solvers[idx]
        var model = Ridge(alpha=0.0, fit_intercept=False, solver=solver_name)
        model.fit(X, y)

        assert_equal(model.intercept_, 0.0)
        assert_almost_equal(model.coef_[0], 3.0, rtol=1e-4)
        assert_almost_equal(model.coef_[1], 4.0, rtol=1e-4)


def test_ridge_regularization_shrinkage() raises:
    # Verify coefficients shrink monotonically as alpha increases
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 0.0
    X[1, 0] = 2.0
    X[1, 1] = 1.0
    X[2, 0] = 3.0
    X[2, 1] = 2.0
    X[3, 0] = 4.0
    X[3, 1] = 3.0

    var y: List[Scalar[DType.float64]] = [2.0, 5.0, 8.0, 11.0]

    var m1 = Ridge(alpha=0.1)
    m1.fit(X, y)

    var m2 = Ridge(alpha=1.0)
    m2.fit(X, y)

    var m3 = Ridge(alpha=10.0)
    m3.fit(X, y)

    var m4 = Ridge(alpha=100.0)
    m4.fit(X, y)

    var norm1 = m1.coef_[0] * m1.coef_[0] + m1.coef_[1] * m1.coef_[1]
    var norm2 = m2.coef_[0] * m2.coef_[0] + m2.coef_[1] * m2.coef_[1]
    var norm3 = m3.coef_[0] * m3.coef_[0] + m3.coef_[1] * m3.coef_[1]
    var norm4 = m4.coef_[0] * m4.coef_[0] + m4.coef_[1] * m4.coef_[1]

    # Strictly monotonic shrinkage
    assert_true(norm1 > norm2)
    assert_true(norm2 > norm3)
    assert_true(norm3 > norm4)


def test_ridge_asymptotic_infinite_alpha() raises:
    # As alpha -> infinity, beta -> 0 and intercept -> y_mean
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 2.0
    X[1, 0] = 2.0
    X[1, 1] = 1.0
    X[2, 0] = 3.0
    X[2, 1] = 4.0
    X[3, 0] = 4.0
    X[3, 1] = 3.0

    var y: List[Scalar[DType.float64]] = [4.0, 8.0, 12.0, 16.0]  # y_mean = 10.0

    var model = Ridge(alpha=1e12, fit_intercept=True)
    model.fit(X, y)

    assert_almost_equal(model.intercept_, 10.0, rtol=1e-4)
    assert_almost_equal(model.coef_[0], 0.0, atol=1e-6)
    assert_almost_equal(model.coef_[1], 0.0, atol=1e-6)


def test_ridge_multivariate_collinear_stabilization() raises:
    # Collinear columns: Col 1 = 2 * Col 0
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 2.0
    X[1, 0] = 2.0
    X[1, 1] = 4.0
    X[2, 0] = 3.0
    X[2, 1] = 6.0
    X[3, 0] = 4.0
    X[3, 1] = 8.0

    var y: List[Scalar[DType.float64]] = [5.0, 10.0, 15.0, 20.0]

    # With alpha > 0, XtX + alpha*I is strictly positive definite
    var solvers: List[String] = ["auto", "cholesky", "svd", "solve"]
    for idx in range(len(solvers)):
        var solver_name = solvers[idx]
        var model = Ridge(alpha=1.0, fit_intercept=True, solver=solver_name)
        model.fit(X, y)

        assert_equal(len(model.coef_), 2)
        var preds = model.predict(X)
        assert_equal(len(preds), 4)


def test_ridge_fat_matrix_underdetermined() raises:
    # Fat matrix: N=2, D=4 (n < p)
    var X = Matrix[DType.float64](2, 4, 0)
    X[0, 0] = 1.0
    X[0, 1] = 0.0
    X[0, 2] = 1.0
    X[0, 3] = 0.0
    X[1, 0] = 0.0
    X[1, 1] = 1.0
    X[1, 2] = 0.0
    X[1, 3] = 1.0

    var y: List[Scalar[DType.float64]] = [3.0, 5.0]

    # SVD and Cholesky with regularization handle fat matrix
    var model_svd = Ridge(alpha=0.5, fit_intercept=False, solver="svd")
    model_svd.fit(X, y)
    assert_equal(len(model_svd.coef_), 4)

    var preds = model_svd.predict(X)
    assert_equal(len(preds), 2)


def test_ridge_sample_size_discrepancies() raises:
    # 1. Empty inputs raise DimensionMismatchError
    var X_empty = Matrix[DType.float64](0, 2, 0)
    var y_empty = List[Scalar[DType.float64]]()
    var model = Ridge()
    with assert_raises():
        model.fit(X_empty, y_empty)

    # 2. Single sample (N = 1) with fit_intercept=True:
    # Centering leaves X_c = [0, 0], y_c = [0] -> coef_ == [0, 0] and intercept_ == 42.0
    var X_single = Matrix[DType.float64](1, 2, 0)
    X_single[0, 0] = 5.0
    X_single[0, 1] = 10.0
    var y_single: List[Scalar[DType.float64]] = [42.0]

    var model_single = Ridge(alpha=1.0, fit_intercept=True)
    model_single.fit(X_single, y_single)
    assert_almost_equal(model_single.intercept_, 42.0, rtol=1e-4)
    assert_equal(len(model_single.coef_), 2)
    assert_almost_equal(model_single.coef_[0], 0.0, rtol=1e-4)
    assert_almost_equal(model_single.coef_[1], 0.0, rtol=1e-4)

    var preds_single = model_single.predict(X_single)
    assert_almost_equal(preds_single[0], 42.0, rtol=1e-4)


def test_ridge_constant_zero_variance_column() raises:
    # Column 1 has constant value 1.0 (zero variance)
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 1.0
    X[1, 0] = 2.0
    X[1, 1] = 1.0
    X[2, 0] = 3.0
    X[2, 1] = 1.0
    X[3, 0] = 4.0
    X[3, 1] = 1.0

    # y = 3.0 * x0 + 5.0
    var y: List[Scalar[DType.float64]] = [8.0, 11.0, 14.0, 17.0]

    var model = Ridge(alpha=0.5, fit_intercept=True)
    model.fit(X, y)

    var preds = model.predict(X)
    for i in range(4):
        assert_true(preds[i] > 0.0)


def test_ridge_all_zero_target() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 2.0
    X[1, 0] = 3.0
    X[1, 1] = 4.0
    X[2, 0] = 5.0
    X[2, 1] = 6.0
    X[3, 0] = 7.0
    X[3, 1] = 8.0

    var y: List[Scalar[DType.float64]] = [0.0, 0.0, 0.0, 0.0]

    var model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit(X, y)

    assert_almost_equal(model.intercept_, 0.0, rtol=1e-5)
    assert_almost_equal(model.coef_[0], 0.0, rtol=1e-5)
    assert_almost_equal(model.coef_[1], 0.0, rtol=1e-5)


def test_ridge_dtype_flexibility_and_float32() raises:
    # 1. Integer inputs promoted upfront
    var X_int = Matrix[DType.int32](3, 2, 0)
    X_int[0, 0] = 1
    X_int[0, 1] = 1
    X_int[1, 0] = 2
    X_int[1, 1] = 1
    X_int[2, 0] = 3
    X_int[2, 1] = 2
    var y_int: List[Scalar[DType.int32]] = [6, 8, 13]

    var model = Ridge(alpha=0.1)
    model.fit(X_int, y_int)

    var preds_int = model.predict(X_int)
    assert_equal(len(preds_int), 3)

    # 2. Native Float32 model
    var X_f32 = X_int.cast[DType.float32]()
    var y_f32: List[Scalar[DType.float32]] = [
        Float32(6.0),
        Float32(8.0),
        Float32(13.0),
    ]

    var model_f32 = Ridge[DType.float32](alpha=Float32(0.5))
    model_f32.fit(X_f32, y_f32)
    assert_equal(len(model_f32.coef_), 2)

    var preds_f32 = model_f32.predict(X_f32)
    assert_equal(len(preds_f32), 3)


def test_ridge_centering_and_shift_invariance() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 2.0
    X[1, 0] = 2.0
    X[1, 1] = 3.0
    X[2, 0] = 3.0
    X[2, 1] = 1.0
    X[3, 0] = 4.0
    X[3, 1] = 5.0

    var y: List[Scalar[DType.float64]] = [5.0, 8.0, 7.0, 14.0]

    var model1 = Ridge(alpha=1.0, fit_intercept=True)
    model1.fit(X, y)

    # Shift targets: y_shifted = y + 50
    var y_shifted = List[Scalar[DType.float64]](capacity=4)
    for i in range(4):
        y_shifted.append(y[i] + 50.0)

    var model2 = Ridge(alpha=1.0, fit_intercept=True)
    model2.fit(X, y_shifted)

    # Slopes are identical
    assert_almost_equal(model1.coef_[0], model2.coef_[0], rtol=1e-4)
    assert_almost_equal(model1.coef_[1], model2.coef_[1], rtol=1e-4)

    # Intercept shifts by +50.0
    assert_almost_equal(model2.intercept_, model1.intercept_ + 50.0, rtol=1e-4)


def test_ridge_transposed_inputs() raises:
    var X_raw = Matrix[DType.float64](2, 4, 0)
    X_raw[0, 0] = 1.0
    X_raw[0, 1] = 2.0
    X_raw[0, 2] = 3.0
    X_raw[0, 3] = 4.0
    X_raw[1, 0] = 2.0
    X_raw[1, 1] = 1.0
    X_raw[1, 2] = 4.0
    X_raw[1, 3] = 3.0

    var X = X_raw.transpose()  # 4x2
    var y: List[Scalar[DType.float64]] = [4.0, 9.0, 6.0, 11.0]

    var model = Ridge(alpha=0.5, fit_intercept=True)
    model.fit(X, y)

    var preds = model.predict(X)
    assert_equal(len(preds), 4)


def test_ridge_model_refitting_state_reset() raises:
    # 1. Fit on 3 features
    var X1 = Matrix[DType.float64](4, 3, 0)
    var y1: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0, 4.0]
    var model = Ridge(alpha=1.0)
    model.fit(X1, y1)
    assert_equal(len(model.coef_), 3)

    # 2. Refit same instance on 1 feature
    var X2 = Matrix[DType.float64](4, 1, 0)
    X2[0, 0] = 1.0
    X2[1, 0] = 2.0
    X2[2, 0] = 3.0
    X2[3, 0] = 4.0
    var y2: List[Scalar[DType.float64]] = [2.0, 4.0, 6.0, 8.0]

    model.fit(X2, y2)
    assert_equal(len(model.coef_), 1)


def test_ridge_dataset_and_pipeline() raises:
    # 1. Dataset container helpers
    var X = Matrix[DType.float64](3, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 0.0
    X[1, 0] = 0.0
    X[1, 1] = 1.0
    X[2, 0] = 1.0
    X[2, 1] = 1.0

    var y: List[Scalar[DType.float64]] = [2.0, 3.0, 5.0]
    var ds = Dataset[DType.float64, DType.float64](X.copy(), y.copy())

    var model = Ridge(alpha=0.5)
    fit(model, ds)
    var preds = predict(model, ds)
    assert_equal(len(preds), 3)

    # 2. Pipeline composition
    var scaler = StandardScaler()
    var ridge_reg = Ridge(alpha=1.0)
    var pipe = PipelineRegressor(scaler^, ridge_reg^)

    pipe.fit(X, y)
    var pipe_preds = pipe.predict(X)
    assert_equal(len(pipe_preds), 3)


def test_ridge_error_handling() raises:
    # 1. Negative alpha raises InvalidParameterError
    with assert_raises():
        _ = Ridge(alpha=-1.0)

    # 2. Predict before fit raises NotFittedError
    var model = Ridge()
    var X = Matrix[DType.float64].ones(3, 2)
    with assert_raises():
        _ = model.predict(X)

    # 3. Feature dimension mismatch on predict
    var y: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0]
    model.fit(X, y)
    var X_bad = Matrix[DType.float64].ones(2, 4)
    with assert_raises():
        _ = model.predict(X_bad)

    # 4. Invalid solver raises InvalidParameterError
    var bad_solver = Ridge(solver="invalid")
    with assert_raises():
        bad_solver.fit(X, y)

    # 5. Non-finite input raises InvalidParameterError
    var nan_val = Float64(0.0) / Float64(0.0)
    var X_nan = Matrix[DType.float64](3, 2, 1.0)
    X_nan[0, 0] = nan_val
    with assert_raises():
        model.fit(X_nan, y)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
