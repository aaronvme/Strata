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
    LinearRegression,
    StandardScaler,
    PipelineRegressor,
    DimensionMismatchError,
    InvalidParameterError,
    NotFittedError,
)
from strata.base import fit, predict


def test_linear_regression_1d_fit_intercept() raises:
    # y = 2.5 * x + 1.0
    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = 0.0
    X[1, 0] = 1.0
    X[2, 0] = 2.0
    X[3, 0] = 3.0

    var y: List[Scalar[DType.float64]] = [1.0, 3.5, 6.0, 8.5]

    var solvers: List[String] = ["lstsq", "qr", "cholesky", "solve"]
    for idx in range(len(solvers)):
        var solver_name = solvers[idx]
        var model = LinearRegression(fit_intercept=True, solver=solver_name)
        model.fit(X, y)

        assert_almost_equal(model.intercept_, 1.0, rtol=1e-4)
        assert_equal(len(model.coef_), 1)
        assert_almost_equal(model.coef_[0], 2.5, rtol=1e-4)

        var preds = model.predict(X)
        assert_equal(len(preds), 4)
        assert_almost_equal(preds[0], 1.0, rtol=1e-4)
        assert_almost_equal(preds[1], 3.5, rtol=1e-4)
        assert_almost_equal(preds[2], 6.0, rtol=1e-4)
        assert_almost_equal(preds[3], 8.5, rtol=1e-4)


def test_linear_regression_multivariate() raises:
    # y = 1.5*x1 - 2.0*x2 + 3.0*x3 + 4.0
    var X = Matrix[DType.float64](5, 3, 0)
    X[0, 0] = 1.0
    X[0, 1] = 0.0
    X[0, 2] = 2.0
    X[1, 0] = 2.0
    X[1, 1] = 1.0
    X[1, 2] = 0.0
    X[2, 0] = 0.0
    X[2, 1] = 3.0
    X[2, 2] = 1.0
    X[3, 0] = 1.0
    X[3, 1] = 1.0
    X[3, 2] = 1.0
    X[4, 0] = 2.0
    X[4, 1] = 2.0
    X[4, 2] = 2.0

    var y = List[Scalar[DType.float64]](capacity=5)
    for i in range(5):
        var val = (
            1.5 * Float64(X[i, 0])
            - 2.0 * Float64(X[i, 1])
            + 3.0 * Float64(X[i, 2])
            + 4.0
        )
        y.append(val)

    var solvers: List[String] = ["lstsq", "qr", "cholesky", "solve"]
    for idx in range(len(solvers)):
        var solver_name = solvers[idx]
        var model = LinearRegression(fit_intercept=True, solver=solver_name)
        model.fit(X, y)

        assert_almost_equal(model.intercept_, 4.0, rtol=1e-4)
        assert_equal(len(model.coef_), 3)
        assert_almost_equal(model.coef_[0], 1.5, rtol=1e-4)
        assert_almost_equal(model.coef_[1], -2.0, rtol=1e-4)
        assert_almost_equal(model.coef_[2], 3.0, rtol=1e-4)

        var preds = model.predict(X)
        for i in range(5):
            assert_almost_equal(preds[i], y[i], rtol=1e-4)


def test_linear_regression_no_intercept() raises:
    # y = 3.0*x1 + 4.0*x2 (passes through origin)
    var X = Matrix[DType.float64](3, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 2.0
    X[1, 0] = 2.0
    X[1, 1] = 1.0
    X[2, 0] = 3.0
    X[2, 1] = 0.0

    var y: List[Scalar[DType.float64]] = [11.0, 10.0, 9.0]

    var solvers: List[String] = ["lstsq", "qr", "cholesky", "solve"]
    for idx in range(len(solvers)):
        var solver_name = solvers[idx]
        var model = LinearRegression(fit_intercept=False, solver=solver_name)
        model.fit(X, y)

        assert_equal(model.intercept_, 0.0)
        assert_almost_equal(model.coef_[0], 3.0, rtol=1e-4)
        assert_almost_equal(model.coef_[1], 4.0, rtol=1e-4)


def test_linear_regression_dtype_flexibility() raises:
    # 1. Integer inputs automatically promoted upfront
    var X_int = Matrix[DType.int32](3, 2, 0)
    X_int[0, 0] = 1
    X_int[0, 1] = 1
    X_int[1, 0] = 2
    X_int[1, 1] = 1
    X_int[2, 0] = 3
    X_int[2, 1] = 2

    # y = 2*x1 + 3*x2 + 1
    var y_int: List[Scalar[DType.int32]] = [6, 8, 13]

    var model = LinearRegression()
    model.fit(X_int, y_int)

    assert_almost_equal(model.intercept_, 1.0, rtol=1e-4)
    assert_almost_equal(model.coef_[0], 2.0, rtol=1e-4)
    assert_almost_equal(model.coef_[1], 3.0, rtol=1e-4)

    # Predictions return matching integer type
    var preds_int = model.predict(X_int)
    assert_equal(preds_int[0], 6)
    assert_equal(preds_int[1], 8)
    assert_equal(preds_int[2], 13)

    # 2. Float32 precision prediction matching
    var X_f32 = X_int.cast[DType.float32]()
    var preds_f32 = model.predict(X_f32)
    assert_almost_equal(preds_f32[0], Float32(6.0), rtol=1e-3)


def test_linear_regression_float32_native_model() raises:
    # Native Float32 model with Float32 inputs
    var X = Matrix[DType.float32](3, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 0.0
    X[1, 0] = 0.0
    X[1, 1] = 1.0
    X[2, 0] = 1.0
    X[2, 1] = 1.0

    # y = 2.0*x1 + 3.0*x2 + 1.0
    var y: List[Scalar[DType.float32]] = [3.0, 4.0, 6.0]

    var model = LinearRegression[DType.float32](fit_intercept=True)
    model.fit(X, y)

    assert_almost_equal(model.intercept_, Float32(1.0), rtol=1e-3)
    assert_almost_equal(model.coef_[0], Float32(2.0), rtol=1e-3)
    assert_almost_equal(model.coef_[1], Float32(3.0), rtol=1e-3)

    var preds = model.predict(X)
    assert_almost_equal(preds[0], Float32(3.0), rtol=1e-3)
    assert_almost_equal(preds[1], Float32(4.0), rtol=1e-3)
    assert_almost_equal(preds[2], Float32(6.0), rtol=1e-3)


def test_linear_regression_underdetermined_fat_matrix() raises:
    # Underdetermined system: N=2 samples, D=4 features (fat matrix, n < p)
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

    # 1. lstsq (SVD) finds minimum Euclidean norm solution
    var model_lstsq = LinearRegression(fit_intercept=False, solver="lstsq")
    model_lstsq.fit(X, y)

    var preds = model_lstsq.predict(X)
    assert_almost_equal(preds[0], 3.0, rtol=1e-4)
    assert_almost_equal(preds[1], 5.0, rtol=1e-4)
    assert_equal(len(model_lstsq.coef_), 4)

    # 2. Cholesky fails on singular XtX
    var model_cholesky = LinearRegression(
        fit_intercept=False, solver="cholesky"
    )
    with assert_raises():
        model_cholesky.fit(X, y)

    # 3. Direct LU solve fails on singular XtX
    var model_solve = LinearRegression(fit_intercept=False, solver="solve")
    with assert_raises():
        model_solve.fit(X, y)


def test_linear_regression_prediction_dimension_mismatch() raises:
    # Fit on 3 features
    var X_train = Matrix[DType.float64].ones(4, 3)
    var y_train: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0, 4.0]

    var model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on 2 features (mismatch: expected 3, got 2)
    var X_bad = Matrix[DType.float64].ones(2, 2)
    with assert_raises():
        _ = model.predict(X_bad)


def test_linear_regression_sample_size_discrepancies() raises:
    # 1. Empty inputs raise DimensionMismatchError
    var X_empty = Matrix[DType.float64](0, 2, 0)
    var y_empty = List[Scalar[DType.float64]]()
    var model = LinearRegression()
    with assert_raises():
        model.fit(X_empty, y_empty)

    # 2. Single sample (N = 1) with fit_intercept=True:
    # Centering leaves X_c = [0, 0], y_c = [0] -> coef_ == [0, 0] and intercept_ == 42.0
    var X_single = Matrix[DType.float64](1, 2, 0)
    X_single[0, 0] = 5.0
    X_single[0, 1] = 10.0
    var y_single: List[Scalar[DType.float64]] = [42.0]

    var model_single = LinearRegression(fit_intercept=True)
    model_single.fit(X_single, y_single)
    assert_almost_equal(model_single.intercept_, 42.0, rtol=1e-4)
    assert_equal(len(model_single.coef_), 2)
    assert_almost_equal(model_single.coef_[0], 0.0, rtol=1e-4)
    assert_almost_equal(model_single.coef_[1], 0.0, rtol=1e-4)

    var preds_single = model_single.predict(X_single)
    assert_almost_equal(preds_single[0], 42.0, rtol=1e-4)


def test_linear_regression_centering_and_shift_invariance() raises:
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

    var model1 = LinearRegression(fit_intercept=True)
    model1.fit(X, y)

    # Shift targets by constant c = 100.0: y_shifted = y + 100
    var y_shifted = List[Scalar[DType.float64]](capacity=4)
    for i in range(4):
        y_shifted.append(y[i] + 100.0)

    var model2 = LinearRegression(fit_intercept=True)
    model2.fit(X, y_shifted)

    # Slopes (coef_) must be identical
    assert_almost_equal(model1.coef_[0], model2.coef_[0], rtol=1e-4)
    assert_almost_equal(model1.coef_[1], model2.coef_[1], rtol=1e-4)

    # Intercept must be shifted by exactly +100.0
    assert_almost_equal(model2.intercept_, model1.intercept_ + 100.0, rtol=1e-4)


def test_linear_regression_constant_zero_variance_column() raises:
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

    # 1. Default lstsq (SVD) absorbs constant column into intercept
    var model = LinearRegression(fit_intercept=True, solver="lstsq")
    model.fit(X, y)

    assert_almost_equal(model.intercept_, 5.0, rtol=1e-3)
    assert_almost_equal(model.coef_[0], 3.0, rtol=1e-3)
    assert_almost_equal(model.coef_[1], 0.0, rtol=1e-3)

    var preds = model.predict(X)
    for i in range(4):
        assert_almost_equal(preds[i], y[i], rtol=1e-3)

    # 2. Non-SVD solvers fail due to singular centered matrix
    var model_cholesky = LinearRegression(fit_intercept=True, solver="cholesky")
    with assert_raises():
        model_cholesky.fit(X, y)

    var model_solve = LinearRegression(fit_intercept=True, solver="solve")
    with assert_raises():
        model_solve.fit(X, y)


def test_linear_regression_ill_conditioned_matrix() raises:
    # Near-singular columns: x1 = x0 + 1e-7
    var X = Matrix[DType.float64](3, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 1.0000001
    X[1, 0] = 2.0
    X[1, 1] = 2.0000001
    X[2, 0] = 3.0
    X[2, 1] = 3.0000001

    var y: List[Scalar[DType.float64]] = [2.0, 4.0, 6.0]

    # lstsq solves without crashing or producing NaNs
    var model_lstsq = LinearRegression(fit_intercept=False, solver="lstsq")
    model_lstsq.fit(X, y)

    var preds = model_lstsq.predict(X)
    for i in range(3):
        assert_almost_equal(preds[i], y[i], rtol=1e-3)


def test_linear_regression_transposed_and_sliced_inputs() raises:
    # Build raw 2x4 matrix and transpose to 4x2
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

    var model = LinearRegression(fit_intercept=True)
    model.fit(X, y)

    assert_almost_equal(model.intercept_, 5.0, rtol=1e-4)
    assert_almost_equal(model.coef_[0], 3.0, rtol=1e-4)
    assert_almost_equal(model.coef_[1], -2.0, rtol=1e-4)


def test_linear_regression_non_finite_inputs() raises:
    var nan_val = Float64(0.0) / Float64(0.0)
    var inf_val = Float64(1.0) / Float64(0.0)

    # 1. NaN in X
    var X_nan = Matrix[DType.float64](3, 2, 1.0)
    X_nan[1, 0] = nan_val
    var y_valid: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0]
    var model = LinearRegression()
    with assert_raises():
        model.fit(X_nan, y_valid)

    # 2. Inf in X
    var X_inf = Matrix[DType.float64](3, 2, 1.0)
    X_inf[0, 1] = inf_val
    with assert_raises():
        model.fit(X_inf, y_valid)

    # 3. NaN in y
    var X_valid = Matrix[DType.float64].ones(3, 2)
    var y_nan: List[Scalar[DType.float64]] = [1.0, nan_val, 3.0]
    with assert_raises():
        model.fit(X_valid, y_nan)


def test_linear_regression_all_zero_target() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 2.0
    X[1, 0] = 3.0
    X[1, 1] = 4.0
    X[2, 0] = 5.0
    X[2, 1] = 6.0
    X[3, 0] = 7.0
    X[3, 1] = 8.0

    # y = [0, 0, 0, 0]
    var y: List[Scalar[DType.float64]] = [0.0, 0.0, 0.0, 0.0]

    var model = LinearRegression(fit_intercept=True)
    model.fit(X, y)

    assert_almost_equal(model.intercept_, 0.0, rtol=1e-5)
    assert_almost_equal(model.coef_[0], 0.0, rtol=1e-5)
    assert_almost_equal(model.coef_[1], 0.0, rtol=1e-5)

    var preds = model.predict(X)
    for i in range(4):
        assert_almost_equal(preds[i], 0.0, rtol=1e-5)


def test_linear_regression_model_refitting_state_reset() raises:
    # 1. Fit on 3 features
    var X1 = Matrix[DType.float64](4, 3, 0)
    var y1: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0, 4.0]
    var model = LinearRegression()
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
    assert_almost_equal(model.coef_[0], 2.0, rtol=1e-4)
    assert_almost_equal(model.intercept_, 0.0, rtol=1e-4)


def test_linear_regression_dataset_and_pipeline() raises:
    # 1. Generic Dataset container helpers: fit(model, ds) and predict(model, ds)
    var X = Matrix[DType.float64](3, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 0.0
    X[1, 0] = 0.0
    X[1, 1] = 1.0
    X[2, 0] = 1.0
    X[2, 1] = 1.0

    var y: List[Scalar[DType.float64]] = [2.0, 3.0, 5.0]
    var ds = Dataset[DType.float64, DType.float64](X.copy(), y.copy())

    var model = LinearRegression()
    fit(model, ds)
    var preds = predict(model, ds)
    assert_almost_equal(preds[0], 2.0, rtol=1e-4)
    assert_almost_equal(preds[1], 3.0, rtol=1e-4)
    assert_almost_equal(preds[2], 5.0, rtol=1e-4)

    # 2. Pipeline composition: StandardScaler -> LinearRegression
    var scaler = StandardScaler()
    var reg = LinearRegression()
    var pipe = PipelineRegressor(scaler^, reg^)

    pipe.fit(X, y)
    var pipe_preds = pipe.predict(X)
    assert_equal(len(pipe_preds), 3)
    assert_almost_equal(pipe_preds[0], 2.0, rtol=1e-4)


def test_linear_regression_error_handling() raises:
    var model = LinearRegression()
    var X = Matrix[DType.float64].ones(3, 2)

    # 1. Predict before fit raises NotFittedError
    with assert_raises():
        _ = model.predict(X)

    # 2. Dimension mismatch between X and y
    var bad_y: List[Scalar[DType.float64]] = [1.0, 2.0]  # Expected len=3
    with assert_raises():
        model.fit(X, bad_y)

    # 3. Invalid solver name raises InvalidParameterError
    var bad_solver_model = LinearRegression(solver="invalid_solver")
    var valid_y: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0]
    with assert_raises():
        bad_solver_model.fit(X, valid_y)


def test_linear_regression_large_scale_conditioning() raises:
    # High dynamic range across features (scale disparity)
    var X = Matrix[DType.float64](3, 2, 0)
    X[0, 0] = 1e-6
    X[0, 1] = 1e6
    X[1, 0] = 2e-6
    X[1, 1] = 2e6
    X[2, 0] = 3e-6
    X[2, 1] = 1e6

    # y = 1e6 * x0 + 1e-6 * x1 = 1.0 + 1.0 = 2.0, etc.
    var y: List[Scalar[DType.float64]] = [2.0, 4.0, 4.0]

    var model = LinearRegression(fit_intercept=False, solver="lstsq")
    model.fit(X, y)

    var preds = model.predict(X)
    for i in range(3):
        assert_almost_equal(preds[i], y[i], rtol=1e-3)


def test_linear_regression_predict_feature_count_mismatch_greater() raises:
    var X_train = Matrix[DType.float64].ones(4, 2)
    var y_train: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0, 4.0]
    var model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on 4 features (expected 2)
    var X_bad = Matrix[DType.float64].ones(2, 4)
    with assert_raises():
        _ = model.predict(X_bad)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
