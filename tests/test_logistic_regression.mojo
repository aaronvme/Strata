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
    LogisticRegression,
    StandardScaler,
    PipelineClassifier,
    DimensionMismatchError,
    InvalidParameterError,
    NotFittedError,
)
from strata.base import fit, predict, predict_proba


def test_logistic_regression_binary_separable() raises:
    var X = Matrix[DType.float64](6, 2, 0)
    X[0, 0] = -2.0
    X[0, 1] = -2.0
    X[1, 0] = -1.5
    X[1, 1] = -1.0
    X[2, 0] = -1.0
    X[2, 1] = -1.5
    X[3, 0] = 1.0
    X[3, 1] = 1.5
    X[4, 0] = 1.5
    X[4, 1] = 1.0
    X[5, 0] = 2.0
    X[5, 1] = 2.0

    var y: List[Scalar[DType.int32]] = [0, 0, 0, 1, 1, 1]

    var model = LogisticRegression(C=1.0, max_iter=200, learning_rate=0.5)
    model.fit(X, y)

    var preds = model.predict(X)
    for i in range(6):
        assert_equal(preds[i], Int(y[i]))

    var probs = model.predict_proba(X)
    assert_equal(probs.rows, 6)
    assert_equal(probs.cols, 2)
    for i in range(6):
        var row_sum = probs[i, 0] + probs[i, 1]
        assert_almost_equal(row_sum, 1.0, rtol=1e-4)
        assert_true(probs[i, Int(y[i])] > 0.5)


def test_logistic_regression_multiclass() raises:
    var X = Matrix[DType.float64](6, 2, 0)
    X[0, 0] = -2.0
    X[0, 1] = -2.0  # class 0
    X[1, 0] = -2.5
    X[1, 1] = -1.5  # class 0
    X[2, 0] = 0.0
    X[2, 1] = 2.0  # class 1
    X[3, 0] = 0.5
    X[3, 1] = 2.5  # class 1
    X[4, 0] = 2.0
    X[4, 1] = -2.0  # class 2
    X[5, 0] = 2.5
    X[5, 1] = -1.5  # class 2

    var y: List[Scalar[DType.int32]] = [0, 0, 1, 1, 2, 2]

    var model = LogisticRegression(C=10.0, max_iter=300, learning_rate=0.5)
    model.fit(X, y)

    var preds = model.predict(X)
    for i in range(6):
        assert_equal(preds[i], Int(y[i]))

    var probs = model.predict_proba(X)
    assert_equal(probs.rows, 6)
    assert_equal(probs.cols, 3)
    for i in range(6):
        var row_sum = probs[i, 0] + probs[i, 1] + probs[i, 2]
        assert_almost_equal(row_sum, 1.0, rtol=1e-4)
        assert_true(probs[i, Int(y[i])] > 0.6)


def test_logistic_regression_arbitrary_labels() raises:
    # Test discrete non-standard labels: [-1, 1] and [10, 20, 30]
    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = -2.0
    X[1, 0] = -1.0
    X[2, 0] = 1.0
    X[3, 0] = 2.0

    var y_neg_pos: List[Scalar[DType.int32]] = [-1, -1, 1, 1]

    var model = LogisticRegression(C=1.0, max_iter=200, learning_rate=0.5)
    model.fit(X, y_neg_pos)

    var preds = model.predict(X)
    assert_equal(preds[0], -1)
    assert_equal(preds[1], -1)
    assert_equal(preds[2], 1)
    assert_equal(preds[3], 1)


def test_logistic_regression_unregularized_none_penalty() raises:
    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = -2.0
    X[1, 0] = -1.0
    X[2, 0] = 1.0
    X[3, 0] = 2.0

    var y: List[Scalar[DType.int32]] = [0, 0, 1, 1]

    var model = LogisticRegression(
        penalty="none", max_iter=200, learning_rate=0.5
    )
    model.fit(X, y)

    var preds = model.predict(X)
    assert_equal(preds[0], 0)
    assert_equal(preds[1], 0)
    assert_equal(preds[2], 1)
    assert_equal(preds[3], 1)


def test_logistic_regression_regularization_shrinkage_C() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = -2.0
    X[0, 1] = -1.0
    X[1, 0] = -1.0
    X[1, 1] = -2.0
    X[2, 0] = 1.0
    X[2, 1] = 2.0
    X[3, 0] = 2.0
    X[3, 1] = 1.0

    var y: List[Scalar[DType.int32]] = [0, 0, 1, 1]

    # Strong regularization (C = 0.1) vs weak regularization (C = 100.0)
    var m_strong = LogisticRegression(
        penalty="l2", C=0.1, max_iter=200, learning_rate=0.05
    )
    m_strong.fit(X, y)

    var m_weak = LogisticRegression(
        penalty="l2", C=100.0, max_iter=200, learning_rate=0.05
    )
    m_weak.fit(X, y)

    var norm_strong: Float64 = 0.0
    for k in range(m_strong.coef_.rows):
        for j in range(m_strong.coef_.cols):
            norm_strong += m_strong.coef_[k, j] * m_strong.coef_[k, j]

    var norm_weak: Float64 = 0.0
    for k in range(m_weak.coef_.rows):
        for j in range(m_weak.coef_.cols):
            norm_weak += m_weak.coef_[k, j] * m_weak.coef_[k, j]

    assert_true(norm_strong < norm_weak)


def test_logistic_regression_fit_intercept_false() raises:
    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = -2.0
    X[1, 0] = -1.0
    X[2, 0] = 1.0
    X[3, 0] = 2.0

    var y: List[Scalar[DType.int32]] = [0, 0, 1, 1]

    var model = LogisticRegression(
        fit_intercept=False, max_iter=200, learning_rate=0.5
    )
    model.fit(X, y)

    for k in range(len(model.intercept_)):
        assert_equal(model.intercept_[k], 0.0)


def test_logistic_regression_dtype_flexibility_and_float32() raises:
    # 1. Integer feature matrix
    var X_int = Matrix[DType.int32](4, 1, 0)
    X_int[0, 0] = -2
    X_int[1, 0] = -1
    X_int[2, 0] = 1
    X_int[3, 0] = 2

    var y: List[Scalar[DType.int32]] = [0, 0, 1, 1]

    var model = LogisticRegression(max_iter=200, learning_rate=0.5)
    model.fit(X_int, y)

    var preds_int = model.predict(X_int)
    assert_equal(preds_int[0], 0)
    assert_equal(preds_int[3], 1)

    # 2. Native Float32 model
    var X_f32 = X_int.cast[DType.float32]()
    var model_f32 = LogisticRegression[DType.float32](
        C=Float32(1.0),
        max_iter=200,
        learning_rate=Float32(0.5),
    )
    model_f32.fit(X_f32, y)

    var preds_f32 = model_f32.predict(X_f32)
    assert_equal(preds_f32[0], 0)
    assert_equal(preds_f32[3], 1)


def test_logistic_regression_transposed_inputs() raises:
    var X_raw = Matrix[DType.float64](2, 4, 0)
    X_raw[0, 0] = -2.0
    X_raw[0, 1] = -1.0
    X_raw[0, 2] = 1.0
    X_raw[0, 3] = 2.0
    X_raw[1, 0] = -1.0
    X_raw[1, 1] = -2.0
    X_raw[1, 2] = 2.0
    X_raw[1, 3] = 1.0

    var X = X_raw.transpose()  # 4x2
    var y: List[Scalar[DType.int32]] = [0, 0, 1, 1]

    var model = LogisticRegression(max_iter=200, learning_rate=0.5)
    model.fit(X, y)

    var preds = model.predict(X)
    assert_equal(preds[0], 0)
    assert_equal(preds[3], 1)


def test_logistic_regression_model_refitting_state_reset() raises:
    # 1. Fit on 3 classes, 2 features
    var X1 = Matrix[DType.float64](6, 2, 0)
    X1[0, 0] = -2.0
    X1[0, 1] = -2.0
    X1[1, 0] = -2.0
    X1[1, 1] = -1.0
    X1[2, 0] = 0.0
    X1[2, 1] = 2.0
    X1[3, 0] = 0.0
    X1[3, 1] = 3.0
    X1[4, 0] = 2.0
    X1[4, 1] = -2.0
    X1[5, 0] = 2.0
    X1[5, 1] = -1.0
    var y1: List[Scalar[DType.int32]] = [0, 0, 1, 1, 2, 2]

    var model = LogisticRegression(max_iter=200, learning_rate=0.5)
    model.fit(X1, y1)
    assert_equal(len(model.classes_), 3)
    assert_equal(model.coef_.rows, 3)

    # 2. Refit same instance on 2 classes, 1 feature
    var X2 = Matrix[DType.float64](4, 1, 0)
    X2[0, 0] = -2.0
    X2[1, 0] = -1.0
    X2[2, 0] = 1.0
    X2[3, 0] = 2.0
    var y2: List[Scalar[DType.int32]] = [10, 10, 20, 20]

    model.fit(X2, y2)
    assert_equal(len(model.classes_), 2)
    assert_equal(model.classes_[0], 10)
    assert_equal(model.classes_[1], 20)
    assert_equal(model.coef_.rows, 2)
    assert_equal(model.coef_.cols, 1)


def test_logistic_regression_dataset_and_pipeline() raises:
    # 1. Dataset container helpers
    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = -2.0
    X[1, 0] = -1.0
    X[2, 0] = 1.0
    X[3, 0] = 2.0

    var y: List[Scalar[DType.int32]] = [0, 0, 1, 1]
    var ds = Dataset[DType.float64, DType.int32](X.copy(), y.copy())

    var model = LogisticRegression(max_iter=200, learning_rate=0.5)
    fit(model, ds)

    var preds = predict(model, ds)
    assert_equal(len(preds), 4)
    assert_equal(preds[0], 0)
    assert_equal(preds[3], 1)

    var probs = predict_proba(model, ds)
    assert_equal(probs.rows, 4)
    assert_equal(probs.cols, 2)

    # 2. Pipeline composition: StandardScaler -> LogisticRegression
    var scaler = StandardScaler()
    var clf = LogisticRegression(max_iter=200, learning_rate=0.5)
    var pipe = PipelineClassifier(scaler^, clf^)

    pipe.fit(X, y)
    var pipe_preds = pipe.predict(X)
    assert_equal(len(pipe_preds), 4)
    assert_equal(pipe_preds[0], 0)
    assert_equal(pipe_preds[3], 1)


def test_logistic_regression_error_handling() raises:
    # 1. Single-class target raises InvalidParameterError
    var X = Matrix[DType.float64].ones(3, 2)
    var y_single: List[Scalar[DType.int32]] = [1, 1, 1]
    var model = LogisticRegression()
    with assert_raises():
        model.fit(X, y_single)

    # 2. Predict before fit raises NotFittedError
    var model_unfitted = LogisticRegression()
    with assert_raises():
        _ = model_unfitted.predict(X)

    # 3. Feature dimension mismatch
    var y_valid: List[Scalar[DType.int32]] = [0, 1, 0]
    model.fit(X, y_valid)
    var X_bad = Matrix[DType.float64].ones(2, 4)
    with assert_raises():
        _ = model.predict(X_bad)

    # 4. Invalid parameter values raise InvalidParameterError
    with assert_raises():
        _ = LogisticRegression(C=-1.0)
    with assert_raises():
        _ = LogisticRegression(max_iter=0)
    with assert_raises():
        _ = LogisticRegression(penalty="invalid")

    # 5. Non-finite inputs
    var nan_val = Float64(0.0) / Float64(0.0)
    var X_nan = Matrix[DType.float64](3, 2, 1.0)
    X_nan[0, 0] = nan_val
    with assert_raises():
        model.fit(X_nan, y_valid)


def test_logistic_regression_copy_semantics_state_isolation() raises:
    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = -2.0
    X[1, 0] = -1.0
    X[2, 0] = 1.0
    X[3, 0] = 2.0
    var y: List[Scalar[DType.int32]] = [0, 0, 1, 1]

    var clf1 = LogisticRegression(C=2.0, max_iter=100)
    clf1.fit(X, y)

    var clf2 = clf1.copy()
    assert_equal(clf2.C, 2.0)
    assert_equal(clf2.is_fitted, True)
    assert_equal(len(clf2.classes_), 2)

    var y_new: List[Scalar[DType.int32]] = [1, 1, 0, 0]
    clf2.fit(X, y_new)

    var p1 = clf1.predict(X)
    var p2 = clf2.predict(X)
    assert_equal(p1[0], 0)
    assert_equal(p2[0], 1)


def test_logistic_regression_probability_calibration_sum_to_one() raises:
    var X = Matrix[DType.float64](8, 2, 0)
    for i in range(8):
        X[i, 0] = Float64(i)
        X[i, 1] = Float64(i * 2)
    var y: List[Scalar[DType.int32]] = [0, 0, 1, 1, 2, 2, 3, 3]

    var model = LogisticRegression(max_iter=150)
    model.fit(X, y)

    var probs = model.predict_proba(X)
    assert_equal(probs.rows, 8)
    assert_equal(probs.cols, 4)

    for i in range(8):
        var row_sum: Float64 = 0.0
        for c in range(4):
            assert_true(probs[i, c] >= 0.0)
            assert_true(probs[i, c] <= 1.0)
            row_sum += probs[i, c]
        assert_almost_equal(row_sum, 1.0, rtol=1e-4)


def test_logistic_regression_high_dimensional_features() raises:
    var N = 20
    var D = 6
    var X = Matrix[DType.float64](N, D, 0)
    var y = List[Scalar[DType.int32]](capacity=N)
    for i in range(N):
        for j in range(D):
            X[i, j] = Float64((i + 1) * (j + 1)) * (1.0 if i >= 10 else -1.0)
        y.append(Int32(0 if i < 10 else 1))

    var model = LogisticRegression(max_iter=200, learning_rate=0.1)
    model.fit(X, y)
    var preds = model.predict(X)
    for i in range(N):
        assert_equal(preds[i], Int(y[i]))


def test_logistic_regression_float32_native_model() raises:
    var X = Matrix[DType.float32](4, 1, 0)
    X[0, 0] = -2.0
    X[1, 0] = -1.0
    X[2, 0] = 1.0
    X[3, 0] = 2.0
    var y: List[Scalar[DType.int32]] = [0, 0, 1, 1]

    var model = LogisticRegression[DType.float32](max_iter=100)
    model.fit(X, y)
    var preds = model.predict(X)
    assert_equal(preds[0], 0)
    assert_equal(preds[3], 1)


def test_logistic_regression_asymmetric_binary_classes() raises:
    var X = Matrix[DType.float64](10, 1, 0)
    var y = List[Scalar[DType.int32]](capacity=10)
    for i in range(8):
        X[i, 0] = Float64(i - 10)
        y.append(0)
    for i in range(8, 10):
        X[i, 0] = Float64(i + 5)
        y.append(1)

    var model = LogisticRegression(max_iter=150)
    model.fit(X, y)
    var preds = model.predict(X)
    assert_equal(preds[0], 0)
    assert_equal(preds[9], 1)


def test_logistic_regression_discontinuous_labels() raises:
    var X = Matrix[DType.float64](6, 1, 0)
    X[0, 0] = -5.0
    X[1, 0] = -4.0
    X[2, 0] = 0.0
    X[3, 0] = 1.0
    X[4, 0] = 6.0
    X[5, 0] = 7.0
    var y: List[Scalar[DType.int32]] = [10, 10, 50, 50, 100, 100]

    var model = LogisticRegression(max_iter=150)
    model.fit(X, y)
    var preds = model.predict(X)
    assert_equal(preds[0], 10)
    assert_equal(preds[2], 50)
    assert_equal(preds[4], 100)


def test_logistic_regression_negative_class_labels() raises:
    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = -2.0
    X[1, 0] = -1.0
    X[2, 0] = 1.0
    X[3, 0] = 2.0
    var y: List[Scalar[DType.int32]] = [-1, -1, 1, 1]

    var model = LogisticRegression(max_iter=150)
    model.fit(X, y)
    var preds = model.predict(X)
    assert_equal(preds[0], -1)
    assert_equal(preds[3], 1)


def test_logistic_regression_decision_boundary_at_origin() raises:
    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = -3.0
    X[1, 0] = -1.0
    X[2, 0] = 1.0
    X[3, 0] = 3.0
    var y: List[Scalar[DType.int32]] = [0, 0, 1, 1]

    var model = LogisticRegression(max_iter=200, fit_intercept=True)
    model.fit(X, y)

    var X_origin = Matrix[DType.float64](1, 1, 0.0)
    var prob_origin = model.predict_proba(X_origin)
    assert_almost_equal(prob_origin[0, 0], 0.5, atol=0.1)
    assert_almost_equal(prob_origin[0, 1], 0.5, atol=0.1)


def test_logistic_regression_no_intercept_boundary() raises:
    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = -2.0
    X[1, 0] = -1.0
    X[2, 0] = 1.0
    X[3, 0] = 2.0
    var y: List[Scalar[DType.int32]] = [0, 0, 1, 1]

    var model = LogisticRegression(fit_intercept=False, max_iter=150)
    model.fit(X, y)
    for c in range(len(model.intercept_)):
        assert_equal(model.intercept_[c], 0.0)


def test_logistic_regression_predict_proba_unfitted_raises() raises:
    var model = LogisticRegression()
    var X = Matrix[DType.float64].ones(3, 2)
    with assert_raises():
        _ = model.predict_proba(X)


def test_logistic_regression_extreme_C_unregularized() raises:
    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = -2.0
    X[1, 0] = -1.0
    X[2, 0] = 1.0
    X[3, 0] = 2.0
    var y: List[Scalar[DType.int32]] = [0, 0, 1, 1]

    var model = LogisticRegression(C=1e6, max_iter=150)
    model.fit(X, y)
    var preds = model.predict(X)
    assert_equal(preds[0], 0)
    assert_equal(preds[3], 1)


def test_logistic_regression_heavy_shrinkage_small_C() raises:
    var X = Matrix[DType.float64](6, 1, 0)
    X[0, 0] = -3.0
    X[1, 0] = -2.0
    X[2, 0] = -1.0
    X[3, 0] = 1.0
    X[4, 0] = 2.0
    X[5, 0] = 3.0
    var y: List[Scalar[DType.int32]] = [0, 0, 0, 1, 1, 1]

    var m_reg = LogisticRegression(C=0.01, max_iter=200, learning_rate=0.1)
    var m_unreg = LogisticRegression(C=100.0, max_iter=200, learning_rate=0.1)
    m_reg.fit(X, y)
    m_unreg.fit(X, y)

    var norm_reg = abs(m_reg.coef_[1, 0] - m_reg.coef_[0, 0])
    var norm_unreg = abs(m_unreg.coef_[1, 0] - m_unreg.coef_[0, 0])
    assert_true(norm_reg < norm_unreg)


def test_logistic_regression_multiclass_four_classes() raises:
    var X = Matrix[DType.float64](8, 2, 0)
    X[0, 0] = -3.0
    X[0, 1] = -3.0
    X[1, 0] = -3.0
    X[1, 1] = -2.0
    X[2, 0] = -3.0
    X[2, 1] = 3.0
    X[3, 0] = -2.0
    X[3, 1] = 3.0
    X[4, 0] = 3.0
    X[4, 1] = -3.0
    X[5, 0] = 2.0
    X[5, 1] = -3.0
    X[6, 0] = 3.0
    X[6, 1] = 3.0
    X[7, 0] = 3.0
    X[7, 1] = 2.0
    var y: List[Scalar[DType.int32]] = [0, 0, 1, 1, 2, 2, 3, 3]

    var model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    var preds = model.predict(X)
    assert_equal(preds[0], 0)
    assert_equal(preds[2], 1)
    assert_equal(preds[4], 2)
    assert_equal(preds[6], 3)


def test_logistic_regression_repeated_predict_consistency() raises:
    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = -2.0
    X[1, 0] = -1.0
    X[2, 0] = 1.0
    X[3, 0] = 2.0
    var y: List[Scalar[DType.int32]] = [0, 0, 1, 1]

    var model = LogisticRegression(max_iter=100)
    model.fit(X, y)

    var p1 = model.predict(X)
    var p2 = model.predict(X)
    for i in range(4):
        assert_equal(p1[i], p2[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
