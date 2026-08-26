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
    StandardScaler,
    NotFittedError,
    DataConversionError,
)


def test_standard_scaler() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 10.0
    X[1, 0] = 20.0
    X[2, 0] = 30.0
    X[3, 0] = 40.0

    X[0, 1] = 1.0
    X[1, 1] = 1.0
    X[2, 1] = 1.0
    X[3, 1] = 1.0

    var scaler = StandardScaler()
    var X_scaled = scaler.fit_transform(X)

    assert_true(scaler.is_fitted)
    assert_equal(scaler.mean_[0], 25.0)
    assert_equal(scaler.mean_[1], 1.0)
    assert_equal(X_scaled[0, 1], 0.0)
    assert_equal(X_scaled[3, 1], 0.0)


def test_standard_scaler_dataset() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 10.0
    X[1, 0] = 20.0
    X[2, 0] = 30.0
    X[3, 0] = 40.0
    X[0, 1] = 2.0
    X[1, 1] = 4.0
    X[2, 1] = 6.0
    X[3, 1] = 8.0

    var y = List[Scalar[DType.float64]](capacity=4)
    y.append(0.0)
    y.append(1.0)
    y.append(0.0)
    y.append(1.0)

    var ds = Dataset(X^, y^)
    var scaler = StandardScaler()
    var ds_scaled = scaler.fit_transform(ds)

    assert_true(scaler.is_fitted)
    assert_equal(ds_scaled.n_samples(), 4)
    assert_equal(ds_scaled.n_features(), 2)
    assert_equal(ds_scaled.targets[1], 1.0)


def test_standard_scaler_dtype_incoherence_prevention() raises:
    var scaler = StandardScaler()
    var X32 = Matrix[DType.float32](2, 2, 1.0)
    scaler.fit(X32)

    var X64 = Matrix[DType.float64](2, 2, 1.0)
    with assert_raises():
        _ = scaler.transform(X64)


def test_standard_scaler_not_fitted() raises:
    var scaler = StandardScaler()
    var X = Matrix[DType.float64](2, 2, 1.0)
    with assert_raises():
        _ = scaler.transform(X)


def test_standard_scaler_options() raises:
    var X = Matrix[DType.float64](2, 1, 0)
    X[0, 0] = 10.0
    X[1, 0] = 20.0

    # with_mean=False, with_std=False -> Identity
    var scaler_noop = StandardScaler(with_mean=False, with_std=False)
    var X_noop = scaler_noop.fit_transform(X)
    assert_equal(X_noop[0, 0], 10.0)
    assert_equal(X_noop[1, 0], 20.0)


def test_standard_scaler_unit_variance_and_zero_mean_properties() raises:
    var X = Matrix[DType.float64](5, 2, 0)
    for i in range(5):
        var fi = Float64(i + 1)
        X[i, 0] = fi * 10.0
        X[i, 1] = fi * 100.0

    var scaler = StandardScaler()
    var X_scaled = scaler.fit_transform(X)

    var means = X_scaled.mean_along_axis_0()
    assert_almost_equal(means[0], 0.0, atol=1e-5)
    assert_almost_equal(means[1], 0.0, atol=1e-5)


def test_standard_scaler_constant_columns_scale_one() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    for i in range(4):
        X[i, 0] = Float64(i + 1)
        X[i, 1] = 55.0

    var scaler = StandardScaler()
    var X_scaled = scaler.fit_transform(X)

    assert_equal(scaler.scale_[1], 1.0)
    assert_equal(scaler.mean_[1], 55.0)
    for i in range(4):
        assert_equal(X_scaled[i, 1], 0.0)


def test_standard_scaler_with_mean_only() raises:
    var X = Matrix[DType.float64](3, 1, 0)
    X[0, 0] = 10.0
    X[1, 0] = 20.0
    X[2, 0] = 30.0

    var scaler = StandardScaler(with_mean=True, with_std=False)
    var X_scaled = scaler.fit_transform(X)

    assert_equal(scaler.mean_[0], 20.0)
    assert_equal(X_scaled[0, 0], -10.0)
    assert_equal(X_scaled[1, 0], 0.0)
    assert_equal(X_scaled[2, 0], 10.0)


def test_standard_scaler_with_std_only() raises:
    var X = Matrix[DType.float64](3, 1, 0)
    X[0, 0] = 2.0
    X[1, 0] = 4.0
    X[2, 0] = 6.0

    var scaler = StandardScaler(with_mean=False, with_std=True)
    var X_scaled = scaler.fit_transform(X)

    assert_equal(len(scaler.mean_), 1)
    assert_true(scaler.scale_[0] > 0)


def test_standard_scaler_copy_constructor() raises:
    var X = Matrix[DType.float64](3, 1, 0)
    X[0, 0] = 1.0
    X[1, 0] = 2.0
    X[2, 0] = 3.0

    var s1 = StandardScaler()
    s1.fit(X)

    var s2 = s1.copy()
    assert_equal(s2.is_fitted, True)
    assert_equal(s2.mean_[0], 2.0)

    var X_new = Matrix[DType.float64](3, 1, 0)
    X_new[0, 0] = 10.0
    X_new[1, 0] = 20.0
    X_new[2, 0] = 30.0
    s2.fit(X_new)

    assert_equal(s1.mean_[0], 2.0)
    assert_equal(s2.mean_[0], 20.0)


def test_standard_scaler_single_sample() raises:
    var X = Matrix[DType.float64](1, 2, 42.0)
    var scaler = StandardScaler()
    var X_scaled = scaler.fit_transform(X)

    assert_equal(scaler.mean_[0], 42.0)
    assert_equal(scaler.scale_[0], 1.0)
    assert_equal(X_scaled[0, 0], 0.0)


def test_standard_scaler_single_feature() raises:
    var X = Matrix[DType.float64](5, 1, 0)
    for i in range(5):
        X[i, 0] = Float64(i * 10)
    var scaler = StandardScaler()
    var X_scaled = scaler.fit_transform(X)

    assert_equal(scaler.mean_[0], 20.0)
    assert_almost_equal(X_scaled[2, 0], 0.0, atol=1e-5)


def test_standard_scaler_float32_native() raises:
    var X = Matrix[DType.float32](4, 1, 0)
    X[0, 0] = 1.0
    X[1, 0] = 2.0
    X[2, 0] = 3.0
    X[3, 0] = 4.0

    var scaler = StandardScaler()
    var X_scaled = scaler.fit_transform(X)
    assert_equal(scaler.mean_[0], 2.5)
    assert_equal(X_scaled.rows, 4)


def test_standard_scaler_bfloat16_native() raises:
    var X = Matrix[DType.bfloat16](3, 1, 0)
    X[0, 0] = 2.0
    X[1, 0] = 4.0
    X[2, 0] = 6.0

    var scaler = StandardScaler()
    var X_scaled = scaler.fit_transform(X)
    assert_almost_equal(Float64(scaler.mean_[0]), 4.0, rtol=1e-2)


def test_standard_scaler_int32_input_scaling() raises:
    var X = Matrix[DType.int32](3, 1, 0)
    X[0, 0] = 10
    X[1, 0] = 20
    X[2, 0] = 30

    var scaler = StandardScaler()
    var X_scaled = scaler.fit_transform(X)
    assert_equal(scaler.mean_[0], 20.0)


def test_standard_scaler_dimension_mismatch_transform() raises:
    var X_train = Matrix[DType.float64].ones(4, 3)
    var scaler = StandardScaler()
    scaler.fit(X_train)

    var X_bad = Matrix[DType.float64].ones(2, 5)
    with assert_raises():
        _ = scaler.transform(X_bad)


def test_standard_scaler_separate_fit_and_transform() raises:
    var X = Matrix[DType.float64](3, 1, 0)
    X[0, 0] = 1.0
    X[1, 0] = 2.0
    X[2, 0] = 3.0

    var scaler = StandardScaler()
    scaler.fit(X)
    var X_trans = scaler.transform(X)
    assert_almost_equal(X_trans[1, 0], 0.0, atol=1e-5)


def test_standard_scaler_transform_unseen_data() raises:
    var X_train = Matrix[DType.float64](3, 1, 0)
    X_train[0, 0] = 0.0
    X_train[1, 0] = 10.0
    X_train[2, 0] = 20.0
    var scaler = StandardScaler()
    scaler.fit(X_train)

    var X_test = Matrix[DType.float64](1, 1, 30.0)
    var X_test_scaled = scaler.transform(X_test)
    assert_true(X_test_scaled[0, 0] > 0.0)


def test_standard_scaler_dataset_metadata_preservation() raises:
    var X = Matrix[DType.float64](4, 2, 1.0)
    var y: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0, 4.0]
    var fnames: List[String] = ["feat1", "feat2"]
    var tnames: List[String] = ["tgt"]
    var ds = Dataset(X^, y^, fnames^, tnames^)

    var scaler = StandardScaler()
    var ds_scaled = scaler.fit_transform(ds)
    assert_equal(ds_scaled.feature_names[0], "feat1")
    assert_equal(ds_scaled.target_names[0], "tgt")


def test_standard_scaler_large_scale_matrix() raises:
    var N = 100
    var D = 10
    var X = Matrix[DType.float64](N, D, 0)
    for i in range(N):
        for j in range(D):
            X[i, j] = Float64((i + 1) * (j + 1))

    var scaler = StandardScaler()
    var X_scaled = scaler.fit_transform(X)
    assert_equal(X_scaled.rows, N)
    assert_equal(X_scaled.cols, D)


def test_standard_scaler_refitting_resets_parameters() raises:
    var X1 = Matrix[DType.float64](3, 1, 5.0)
    var scaler = StandardScaler()
    scaler.fit(X1)
    assert_equal(scaler.mean_[0], 5.0)

    var X2 = Matrix[DType.float64](3, 1, 50.0)
    scaler.fit(X2)
    assert_equal(scaler.mean_[0], 50.0)


def test_standard_scaler_symmetric_negative_inputs() raises:
    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = -10.0
    X[1, 0] = -5.0
    X[2, 0] = 5.0
    X[3, 0] = 10.0

    var scaler = StandardScaler()
    var X_scaled = scaler.fit_transform(X)
    assert_almost_equal(scaler.mean_[0], 0.0, atol=1e-5)
    assert_almost_equal(X_scaled[0, 0], -X_scaled[3, 0], atol=1e-5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
