from std.math import isnan
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
    MinMaxScaler,
    RobustScaler,
    Binarizer,
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder,
    SimpleImputer,
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


def test_min_max_scaler_default_range() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 10.0
    X[1, 0] = 20.0
    X[2, 0] = 30.0
    X[3, 0] = 40.0
    X[0, 1] = -1.0
    X[1, 1] = 0.0
    X[2, 1] = 1.0
    X[3, 1] = 2.0

    var scaler = MinMaxScaler()
    var Xs = scaler.fit_transform(X)

    assert_true(scaler.is_fitted)
    assert_equal(scaler.data_min_[0], 10.0)
    assert_equal(scaler.data_max_[0], 40.0)
    assert_equal(scaler.data_range_[0], 30.0)
    assert_almost_equal(Xs[0, 0], 0.0, atol=1e-12)
    assert_almost_equal(Xs[3, 0], 1.0, atol=1e-12)
    assert_almost_equal(Xs[1, 0], 1.0 / 3.0, atol=1e-12)
    assert_almost_equal(Xs[0, 1], 0.0, atol=1e-12)
    assert_almost_equal(Xs[3, 1], 1.0, atol=1e-12)


def test_min_max_scaler_custom_range() raises:
    var X = Matrix[DType.float64](3, 1, 0)
    X[0, 0] = 0.0
    X[1, 0] = 5.0
    X[2, 0] = 10.0

    var scaler = MinMaxScaler(-1.0, 1.0)
    var Xs = scaler.fit_transform(X)
    assert_almost_equal(Xs[0, 0], -1.0, atol=1e-12)
    assert_almost_equal(Xs[1, 0], 0.0, atol=1e-12)
    assert_almost_equal(Xs[2, 0], 1.0, atol=1e-12)


def test_min_max_scaler_constant_column() raises:
    var X = Matrix[DType.float64](3, 1, 7.0)
    var scaler = MinMaxScaler()
    var Xs = scaler.fit_transform(X)
    assert_equal(scaler.data_range_[0], 0.0)
    assert_equal(scaler.scale_[0], 1.0)
    assert_almost_equal(Xs[0, 0], 0.0, atol=1e-12)
    assert_almost_equal(Xs[2, 0], 0.0, atol=1e-12)


def test_min_max_scaler_clip_unseen_data() raises:
    var X = Matrix[DType.float64](2, 1, 0)
    X[0, 0] = 0.0
    X[1, 0] = 10.0

    var unclipped = MinMaxScaler()
    _ = unclipped.fit_transform(X)
    var X_new = Matrix[DType.float64](2, 1, 0)
    X_new[0, 0] = -10.0
    X_new[1, 0] = 20.0
    var Xu = unclipped.transform(X_new)
    assert_almost_equal(Xu[0, 0], -1.0, atol=1e-12)
    assert_almost_equal(Xu[1, 0], 2.0, atol=1e-12)

    var clipped = MinMaxScaler(0.0, 1.0, True)
    _ = clipped.fit_transform(X)
    var Xc = clipped.transform(X_new)
    assert_almost_equal(Xc[0, 0], 0.0, atol=1e-12)
    assert_almost_equal(Xc[1, 0], 1.0, atol=1e-12)


def test_min_max_scaler_inverse_transform_roundtrip() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    for i in range(4):
        X[i, 0] = Float64(i) * 3.0 - 4.0
        X[i, 1] = Float64(i) * Float64(i)

    var scaler = MinMaxScaler(2.0, 8.0)
    var Xs = scaler.fit_transform(X)
    var Xr = scaler.inverse_transform(Xs)
    for i in range(4):
        assert_almost_equal(Xr[i, 0], X[i, 0], atol=1e-10)
        assert_almost_equal(Xr[i, 1], X[i, 1], atol=1e-10)


def test_min_max_scaler_dataset_metadata_preservation() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    for i in range(4):
        X[i, 0] = Float64(i)
        X[i, 1] = Float64(i) * 2.0
    var y: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0, 4.0]
    var fnames: List[String] = ["a", "b"]
    var tnames: List[String] = ["t"]
    var ds = Dataset(X^, y^, fnames^, tnames^)

    var scaler = MinMaxScaler()
    var ds_scaled = scaler.fit_transform(ds)
    assert_equal(ds_scaled.n_samples(), 4)
    assert_equal(ds_scaled.n_features(), 2)
    assert_equal(ds_scaled.feature_names[1], "b")
    assert_equal(ds_scaled.target_names[0], "t")
    assert_almost_equal(ds_scaled.records[3, 0], 1.0, atol=1e-12)


def test_min_max_scaler_not_fitted() raises:
    var scaler = MinMaxScaler()
    var X = Matrix[DType.float64](2, 2, 1.0)
    with assert_raises():
        _ = scaler.transform(X)
    with assert_raises():
        _ = scaler.inverse_transform(X)


def test_min_max_scaler_invalid_feature_range() raises:
    with assert_raises():
        var bad = MinMaxScaler(1.0, 1.0)
        _ = bad.clip
    with assert_raises():
        var worse = MinMaxScaler(2.0, 1.0)
        _ = worse.clip


def test_min_max_scaler_dtype_incoherence_prevention() raises:
    var scaler = MinMaxScaler()
    var X32 = Matrix[DType.float32](2, 2, 1.0)
    scaler.fit(X32)
    var X64 = Matrix[DType.float64](2, 2, 1.0)
    with assert_raises():
        _ = scaler.transform(X64)


def test_min_max_scaler_dimension_mismatch_transform() raises:
    var X = Matrix[DType.float64](3, 2, 1.0)
    var scaler = MinMaxScaler()
    scaler.fit(X)
    var X_wrong = Matrix[DType.float64](3, 3, 1.0)
    with assert_raises():
        _ = scaler.transform(X_wrong)


def test_min_max_scaler_float32_native() raises:
    var X = Matrix[DType.float32](3, 1, 0)
    X[0, 0] = 1.0
    X[1, 0] = 2.0
    X[2, 0] = 3.0
    var scaler = MinMaxScaler()
    var Xs = scaler.fit_transform(X)
    assert_almost_equal(Xs[0, 0], Float32(0.0), atol=1e-6)
    assert_almost_equal(Xs[2, 0], Float32(1.0), atol=1e-6)


def test_min_max_scaler_int32_input_scaling() raises:
    var X = Matrix[DType.int32](3, 1, 0)
    X[0, 0] = 0
    X[1, 0] = 5
    X[2, 0] = 10
    var scaler = MinMaxScaler(0.0, 100.0)
    var Xs = scaler.fit_transform(X)
    assert_equal(Xs[0, 0], 0)
    assert_equal(Xs[1, 0], 50)
    assert_equal(Xs[2, 0], 100)


def test_min_max_scaler_copy_constructor() raises:
    var X = Matrix[DType.float64](3, 1, 0)
    X[0, 0] = 1.0
    X[1, 0] = 2.0
    X[2, 0] = 4.0
    var scaler = MinMaxScaler(-5.0, 5.0, True)
    scaler.fit(X)

    var clone = MinMaxScaler(copy=scaler)
    assert_true(clone.is_fitted)
    assert_equal(clone.clip, True)
    assert_equal(clone.feature_range_min, -5.0)
    assert_equal(clone.data_max_[0], 4.0)
    var Xs = clone.transform(X)
    assert_almost_equal(Xs[0, 0], -5.0, atol=1e-12)
    assert_almost_equal(Xs[2, 0], 5.0, atol=1e-12)


def test_min_max_scaler_negative_only_column() raises:
    var X = Matrix[DType.float64](3, 1, 0)
    X[0, 0] = -30.0
    X[1, 0] = -20.0
    X[2, 0] = -10.0
    var scaler = MinMaxScaler()
    var Xs = scaler.fit_transform(X)
    assert_almost_equal(Xs[0, 0], 0.0, atol=1e-12)
    assert_almost_equal(Xs[1, 0], 0.5, atol=1e-12)
    assert_almost_equal(Xs[2, 0], 1.0, atol=1e-12)


def test_min_max_scaler_single_sample() raises:
    var X = Matrix[DType.float64](1, 3, 0)
    X[0, 0] = 1.0
    X[0, 1] = -2.0
    X[0, 2] = 3.0
    var scaler = MinMaxScaler()
    var Xs = scaler.fit_transform(X)
    for c in range(3):
        assert_equal(scaler.scale_[c], 1.0)
        assert_almost_equal(Xs[0, c], 0.0, atol=1e-12)


def test_min_max_scaler_refitting_resets_parameters() raises:
    var X1 = Matrix[DType.float64](2, 1, 0)
    X1[0, 0] = 0.0
    X1[1, 0] = 1.0
    var scaler = MinMaxScaler()
    scaler.fit(X1)
    assert_equal(scaler.data_max_[0], 1.0)

    var X2 = Matrix[DType.float64](2, 1, 0)
    X2[0, 0] = 100.0
    X2[1, 0] = 200.0
    scaler.fit(X2)
    assert_equal(scaler.data_min_[0], 100.0)
    assert_equal(scaler.data_max_[0], 200.0)


def test_robust_scaler_median_and_iqr() raises:
    var X = Matrix[DType.float64](5, 1, 0)
    for i in range(5):
        X[i, 0] = Float64(i + 1)

    var scaler = RobustScaler()
    var Xs = scaler.fit_transform(X)

    assert_true(scaler.is_fitted)
    assert_almost_equal(scaler.center_[0], 3.0, atol=1e-12)
    assert_almost_equal(scaler.scale_[0], 2.0, atol=1e-12)
    assert_almost_equal(Xs[0, 0], -1.0, atol=1e-12)
    assert_almost_equal(Xs[1, 0], -0.5, atol=1e-12)
    assert_almost_equal(Xs[2, 0], 0.0, atol=1e-12)
    assert_almost_equal(Xs[4, 0], 1.0, atol=1e-12)


def test_robust_scaler_even_sample_count_interpolation() raises:
    var X = Matrix[DType.float64](6, 1, 0)
    var vals: List[Float64] = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]
    for i in range(6):
        X[i, 0] = vals[i]

    var scaler = RobustScaler()
    scaler.fit(X)
    assert_almost_equal(scaler.center_[0], 3.5, atol=1e-12)
    assert_almost_equal(scaler.scale_[0], 2.5, atol=1e-12)


def test_robust_scaler_unsorted_input_ordering() raises:
    var X = Matrix[DType.float64](5, 1, 0)
    var vals: List[Float64] = [5.0, 1.0, 4.0, 2.0, 3.0]
    for i in range(5):
        X[i, 0] = vals[i]

    var scaler = RobustScaler()
    scaler.fit(X)
    assert_almost_equal(scaler.center_[0], 3.0, atol=1e-12)
    assert_almost_equal(scaler.scale_[0], 2.0, atol=1e-12)


def test_robust_scaler_outlier_resistance() raises:
    var X = Matrix[DType.float64](5, 1, 0)
    var vals: List[Float64] = [1.0, 2.0, 3.0, 4.0, 1000.0]
    for i in range(5):
        X[i, 0] = vals[i]

    var scaler = RobustScaler()
    var Xs = scaler.fit_transform(X)
    assert_almost_equal(scaler.center_[0], 3.0, atol=1e-12)
    assert_almost_equal(scaler.scale_[0], 2.0, atol=1e-12)
    assert_almost_equal(Xs[2, 0], 0.0, atol=1e-12)
    assert_almost_equal(Xs[4, 0], 498.5, atol=1e-12)


def test_robust_scaler_custom_quantile_range() raises:
    var X = Matrix[DType.float64](11, 1, 0)
    for i in range(11):
        X[i, 0] = Float64(i)

    var scaler = RobustScaler(True, True, 10.0, 90.0)
    scaler.fit(X)
    assert_almost_equal(scaler.center_[0], 5.0, atol=1e-12)
    assert_almost_equal(scaler.scale_[0], 8.0, atol=1e-12)


def test_robust_scaler_constant_column_scale_one() raises:
    var X = Matrix[DType.float64](4, 1, 9.0)
    var scaler = RobustScaler()
    var Xs = scaler.fit_transform(X)
    assert_almost_equal(scaler.center_[0], 9.0, atol=1e-12)
    assert_equal(scaler.scale_[0], 1.0)
    for i in range(4):
        assert_almost_equal(Xs[i, 0], 0.0, atol=1e-12)


def test_robust_scaler_with_centering_only() raises:
    var X = Matrix[DType.float64](5, 1, 0)
    for i in range(5):
        X[i, 0] = Float64(i + 1)

    var scaler = RobustScaler(True, False)
    var Xs = scaler.fit_transform(X)
    assert_almost_equal(Xs[0, 0], -2.0, atol=1e-12)
    assert_almost_equal(Xs[4, 0], 2.0, atol=1e-12)


def test_robust_scaler_with_scaling_only() raises:
    var X = Matrix[DType.float64](5, 1, 0)
    for i in range(5):
        X[i, 0] = Float64(i + 1)

    var scaler = RobustScaler(False, True)
    var Xs = scaler.fit_transform(X)
    assert_almost_equal(Xs[0, 0], 0.5, atol=1e-12)
    assert_almost_equal(Xs[4, 0], 2.5, atol=1e-12)


def test_robust_scaler_multi_column_independence() raises:
    var X = Matrix[DType.float64](5, 2, 0)
    for i in range(5):
        X[i, 0] = Float64(i + 1)
        X[i, 1] = Float64(i + 1) * 100.0

    var scaler = RobustScaler()
    var Xs = scaler.fit_transform(X)
    assert_almost_equal(scaler.center_[1], 300.0, atol=1e-12)
    assert_almost_equal(scaler.scale_[1], 200.0, atol=1e-12)
    for i in range(5):
        assert_almost_equal(Xs[i, 0], Xs[i, 1], atol=1e-12)


def test_robust_scaler_inverse_transform_roundtrip() raises:
    var X = Matrix[DType.float64](6, 2, 0)
    for i in range(6):
        X[i, 0] = Float64(i) * 2.5 - 7.0
        X[i, 1] = Float64(i * i) + 1.0

    var scaler = RobustScaler(True, True, 20.0, 80.0)
    var Xs = scaler.fit_transform(X)
    var Xr = scaler.inverse_transform(Xs)
    for i in range(6):
        assert_almost_equal(Xr[i, 0], X[i, 0], atol=1e-10)
        assert_almost_equal(Xr[i, 1], X[i, 1], atol=1e-10)


def test_robust_scaler_dataset_metadata_preservation() raises:
    var X = Matrix[DType.float64](5, 2, 0)
    for i in range(5):
        X[i, 0] = Float64(i + 1)
        X[i, 1] = Float64(i + 1)
    var y: List[Scalar[DType.float64]] = [0.0, 1.0, 0.0, 1.0, 0.0]
    var fnames: List[String] = ["p", "q"]
    var tnames: List[String] = ["z"]
    var ds = Dataset(X^, y^, fnames^, tnames^)

    var scaler = RobustScaler()
    var ds_scaled = scaler.fit_transform(ds)
    assert_equal(ds_scaled.n_samples(), 5)
    assert_equal(ds_scaled.feature_names[0], "p")
    assert_equal(ds_scaled.target_names[0], "z")
    assert_almost_equal(ds_scaled.records[2, 0], 0.0, atol=1e-12)


def test_robust_scaler_not_fitted() raises:
    var scaler = RobustScaler()
    var X = Matrix[DType.float64](2, 2, 1.0)
    with assert_raises():
        _ = scaler.transform(X)
    with assert_raises():
        _ = scaler.inverse_transform(X)


def test_robust_scaler_invalid_quantile_range() raises:
    with assert_raises():
        var a = RobustScaler(True, True, 75.0, 25.0)
        _ = a.with_scaling
    with assert_raises():
        var b = RobustScaler(True, True, -1.0, 75.0)
        _ = b.with_scaling
    with assert_raises():
        var c = RobustScaler(True, True, 25.0, 101.0)
        _ = c.with_scaling


def test_robust_scaler_dtype_incoherence_prevention() raises:
    var scaler = RobustScaler()
    var X32 = Matrix[DType.float32](3, 2, 1.0)
    scaler.fit(X32)
    var X64 = Matrix[DType.float64](3, 2, 1.0)
    with assert_raises():
        _ = scaler.transform(X64)


def test_robust_scaler_dimension_mismatch_transform() raises:
    var X = Matrix[DType.float64](3, 2, 1.0)
    var scaler = RobustScaler()
    scaler.fit(X)
    var X_wrong = Matrix[DType.float64](3, 5, 1.0)
    with assert_raises():
        _ = scaler.transform(X_wrong)


def test_robust_scaler_float32_native() raises:
    var X = Matrix[DType.float32](5, 1, 0)
    for i in range(5):
        X[i, 0] = Float32(i + 1)
    var scaler = RobustScaler()
    var Xs = scaler.fit_transform(X)
    assert_almost_equal(Xs[0, 0], Float32(-1.0), atol=1e-6)
    assert_almost_equal(Xs[4, 0], Float32(1.0), atol=1e-6)


def test_robust_scaler_int32_input_scaling() raises:
    var X = Matrix[DType.int32](5, 1, 0)
    for i in range(5):
        X[i, 0] = Int32(i + 1)
    var scaler = RobustScaler()
    var Xs = scaler.fit_transform(X)
    assert_equal(Xs[0, 0], -1)
    assert_equal(Xs[2, 0], 0)
    assert_equal(Xs[4, 0], 1)


def test_robust_scaler_copy_constructor() raises:
    var X = Matrix[DType.float64](5, 1, 0)
    for i in range(5):
        X[i, 0] = Float64(i + 1)
    var scaler = RobustScaler(True, True, 20.0, 80.0)
    scaler.fit(X)

    var clone = RobustScaler(copy=scaler)
    assert_true(clone.is_fitted)
    assert_equal(clone.quantile_min, 20.0)
    assert_equal(clone.quantile_max, 80.0)
    assert_almost_equal(clone.center_[0], 3.0, atol=1e-12)
    var Xs = clone.transform(X)
    assert_almost_equal(Xs[2, 0], 0.0, atol=1e-12)


def test_robust_scaler_single_sample() raises:
    var X = Matrix[DType.float64](1, 2, 0)
    X[0, 0] = 4.0
    X[0, 1] = -8.0
    var scaler = RobustScaler()
    var Xs = scaler.fit_transform(X)
    assert_almost_equal(scaler.center_[0], 4.0, atol=1e-12)
    assert_equal(scaler.scale_[0], 1.0)
    assert_almost_equal(Xs[0, 0], 0.0, atol=1e-12)
    assert_almost_equal(Xs[0, 1], 0.0, atol=1e-12)


def test_robust_scaler_refitting_resets_parameters() raises:
    var X1 = Matrix[DType.float64](5, 1, 0)
    for i in range(5):
        X1[i, 0] = Float64(i + 1)
    var scaler = RobustScaler()
    scaler.fit(X1)
    assert_almost_equal(scaler.center_[0], 3.0, atol=1e-12)

    var X2 = Matrix[DType.float64](5, 1, 0)
    for i in range(5):
        X2[i, 0] = Float64(i + 1) * 10.0
    scaler.fit(X2)
    assert_almost_equal(scaler.center_[0], 30.0, atol=1e-12)
    assert_almost_equal(scaler.scale_[0], 20.0, atol=1e-12)


def test_binarizer_default_threshold() raises:
    var X = Matrix[DType.float64](2, 3, 0)
    X[0, 0] = 1.0
    X[0, 1] = -1.0
    X[0, 2] = 2.0
    X[1, 0] = 2.0
    X[1, 1] = 0.0
    X[1, 2] = -3.0

    var binarizer = Binarizer()
    var Xb = binarizer.fit_transform(X)

    assert_true(binarizer.is_fitted)
    assert_equal(binarizer.n_features_in_, 3)
    assert_equal(Xb[0, 0], 1.0)
    assert_equal(Xb[0, 1], 0.0)
    assert_equal(Xb[0, 2], 1.0)
    assert_equal(Xb[1, 0], 1.0)
    assert_equal(Xb[1, 1], 0.0)
    assert_equal(Xb[1, 2], 0.0)


def test_binarizer_custom_threshold() raises:
    var X = Matrix[DType.float64](1, 4, 0)
    X[0, 0] = 0.5
    X[0, 1] = 1.0
    X[0, 2] = 1.5
    X[0, 3] = 2.0

    var binarizer = Binarizer(1.0)
    var Xb = binarizer.fit_transform(X)
    assert_equal(Xb[0, 0], 0.0)
    assert_equal(Xb[0, 1], 0.0)
    assert_equal(Xb[0, 2], 1.0)
    assert_equal(Xb[0, 3], 1.0)


def test_binarizer_negative_threshold() raises:
    var X = Matrix[DType.float64](1, 3, 0)
    X[0, 0] = -5.0
    X[0, 1] = -2.0
    X[0, 2] = 0.0

    var binarizer = Binarizer(-2.0)
    var Xb = binarizer.fit_transform(X)
    assert_equal(Xb[0, 0], 0.0)
    assert_equal(Xb[0, 1], 0.0)
    assert_equal(Xb[0, 2], 1.0)


def test_binarizer_threshold_is_exclusive() raises:
    var X = Matrix[DType.float64](1, 1, 3.0)
    var binarizer = Binarizer(3.0)
    var Xb = binarizer.fit_transform(X)
    assert_equal(Xb[0, 0], 0.0)


def test_binarizer_output_is_only_zeros_and_ones() raises:
    var X = Matrix[DType.float64](5, 4, 0)
    for i in range(5):
        for j in range(4):
            X[i, j] = Float64(i * 4 + j) - 7.5

    var binarizer = Binarizer()
    var Xb = binarizer.fit_transform(X)
    for i in range(5):
        for j in range(4):
            assert_true(Xb[i, j] == 0.0 or Xb[i, j] == 1.0)


def test_binarizer_shape_is_preserved() raises:
    var X = Matrix[DType.float64](7, 3, 1.0)
    var binarizer = Binarizer()
    var Xb = binarizer.fit_transform(X)
    assert_equal(Xb.rows, 7)
    assert_equal(Xb.cols, 3)


def test_binarizer_separate_fit_and_transform() raises:
    var X_train = Matrix[DType.float64](3, 1, 0)
    X_train[0, 0] = 1.0
    X_train[1, 0] = 2.0
    X_train[2, 0] = 3.0

    var binarizer = Binarizer(2.5)
    binarizer.fit(X_train)

    var X_test = Matrix[DType.float64](2, 1, 0)
    X_test[0, 0] = 2.4
    X_test[1, 0] = 2.6
    var Xb = binarizer.transform(X_test)
    assert_equal(Xb[0, 0], 0.0)
    assert_equal(Xb[1, 0], 1.0)


def test_binarizer_dataset_metadata_preservation() raises:
    var X = Matrix[DType.float64](3, 2, 0)
    for i in range(3):
        X[i, 0] = Float64(i) - 1.0
        X[i, 1] = Float64(i)
    var y: List[Scalar[DType.float64]] = [1.0, 0.0, 1.0]
    var fnames: List[String] = ["u", "v"]
    var tnames: List[String] = ["w"]
    var ds = Dataset(X^, y^, fnames^, tnames^)

    var binarizer = Binarizer()
    var ds_bin = binarizer.fit_transform(ds)
    assert_equal(ds_bin.n_samples(), 3)
    assert_equal(ds_bin.n_features(), 2)
    assert_equal(ds_bin.feature_names[1], "v")
    assert_equal(ds_bin.target_names[0], "w")
    assert_equal(ds_bin.records[0, 0], 0.0)
    assert_equal(ds_bin.records[2, 0], 1.0)


def test_binarizer_not_fitted() raises:
    var binarizer = Binarizer()
    var X = Matrix[DType.float64](2, 2, 1.0)
    with assert_raises():
        _ = binarizer.transform(X)


def test_binarizer_dtype_incoherence_prevention() raises:
    var binarizer = Binarizer()
    var X32 = Matrix[DType.float32](2, 2, 1.0)
    binarizer.fit(X32)
    var X64 = Matrix[DType.float64](2, 2, 1.0)
    with assert_raises():
        _ = binarizer.transform(X64)


def test_binarizer_dimension_mismatch_transform() raises:
    var X = Matrix[DType.float64](2, 2, 1.0)
    var binarizer = Binarizer()
    binarizer.fit(X)
    var X_wrong = Matrix[DType.float64](2, 4, 1.0)
    with assert_raises():
        _ = binarizer.transform(X_wrong)


def test_binarizer_float32_native() raises:
    var X = Matrix[DType.float32](1, 2, 0)
    X[0, 0] = -0.25
    X[0, 1] = 0.25
    var binarizer = Binarizer()
    var Xb = binarizer.fit_transform(X)
    assert_equal(Xb[0, 0], Float32(0.0))
    assert_equal(Xb[0, 1], Float32(1.0))


def test_binarizer_int32_input() raises:
    var X = Matrix[DType.int32](1, 3, 0)
    X[0, 0] = -4
    X[0, 1] = 0
    X[0, 2] = 7
    var binarizer = Binarizer()
    var Xb = binarizer.fit_transform(X)
    assert_equal(Xb[0, 0], 0)
    assert_equal(Xb[0, 1], 0)
    assert_equal(Xb[0, 2], 1)


def test_binarizer_copy_constructor() raises:
    var X = Matrix[DType.float64](1, 2, 0)
    X[0, 0] = 4.0
    X[0, 1] = 6.0
    var binarizer = Binarizer(5.0)
    binarizer.fit(X)

    var clone = Binarizer(copy=binarizer)
    assert_true(clone.is_fitted)
    assert_equal(clone.threshold, 5.0)
    assert_equal(clone.n_features_in_, 2)
    var Xb = clone.transform(X)
    assert_equal(Xb[0, 0], 0.0)
    assert_equal(Xb[0, 1], 1.0)


def test_binarizer_idempotent_on_binary_input() raises:
    var X = Matrix[DType.float64](2, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 0.0
    X[1, 0] = 0.0
    X[1, 1] = 1.0

    var binarizer = Binarizer()
    var once = binarizer.fit_transform(X)
    var twice = binarizer.transform(once)
    for i in range(2):
        for j in range(2):
            assert_equal(once[i, j], twice[i, j])


def test_binarizer_all_below_threshold() raises:
    var X = Matrix[DType.float64](3, 2, -1.0)
    var binarizer = Binarizer()
    var Xb = binarizer.fit_transform(X)
    for i in range(3):
        for j in range(2):
            assert_equal(Xb[i, j], 0.0)


def test_binarizer_refitting_updates_feature_count() raises:
    var X1 = Matrix[DType.float64](2, 2, 1.0)
    var binarizer = Binarizer()
    binarizer.fit(X1)
    assert_equal(binarizer.n_features_in_, 2)

    var X2 = Matrix[DType.float64](2, 5, 1.0)
    binarizer.fit(X2)
    assert_equal(binarizer.n_features_in_, 5)


def _ohe_fixture() raises -> Matrix[DType.float64]:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 0.0
    X[1, 0] = 1.0
    X[2, 0] = 0.0
    X[3, 0] = 2.0
    X[0, 1] = 5.0
    X[1, 1] = 5.0
    X[2, 1] = 7.0
    X[3, 1] = 7.0
    return X^


def test_one_hot_encoder_categories_discovered_sorted() raises:
    var X = _ohe_fixture()
    var encoder = OneHotEncoder()
    encoder.fit(X)

    assert_true(encoder.is_fitted)
    assert_equal(encoder.n_features_in_, 2)
    assert_equal(len(encoder.categories_[0]), 3)
    assert_equal(encoder.categories_[0][0], 0.0)
    assert_equal(encoder.categories_[0][1], 1.0)
    assert_equal(encoder.categories_[0][2], 2.0)
    assert_equal(len(encoder.categories_[1]), 2)
    assert_equal(encoder.categories_[1][0], 5.0)
    assert_equal(encoder.categories_[1][1], 7.0)


def test_one_hot_encoder_dense_expansion() raises:
    var X = _ohe_fixture()
    var encoder = OneHotEncoder()
    var Xe = encoder.fit_transform(X)

    assert_equal(encoder.n_features_out(), 5)
    assert_equal(Xe.rows, 4)
    assert_equal(Xe.cols, 5)

    var expected: List[List[Float64]] = [
        [1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 1.0],
    ]
    for r in range(4):
        for c in range(5):
            assert_equal(Xe[r, c], expected[r][c])


def test_one_hot_encoder_rows_sum_to_feature_count() raises:
    var X = _ohe_fixture()
    var encoder = OneHotEncoder()
    var Xe = encoder.fit_transform(X)
    for r in range(Xe.rows):
        var total = 0.0
        for c in range(Xe.cols):
            total += Xe[r, c]
        assert_equal(total, 2.0)


def test_one_hot_encoder_drop_first() raises:
    var X = _ohe_fixture()
    var encoder = OneHotEncoder("first")
    var Xe = encoder.fit_transform(X)

    assert_equal(encoder.n_features_out(), 3)
    assert_equal(Xe.cols, 3)

    var expected: List[List[Float64]] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
    for r in range(4):
        for c in range(3):
            assert_equal(Xe[r, c], expected[r][c])


def test_one_hot_encoder_drop_if_binary() raises:
    var X = _ohe_fixture()
    var encoder = OneHotEncoder("if_binary")
    var Xe = encoder.fit_transform(X)

    assert_equal(encoder.drop_idx_[0], -1)
    assert_equal(encoder.drop_idx_[1], 0)
    assert_equal(Xe.cols, 4)

    var expected: List[List[Float64]] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
    ]
    for r in range(4):
        for c in range(4):
            assert_equal(Xe[r, c], expected[r][c])


def test_one_hot_encoder_unknown_category_raises() raises:
    var X = Matrix[DType.float64](2, 1, 0)
    X[0, 0] = 0.0
    X[1, 0] = 1.0

    var encoder = OneHotEncoder()
    encoder.fit(X)

    var X_new = Matrix[DType.float64](1, 1, 9.0)
    with assert_raises(contains="found unknown category"):
        _ = encoder.transform(X_new)


def test_one_hot_encoder_unknown_category_ignored() raises:
    var X = Matrix[DType.float64](2, 1, 0)
    X[0, 0] = 0.0
    X[1, 0] = 1.0

    var encoder = OneHotEncoder("none", "ignore")
    encoder.fit(X)

    var X_new = Matrix[DType.float64](2, 1, 0)
    X_new[0, 0] = 9.0
    X_new[1, 0] = 1.0
    var Xe = encoder.transform(X_new)
    assert_equal(Xe.cols, 2)
    assert_equal(Xe[0, 0], 0.0)
    assert_equal(Xe[0, 1], 0.0)
    assert_equal(Xe[1, 0], 0.0)
    assert_equal(Xe[1, 1], 1.0)


def test_one_hot_encoder_single_category_column() raises:
    var X = Matrix[DType.float64](3, 1, 4.0)
    var encoder = OneHotEncoder()
    var Xe = encoder.fit_transform(X)
    assert_equal(Xe.cols, 1)
    for r in range(3):
        assert_equal(Xe[r, 0], 1.0)


def test_one_hot_encoder_negative_categories() raises:
    var X = Matrix[DType.float64](3, 1, 0)
    X[0, 0] = 1.0
    X[1, 0] = -2.0
    X[2, 0] = 0.0

    var encoder = OneHotEncoder()
    var Xe = encoder.fit_transform(X)
    assert_equal(encoder.categories_[0][0], -2.0)
    assert_equal(encoder.categories_[0][1], 0.0)
    assert_equal(encoder.categories_[0][2], 1.0)
    assert_equal(Xe[0, 2], 1.0)
    assert_equal(Xe[1, 0], 1.0)
    assert_equal(Xe[2, 1], 1.0)


def test_one_hot_encoder_feature_names_default() raises:
    var X = _ohe_fixture()
    var encoder = OneHotEncoder()
    encoder.fit(X)

    var names = encoder.get_feature_names_out()
    assert_equal(len(names), 5)
    assert_equal(names[0], "x0_0.0")
    assert_equal(names[2], "x0_2.0")
    assert_equal(names[3], "x1_5.0")
    assert_equal(names[4], "x1_7.0")


def test_one_hot_encoder_feature_names_from_input() raises:
    var X = _ohe_fixture()
    var encoder = OneHotEncoder("first")
    encoder.fit(X)

    var input_names: List[String] = ["color", "size"]
    var names = encoder.get_feature_names_out(input_names)
    assert_equal(len(names), 3)
    assert_equal(names[0], "color_1.0")
    assert_equal(names[1], "color_2.0")
    assert_equal(names[2], "size_7.0")


def test_one_hot_encoder_feature_names_length_mismatch() raises:
    var X = _ohe_fixture()
    var encoder = OneHotEncoder()
    encoder.fit(X)
    var bad: List[String] = ["only_one"]
    with assert_raises():
        _ = encoder.get_feature_names_out(bad)


def test_one_hot_encoder_dataset_expands_feature_names() raises:
    var X = _ohe_fixture()
    var y: List[Scalar[DType.float64]] = [0.0, 1.0, 1.0, 0.0]
    var fnames: List[String] = ["color", "size"]
    var tnames: List[String] = ["label"]
    var ds = Dataset(X^, y^, fnames^, tnames^)

    var encoder = OneHotEncoder()
    var ds_enc = encoder.fit_transform(ds)
    assert_equal(ds_enc.n_samples(), 4)
    assert_equal(ds_enc.n_features(), 5)
    assert_equal(len(ds_enc.feature_names), 5)
    assert_equal(ds_enc.feature_names[0], "color_0.0")
    assert_equal(ds_enc.feature_names[4], "size_7.0")
    assert_equal(ds_enc.target_names[0], "label")
    assert_equal(ds_enc.targets[1], 1.0)


def test_one_hot_encoder_inverse_transform_roundtrip() raises:
    var X = _ohe_fixture()
    var encoder = OneHotEncoder()
    var Xe = encoder.fit_transform(X)
    var Xr = encoder.inverse_transform(Xe)
    assert_equal(Xr.rows, 4)
    assert_equal(Xr.cols, 2)
    for r in range(4):
        for c in range(2):
            assert_equal(Xr[r, c], X[r, c])


def test_one_hot_encoder_inverse_transform_with_drop() raises:
    var X = _ohe_fixture()
    var encoder = OneHotEncoder("first")
    var Xe = encoder.fit_transform(X)
    var Xr = encoder.inverse_transform(Xe)
    for r in range(4):
        for c in range(2):
            assert_equal(Xr[r, c], X[r, c])


def test_one_hot_encoder_inverse_transform_all_zero_block_raises() raises:
    var X = Matrix[DType.float64](2, 1, 0)
    X[0, 0] = 0.0
    X[1, 0] = 1.0
    var encoder = OneHotEncoder()
    encoder.fit(X)

    var Xe = Matrix[DType.float64](1, 2, 0)
    with assert_raises(contains="all-zero block"):
        _ = encoder.inverse_transform(Xe)


def test_one_hot_encoder_not_fitted() raises:
    var encoder = OneHotEncoder()
    var X = Matrix[DType.float64](2, 2, 1.0)
    with assert_raises():
        _ = encoder.transform(X)
    with assert_raises():
        _ = encoder.inverse_transform(X)
    with assert_raises():
        _ = encoder.get_feature_names_out()


def test_one_hot_encoder_invalid_parameters() raises:
    with assert_raises():
        var a = OneHotEncoder("last")
        _ = a.drop
    with assert_raises():
        var b = OneHotEncoder("none", "skip")
        _ = b.drop


def test_one_hot_encoder_dtype_incoherence_prevention() raises:
    var encoder = OneHotEncoder()
    var X32 = Matrix[DType.float32](2, 1, 1.0)
    encoder.fit(X32)
    var X64 = Matrix[DType.float64](2, 1, 1.0)
    with assert_raises():
        _ = encoder.transform(X64)


def test_one_hot_encoder_dimension_mismatch_transform() raises:
    var X = _ohe_fixture()
    var encoder = OneHotEncoder()
    encoder.fit(X)
    var X_wrong = Matrix[DType.float64](2, 3, 0.0)
    with assert_raises():
        _ = encoder.transform(X_wrong)


def test_one_hot_encoder_int32_input() raises:
    var X = Matrix[DType.int32](3, 1, 0)
    X[0, 0] = 10
    X[1, 0] = 20
    X[2, 0] = 10
    var encoder = OneHotEncoder()
    var Xe = encoder.fit_transform(X)
    assert_equal(Xe.cols, 2)
    assert_equal(Xe[0, 0], 1)
    assert_equal(Xe[1, 1], 1)
    assert_equal(Xe[2, 0], 1)


def test_one_hot_encoder_float32_native() raises:
    var X = Matrix[DType.float32](2, 1, 0)
    X[0, 0] = 1.5
    X[1, 0] = 2.5
    var encoder = OneHotEncoder()
    var Xe = encoder.fit_transform(X)
    assert_equal(Xe.cols, 2)
    assert_equal(Xe[0, 0], Float32(1.0))
    assert_equal(Xe[1, 1], Float32(1.0))


def test_one_hot_encoder_copy_constructor() raises:
    var X = _ohe_fixture()
    var encoder = OneHotEncoder("if_binary", "ignore")
    encoder.fit(X)

    var clone = OneHotEncoder(copy=encoder)
    assert_true(clone.is_fitted)
    assert_equal(clone.drop, "if_binary")
    assert_equal(clone.handle_unknown, "ignore")
    assert_equal(clone.n_features_out(), 4)
    var Xe = clone.transform(X)
    assert_equal(Xe.cols, 4)


def test_one_hot_encoder_refitting_resets_categories() raises:
    var X1 = _ohe_fixture()
    var encoder = OneHotEncoder()
    encoder.fit(X1)
    assert_equal(encoder.n_features_out(), 5)

    var X2 = Matrix[DType.float64](2, 1, 0)
    X2[0, 0] = 3.0
    X2[1, 0] = 4.0
    encoder.fit(X2)
    assert_equal(encoder.n_features_in_, 1)
    assert_equal(encoder.n_features_out(), 2)
    assert_equal(encoder.categories_[0][0], 3.0)


def test_one_hot_encoder_full_drop_produces_zero_columns() raises:
    var X = Matrix[DType.float64](3, 1, 4.0)
    var encoder = OneHotEncoder("first")
    var Xe = encoder.fit_transform(X)

    assert_equal(encoder.n_features_out(), 0)
    assert_equal(Xe.cols, 0)
    assert_equal(len(encoder.get_feature_names_out()), 0)

    var Xr = encoder.inverse_transform(Xe)
    assert_equal(Xr.rows, 3)
    assert_equal(Xr.cols, 1)
    for r in range(3):
        assert_equal(Xr[r, 0], 4.0)


def test_one_hot_encoder_inverse_transform_rejects_empty_rows() raises:
    var X = _ohe_fixture()
    var encoder = OneHotEncoder()
    encoder.fit(X)
    var Xe = Matrix[DType.float64](0, 5, 0)
    with assert_raises():
        _ = encoder.inverse_transform(Xe)


def test_one_hot_encoder_mixed_drop_keeps_nonempty_blocks() raises:
    var X = Matrix[DType.float64](3, 2, 0)
    X[0, 0] = 7.0
    X[1, 0] = 7.0
    X[2, 0] = 7.0
    X[0, 1] = 1.0
    X[1, 1] = 2.0
    X[2, 1] = 3.0

    var encoder = OneHotEncoder("first")
    var Xe = encoder.fit_transform(X)
    assert_equal(Xe.cols, 2)

    var Xr = encoder.inverse_transform(Xe)
    for r in range(3):
        assert_equal(Xr[r, 0], X[r, 0])
        assert_equal(Xr[r, 1], X[r, 1])


def test_scalers_honor_non_default_compute_dtype() raises:
    var X = Matrix[DType.float32](5, 1, 0)
    for i in range(5):
        X[i, 0] = Float32(i + 1)

    var mm = MinMaxScaler[DType.float32]()
    var mm_out = mm.fit_transform(X)
    assert_almost_equal(mm.scale_[0], Float32(0.25), atol=1e-6)
    assert_almost_equal(mm_out[0, 0], Float32(0.0), atol=1e-6)
    assert_almost_equal(mm_out[4, 0], Float32(1.0), atol=1e-6)

    var rs = RobustScaler[DType.float32]()
    var rs_out = rs.fit_transform(X)
    assert_almost_equal(rs.center_[0], Float32(3.0), atol=1e-6)
    assert_almost_equal(rs.scale_[0], Float32(2.0), atol=1e-6)
    assert_almost_equal(rs_out[0, 0], Float32(-1.0), atol=1e-6)


def test_binarizer_and_encoder_honor_non_default_compute_dtype() raises:
    var X = Matrix[DType.float32](5, 1, 0)
    for i in range(5):
        X[i, 0] = Float32(i + 1)

    var bz = Binarizer[DType.float32](3.0)
    var bz_out = bz.fit_transform(X)
    assert_equal(bz_out[2, 0], Float32(0.0))
    assert_equal(bz_out[3, 0], Float32(1.0))

    var enc = OneHotEncoder[DType.float32]()
    var enc_out = enc.fit_transform(X)
    assert_equal(enc_out.cols, 5)
    assert_equal(enc.categories_[0][0], Float32(1.0))
    assert_equal(enc_out[0, 0], Float32(1.0))
    assert_equal(enc_out[4, 4], Float32(1.0))


def test_one_hot_encoder_zero_width_block_between_populated_blocks() raises:
    var X = Matrix[DType.float64](3, 3, 0)
    X[0, 0] = 1.0
    X[1, 0] = 2.0
    X[2, 0] = 3.0
    X[0, 1] = 9.0
    X[1, 1] = 9.0
    X[2, 1] = 9.0
    X[0, 2] = 4.0
    X[1, 2] = 5.0
    X[2, 2] = 4.0

    var encoder = OneHotEncoder("first")
    var Xe = encoder.fit_transform(X)
    assert_equal(Xe.cols, 3)

    var names = encoder.get_feature_names_out()
    assert_equal(len(names), 3)
    assert_equal(names[0], "x0_2.0")
    assert_equal(names[1], "x0_3.0")
    assert_equal(names[2], "x2_5.0")

    assert_equal(Xe[1, 0], 1.0)
    assert_equal(Xe[1, 2], 1.0)
    assert_equal(Xe[2, 1], 1.0)
    assert_equal(Xe[2, 2], 0.0)

    var Xr = encoder.inverse_transform(Xe)
    for r in range(3):
        for c in range(3):
            assert_equal(Xr[r, c], X[r, c])


def test_one_hot_encoder_multiple_zero_width_blocks() raises:
    var X = Matrix[DType.float64](2, 3, 0)
    X[0, 0] = 7.0
    X[1, 0] = 7.0
    X[0, 1] = 1.0
    X[1, 1] = 2.0
    X[0, 2] = 8.0
    X[1, 2] = 8.0

    var encoder = OneHotEncoder("first")
    var Xe = encoder.fit_transform(X)
    assert_equal(Xe.cols, 1)

    var Xr = encoder.inverse_transform(Xe)
    for r in range(2):
        for c in range(3):
            assert_equal(Xr[r, c], X[r, c])


def test_transformers_reject_empty_input_on_fit() raises:
    var X = Matrix[DType.float64](0, 2, 0)
    var mm = MinMaxScaler()
    with assert_raises():
        mm.fit(X)
    var rs = RobustScaler()
    with assert_raises():
        rs.fit(X)
    var bz = Binarizer()
    with assert_raises():
        bz.fit(X)
    var enc = OneHotEncoder("first")
    with assert_raises():
        enc.fit(X)


def test_transformers_validate_array_before_shape() raises:
    var X = Matrix[DType.float64](3, 2, 1.0)
    var mm = MinMaxScaler()
    mm.fit(X)
    var rs = RobustScaler()
    rs.fit(X)
    var bz = Binarizer()
    bz.fit(X)

    var empty = Matrix[DType.float64](0, 5, 0)
    with assert_raises(contains="non-empty 2D array"):
        _ = mm.transform(empty)
    with assert_raises(contains="non-empty 2D array"):
        _ = rs.transform(empty)
    with assert_raises(contains="non-empty 2D array"):
        _ = bz.transform(empty)
    with assert_raises(contains="non-empty 2D array"):
        _ = mm.inverse_transform(empty)
    with assert_raises(contains="non-empty 2D array"):
        _ = rs.inverse_transform(empty)

    var wrong = Matrix[DType.float64](3, 5, 1.0)
    with assert_raises(contains="MinMaxScaler.transform"):
        _ = mm.transform(wrong)
    with assert_raises(contains="RobustScaler.transform"):
        _ = rs.transform(wrong)
    with assert_raises(contains="Binarizer.transform"):
        _ = bz.transform(wrong)


def test_one_hot_encoder_validates_array_before_shape() raises:
    var X = _ohe_fixture()
    var encoder = OneHotEncoder()
    encoder.fit(X)

    var empty = Matrix[DType.float64](0, 9, 0)
    with assert_raises(contains="non-empty 2D array"):
        _ = encoder.transform(empty)
    with assert_raises(contains="non-empty 2D array"):
        _ = encoder.inverse_transform(empty)

    var wrong = Matrix[DType.float64](2, 3, 0.0)
    with assert_raises(contains="OneHotEncoder.transform"):
        _ = encoder.transform(wrong)


def test_one_hot_encoder_drop_never_indexes_missing_category() raises:
    var X = _ohe_fixture()
    var encoder = OneHotEncoder("first")
    encoder.fit(X)
    for f in range(encoder.n_features_in_):
        assert_true(encoder.drop_idx_[f] < len(encoder.categories_[f]))
    assert_true(encoder.n_features_out() >= 0)


def test_one_hot_encoder_signed_zero_collapses_to_one_category() raises:
    var X = Matrix[DType.float64](3, 1, 0)
    X[0, 0] = -0.0
    X[1, 0] = 0.0
    X[2, 0] = 1.0

    var encoder = OneHotEncoder()
    var Xe = encoder.fit_transform(X)
    assert_equal(len(encoder.categories_[0]), 2)
    assert_equal(Xe.cols, 2)
    assert_equal(Xe[0, 0], 1.0)
    assert_equal(Xe[1, 0], 1.0)
    assert_equal(Xe[2, 1], 1.0)

    var probe = Matrix[DType.float64](1, 1, 0)
    probe[0, 0] = -0.0
    var neg = encoder.transform(probe)
    probe[0, 0] = 0.0
    var pos = encoder.transform(probe)
    assert_equal(neg[0, 0], 1.0)
    assert_equal(pos[0, 0], 1.0)


def test_robust_scaler_single_row_quantiles() raises:
    var X = Matrix[DType.float64](1, 2, 0)
    X[0, 0] = 5.0
    X[0, 1] = -3.0
    var scaler = RobustScaler()
    var Xs = scaler.fit_transform(X)
    assert_equal(scaler.center_[0], 5.0)
    assert_equal(scaler.center_[1], -3.0)
    assert_equal(scaler.scale_[0], 1.0)
    assert_equal(Xs[0, 0], 0.0)
    assert_equal(Xs[0, 1], 0.0)


def test_min_max_scaler_single_row_uses_unit_scale() raises:
    var X = Matrix[DType.float64](1, 1, 5.0)
    var scaler = MinMaxScaler()
    scaler.fit(X)
    assert_equal(scaler.scale_[0], 1.0)
    assert_equal(scaler.min_[0], -5.0)
    assert_equal(scaler.data_range_[0], 0.0)


def test_robust_scaler_full_quantile_range_boundaries() raises:
    var X = Matrix[DType.float64](6, 1, 0)
    var vals: List[Float64] = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]
    for i in range(6):
        X[i, 0] = vals[i]

    var scaler = RobustScaler(True, True, 0.0, 100.0)
    scaler.fit(X)
    assert_almost_equal(scaler.center_[0], 3.5, atol=1e-12)
    assert_almost_equal(scaler.scale_[0], 99.0, atol=1e-12)

    var Xs = scaler.transform(X)
    assert_almost_equal(Xs[0, 0], (1.0 - 3.5) / 99.0, atol=1e-12)
    assert_almost_equal(Xs[5, 0], (100.0 - 3.5) / 99.0, atol=1e-12)


def test_robust_scaler_quantile_100_uses_maximum() raises:
    var X = Matrix[DType.float64](11, 1, 0)
    for i in range(11):
        X[i, 0] = Float64(i)

    var scaler = RobustScaler(True, True, 0.0, 100.0)
    scaler.fit(X)
    assert_almost_equal(scaler.center_[0], 5.0, atol=1e-12)
    assert_almost_equal(scaler.scale_[0], 10.0, atol=1e-12)

    var half = RobustScaler(True, True, 0.0, 50.0)
    half.fit(X)
    assert_almost_equal(half.scale_[0], 5.0, atol=1e-12)


def _ordinal_fixture() raises -> Matrix[DType.float64]:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 20.0
    X[1, 0] = 10.0
    X[2, 0] = 20.0
    X[3, 0] = 30.0
    X[0, 1] = 5.0
    X[1, 1] = 7.0
    X[2, 1] = 7.0
    X[3, 1] = 5.0
    return X^


def test_ordinal_encoder_categories_discovered_sorted() raises:
    var X = _ordinal_fixture()
    var encoder = OrdinalEncoder()
    encoder.fit(X)

    assert_true(encoder.is_fitted)
    assert_equal(encoder.n_features_in_, 2)
    assert_equal(len(encoder.categories_[0]), 3)
    assert_equal(encoder.categories_[0][0], 10.0)
    assert_equal(encoder.categories_[0][1], 20.0)
    assert_equal(encoder.categories_[0][2], 30.0)
    assert_equal(len(encoder.categories_[1]), 2)
    assert_equal(encoder.categories_[1][0], 5.0)
    assert_equal(encoder.categories_[1][1], 7.0)


def test_ordinal_encoder_preserves_shape() raises:
    var X = _ordinal_fixture()
    var encoder = OrdinalEncoder()
    var Xe = encoder.fit_transform(X)
    assert_equal(Xe.rows, 4)
    assert_equal(Xe.cols, 2)


def test_ordinal_encoder_codes() raises:
    var X = _ordinal_fixture()
    var encoder = OrdinalEncoder()
    var Xe = encoder.fit_transform(X)

    var expected: List[List[Float64]] = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 0.0],
    ]
    for r in range(4):
        for c in range(2):
            assert_equal(Xe[r, c], expected[r][c])


def test_ordinal_encoder_columns_encoded_independently() raises:
    var X = Matrix[DType.float64](2, 2, 0)
    X[0, 0] = 100.0
    X[1, 0] = 200.0
    X[0, 1] = 3.0
    X[1, 1] = 4.0

    var encoder = OrdinalEncoder()
    var Xe = encoder.fit_transform(X)
    assert_equal(Xe[0, 0], 0.0)
    assert_equal(Xe[1, 0], 1.0)
    assert_equal(Xe[0, 1], 0.0)
    assert_equal(Xe[1, 1], 1.0)


def test_ordinal_encoder_fit_transform_matches_fit_then_transform() raises:
    var X = _ordinal_fixture()
    var a = OrdinalEncoder()
    var combined = a.fit_transform(X)

    var b = OrdinalEncoder()
    b.fit(X)
    var separate = b.transform(X)

    for r in range(4):
        for c in range(2):
            assert_equal(combined[r, c], separate[r, c])


def test_ordinal_encoder_single_category_column() raises:
    var X = Matrix[DType.float64](3, 1, 7.0)
    var encoder = OrdinalEncoder()
    var Xe = encoder.fit_transform(X)
    assert_equal(len(encoder.categories_[0]), 1)
    for r in range(3):
        assert_equal(Xe[r, 0], 0.0)


def test_ordinal_encoder_negative_categories() raises:
    var X = Matrix[DType.float64](3, 1, 0)
    X[0, 0] = -5.0
    X[1, 0] = 0.0
    X[2, 0] = -9.0

    var encoder = OrdinalEncoder()
    var Xe = encoder.fit_transform(X)
    assert_equal(encoder.categories_[0][0], -9.0)
    assert_equal(encoder.categories_[0][1], -5.0)
    assert_equal(encoder.categories_[0][2], 0.0)
    assert_equal(Xe[0, 0], 1.0)
    assert_equal(Xe[1, 0], 2.0)
    assert_equal(Xe[2, 0], 0.0)


def test_ordinal_encoder_unknown_category_raises() raises:
    var X = _ordinal_fixture()
    var encoder = OrdinalEncoder()
    encoder.fit(X)

    var Xn = Matrix[DType.float64](1, 2, 0)
    Xn[0, 0] = 99.0
    Xn[0, 1] = 5.0
    with assert_raises():
        _ = encoder.transform(Xn)


def test_ordinal_encoder_unknown_uses_encoded_value() raises:
    var X = _ordinal_fixture()
    var encoder = OrdinalEncoder(handle_unknown="use_encoded_value")
    encoder.fit(X)

    var Xn = Matrix[DType.float64](2, 2, 0)
    Xn[0, 0] = 99.0
    Xn[0, 1] = 5.0
    Xn[1, 0] = 10.0
    Xn[1, 1] = 42.0

    var Xe = encoder.transform(Xn)
    assert_equal(Xe[0, 0], -1.0)
    assert_equal(Xe[0, 1], 0.0)
    assert_equal(Xe[1, 0], 0.0)
    assert_equal(Xe[1, 1], -1.0)


def test_ordinal_encoder_custom_unknown_value() raises:
    var X = _ordinal_fixture()
    var encoder = OrdinalEncoder("use_encoded_value", -42.0)
    encoder.fit(X)

    var Xn = Matrix[DType.float64](1, 2, 0)
    Xn[0, 0] = 99.0
    Xn[0, 1] = 7.0
    var Xe = encoder.transform(Xn)
    assert_equal(Xe[0, 0], -42.0)
    assert_equal(Xe[0, 1], 1.0)


def test_ordinal_encoder_unknown_value_collision_raises() raises:
    var X = _ordinal_fixture()
    var encoder = OrdinalEncoder("use_encoded_value", 1.0)
    with assert_raises():
        encoder.fit(X)
    assert_true(not encoder.is_fitted)


def test_ordinal_encoder_failed_refit_preserves_state() raises:
    var small = Matrix[DType.float64](2, 1, 0)
    small[0, 0] = 4.0
    small[1, 0] = 5.0

    var encoder = OrdinalEncoder("use_encoded_value", 2.0)
    encoder.fit(small)
    assert_equal(len(encoder.categories_[0]), 2)

    var big = _ordinal_fixture()
    with assert_raises():
        encoder.fit(big)

    assert_true(encoder.is_fitted)
    assert_equal(encoder.n_features_in_, 1)
    assert_equal(len(encoder.categories_), 1)
    assert_equal(len(encoder.categories_[0]), 2)


def test_ordinal_encoder_feature_names_default() raises:
    var X = _ordinal_fixture()
    var encoder = OrdinalEncoder()
    encoder.fit(X)

    var names = encoder.get_feature_names_out()
    assert_equal(len(names), 2)
    assert_equal(names[0], "x0")
    assert_equal(names[1], "x1")


def test_ordinal_encoder_feature_names_passthrough() raises:
    var X = _ordinal_fixture()
    var encoder = OrdinalEncoder()
    encoder.fit(X)

    var given: List[String] = ["city", "size"]
    var names = encoder.get_feature_names_out(given)
    assert_equal(len(names), 2)
    assert_equal(names[0], "city")
    assert_equal(names[1], "size")


def test_ordinal_encoder_feature_names_length_mismatch() raises:
    var X = _ordinal_fixture()
    var encoder = OrdinalEncoder()
    encoder.fit(X)

    var given: List[String] = ["only_one"]
    with assert_raises():
        _ = encoder.get_feature_names_out(given)


def test_ordinal_encoder_dataset_keeps_targets_and_names() raises:
    var X = _ordinal_fixture()
    var y: List[Scalar[DType.float64]] = [0.0, 1.0, 1.0, 0.0]
    var fnames: List[String] = ["city", "size"]
    var tnames: List[String] = ["label"]
    var ds = Dataset(X^, y^, fnames^, tnames^)

    var encoder = OrdinalEncoder()
    var ds_enc = encoder.fit_transform(ds)
    assert_equal(ds_enc.n_samples(), 4)
    assert_equal(ds_enc.n_features(), 2)
    assert_equal(len(ds_enc.feature_names), 2)
    assert_equal(ds_enc.feature_names[0], "city")
    assert_equal(ds_enc.feature_names[1], "size")
    assert_equal(ds_enc.target_names[0], "label")
    assert_equal(ds_enc.targets[1], 1.0)
    assert_equal(ds_enc.records[0, 0], 1.0)
    assert_equal(ds_enc.records[3, 0], 2.0)


def test_ordinal_encoder_inverse_transform_roundtrip() raises:
    var X = _ordinal_fixture()
    var encoder = OrdinalEncoder()
    var Xe = encoder.fit_transform(X)
    var Xr = encoder.inverse_transform(Xe)

    assert_equal(Xr.rows, 4)
    assert_equal(Xr.cols, 2)
    for r in range(4):
        for c in range(2):
            assert_equal(Xr[r, c], X[r, c])


def test_ordinal_encoder_inverse_transform_rejects_bad_codes() raises:
    var X = _ordinal_fixture()
    var encoder = OrdinalEncoder()
    encoder.fit(X)

    var too_big = Matrix[DType.float64](1, 2, 0)
    too_big[0, 0] = 3.0
    with assert_raises():
        _ = encoder.inverse_transform(too_big)

    var negative = Matrix[DType.float64](1, 2, 0)
    negative[0, 0] = -1.0
    with assert_raises():
        _ = encoder.inverse_transform(negative)

    var fractional = Matrix[DType.float64](1, 2, 0)
    fractional[0, 0] = 1.5
    with assert_raises():
        _ = encoder.inverse_transform(fractional)


def test_ordinal_encoder_inverse_transform_unknown_raises() raises:
    var X = _ordinal_fixture()
    var encoder = OrdinalEncoder(handle_unknown="use_encoded_value")
    encoder.fit(X)

    var Xn = Matrix[DType.float64](1, 2, 0)
    Xn[0, 0] = 99.0
    Xn[0, 1] = 5.0
    var Xe = encoder.transform(Xn)
    with assert_raises():
        _ = encoder.inverse_transform(Xe)


def test_ordinal_encoder_not_fitted() raises:
    var X = _ordinal_fixture()
    var encoder = OrdinalEncoder()

    with assert_raises():
        _ = encoder.transform(X)
    with assert_raises():
        _ = encoder.inverse_transform(X)
    with assert_raises():
        _ = encoder.get_feature_names_out()


def test_ordinal_encoder_invalid_parameters() raises:
    with assert_raises():
        _ = OrdinalEncoder(handle_unknown="ignore")


def test_ordinal_encoder_dtype_incoherence_prevention() raises:
    var X = _ordinal_fixture()
    var encoder = OrdinalEncoder()
    encoder.fit(X)

    var X32 = Matrix[DType.float32](4, 2, 0)
    with assert_raises():
        _ = encoder.transform(X32)
    with assert_raises():
        _ = encoder.inverse_transform(X32)


def test_ordinal_encoder_dimension_mismatch_transform() raises:
    var X = _ordinal_fixture()
    var encoder = OrdinalEncoder()
    encoder.fit(X)

    var wide = Matrix[DType.float64](2, 3, 0)
    with assert_raises():
        _ = encoder.transform(wide)
    with assert_raises():
        _ = encoder.inverse_transform(wide)


def test_ordinal_encoder_copy_constructor() raises:
    var X = _ordinal_fixture()
    var encoder = OrdinalEncoder("use_encoded_value", -3.0)
    encoder.fit(X)

    var clone = OrdinalEncoder[DType.float64](copy=encoder)
    assert_true(clone.is_fitted)
    assert_equal(clone.handle_unknown, "use_encoded_value")
    assert_equal(clone.unknown_value, -3.0)
    assert_equal(clone.n_features_in_, 2)
    assert_equal(len(clone.categories_[0]), 3)
    assert_equal(clone.categories_[0][1], 20.0)


def _label_fixture() -> List[Scalar[DType.float64]]:
    var y: List[Scalar[DType.float64]] = [30.0, 10.0, 30.0, 20.0, 10.0]
    return y^


def test_label_encoder_classes_discovered_sorted() raises:
    var y = _label_fixture()
    var encoder = LabelEncoder()
    encoder.fit(y)

    assert_true(encoder.is_fitted)
    assert_equal(len(encoder.classes_), 3)
    assert_equal(encoder.classes_[0], 10.0)
    assert_equal(encoder.classes_[1], 20.0)
    assert_equal(encoder.classes_[2], 30.0)


def test_label_encoder_codes() raises:
    var y = _label_fixture()
    var encoder = LabelEncoder()
    var codes = encoder.fit_transform(y)

    var expected: List[Float64] = [2.0, 0.0, 2.0, 1.0, 0.0]
    assert_equal(len(codes), 5)
    for i in range(5):
        assert_equal(codes[i], expected[i])


def test_label_encoder_fit_transform_matches_fit_then_transform() raises:
    var y = _label_fixture()
    var a = LabelEncoder()
    var combined = a.fit_transform(y)

    var b = LabelEncoder()
    b.fit(y)
    var separate = b.transform(y)

    for i in range(len(y)):
        assert_equal(combined[i], separate[i])


def test_label_encoder_single_class() raises:
    var y: List[Scalar[DType.float64]] = [7.0, 7.0, 7.0]
    var encoder = LabelEncoder()
    var codes = encoder.fit_transform(y)

    assert_equal(len(encoder.classes_), 1)
    for i in range(3):
        assert_equal(codes[i], 0.0)


def test_label_encoder_negative_and_sparse_labels() raises:
    var y: List[Scalar[DType.float64]] = [0.0, -5.0, 3.0, -5.0]
    var encoder = LabelEncoder()
    var codes = encoder.fit_transform(y)

    assert_equal(encoder.classes_[0], -5.0)
    assert_equal(encoder.classes_[1], 0.0)
    assert_equal(encoder.classes_[2], 3.0)
    assert_equal(codes[0], 1.0)
    assert_equal(codes[1], 0.0)
    assert_equal(codes[2], 2.0)
    assert_equal(codes[3], 0.0)


def test_label_encoder_transform_subset() raises:
    var y = _label_fixture()
    var encoder = LabelEncoder()
    encoder.fit(y)

    var subset: List[Scalar[DType.float64]] = [20.0, 20.0, 10.0]
    var codes = encoder.transform(subset)
    assert_equal(len(codes), 3)
    assert_equal(codes[0], 1.0)
    assert_equal(codes[1], 1.0)
    assert_equal(codes[2], 0.0)


def test_label_encoder_transform_empty_is_allowed() raises:
    var y = _label_fixture()
    var encoder = LabelEncoder()
    encoder.fit(y)

    var empty = List[Scalar[DType.float64]]()
    assert_equal(len(encoder.transform(empty)), 0)
    assert_equal(len(encoder.inverse_transform(empty)), 0)


def test_label_encoder_unseen_label_raises() raises:
    var y = _label_fixture()
    var encoder = LabelEncoder()
    encoder.fit(y)

    var unseen: List[Scalar[DType.float64]] = [30.0, 99.0]
    with assert_raises():
        _ = encoder.transform(unseen)


def test_label_encoder_fit_rejects_empty() raises:
    var empty = List[Scalar[DType.float64]]()
    var encoder = LabelEncoder()
    with assert_raises():
        encoder.fit(empty)
    assert_true(not encoder.is_fitted)


def test_label_encoder_rejects_non_finite() raises:
    var nan = Float64(0.0) / Float64(0.0)
    var inf = Float64(1.0) / Float64(0.0)

    var with_nan: List[Scalar[DType.float64]] = [1.0, nan]
    var a = LabelEncoder()
    with assert_raises():
        a.fit(with_nan)

    var with_inf: List[Scalar[DType.float64]] = [1.0, inf]
    var b = LabelEncoder()
    with assert_raises():
        b.fit(with_inf)

    var y = _label_fixture()
    var c = LabelEncoder()
    c.fit(y)
    var bad: List[Scalar[DType.float64]] = [10.0, nan]
    with assert_raises():
        _ = c.transform(bad)
    with assert_raises():
        _ = c.inverse_transform(bad)


def test_label_encoder_refit_replaces_classes() raises:
    var y = _label_fixture()
    var encoder = LabelEncoder()
    encoder.fit(y)
    assert_equal(len(encoder.classes_), 3)

    var other: List[Scalar[DType.float64]] = [1.0, 2.0]
    encoder.fit(other)
    assert_equal(len(encoder.classes_), 2)
    assert_equal(encoder.classes_[0], 1.0)
    assert_equal(encoder.classes_[1], 2.0)


def test_label_encoder_inverse_transform_roundtrip() raises:
    var y = _label_fixture()
    var encoder = LabelEncoder()
    var codes = encoder.fit_transform(y)
    var back = encoder.inverse_transform(codes)

    assert_equal(len(back), len(y))
    for i in range(len(y)):
        assert_equal(back[i], y[i])


def test_label_encoder_inverse_transform_rejects_bad_codes() raises:
    var y = _label_fixture()
    var encoder = LabelEncoder()
    encoder.fit(y)

    var too_big: List[Scalar[DType.float64]] = [3.0]
    with assert_raises():
        _ = encoder.inverse_transform(too_big)

    var negative: List[Scalar[DType.float64]] = [-1.0]
    with assert_raises():
        _ = encoder.inverse_transform(negative)

    var fractional: List[Scalar[DType.float64]] = [1.5]
    with assert_raises():
        _ = encoder.inverse_transform(fractional)


def test_label_encoder_not_fitted() raises:
    var y = _label_fixture()
    var encoder = LabelEncoder()

    with assert_raises():
        _ = encoder.transform(y)
    with assert_raises():
        _ = encoder.inverse_transform(y)


def test_label_encoder_dtype_incoherence_prevention() raises:
    var y = _label_fixture()
    var encoder = LabelEncoder()
    encoder.fit(y)

    var y32: List[Scalar[DType.float32]] = [30.0, 10.0]
    with assert_raises():
        _ = encoder.transform(y32)
    with assert_raises():
        _ = encoder.inverse_transform(y32)


def test_label_encoder_integer_labels_roundtrip() raises:
    var y: List[Scalar[DType.int32]] = [5, 1, 5, 3]
    var encoder = LabelEncoder()
    var codes = encoder.fit_transform(y)

    assert_equal(codes[0], 2)
    assert_equal(codes[1], 0)
    assert_equal(codes[2], 2)
    assert_equal(codes[3], 1)

    var back = encoder.inverse_transform(codes)
    for i in range(len(y)):
        assert_equal(back[i], y[i])


def test_label_encoder_copy_constructor() raises:
    var y = _label_fixture()
    var encoder = LabelEncoder()
    encoder.fit(y)

    var clone = LabelEncoder[DType.float64](copy=encoder)
    assert_true(clone.is_fitted)
    assert_equal(len(clone.classes_), 3)
    assert_equal(clone.classes_[0], 10.0)
    assert_equal(clone.classes_[2], 30.0)

    var codes = clone.transform(y)
    assert_equal(codes[0], 2.0)


def _imputer_fixture() raises -> Matrix[DType.float64]:
    var nan = Float64(0.0) / Float64(0.0)
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1000.0
    X[1, 0] = nan
    X[2, 0] = 2000.0
    X[3, 0] = 3000.0
    X[0, 1] = 3.0
    X[1, 1] = 2.0
    X[2, 1] = nan
    X[3, 1] = 3.0
    return X^


def test_simple_imputer_mean_statistics() raises:
    var X = _imputer_fixture()
    var imputer = SimpleImputer("mean")
    imputer.fit(X)

    assert_true(imputer.is_fitted)
    assert_equal(imputer.n_features_in_, 2)
    assert_equal(len(imputer.statistics_), 2)
    assert_almost_equal(imputer.statistics_[0], 2000.0)
    assert_almost_equal(imputer.statistics_[1], 2.6666666666666665)


def test_simple_imputer_median_statistics() raises:
    var X = _imputer_fixture()
    var imputer = SimpleImputer("median")
    imputer.fit(X)
    assert_almost_equal(imputer.statistics_[0], 2000.0)
    assert_almost_equal(imputer.statistics_[1], 3.0)


def test_simple_imputer_median_even_count_averages_middles() raises:
    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = 1.0
    X[1, 0] = 2.0
    X[2, 0] = 3.0
    X[3, 0] = 4.0

    var imputer = SimpleImputer("median")
    imputer.fit(X)
    assert_almost_equal(imputer.statistics_[0], 2.5)


def test_simple_imputer_most_frequent_statistics() raises:
    var X = _imputer_fixture()
    var imputer = SimpleImputer("most_frequent")
    imputer.fit(X)
    assert_equal(imputer.statistics_[0], 1000.0)
    assert_equal(imputer.statistics_[1], 3.0)


def test_simple_imputer_most_frequent_tie_takes_smallest() raises:
    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = 7.0
    X[1, 0] = 3.0
    X[2, 0] = 7.0
    X[3, 0] = 3.0

    var imputer = SimpleImputer("most_frequent")
    imputer.fit(X)
    assert_equal(imputer.statistics_[0], 3.0)


def test_simple_imputer_constant_ignores_data() raises:
    var nan = Float64(0.0) / Float64(0.0)
    var X = _imputer_fixture()
    var imputer = SimpleImputer("constant", nan, -7.0)
    imputer.fit(X)
    assert_equal(imputer.statistics_[0], -7.0)
    assert_equal(imputer.statistics_[1], -7.0)


def test_simple_imputer_fills_holes() raises:
    var X = _imputer_fixture()
    var imputer = SimpleImputer("mean")
    var Xf = imputer.fit_transform(X)

    assert_equal(Xf.rows, 4)
    assert_equal(Xf.cols, 2)
    assert_almost_equal(Xf[1, 0], 2000.0)
    assert_almost_equal(Xf[2, 1], 2.6666666666666665)


def test_simple_imputer_output_has_no_missing_values() raises:
    var X = _imputer_fixture()
    var imputer = SimpleImputer("mean")
    var Xf = imputer.fit_transform(X)
    for r in range(Xf.rows):
        for c in range(Xf.cols):
            assert_true(not isnan(Xf[r, c]))


def test_simple_imputer_preserves_observed_values() raises:
    var X = _imputer_fixture()
    var imputer = SimpleImputer("mean")
    var Xf = imputer.fit_transform(X)

    assert_equal(Xf[0, 0], 1000.0)
    assert_equal(Xf[2, 0], 2000.0)
    assert_equal(Xf[3, 0], 3000.0)
    assert_equal(Xf[0, 1], 3.0)
    assert_equal(Xf[1, 1], 2.0)
    assert_equal(Xf[3, 1], 3.0)


def test_simple_imputer_fit_transform_matches_fit_then_transform() raises:
    var X = _imputer_fixture()
    var a = SimpleImputer("mean")
    var combined = a.fit_transform(X)

    var b = SimpleImputer("mean")
    b.fit(X)
    var separate = b.transform(X)

    for r in range(4):
        for c in range(2):
            assert_equal(combined[r, c], separate[r, c])


def test_simple_imputer_transform_uses_fitted_statistics() raises:
    var nan = Float64(0.0) / Float64(0.0)
    var X = _imputer_fixture()
    var imputer = SimpleImputer("mean")
    imputer.fit(X)

    var fresh = Matrix[DType.float64](1, 2, 0)
    fresh[0, 0] = nan
    fresh[0, 1] = 9.0

    var out = imputer.transform(fresh)
    assert_almost_equal(out[0, 0], 2000.0)
    assert_equal(out[0, 1], 9.0)


def test_simple_imputer_sentinel_missing_values() raises:
    var X = Matrix[DType.float64](3, 1, 0)
    X[0, 0] = -1.0
    X[1, 0] = 4.0
    X[2, 0] = 6.0

    var imputer = SimpleImputer("mean", -1.0)
    var Xf = imputer.fit_transform(X)
    assert_almost_equal(imputer.statistics_[0], 5.0)
    assert_almost_equal(Xf[0, 0], 5.0)
    assert_equal(Xf[1, 0], 4.0)
    assert_equal(Xf[2, 0], 6.0)


def test_simple_imputer_zero_sentinel() raises:
    var X = Matrix[DType.float64](3, 1, 0)
    X[0, 0] = 0.0
    X[1, 0] = 2.0
    X[2, 0] = 4.0

    var imputer = SimpleImputer("mean", 0.0)
    var Xf = imputer.fit_transform(X)
    assert_almost_equal(imputer.statistics_[0], 3.0)
    assert_almost_equal(Xf[0, 0], 3.0)


def test_simple_imputer_all_missing_column_raises() raises:
    var nan = Float64(0.0) / Float64(0.0)
    var X = Matrix[DType.float64](2, 1, nan)

    var mean = SimpleImputer("mean")
    with assert_raises():
        mean.fit(X)
    assert_true(not mean.is_fitted)

    var median = SimpleImputer("median")
    with assert_raises():
        median.fit(X)

    var mode = SimpleImputer("most_frequent")
    with assert_raises():
        mode.fit(X)


def test_simple_imputer_all_missing_column_constant_fills() raises:
    var nan = Float64(0.0) / Float64(0.0)
    var X = Matrix[DType.float64](2, 1, nan)

    var imputer = SimpleImputer("constant", nan, 7.0)
    var Xf = imputer.fit_transform(X)
    assert_equal(imputer.statistics_[0], 7.0)
    assert_equal(Xf[0, 0], 7.0)
    assert_equal(Xf[1, 0], 7.0)


def test_simple_imputer_rejects_infinity() raises:
    var inf = Float64(1.0) / Float64(0.0)
    var X = Matrix[DType.float64](2, 1, 0)
    X[0, 0] = 1.0
    X[1, 0] = inf

    var imputer = SimpleImputer("mean")
    with assert_raises():
        imputer.fit(X)


def test_simple_imputer_rejects_stray_nan_with_sentinel() raises:
    var nan = Float64(0.0) / Float64(0.0)
    var X = Matrix[DType.float64](2, 1, 0)
    X[0, 0] = 5.0
    X[1, 0] = nan

    var imputer = SimpleImputer("mean", -1.0)
    with assert_raises():
        imputer.fit(X)


def test_simple_imputer_failed_refit_preserves_state() raises:
    var nan = Float64(0.0) / Float64(0.0)
    var good = Matrix[DType.float64](2, 1, 0)
    good[0, 0] = 4.0
    good[1, 0] = 6.0

    var imputer = SimpleImputer("mean")
    imputer.fit(good)
    assert_almost_equal(imputer.statistics_[0], 5.0)

    var bad = Matrix[DType.float64](2, 2, nan)
    with assert_raises():
        imputer.fit(bad)

    assert_true(imputer.is_fitted)
    assert_equal(imputer.n_features_in_, 1)
    assert_equal(len(imputer.statistics_), 1)
    assert_almost_equal(imputer.statistics_[0], 5.0)


def test_simple_imputer_feature_names_default() raises:
    var X = _imputer_fixture()
    var imputer = SimpleImputer("mean")
    imputer.fit(X)

    var names = imputer.get_feature_names_out()
    assert_equal(len(names), 2)
    assert_equal(names[0], "x0")
    assert_equal(names[1], "x1")


def test_simple_imputer_feature_names_passthrough() raises:
    var X = _imputer_fixture()
    var imputer = SimpleImputer("mean")
    imputer.fit(X)

    var given: List[String] = ["sqft", "beds"]
    var names = imputer.get_feature_names_out(given)
    assert_equal(names[0], "sqft")
    assert_equal(names[1], "beds")


def test_simple_imputer_feature_names_length_mismatch() raises:
    var X = _imputer_fixture()
    var imputer = SimpleImputer("mean")
    imputer.fit(X)

    var given: List[String] = ["only_one"]
    with assert_raises():
        _ = imputer.get_feature_names_out(given)


def test_simple_imputer_dataset_keeps_targets_and_names() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1000.0
    X[1, 0] = -1.0
    X[2, 0] = 2000.0
    X[3, 0] = 3000.0
    X[0, 1] = 3.0
    X[1, 1] = 2.0
    X[2, 1] = -1.0
    X[3, 1] = 3.0

    var y: List[Scalar[DType.float64]] = [10.0, 20.0, 30.0, 40.0]
    var fnames: List[String] = ["sqft", "beds"]
    var tnames: List[String] = ["price"]
    var ds = Dataset(X^, y^, fnames^, tnames^)

    var imputer = SimpleImputer("mean", -1.0)
    var ds_out = imputer.fit_transform(ds)

    assert_equal(ds_out.n_samples(), 4)
    assert_equal(ds_out.n_features(), 2)
    assert_equal(ds_out.feature_names[0], "sqft")
    assert_equal(ds_out.feature_names[1], "beds")
    assert_equal(ds_out.target_names[0], "price")
    assert_equal(ds_out.targets[3], 40.0)
    assert_almost_equal(ds_out.records[1, 0], 2000.0)
    assert_almost_equal(ds_out.records[2, 1], 2.6666666666666665)


def test_simple_imputer_not_fitted() raises:
    var X = _imputer_fixture()
    var imputer = SimpleImputer("mean")

    with assert_raises():
        _ = imputer.transform(X)
    with assert_raises():
        _ = imputer.get_feature_names_out()


def test_simple_imputer_invalid_strategy() raises:
    with assert_raises():
        _ = SimpleImputer("mode")
    with assert_raises():
        _ = SimpleImputer("MEAN")


def test_simple_imputer_dtype_incoherence_prevention() raises:
    var X = _imputer_fixture()
    var imputer = SimpleImputer("mean")
    imputer.fit(X)

    var X32 = Matrix[DType.float32](4, 2, 0)
    with assert_raises():
        _ = imputer.transform(X32)


def test_simple_imputer_dimension_mismatch_transform() raises:
    var X = _imputer_fixture()
    var imputer = SimpleImputer("mean")
    imputer.fit(X)

    var wide = Matrix[DType.float64](2, 3, 0)
    with assert_raises():
        _ = imputer.transform(wide)


def test_simple_imputer_copy_constructor() raises:
    var X = _imputer_fixture()
    var imputer = SimpleImputer("median")
    imputer.fit(X)

    var clone = SimpleImputer[DType.float64](copy=imputer)
    assert_true(clone.is_fitted)
    assert_equal(clone.strategy, "median")
    assert_equal(clone.n_features_in_, 2)
    assert_almost_equal(clone.statistics_[0], 2000.0)

    var Xf = clone.transform(X)
    assert_almost_equal(Xf[1, 0], 2000.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
