from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_almost_equal,
)
from strata import Matrix, Dataset, StandardScaler


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
    from std.testing import assert_raises

    var scaler = StandardScaler()
    var X32 = Matrix[DType.float32](2, 2, 1.0)
    scaler.fit(X32)

    var X64 = Matrix[DType.float64](2, 2, 1.0)
    with assert_raises():
        _ = scaler.transform(X64)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
