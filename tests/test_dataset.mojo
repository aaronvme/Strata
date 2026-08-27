from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_raises,
    assert_almost_equal,
)
from strata import (
    Matrix,
    Dataset,
    DatasetSplit,
    train_test_split,
    DimensionMismatchError,
    InvalidParameterError,
)


def test_dataset_construction_and_validation() raises:
    var X = Matrix[DType.float64](5, 2, 1.0)
    var y: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0, 4.0, 5.0]
    var feat_names: List[String] = ["f1", "f2"]
    var target_names: List[String] = ["t1"]

    var ds = Dataset(X^, y^, feat_names^, target_names^)
    assert_equal(ds.n_samples(), 5)
    assert_equal(ds.n_features(), 2)
    assert_equal(len(ds.targets), 5)
    assert_equal(len(ds.feature_names), 2)
    assert_equal(len(ds.target_names), 1)

    # Dimension mismatch: records.rows != len(targets)
    var X_bad = Matrix[DType.float64](4, 2, 1.0)
    var y_bad: List[Scalar[DType.float64]] = [1.0, 2.0]
    with assert_raises():
        _ = Dataset(X_bad^, y_bad^)


def test_dataset_split_with_ratio_properties() raises:
    var X = Matrix[DType.float64](20, 3, 0)
    var y = List[Scalar[DType.float64]](capacity=20)
    for i in range(20):
        X[i, 0] = Float64(i)
        y.append(Float64(i * 2))

    var ds = Dataset(X^, y^)

    # 1. 25% split (5 test, 15 train)
    var split = ds.split_with_ratio(0.25, shuffle=True, seed=42)
    assert_equal(split.train.n_samples(), 15)
    assert_equal(split.test.n_samples(), 5)
    assert_equal(split.train.n_features(), 3)
    assert_equal(split.test.n_features(), 3)
    assert_equal(len(split.train.targets), 15)
    assert_equal(len(split.test.targets), 5)

    # 2. Sequential split without shuffle
    var seq_split = ds.split_with_ratio(0.25, shuffle=False)
    assert_equal(seq_split.train.targets[0], 0.0)
    assert_equal(seq_split.train.targets[14], 28.0)
    assert_equal(seq_split.test.targets[0], 30.0)
    assert_equal(seq_split.test.targets[4], 38.0)

    # 3. Small dataset guarantee (total >= 2 with small ratio guarantees n_test >= 1)
    var X_small = Matrix[DType.float64](2, 2, 1.0)
    var y_small: List[Scalar[DType.float64]] = [1.0, 2.0]
    var ds_small = Dataset(X_small^, y_small^)
    var small_split = ds_small.split_with_ratio(0.1, shuffle=False)
    assert_equal(small_split.test.n_samples(), 1)
    assert_equal(small_split.train.n_samples(), 1)

    # 4. Invalid ratio validation
    with assert_raises():
        _ = ds.split_with_ratio(0.0)
    with assert_raises():
        _ = ds.split_with_ratio(1.0)
    with assert_raises():
        _ = ds.split_with_ratio(-0.5)
    with assert_raises():
        _ = ds.split_with_ratio(1.5)


def test_train_test_split_matrix_overloads() raises:
    var X = Matrix[DType.float64](10, 2, 0)
    var y = List[Scalar[DType.float64]](capacity=10)
    for i in range(10):
        X[i, 0] = Float64(i)
        y.append(Float64(i))

    var split = train_test_split(X, y, test_size=0.3, shuffle=False)
    assert_equal(split.train.n_samples(), 7)
    assert_equal(split.test.n_samples(), 3)


def test_dataset_copy_constructor_isolation() raises:
    var X = Matrix[DType.float64](3, 2, 1.0)
    var y: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0]
    var ds1 = Dataset(X^, y^)
    var ds2 = ds1.copy()

    assert_equal(ds2.n_samples(), 3)
    assert_equal(ds2.n_features(), 2)
    assert_equal(ds2.records[0, 0], 1.0)

    ds2.records[0, 0] = 999.0
    assert_equal(ds1.records[0, 0], 1.0)
    assert_equal(ds2.records[0, 0], 999.0)


def test_dataset_integer_targets() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    var y: List[Scalar[DType.int32]] = [0, 1, 0, 1]
    var ds = Dataset[DType.float64, DType.int32](X^, y^)

    assert_equal(ds.n_samples(), 4)
    assert_equal(ds.targets[1], 1)
    assert_equal(ds.targets[2], 0)


def test_dataset_float32_features() raises:
    var X = Matrix[DType.float32](3, 3, 2.5)
    var y: List[Scalar[DType.float32]] = [1.0, 2.0, 3.0]
    var ds = Dataset[DType.float32, DType.float32](X^, y^)

    assert_equal(ds.n_samples(), 3)
    assert_equal(ds.n_features(), 3)
    assert_almost_equal(Float64(ds.records[1, 1]), 2.5, rtol=1e-3)


def test_dataset_bfloat16_features() raises:
    var X = Matrix[DType.bfloat16](2, 2, 4.0)
    var y: List[Scalar[DType.int32]] = [1, 2]
    var ds = Dataset[DType.bfloat16, DType.int32](X^, y^)

    assert_equal(ds.n_samples(), 2)
    assert_equal(ds.n_features(), 2)
    assert_equal(ds.targets[0], 1)


def test_dataset_split_feature_names_preservation() raises:
    var X = Matrix[DType.float64](6, 2, 1.0)
    var y: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    var fnames: List[String] = ["age", "income"]
    var tnames: List[String] = ["risk"]
    var ds = Dataset(X^, y^, fnames^, tnames^)

    var split = ds.split_with_ratio(0.5, shuffle=False)
    assert_equal(len(split.train.feature_names), 2)
    assert_equal(split.train.feature_names[0], "age")
    assert_equal(split.train.feature_names[1], "income")
    assert_equal(len(split.test.feature_names), 2)
    assert_equal(len(split.train.target_names), 1)
    assert_equal(split.train.target_names[0], "risk")


def test_dataset_split_deterministic_seed() raises:
    var X = Matrix[DType.float64](10, 2, 0)
    var y = List[Scalar[DType.float64]](capacity=10)
    for i in range(10):
        X[i, 0] = Float64(i)
        y.append(Float64(i))
    var ds = Dataset(X^, y^)

    var split1 = ds.split_with_ratio(0.3, shuffle=True, seed=123)
    var split2 = ds.split_with_ratio(0.3, shuffle=True, seed=123)

    assert_equal(split1.train.n_samples(), split2.train.n_samples())
    for i in range(split1.train.n_samples()):
        assert_equal(split1.train.targets[i], split2.train.targets[i])


def test_dataset_split_different_seed() raises:
    var X = Matrix[DType.float64](20, 2, 0)
    var y = List[Scalar[DType.float64]](capacity=20)
    for i in range(20):
        X[i, 0] = Float64(i)
        y.append(Float64(i))
    var ds = Dataset(X^, y^)

    var split1 = ds.split_with_ratio(0.5, shuffle=True, seed=1)
    var split2 = ds.split_with_ratio(0.5, shuffle=True, seed=2)

    var has_diff = False
    for i in range(split1.train.n_samples()):
        if split1.train.targets[i] != split2.train.targets[i]:
            has_diff = True
            break
    assert_true(has_diff)


def test_dataset_split_partition_coverage_and_disjointness() raises:
    var N = 30
    var X = Matrix[DType.float64](N, 1, 0)
    var y = List[Scalar[DType.float64]](capacity=N)
    for i in range(N):
        X[i, 0] = Float64(i)
        y.append(Float64(i))
    var ds = Dataset(X^, y^)

    var split = ds.split_with_ratio(0.3, shuffle=True, seed=99)
    assert_equal(split.train.n_samples() + split.test.n_samples(), N)

    var seen = List[Bool](capacity=N)
    for _ in range(N):
        seen.append(False)

    for i in range(split.train.n_samples()):
        var val = Int(split.train.targets[i])
        assert_true(not seen[val])
        seen[val] = True

    for i in range(split.test.n_samples()):
        var val = Int(split.test.targets[i])
        assert_true(not seen[val])
        seen[val] = True

    for i in range(N):
        assert_true(seen[i])


def test_dataset_split_50_50() raises:
    var X = Matrix[DType.float64](10, 2, 1.0)
    var y: List[Scalar[DType.float64]] = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
    ]
    var ds = Dataset(X^, y^)

    var split = ds.split_with_ratio(0.5, shuffle=False)
    assert_equal(split.train.n_samples(), 5)
    assert_equal(split.test.n_samples(), 5)


def test_dataset_high_dimensional() raises:
    var N = 40
    var D = 15
    var X = Matrix[DType.float64](N, D, 2.0)
    var y = List[Scalar[DType.float64]](capacity=N)
    for i in range(N):
        y.append(Float64(i))
    var ds = Dataset(X^, y^)

    assert_equal(ds.n_samples(), N)
    assert_equal(ds.n_features(), D)

    var split = ds.split_with_ratio(0.25)
    assert_equal(split.train.n_features(), D)
    assert_equal(split.test.n_features(), D)


def test_dataset_single_feature_column() raises:
    var X = Matrix[DType.float64](5, 1, 3.14)
    var y: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0, 4.0, 5.0]
    var ds = Dataset(X^, y^)
    assert_equal(ds.n_features(), 1)
    assert_equal(ds.n_samples(), 5)


def test_dataset_large_sample_count() raises:
    var N = 500
    var X = Matrix[DType.float64](N, 2, 0)
    var y = List[Scalar[DType.float64]](capacity=N)
    for i in range(N):
        y.append(Float64(i))
    var ds = Dataset(X^, y^)

    var split = ds.split_with_ratio(0.2, shuffle=True, seed=42)
    assert_equal(split.test.n_samples(), 100)
    assert_equal(split.train.n_samples(), 400)


def test_dataset_empty_names_defaults() raises:
    var X = Matrix[DType.float64](2, 2, 1.0)
    var y: List[Scalar[DType.float64]] = [1.0, 2.0]
    var ds = Dataset(X^, y^)
    assert_equal(len(ds.feature_names), 0)
    assert_equal(len(ds.target_names), 0)


def test_dataset_train_test_split_shuffled() raises:
    var X = Matrix[DType.float64](10, 2, 0)
    var y = List[Scalar[DType.float64]](capacity=10)
    for i in range(10):
        X[i, 0] = Float64(i)
        y.append(Float64(i))

    var split = train_test_split(X, y, test_size=0.4, shuffle=True, seed=42)
    assert_equal(split.train.n_samples(), 6)
    assert_equal(split.test.n_samples(), 4)


def test_dataset_zero_variance_target() raises:
    var X = Matrix[DType.float64](5, 2, 1.0)
    var y: List[Scalar[DType.float64]] = [42.0, 42.0, 42.0, 42.0, 42.0]
    var ds = Dataset(X^, y^)
    assert_equal(ds.targets[0], 42.0)
    assert_equal(ds.targets[4], 42.0)


def test_dataset_negative_target_values() raises:
    var X = Matrix[DType.float64](3, 1, 0)
    var y: List[Scalar[DType.float64]] = [-10.0, -20.0, -30.0]
    var ds = Dataset(X^, y^)
    assert_equal(ds.targets[0], -10.0)
    assert_equal(ds.targets[2], -30.0)


def test_dataset_split_small_dataset_bounds() raises:
    var X = Matrix[DType.float64](3, 1, 0)
    var y: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0]
    var ds = Dataset(X^, y^)
    var split = ds.split_with_ratio(0.1, shuffle=False)
    assert_equal(split.test.n_samples(), 1)
    assert_equal(split.train.n_samples(), 2)


def test_dataset_records_access() raises:
    var X = Matrix[DType.float64](2, 2, 0)
    X[0, 0] = 11.0
    X[1, 1] = 22.0
    var y: List[Scalar[DType.float64]] = [1.0, 2.0]
    var ds = Dataset(X^, y^)
    assert_equal(ds.records[0, 0], 11.0)
    assert_equal(ds.records[1, 1], 22.0)


def test_dataset_feature_names_mismatch_error() raises:
    var X = Matrix[DType.float64](4, 2, 1.0)
    var y: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0, 4.0]
    var bad_feat_names: List[String] = ["f1", "f2", "f3"]
    var target_names: List[String] = ["t1"]

    with assert_raises():
        _ = Dataset(X^, y^, bad_feat_names^, target_names^)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

