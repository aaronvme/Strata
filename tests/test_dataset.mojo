from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_raises,
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
