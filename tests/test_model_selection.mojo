from std.testing import TestSuite, assert_equal, assert_true
from strata import Matrix, Dataset, train_test_split


def test_dataset_split_with_ratio() raises:
    var rows = 100
    var cols = 4
    var X = Matrix[DType.float64].ones(rows, cols)
    var y = List[Scalar[DType.float64]](capacity=rows)
    for i in range(rows):
        y.append(Float64(i))

    var ds = Dataset(X^, y^)
    assert_equal(ds.n_samples(), 100)
    assert_equal(ds.n_features(), 4)

    var split = ds.split_with_ratio(0.25, shuffle=True, seed=42)

    assert_equal(split.train.n_samples(), 75)
    assert_equal(split.train.n_features(), 4)
    assert_equal(split.test.n_samples(), 25)
    assert_equal(split.test.n_features(), 4)
    assert_equal(len(split.train.targets), 75)
    assert_equal(len(split.test.targets), 25)


def test_train_test_split_convenience() raises:
    var X = Matrix[DType.float64](10, 2, 0)
    var y = List[Scalar[DType.float64]](capacity=10)
    for i in range(10):
        X[i, 0] = Float64(i)
        X[i, 1] = Float64(i * 10)
        y.append(Float64(i))

    var split1 = train_test_split(X, y, test_size=0.3, shuffle=True, seed=123)
    var split2 = train_test_split(X, y, test_size=0.3, shuffle=True, seed=123)

    assert_equal(split1.train.targets[0], split2.train.targets[0])
    assert_equal(split1.train.targets[1], split2.train.targets[1])
    assert_equal(split1.test.targets[0], split2.test.targets[0])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
