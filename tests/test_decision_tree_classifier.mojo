from std.math import nan, inf, sqrt, log2
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_false,
    assert_almost_equal,
)

from strata.core.matrix import Matrix
from strata.core.dataset import Dataset
from strata.base.estimator import (
    fit as fit_ds,
    predict as predict_ds,
    predict_proba as predict_proba_ds,
)
from strata.base.pipeline import PipelineClassifier
from strata.preprocessing.scaler import StandardScaler
from strata.tree.classifier import DecisionTreeClassifier
from strata.exceptions.errors import (
    NotFittedError,
    DimensionMismatchError,
    InvalidParameterError,
)


def _separable_2d_dataset() -> (
    Tuple[Matrix[DType.float64], List[Scalar[DType.int32]]]
):
    var X = Matrix[DType.float64](12, 2, 0)
    var y = List[Scalar[DType.int32]](capacity=12)
    # Class 0: x0 < 0
    for i in range(6):
        X[i, 0] = -5.0 + Float64(i) * 0.5
        X[i, 1] = Float64(i % 2)
        y.append(Int32(0))
    # Class 1: x0 > 0
    for i in range(6, 12):
        X[i, 0] = 5.0 + Float64(i - 6) * 0.5
        X[i, 1] = Float64(i % 2)
        y.append(Int32(1))
    return (X^, y^)


def test_decision_tree_classifier_binary_simple_1d_split() raises:
    var X = Matrix[DType.float64](6, 1, 0)
    var y = List[Scalar[DType.int32]](capacity=6)
    for i in range(6):
        X[i, 0] = Float64(i)
        y.append(Int32(0 if i < 3 else 1))

    var clf = DecisionTreeClassifier[DType.float64](max_depth=1)
    clf.fit(X, y)

    assert_true(clf.is_fitted)
    assert_equal(clf.n_classes_, 2)
    assert_equal(clf.n_features_in_, 1)

    var preds = clf.predict(X)
    assert_equal(len(preds), 6)
    for i in range(6):
        assert_equal(preds[i], Int(y[i]))


def test_decision_tree_classifier_binary_separable_2d() raises:
    var data = _separable_2d_dataset()
    var clf = DecisionTreeClassifier[DType.float64]()
    clf.fit(data[0], data[1])

    var preds = clf.predict(data[0])
    assert_equal(len(preds), 12)
    for i in range(12):
        assert_equal(preds[i], Int(data[1][i]))


def test_decision_tree_classifier_multiclass_three_classes() raises:
    var X = Matrix[DType.float64](15, 2, 0)
    var y = List[Scalar[DType.int32]](capacity=15)
    for i in range(5):
        X[i, 0] = -10.0 + Float64(i)
        X[i, 1] = 0.0
        y.append(Int32(0))
    for i in range(5, 10):
        X[i, 0] = 0.0 + Float64(i - 5)
        X[i, 1] = 10.0
        y.append(Int32(1))
    for i in range(10, 15):
        X[i, 0] = 10.0 + Float64(i - 10)
        X[i, 1] = -10.0
        y.append(Int32(2))

    var clf = DecisionTreeClassifier[DType.float64](max_depth=3)
    clf.fit(X, y)

    assert_equal(clf.n_classes_, 3)
    var preds = clf.predict(X)
    assert_equal(len(preds), 15)
    for i in range(15):
        assert_equal(preds[i], Int(y[i]))


def test_decision_tree_classifier_multiclass_discontinuous_labels() raises:
    var X = Matrix[DType.float64](9, 1, 0)
    var y = List[Scalar[DType.int32]](capacity=9)
    # Class -7
    for i in range(3):
        X[i, 0] = Float64(i)
        y.append(Int32(-7))
    # Class 42
    for i in range(3, 6):
        X[i, 0] = Float64(i + 10)
        y.append(Int32(42))
    # Class 100
    for i in range(6, 9):
        X[i, 0] = Float64(i + 20)
        y.append(Int32(100))

    var clf = DecisionTreeClassifier[DType.float64]()
    clf.fit(X, y)

    assert_equal(clf.n_classes_, 3)
    var preds = clf.predict(X)
    assert_equal(len(preds), 9)
    for i in range(9):
        assert_equal(preds[i], Int(y[i]))


def test_decision_tree_classifier_gini_criterion() raises:
    var data = _separable_2d_dataset()
    var clf = DecisionTreeClassifier[DType.float64](criterion="gini")
    clf.fit(data[0], data[1])

    var preds = clf.predict(data[0])
    for i in range(12):
        assert_equal(preds[i], Int(data[1][i]))


def test_decision_tree_classifier_entropy_criterion() raises:
    var data = _separable_2d_dataset()
    var clf = DecisionTreeClassifier[DType.float64](criterion="entropy")
    clf.fit(data[0], data[1])

    var preds = clf.predict(data[0])
    for i in range(12):
        assert_equal(preds[i], Int(data[1][i]))


def test_decision_tree_classifier_max_depth_stump() raises:
    var X = Matrix[DType.float64](8, 1, 0)
    var y = List[Scalar[DType.int32]](capacity=8)
    for i in range(8):
        X[i, 0] = Float64(i)
        y.append(Int32(0 if i < 4 else 1))

    var clf = DecisionTreeClassifier[DType.float64](max_depth=1)
    clf.fit(X, y)

    assert_true(clf.is_fitted)
    assert_equal(clf.tree_.max_depth(), 1)
    assert_equal(clf.tree_.node_count(), 3)  # 1 root + 2 leaves


def test_decision_tree_classifier_max_depth_limits_growth() raises:
    # Construct a dataset requiring multiple splits
    var X = Matrix[DType.float64](8, 1, 0)
    var y = List[Scalar[DType.int32]](capacity=8)
    X[0, 0] = 0.0
    y.append(0)
    X[1, 0] = 1.0
    y.append(1)
    X[2, 0] = 2.0
    y.append(0)
    X[3, 0] = 3.0
    y.append(1)
    X[4, 0] = 4.0
    y.append(0)
    X[5, 0] = 5.0
    y.append(1)
    X[6, 0] = 6.0
    y.append(0)
    X[7, 0] = 7.0
    y.append(1)

    var clf = DecisionTreeClassifier[DType.float64](max_depth=2)
    clf.fit(X, y)

    assert_true(clf.tree_.max_depth() <= 2)


def test_decision_tree_classifier_min_samples_split() raises:
    var X = Matrix[DType.float64](6, 1, 0)
    var y = List[Scalar[DType.int32]](capacity=6)
    for i in range(6):
        X[i, 0] = Float64(i)
        y.append(Int32(0 if i < 3 else 1))

    # min_samples_split=10 > 6 => no split occurs
    var clf = DecisionTreeClassifier[DType.float64](min_samples_split=10)
    clf.fit(X, y)

    assert_equal(clf.tree_.node_count(), 1)  # Only root node


def test_decision_tree_classifier_min_samples_leaf() raises:
    var X = Matrix[DType.float64](10, 1, 0)
    var y = List[Scalar[DType.int32]](capacity=10)
    for i in range(10):
        X[i, 0] = Float64(i)
        y.append(Int32(0 if i < 2 else 1))

    # Splitting at index 2 would create a left child with 2 samples
    # With min_samples_leaf=4, that split is rejected
    var clf = DecisionTreeClassifier[DType.float64](min_samples_leaf=4)
    clf.fit(X, y)

    # Every leaf in the tree must have at least 4 samples
    for n in range(clf.tree_.node_count()):
        if clf.tree_.nodes[n].is_leaf:
            assert_true(clf.tree_.nodes[n].n_node_samples >= 4)


def test_decision_tree_classifier_min_impurity_decrease() raises:
    var data = _separable_2d_dataset()
    # Extremely high threshold prevents any split
    var clf = DecisionTreeClassifier[DType.float64](min_impurity_decrease=100.0)
    clf.fit(data[0], data[1])

    assert_equal(clf.tree_.node_count(), 1)


def test_decision_tree_classifier_max_features_sqrt() raises:
    var X = Matrix[DType.float64](20, 16, 0)
    var y = List[Scalar[DType.int32]](capacity=20)
    for i in range(20):
        for j in range(16):
            X[i, j] = Float64(i + j)
        y.append(Int32(0 if i < 10 else 1))

    var clf = DecisionTreeClassifier[DType.float64](
        max_features="sqrt", random_state=42
    )
    clf.fit(X, y)
    assert_true(clf.is_fitted)

    var preds = clf.predict(X)
    assert_equal(len(preds), 20)


def test_decision_tree_classifier_max_features_log2() raises:
    var X = Matrix[DType.float64](20, 16, 0)
    var y = List[Scalar[DType.int32]](capacity=20)
    for i in range(20):
        for j in range(16):
            X[i, j] = Float64(i + j)
        y.append(Int32(0 if i < 10 else 1))

    var clf = DecisionTreeClassifier[DType.float64](
        max_features="log2", random_state=42
    )
    clf.fit(X, y)
    assert_true(clf.is_fitted)


def test_decision_tree_classifier_max_features_explicit_int_and_float() raises:
    var X = Matrix[DType.float64](20, 10, 0)
    var y = List[Scalar[DType.int32]](capacity=20)
    for i in range(20):
        X[i, 0] = Float64(i)
        y.append(Int32(0 if i < 10 else 1))

    var clf_int = DecisionTreeClassifier[DType.float64](
        max_features_count=4, random_state=42
    )
    clf_int.fit(X, y)
    assert_true(clf_int.is_fitted)

    var clf_float = DecisionTreeClassifier[DType.float64](
        max_features_ratio=0.5, random_state=42
    )
    clf_float.fit(X, y)
    assert_true(clf_float.is_fitted)


def test_decision_tree_classifier_splitter_random() raises:
    var data = _separable_2d_dataset()
    var clf = DecisionTreeClassifier[DType.float64](
        splitter="random", random_state=123
    )
    clf.fit(data[0], data[1])
    assert_true(clf.is_fitted)

    var preds = clf.predict(data[0])
    assert_equal(len(preds), 12)


def test_decision_tree_classifier_predict_proba_sums_to_one() raises:
    var data = _separable_2d_dataset()
    var clf = DecisionTreeClassifier[DType.float64]()
    clf.fit(data[0], data[1])

    var proba = clf.predict_proba(data[0])
    assert_equal(proba.rows, 12)
    assert_equal(proba.cols, 2)

    for i in range(12):
        var p0 = proba[i, 0]
        var p1 = proba[i, 1]
        assert_true(p0 >= 0.0 and p0 <= 1.0)
        assert_true(p1 >= 0.0 and p1 <= 1.0)
        assert_almost_equal(p0 + p1, 1.0, atol=1e-6)


def test_decision_tree_classifier_predict_proba_multiclass() raises:
    var X = Matrix[DType.float64](9, 1, 0)
    var y = List[Scalar[DType.int32]](capacity=9)
    for i in range(3):
        X[i, 0] = Float64(i)
        y.append(Int32(0))
    for i in range(3, 6):
        X[i, 0] = Float64(i + 10)
        y.append(Int32(1))
    for i in range(6, 9):
        X[i, 0] = Float64(i + 20)
        y.append(Int32(2))

    var clf = DecisionTreeClassifier[DType.float64]()
    clf.fit(X, y)

    var proba = clf.predict_proba(X)
    assert_equal(proba.rows, 9)
    assert_equal(proba.cols, 3)

    for i in range(9):
        var sum_p: Float64 = 0.0
        for c in range(3):
            var p = proba[i, c]
            assert_true(p >= 0.0 and p <= 1.0)
            sum_p += p
        assert_almost_equal(sum_p, 1.0, atol=1e-6)


def test_decision_tree_classifier_single_class_dataset() raises:
    var X = Matrix[DType.float64](6, 2, 1.0)
    var y = List[Scalar[DType.int32]](capacity=6)
    for _ in range(6):
        y.append(Int32(42))

    var clf = DecisionTreeClassifier[DType.float64]()
    clf.fit(X, y)

    assert_true(clf.is_fitted)
    assert_equal(clf.n_classes_, 1)
    assert_equal(clf.tree_.node_count(), 1)  # Single pure leaf

    var preds = clf.predict(X)
    for i in range(6):
        assert_equal(preds[i], 42)


def test_decision_tree_classifier_all_identical_samples() raises:
    var X = Matrix[DType.float64](8, 2, 3.14)
    var y = List[Scalar[DType.int32]](capacity=8)
    for _ in range(8):
        y.append(Int32(1))

    var clf = DecisionTreeClassifier[DType.float64]()
    clf.fit(X, y)

    assert_true(clf.is_fitted)
    assert_equal(clf.tree_.node_count(), 1)


def test_decision_tree_classifier_constant_features_mixed_labels() raises:
    # Identical features but conflicting labels: leaf should predict majority class
    var X = Matrix[DType.float64](5, 2, 1.0)
    var y = List[Scalar[DType.int32]](capacity=5)
    y.append(Int32(0))
    y.append(Int32(1))
    y.append(Int32(1))
    y.append(Int32(1))
    y.append(Int32(0))

    var clf = DecisionTreeClassifier[DType.float64]()
    clf.fit(X, y)

    assert_true(clf.is_fitted)
    assert_equal(clf.tree_.node_count(), 1)  # Cannot split identical features

    var preds = clf.predict(X)
    for i in range(5):
        assert_equal(preds[i], 1)  # Majority class is 1


def test_decision_tree_classifier_xor_dataset() raises:
    # 2D XOR problem
    var X = Matrix[DType.float64](8, 2, 0)
    var y = List[Scalar[DType.int32]](capacity=8)

    # (-1, -1) -> 0
    X[0, 0] = -1.0
    X[0, 1] = -1.0
    y.append(0)
    X[1, 0] = -2.0
    X[1, 1] = -2.0
    y.append(0)

    # (-1, 1) -> 1
    X[2, 0] = -1.0
    X[2, 1] = 1.0
    y.append(1)
    X[3, 0] = -2.0
    X[3, 1] = 2.0
    y.append(1)

    # (1, -1) -> 1
    X[4, 0] = 1.0
    X[4, 1] = -1.0
    y.append(1)
    X[5, 0] = 2.0
    X[5, 1] = -2.0
    y.append(1)

    # (1, 1) -> 0
    X[6, 0] = 1.0
    X[6, 1] = 1.0
    y.append(0)
    X[7, 0] = 2.0
    X[7, 1] = 2.0
    y.append(0)

    var clf = DecisionTreeClassifier[DType.float64](max_depth=4)
    clf.fit(X, y)

    var preds = clf.predict(X)
    for i in range(8):
        assert_equal(preds[i], Int(y[i]))


def test_decision_tree_classifier_high_dimensional_fat_matrix() raises:
    var X = Matrix[DType.float64](20, 40, 0)
    var y = List[Scalar[DType.int32]](capacity=20)
    for i in range(20):
        for j in range(40):
            X[i, j] = Float64((i + 1) * (j + 1)) * 0.1
        y.append(Int32(0 if i < 10 else 1))

    var clf = DecisionTreeClassifier[DType.float64](max_depth=3)
    clf.fit(X, y)

    var preds = clf.predict(X)
    assert_equal(len(preds), 20)


def test_decision_tree_classifier_copy_constructor_isolation() raises:
    var data = _separable_2d_dataset()
    var clf1 = DecisionTreeClassifier[DType.float64](max_depth=2)
    clf1.fit(data[0], data[1])

    var clf2 = clf1.copy()
    assert_true(clf2.is_fitted)
    assert_equal(clf2.tree_.node_count(), clf1.tree_.node_count())

    var preds1 = clf1.predict(data[0])
    var preds2 = clf2.predict(data[0])
    for i in range(12):
        assert_equal(preds1[i], preds2[i])


def test_decision_tree_classifier_refit_resets_state() raises:
    var data = _separable_2d_dataset()
    var clf = DecisionTreeClassifier[DType.float64]()
    clf.fit(data[0], data[1])
    assert_equal(clf.n_classes_, 2)

    # Refit with 3 classes
    var X3 = Matrix[DType.float64](6, 1, 0)
    var y3 = List[Scalar[DType.int32]](capacity=6)
    for i in range(6):
        X3[i, 0] = Float64(i)
        y3.append(Int32(i % 3))

    clf.fit(X3, y3)
    assert_equal(clf.n_classes_, 3)
    assert_equal(clf.n_features_in_, 1)


def test_decision_tree_classifier_unfitted_predict_raises() raises:
    var clf = DecisionTreeClassifier[DType.float64]()
    var X = Matrix[DType.float64](4, 2, 1.0)

    var caught_pred = False
    try:
        var _ = clf.predict(X)
    except:
        caught_pred = True
    assert_true(caught_pred)

    var caught_proba = False
    try:
        var _ = clf.predict_proba(X)
    except:
        caught_proba = True
    assert_true(caught_proba)


def test_decision_tree_classifier_dimension_mismatch_raises() raises:
    var data = _separable_2d_dataset()
    var clf = DecisionTreeClassifier[DType.float64]()
    clf.fit(data[0], data[1])

    # 3 features instead of 2
    var X_bad = Matrix[DType.float64](4, 3, 1.0)
    var caught = False
    try:
        var _ = clf.predict(X_bad)
    except:
        caught = True
    assert_true(caught)


def test_decision_tree_classifier_invalid_parameters_raises() raises:
    var caught_crit = False
    try:
        var _ = DecisionTreeClassifier[DType.float64](
            criterion="invalid_criterion"
        )
    except:
        caught_crit = True
    assert_true(caught_crit)

    var caught_split = False
    try:
        var _ = DecisionTreeClassifier[DType.float64](min_samples_split=1)
    except:
        caught_split = True
    assert_true(caught_split)

    var caught_leaf = False
    try:
        var _ = DecisionTreeClassifier[DType.float64](min_samples_leaf=0)
    except:
        caught_leaf = True
    assert_true(caught_leaf)

    var caught_depth = False
    try:
        var _ = DecisionTreeClassifier[DType.float64](max_depth=0)
    except:
        caught_depth = True
    assert_true(caught_depth)


def test_decision_tree_classifier_nan_and_inf_rejected() raises:
    var X = Matrix[DType.float64](4, 2, 1.0)
    X[0, 0] = nan
    var y: List[Scalar[DType.int32]] = [0, 1, 0, 1]

    var clf = DecisionTreeClassifier[DType.float64]()
    var caught = False
    try:
        clf.fit(X, y)
    except:
        caught = True
    assert_true(caught)


def test_decision_tree_classifier_float32_native() raises:
    var X = Matrix[DType.float32](8, 2, 0)
    var y = List[Scalar[DType.int32]](capacity=8)
    for i in range(8):
        X[i, 0] = Float32(i)
        X[i, 1] = Float32(i * 2)
        y.append(Int32(0 if i < 4 else 1))

    var clf = DecisionTreeClassifier[DType.float32]()
    clf.fit(X, y)
    assert_true(clf.is_fitted)

    var preds = clf.predict(X)
    assert_equal(len(preds), 8)


def test_decision_tree_classifier_dataset_integration() raises:
    var data = _separable_2d_dataset()
    var ds = Dataset[DType.float64, DType.int32](data[0], data[1])

    var clf = DecisionTreeClassifier[DType.float64]()
    fit_ds(clf, ds)
    assert_true(clf.is_fitted)

    var preds = predict_ds(clf, ds)
    assert_equal(len(preds), 12)

    var proba = predict_proba_ds(clf, ds)
    assert_equal(proba.rows, 12)


def test_decision_tree_classifier_pipeline_integration() raises:
    var data = _separable_2d_dataset()
    var scaler = StandardScaler[DType.float64]()
    var tree = DecisionTreeClassifier[DType.float64](max_depth=3)

    var pipe = PipelineClassifier[
        StandardScaler[DType.float64],
        DecisionTreeClassifier[DType.float64],
        DType.int32,
    ](scaler^, tree^)

    pipe.fit(data[0], data[1])
    assert_true(pipe.is_fitted)

    var preds = pipe.predict(data[0])
    assert_equal(len(preds), 12)
    for i in range(12):
        assert_equal(preds[i], Int(data[1][i]))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
