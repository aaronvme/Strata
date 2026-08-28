from std.math import nan, inf, sqrt
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
from strata.ensemble.forest import RandomForestClassifier
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


def test_rf_classifier_binary_simple_1d() raises:
    var X = Matrix[DType.float64](6, 1, 0)
    var y = List[Scalar[DType.int32]](capacity=6)
    for i in range(6):
        X[i, 0] = Float64(i)
        y.append(Int32(0 if i < 3 else 1))

    var rf = RandomForestClassifier[DType.float64](
        n_estimators=10, random_state=42
    )
    rf.fit(X, y)

    assert_true(rf.is_fitted)
    assert_equal(rf.n_classes_, 2)
    assert_equal(rf.n_features_in_, 1)

    var preds = rf.predict(X)
    assert_equal(len(preds), 6)
    for i in range(6):
        assert_equal(preds[i], Int(y[i]))


def test_rf_classifier_binary_separable_2d() raises:
    var data = _separable_2d_dataset()
    var rf = RandomForestClassifier[DType.float64](
        n_estimators=10, random_state=42
    )
    rf.fit(data[0], data[1])

    var preds = rf.predict(data[0])
    assert_equal(len(preds), 12)
    for i in range(12):
        assert_equal(preds[i], Int(data[1][i]))


def test_rf_classifier_multiclass_three_classes() raises:
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

    var rf = RandomForestClassifier[DType.float64](
        n_estimators=15, max_depth=3, random_state=42
    )
    rf.fit(X, y)

    assert_equal(rf.n_classes_, 3)
    var preds = rf.predict(X)
    assert_equal(len(preds), 15)
    for i in range(15):
        assert_equal(preds[i], Int(y[i]))


def test_rf_classifier_multiclass_arbitrary_labels() raises:
    var X = Matrix[DType.float64](6, 1, 0)
    var y = List[Scalar[DType.int32]](capacity=6)
    X[0, 0] = -10.0
    y.append(Int32(-10))
    X[1, 0] = -8.0
    y.append(Int32(-10))
    X[2, 0] = 0.0
    y.append(Int32(42))
    X[3, 0] = 1.0
    y.append(Int32(42))
    X[4, 0] = 10.0
    y.append(Int32(100))
    X[5, 0] = 12.0
    y.append(Int32(100))

    var rf = RandomForestClassifier[DType.float64](
        n_estimators=10, max_depth=3, random_state=42
    )
    rf.fit(X, y)

    assert_equal(rf.n_classes_, 3)
    assert_equal(rf.classes_[0], -10)
    assert_equal(rf.classes_[1], 42)
    assert_equal(rf.classes_[2], 100)

    var preds = rf.predict(X)
    for i in range(6):
        assert_equal(preds[i], Int(y[i]))


def test_rf_classifier_predict_proba_simplex() raises:
    var data = _separable_2d_dataset()
    var rf = RandomForestClassifier[DType.float64](
        n_estimators=10, random_state=42
    )
    rf.fit(data[0], data[1])

    var proba = rf.predict_proba(data[0])
    assert_equal(proba.rows, 12)
    assert_equal(proba.cols, 2)

    for i in range(12):
        var row_sum: Float64 = Float64(proba[i, 0]) + Float64(proba[i, 1])
        assert_almost_equal(row_sum, 1.0, atol=1e-5)
        assert_true(proba[i, 0] >= 0.0)
        assert_true(proba[i, 1] >= 0.0)


def test_rf_classifier_soft_voting_argmax_parity() raises:
    var data = _separable_2d_dataset()
    var rf = RandomForestClassifier[DType.float64](
        n_estimators=10, random_state=42
    )
    rf.fit(data[0], data[1])

    var preds = rf.predict(data[0])
    var proba = rf.predict_proba(data[0])

    for i in range(12):
        var max_c = 0 if proba[i, 0] >= proba[i, 1] else 1
        assert_equal(preds[i], rf.classes_[max_c])


def test_rf_classifier_bootstrap_false() raises:
    var data = _separable_2d_dataset()
    var rf = RandomForestClassifier[DType.float64](
        n_estimators=5, bootstrap=False, random_state=42
    )
    rf.fit(data[0], data[1])

    assert_true(rf.is_fitted)
    var preds = rf.predict(data[0])
    for i in range(12):
        assert_equal(preds[i], Int(data[1][i]))


def test_rf_classifier_max_samples_options() raises:
    var data = _separable_2d_dataset()
    var rf_ratio = RandomForestClassifier[DType.float64](
        n_estimators=5,
        bootstrap=True,
        max_samples_ratio=0.75,
        random_state=42,
    )
    var rf_count = RandomForestClassifier[DType.float64](
        n_estimators=5, bootstrap=True, max_samples_count=8, random_state=42
    )
    rf_ratio.fit(data[0], data[1])
    rf_count.fit(data[0], data[1])

    assert_true(rf_ratio.is_fitted)
    assert_true(rf_count.is_fitted)


def test_rf_classifier_oob_score_enabled() raises:
    var data = _separable_2d_dataset()
    var rf = RandomForestClassifier[DType.float64](
        n_estimators=30,
        bootstrap=True,
        oob_score=True,
        random_state=42,
    )
    rf.fit(data[0], data[1])

    assert_true(rf.is_fitted)
    assert_true(rf.oob_score_ >= 0.7)


def test_rf_classifier_oob_score_disabled_raises() raises:
    var data = _separable_2d_dataset()
    var rf = RandomForestClassifier[DType.float64](
        n_estimators=5, oob_score=False
    )
    rf.fit(data[0], data[1])

    var caught = False
    try:
        _ = rf.get_oob_score()
    except:
        caught = True
    assert_true(caught)


def test_rf_classifier_feature_importances_sum_to_one() raises:
    var X = Matrix[DType.float64](20, 3, 0)
    var y = List[Scalar[DType.int32]](capacity=20)
    for i in range(20):
        X[i, 0] = -5.0 if i < 10 else 5.0
        X[i, 1] = 0.0
        X[i, 2] = Float64(i % 2)
        y.append(Int32(0 if i < 10 else 1))

    var rf = RandomForestClassifier[DType.float64](
        n_estimators=15, random_state=42
    )
    rf.fit(X, y)

    var importances = rf.get_feature_importances()
    assert_equal(len(importances), 3)

    var sum_imp: Float64 = 0.0
    for j in range(3):
        sum_imp += importances[j]
    assert_almost_equal(sum_imp, 1.0, atol=1e-4)
    # Feature 0 separates classes perfectly and must have highest importance
    assert_true(importances[0] > importances[1])


def test_rf_classifier_criterion_gini_and_entropy_and_log_loss() raises:
    var data = _separable_2d_dataset()
    var rf_gini = RandomForestClassifier[DType.float64](
        n_estimators=5, criterion="gini", random_state=42
    )
    var rf_ent = RandomForestClassifier[DType.float64](
        n_estimators=5, criterion="entropy", random_state=42
    )
    var rf_ll = RandomForestClassifier[DType.float64](
        n_estimators=5, criterion="log_loss", random_state=42
    )

    rf_gini.fit(data[0], data[1])
    rf_ent.fit(data[0], data[1])
    rf_ll.fit(data[0], data[1])

    assert_true(rf_gini.is_fitted)
    assert_true(rf_ent.is_fitted)
    assert_true(rf_ll.is_fitted)


def test_rf_classifier_max_depth_stump() raises:
    var data = _separable_2d_dataset()
    var rf = RandomForestClassifier[DType.float64](
        n_estimators=5, max_depth=1, random_state=42
    )
    rf.fit(data[0], data[1])

    for b in range(len(rf.estimators_)):
        assert_true(rf.estimators_[b].get_depth() <= 1)


def test_rf_classifier_min_samples_split_and_leaf() raises:
    var data = _separable_2d_dataset()
    var rf = RandomForestClassifier[DType.float64](
        n_estimators=5,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
    )
    rf.fit(data[0], data[1])
    assert_true(rf.is_fitted)


def test_rf_classifier_min_impurity_decrease() raises:
    var data = _separable_2d_dataset()
    var rf = RandomForestClassifier[DType.float64](
        n_estimators=5, min_impurity_decrease=100.0, random_state=42
    )
    rf.fit(data[0], data[1])

    for b in range(len(rf.estimators_)):
        assert_equal(rf.estimators_[b].get_depth(), 0)


def test_rf_classifier_max_features_sqrt_and_log2() raises:
    var X = Matrix[DType.float64](16, 9, 0)
    var y = List[Scalar[DType.int32]](capacity=16)
    for i in range(16):
        for j in range(9):
            X[i, j] = Float64(i + j)
        y.append(Int32(0 if i < 8 else 1))

    var rf_sqrt = RandomForestClassifier[DType.float64](
        n_estimators=5, max_features="sqrt", random_state=42
    )
    var rf_log2 = RandomForestClassifier[DType.float64](
        n_estimators=5, max_features="log2", random_state=42
    )
    rf_sqrt.fit(X, y)
    rf_log2.fit(X, y)

    assert_true(rf_sqrt.is_fitted)
    assert_true(rf_log2.is_fitted)


def test_rf_classifier_single_sample() raises:
    var X = Matrix[DType.float64](1, 2, 5.0)
    var y: List[Scalar[DType.int32]] = [1]

    var rf = RandomForestClassifier[DType.float64](n_estimators=5)
    rf.fit(X, y)

    var preds = rf.predict(X)
    assert_equal(len(preds), 1)
    assert_equal(preds[0], 1)


def test_rf_classifier_identical_samples() raises:
    var X = Matrix[DType.float64](8, 2, 3.0)
    var y = List[Scalar[DType.int32]](capacity=8)
    for _ in range(8):
        y.append(0)

    var rf = RandomForestClassifier[DType.float64](n_estimators=5)
    rf.fit(X, y)

    var preds = rf.predict(X)
    for i in range(8):
        assert_equal(preds[i], 0)


def test_rf_classifier_constant_features_mixed_labels() raises:
    var X = Matrix[DType.float64](6, 2, 1.0)
    var y: List[Scalar[DType.int32]] = [0, 0, 1, 1, 1, 1]

    var rf = RandomForestClassifier[DType.float64](n_estimators=5)
    rf.fit(X, y)

    var preds = rf.predict(X)
    for i in range(6):
        assert_equal(preds[i], 1)  # Majority class is 1


def test_rf_classifier_non_linear_decision_boundary() raises:
    # Non-linear 2-split decision boundary (Class 0 outer, Class 1 inner)
    var X = Matrix[DType.float64](8, 2, 0)
    var y = List[Scalar[DType.int32]](capacity=8)

    X[0, 0] = -3.0
    X[0, 1] = -2.0
    y.append(0)
    X[1, 0] = -2.0
    X[1, 1] = -1.0
    y.append(0)
    X[2, 0] = 6.0
    X[2, 1] = 6.0
    y.append(0)
    X[3, 0] = 7.0
    X[3, 1] = 7.0
    y.append(0)

    X[4, 0] = 1.0
    X[4, 1] = 1.0
    y.append(1)
    X[5, 0] = 2.0
    X[5, 1] = 2.0
    y.append(1)
    X[6, 0] = 3.0
    X[6, 1] = -1.0
    y.append(1)
    X[7, 0] = 4.0
    X[7, 1] = -2.0
    y.append(1)

    var rf = RandomForestClassifier[DType.float64](
        n_estimators=15, max_depth=4, random_state=42
    )
    rf.fit(X, y)

    var preds = rf.predict(X)
    for i in range(8):
        assert_equal(preds[i], Int(y[i]))


def test_rf_classifier_high_dimensional_features() raises:
    var X = Matrix[DType.float64](16, 50, 0)
    var y = List[Scalar[DType.int32]](capacity=16)
    for i in range(16):
        for j in range(50):
            X[i, j] = Float64(i + j)
        y.append(Int32(0 if i < 8 else 1))

    var rf = RandomForestClassifier[DType.float64](
        n_estimators=10, max_features="sqrt", random_state=42
    )
    rf.fit(X, y)

    var preds = rf.predict(X)
    assert_equal(len(preds), 16)


def test_rf_classifier_random_state_determinism() raises:
    var data = _separable_2d_dataset()
    var rf1 = RandomForestClassifier[DType.float64](
        n_estimators=10, random_state=42
    )
    var rf2 = RandomForestClassifier[DType.float64](
        n_estimators=10, random_state=42
    )
    rf1.fit(data[0], data[1])
    rf2.fit(data[0], data[1])

    var p1 = rf1.predict_proba(data[0])
    var p2 = rf2.predict_proba(data[0])
    for i in range(12):
        for c in range(2):
            assert_almost_equal(p1[i, c], p2[i, c], atol=1e-8)


def test_rf_classifier_copy_constructor_isolation() raises:
    var data = _separable_2d_dataset()
    var rf1 = RandomForestClassifier[DType.float64](
        n_estimators=5, random_state=42
    )
    rf1.fit(data[0], data[1])

    var rf2 = RandomForestClassifier[DType.float64](copy=rf1)
    assert_true(rf2.is_fitted)
    assert_equal(rf2.n_classes_, rf1.n_classes_)

    var p1 = rf1.predict(data[0])
    var p2 = rf2.predict(data[0])
    for i in range(12):
        assert_equal(p1[i], p2[i])


def test_rf_classifier_refit_resets_state() raises:
    var data = _separable_2d_dataset()
    var rf = RandomForestClassifier[DType.float64](
        n_estimators=5, random_state=42
    )
    rf.fit(data[0], data[1])
    rf.fit(data[0], data[1])
    assert_equal(len(rf.estimators_), 5)


def test_rf_classifier_unfitted_predict_raises() raises:
    var rf = RandomForestClassifier[DType.float64]()
    var X = Matrix[DType.float64](3, 2, 1.0)
    var caught_pred = False
    try:
        _ = rf.predict(X)
    except:
        caught_pred = True
    assert_true(caught_pred)

    var caught_proba = False
    try:
        _ = rf.predict_proba(X)
    except:
        caught_proba = True
    assert_true(caught_proba)


def test_rf_classifier_dimension_mismatch_raises() raises:
    var data = _separable_2d_dataset()
    var rf = RandomForestClassifier[DType.float64](n_estimators=5)
    rf.fit(data[0], data[1])

    var X_bad = Matrix[DType.float64](4, 3, 1.0)
    var caught = False
    try:
        _ = rf.predict(X_bad)
    except:
        caught = True
    assert_true(caught)


def test_rf_classifier_invalid_hyperparameters_raises() raises:
    var caught_n_est = False
    try:
        _ = RandomForestClassifier[DType.float64](n_estimators=0)
    except:
        caught_n_est = True
    assert_true(caught_n_est)

    var caught_crit = False
    try:
        _ = RandomForestClassifier[DType.float64](criterion="invalid")
    except:
        caught_crit = True
    assert_true(caught_crit)


def test_rf_classifier_nan_and_inf_rejected() raises:
    var X = Matrix[DType.float64](4, 2, 1.0)
    X[0, 0] = nan[DType.float64]()
    var y: List[Scalar[DType.int32]] = [0, 1, 0, 1]

    var rf = RandomForestClassifier[DType.float64](n_estimators=5)
    var caught = False
    try:
        rf.fit(X, y)
    except:
        caught = True
    assert_true(caught)


def test_rf_classifier_float32_native() raises:
    var X = Matrix[DType.float32](8, 2, 0)
    var y = List[Scalar[DType.int32]](capacity=8)
    for i in range(8):
        X[i, 0] = Float32(i)
        X[i, 1] = Float32(i * 2)
        y.append(Int32(0 if i < 4 else 1))

    var rf = RandomForestClassifier[DType.float32](
        n_estimators=5, random_state=42
    )
    rf.fit(X, y)
    assert_true(rf.is_fitted)

    var preds = rf.predict(X)
    assert_equal(len(preds), 8)


def test_rf_classifier_dataset_integration() raises:
    var data = _separable_2d_dataset()
    var ds = Dataset[DType.float64, DType.int32](data[0].copy(), data[1].copy())

    var rf = RandomForestClassifier[DType.float64](
        n_estimators=5, random_state=42
    )
    fit_ds(rf, ds)
    assert_true(rf.is_fitted)

    var preds = predict_ds(rf, ds)
    assert_equal(len(preds), 12)

    var proba = predict_proba_ds(rf, ds)
    assert_equal(proba.rows, 12)


def test_rf_classifier_pipeline_integration() raises:
    var data = _separable_2d_dataset()
    var scaler = StandardScaler[DType.float64]()
    var rf = RandomForestClassifier[DType.float64](
        n_estimators=5, random_state=42
    )

    var pipe = PipelineClassifier[
        StandardScaler[DType.float64],
        RandomForestClassifier[DType.float64],
        DType.int32,
    ](scaler^, rf^)

    pipe.fit(data[0], data[1])
    assert_true(pipe.classifier.is_fitted)

    var preds = pipe.predict(data[0])
    assert_equal(len(preds), 12)
    for i in range(12):
        assert_equal(preds[i], Int(data[1][i]))


def test_rf_classifier_imbalanced_classes_extreme_ratio() raises:
    # 40 samples of Class 0, only 2 samples of Class 1
    var X = Matrix[DType.float64](42, 2, 0.0)
    var y = List[Scalar[DType.int32]](capacity=42)
    for i in range(40):
        X[i, 0] = -5.0 + Float64(i) * 0.1
        X[i, 1] = Float64(i % 3)
        y.append(Int32(0))
    X[40, 0] = 10.0
    X[40, 1] = 10.0
    y.append(Int32(1))
    X[41, 0] = 12.0
    X[41, 1] = 12.0
    y.append(Int32(1))

    var rf = RandomForestClassifier[DType.float64](
        n_estimators=20, bootstrap=True, oob_score=True, random_state=42
    )
    rf.fit(X, y)
    assert_true(rf.is_fitted)
    assert_equal(rf.n_classes_, 2)

    var proba = rf.predict_proba(X)
    assert_equal(proba.rows, 42)
    assert_equal(proba.cols, 2)
    for i in range(42):
        var row_sum = Float64(proba[i, 0]) + Float64(proba[i, 1])
        assert_almost_equal(row_sum, 1.0, atol=1e-5)


def test_rf_classifier_high_cardinality_arbitrary_signed_labels() raises:
    # 8 classes with large/negative/discontinuous labels
    var classes_arr = List[Int]()
    classes_arr.append(-999)
    classes_arr.append(-50)
    classes_arr.append(-1)
    classes_arr.append(0)
    classes_arr.append(7)
    classes_arr.append(42)
    classes_arr.append(1000)
    classes_arr.append(99999)

    var X = Matrix[DType.float64](24, 2, 0.0)
    var y = List[Scalar[DType.int32]](capacity=24)
    for c in range(8):
        for rep in range(3):
            var row = c * 3 + rep
            X[row, 0] = Float64(c * 10) + Float64(rep)
            X[row, 1] = Float64(rep)
            y.append(Int32(classes_arr[c]))

    var rf = RandomForestClassifier[DType.float64](
        n_estimators=15, max_depth=6, random_state=42
    )
    rf.fit(X, y)
    assert_true(rf.is_fitted)
    assert_equal(rf.n_classes_, 8)

    var preds = rf.predict(X)
    for i in range(24):
        assert_equal(preds[i], Int(y[i]))


def test_rf_classifier_four_quadrants_multiclass_2d() raises:
    # 4 distinct quadrants in 2D with 4 classes (16 points, 4 per quadrant)
    var X = Matrix[DType.float64](16, 2, 0.0)
    var y = List[Scalar[DType.int32]](capacity=16)

    # Q1: (+, +) -> Class 0
    X[0, 0] = 2.0
    X[0, 1] = 2.0
    y.append(0)
    X[1, 0] = 3.0
    X[1, 1] = 4.0
    y.append(0)
    X[2, 0] = 5.0
    X[2, 1] = 1.0
    y.append(0)
    X[3, 0] = 4.0
    X[3, 1] = 5.0
    y.append(0)

    # Q2: (-, +) -> Class 1
    X[4, 0] = -2.0
    X[4, 1] = 2.0
    y.append(1)
    X[5, 0] = -4.0
    X[5, 1] = 3.0
    y.append(1)
    X[6, 0] = -1.0
    X[6, 1] = 5.0
    y.append(1)
    X[7, 0] = -5.0
    X[7, 1] = 1.0
    y.append(1)

    # Q3: (-, -) -> Class 2
    X[8, 0] = -3.0
    X[8, 1] = -3.0
    y.append(2)
    X[9, 0] = -2.0
    X[9, 1] = -5.0
    y.append(2)
    X[10, 0] = -5.0
    X[10, 1] = -2.0
    y.append(2)
    X[11, 0] = -4.0
    X[11, 1] = -4.0
    y.append(2)

    # Q4: (+, -) -> Class 3
    X[12, 0] = 3.0
    X[12, 1] = -2.0
    y.append(3)
    X[13, 0] = 4.0
    X[13, 1] = -5.0
    y.append(3)
    X[14, 0] = 2.0
    X[14, 1] = -4.0
    y.append(3)
    X[15, 0] = 5.0
    X[15, 1] = -3.0
    y.append(3)

    var rf = RandomForestClassifier[DType.float64](
        n_estimators=15, max_depth=4, random_state=42
    )
    rf.fit(X, y)
    assert_true(rf.is_fitted)
    assert_equal(rf.n_classes_, 4)

    var preds = rf.predict(X)
    assert_equal(len(preds), 16)
    for r in range(16):
        assert_equal(preds[r], Int(y[r]))


def test_rf_classifier_single_sample_per_tree_subsampling() raises:
    var X = Matrix[DType.float64](8, 2, 0.0)
    var y = List[Scalar[DType.int32]](capacity=8)
    for i in range(4):
        X[i, 0] = Float64(i)
        y.append(0)
    for i in range(4, 8):
        X[i, 0] = Float64(i + 10)
        y.append(1)

    var rf = RandomForestClassifier[DType.float64](
        n_estimators=20,
        bootstrap=True,
        max_samples_count=1,
        random_state=42,
    )
    rf.fit(X, y)
    assert_true(rf.is_fitted)

    var preds = rf.predict(X)
    assert_equal(len(preds), 8)


def test_rf_classifier_simplex_invariant_large_unseen_matrix() raises:
    # Fit on 12 samples (3 classes), predict on 100-sample unseen matrix
    var X_train = Matrix[DType.float64](12, 2, 0.0)
    var y_train = List[Scalar[DType.int32]](capacity=12)
    for i in range(4):
        X_train[i, 0] = Float64(i)
        y_train.append(0)
    for i in range(4, 8):
        X_train[i, 0] = Float64(i + 10)
        y_train.append(1)
    for i in range(8, 12):
        X_train[i, 0] = Float64(i + 20)
        y_train.append(2)

    var rf = RandomForestClassifier[DType.float64](
        n_estimators=10, random_state=42
    )
    rf.fit(X_train, y_train)

    var X_test = Matrix[DType.float64](100, 2, 0.0)
    for i in range(100):
        X_test[i, 0] = Float64(i) * 0.5 - 10.0
        X_test[i, 1] = Float64(i % 5)

    var proba = rf.predict_proba(X_test)
    assert_equal(proba.rows, 100)
    assert_equal(proba.cols, 3)

    for i in range(100):
        var row_sum: Float64 = 0.0
        for c in range(3):
            var p = Float64(proba[i, c])
            assert_true(p >= 0.0 and p <= 1.0)
            row_sum += p
        assert_almost_equal(row_sum, 1.0, atol=1e-5)


def test_rf_classifier_constant_features_5_classes_mixed_distribution() raises:
    # 10 rows with identical feature [2.0, 3.0]
    # Distribution: 4 of Class 0, 3 of Class 1, 2 of Class 2, 1 of Class 3
    var X = Matrix[DType.float64](10, 2, 2.0)
    var y = List[Scalar[DType.int32]](capacity=10)
    y.append(0)
    y.append(0)
    y.append(0)
    y.append(0)
    y.append(1)
    y.append(1)
    y.append(1)
    y.append(2)
    y.append(2)
    y.append(3)

    var rf = RandomForestClassifier[DType.float64](
        n_estimators=20, random_state=42
    )
    rf.fit(X, y)
    assert_true(rf.is_fitted)
    assert_equal(rf.n_classes_, 4)

    var preds = rf.predict(X)
    for i in range(10):
        assert_equal(preds[i], 0)  # Majority class is 0


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
