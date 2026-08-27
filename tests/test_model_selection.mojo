from std.testing import TestSuite, assert_equal, assert_true, assert_false
from strata import (
    Matrix,
    Dataset,
    train_test_split,
    LinearRegression,
    Ridge,
    LogisticRegression,
    StandardScaler,
    PipelineRegressor,
    PipelineClassifier,
)
from strata.model_selection import (
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    ShuffleSplit,
    StratifiedShuffleSplit,
    Split,
    cross_val_score,
    cross_val_predict,
    GridSearchRegressor,
    GridSearchClassifier,
    take_rows,
    take_elements,
)
from strata.exceptions import (
    InvalidParameterError,
    NotFittedError,
    DimensionMismatchError,
)


def test_subsampling_take_rows_and_elements_standard() raises:
    var X = Matrix[DType.float64](5, 2, 0)
    var y = List[Scalar[DType.float64]](capacity=5)
    for i in range(5):
        X[i, 0] = Float64(i * 10)
        X[i, 1] = Float64(i * 10 + 1)
        y.append(Float64(i + 100))

    var indices: List[Int] = [4, 1, 3]
    var sub_X = take_rows(X, indices)
    var sub_y = take_elements(y, indices)

    assert_equal(sub_X.rows, 3)
    assert_equal(sub_X.cols, 2)
    assert_equal(sub_X[0, 0], 40.0)
    assert_equal(sub_X[0, 1], 41.0)
    assert_equal(sub_X[1, 0], 10.0)
    assert_equal(sub_X[1, 1], 11.0)
    assert_equal(sub_X[2, 0], 30.0)
    assert_equal(sub_X[2, 1], 31.0)

    assert_equal(len(sub_y), 3)
    assert_equal(sub_y[0], 104.0)
    assert_equal(sub_y[1], 101.0)
    assert_equal(sub_y[2], 103.0)


def test_subsampling_empty_indices() raises:
    var X = Matrix[DType.float64](4, 3, 1.0)
    var y: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0, 4.0]
    var empty_indices = List[Int]()

    var sub_X = take_rows(X, empty_indices)
    assert_equal(sub_X.rows, 0)
    assert_equal(sub_X.cols, 3)

    var sub_y = take_elements(y, empty_indices)
    assert_equal(len(sub_y), 0)


def test_subsampling_repeated_and_single_indices() raises:
    var X = Matrix[DType.float64](3, 1, 0)
    X[0, 0] = 10.0
    X[1, 0] = 20.0
    X[2, 0] = 30.0
    var y: List[Scalar[DType.float64]] = [10.0, 20.0, 30.0]

    var rep_indices: List[Int] = [2, 2, 0, 2, 1]
    var sub_X = take_rows(X, rep_indices)
    var sub_y = take_elements(y, rep_indices)

    assert_equal(sub_X.rows, 5)
    assert_equal(sub_X[0, 0], 30.0)
    assert_equal(sub_X[1, 0], 30.0)
    assert_equal(sub_X[2, 0], 10.0)
    assert_equal(sub_X[3, 0], 30.0)
    assert_equal(sub_X[4, 0], 20.0)

    assert_equal(len(sub_y), 5)
    assert_equal(sub_y[0], 30.0)
    assert_equal(sub_y[1], 30.0)
    assert_equal(sub_y[2], 10.0)
    assert_equal(sub_y[3], 30.0)
    assert_equal(sub_y[4], 20.0)


def test_subsampling_boundary_and_negative_index_errors() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    var y: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0, 4.0]

    var bad_neg: List[Int] = [-1]
    var caught_neg = False
    try:
        var _ = take_rows(X, bad_neg)
    except:
        caught_neg = True
    assert_true(caught_neg)

    var bad_exact_len: List[Int] = [4]
    var caught_exact = False
    try:
        var _ = take_rows(X, bad_exact_len)
    except:
        caught_exact = True
    assert_true(caught_exact)

    var caught_y_neg = False
    try:
        var _ = take_elements(y, bad_neg)
    except:
        caught_y_neg = True
    assert_true(caught_y_neg)

    var caught_y_exact = False
    try:
        var _ = take_elements(y, bad_exact_len)
    except:
        caught_y_exact = True
    assert_true(caught_y_exact)


def test_kfold_balanced_partitions_and_invariants() raises:
    var kf = KFold(n_splits=3, shuffle=False)
    assert_equal(kf.get_n_splits(), 3)

    var splits = kf.split(10)
    assert_equal(len(splits), 3)

    assert_equal(len(splits[0].val_indices), 4)
    assert_equal(len(splits[0].train_indices), 6)
    assert_equal(len(splits[1].val_indices), 3)
    assert_equal(len(splits[1].train_indices), 7)
    assert_equal(len(splits[2].val_indices), 3)
    assert_equal(len(splits[2].train_indices), 7)

    var visited = List[Bool](capacity=10)
    for _ in range(10):
        visited.append(False)

    for s in range(len(splits)):
        for i in range(len(splits[s].val_indices)):
            var idx = splits[s].val_indices[i]
            assert_false(visited[idx])
            visited[idx] = True

    for i in range(10):
        assert_true(visited[i])

    for s in range(len(splits)):
        var in_train = List[Bool](capacity=10)
        for _ in range(10):
            in_train.append(False)
        for i in range(len(splits[s].train_indices)):
            in_train[splits[s].train_indices[i]] = True

        for i in range(len(splits[s].val_indices)):
            var v_idx = splits[s].val_indices[i]
            assert_false(in_train[v_idx])


def test_kfold_leave_one_out_boundary() raises:
    var kf = KFold(n_splits=5, shuffle=False)
    var splits = kf.split(5)
    assert_equal(len(splits), 5)

    for f in range(5):
        assert_equal(len(splits[f].val_indices), 1)
        assert_equal(len(splits[f].train_indices), 4)
        assert_equal(splits[f].val_indices[0], f)


def test_kfold_minimal_two_sample_split() raises:
    var kf = KFold(n_splits=2)
    var splits = kf.split(2)
    assert_equal(len(splits), 2)
    assert_equal(len(splits[0].val_indices), 1)
    assert_equal(len(splits[0].train_indices), 1)
    assert_equal(len(splits[1].val_indices), 1)
    assert_equal(len(splits[1].train_indices), 1)


def test_kfold_large_scale_partition_integrity() raises:
    var kf = KFold(n_splits=10, shuffle=True, random_state=42)
    var splits = kf.split(1000)
    assert_equal(len(splits), 10)

    var visited = List[Bool](capacity=1000)
    for _ in range(1000):
        visited.append(False)

    for f in range(10):
        assert_equal(len(splits[f].val_indices), 100)
        assert_equal(len(splits[f].train_indices), 900)
        for i in range(100):
            var idx = splits[f].val_indices[i]
            assert_false(visited[idx])
            visited[idx] = True

    for i in range(1000):
        assert_true(visited[i])


def test_kfold_deterministic_shuffling_and_seeds() raises:
    var kf1 = KFold(n_splits=4, shuffle=True, random_state=123)
    var kf2 = KFold(n_splits=4, shuffle=True, random_state=123)
    var kf3 = KFold(n_splits=4, shuffle=True, random_state=999)

    var splits1 = kf1.split(20)
    var splits2 = kf2.split(20)
    var splits3 = kf3.split(20)

    for f in range(4):
        for i in range(len(splits1[f].val_indices)):
            assert_equal(splits1[f].val_indices[i], splits2[f].val_indices[i])

    var is_different = False
    for f in range(4):
        for i in range(len(splits1[f].val_indices)):
            if splits1[f].val_indices[i] != splits3[f].val_indices[i]:
                is_different = True
                break
    assert_true(is_different)


def test_kfold_matrix_overload_and_error_handling() raises:
    var X = Matrix[DType.float64](15, 3, 0)
    var kf = KFold(n_splits=5)
    var splits = kf.split(X)
    assert_equal(len(splits), 5)
    for f in range(5):
        assert_equal(len(splits[f].val_indices), 3)
        assert_equal(len(splits[f].train_indices), 12)

    var caught_k = False
    try:
        var _ = KFold(n_splits=1)
    except:
        caught_k = True
    assert_true(caught_k)

    var caught_n = False
    try:
        var _ = kf.split(3)
    except:
        caught_n = True
    assert_true(caught_n)

    var caught_zero = False
    try:
        var _ = kf.split(0)
    except:
        caught_zero = True
    assert_true(caught_zero)


def test_stratified_kfold_multiclass_proportions() raises:
    var y = List[Scalar[DType.int32]](capacity=60)
    for _ in range(30):
        y.append(0)
    for _ in range(20):
        y.append(1)
    for _ in range(10):
        y.append(2)

    var skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    assert_equal(skf.get_n_splits(), 5)
    var splits = skf.split(y)
    assert_equal(len(splits), 5)

    for f in range(5):
        assert_equal(len(splits[f].val_indices), 12)
        assert_equal(len(splits[f].train_indices), 48)

        var count_0 = 0
        var count_1 = 0
        var count_2 = 0
        for i in range(len(splits[f].val_indices)):
            var idx = splits[f].val_indices[i]
            var label = y[idx]
            if label == 0:
                count_0 += 1
            elif label == 1:
                count_1 += 1
            elif label == 2:
                count_2 += 1

        assert_equal(count_0, 6)
        assert_equal(count_1, 4)
        assert_equal(count_2, 2)


def test_stratified_kfold_extreme_imbalance_rare_class() raises:
    var y = List[Scalar[DType.int32]](capacity=51)
    for _ in range(50):
        y.append(0)
    y.append(1)

    var skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
    var splits = skf.split(y)
    assert_equal(len(splits), 5)

    var class_1_count = 0
    for f in range(5):
        for i in range(len(splits[f].val_indices)):
            if y[splits[f].val_indices[i]] == 1:
                class_1_count += 1
    assert_equal(class_1_count, 1)


def test_stratified_kfold_monotonic_indices_across_seeds() raises:
    var y: List[Scalar[DType.int32]] = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]

    for seed in [1, 42, 99, 500, 1337]:
        var skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        var splits = skf.split(y)
        for f in range(3):
            var val_indices = splits[f].val_indices.copy()
            for i in range(len(val_indices) - 1):
                assert_true(val_indices[i] < val_indices[i + 1])


def test_stratified_kfold_float_and_negative_labels() raises:
    var y_flt: List[Scalar[DType.float64]] = [
        0.125,
        0.875,
        0.125,
        0.875,
        0.125,
        0.875,
    ]
    var skf_flt = StratifiedKFold(n_splits=3)
    var splits_flt = skf_flt.split(y_flt)
    assert_equal(len(splits_flt), 3)

    var y_neg: List[Scalar[DType.int32]] = [-1, 1, -1, 1, -1, 1, -1, 1]
    var skf_neg = StratifiedKFold(n_splits=2)
    var splits_neg = skf_neg.split(y_neg)
    assert_equal(len(splits_neg), 2)


def test_stratified_kfold_error_handling() raises:
    var caught_k = False
    try:
        var _ = StratifiedKFold(n_splits=1)
    except:
        caught_k = True
    assert_true(caught_k)

    var skf = StratifiedKFold(n_splits=5)
    var empty_y = List[Scalar[DType.int32]]()
    var caught_empty = False
    try:
        var _ = skf.split(empty_y)
    except:
        caught_empty = True
    assert_true(caught_empty)

    var X = Matrix[DType.float64](4, 2, 0)
    var short_y: List[Scalar[DType.int32]] = [0, 1]
    var caught_mismatch = False
    try:
        var _ = skf.split(X, short_y)
    except:
        caught_mismatch = True
    assert_true(caught_mismatch)


def test_estimator_copy_mutation_isolation_linear_regression() raises:
    var X1 = Matrix[DType.float64](4, 1, 0)
    for i in range(4):
        X1[i, 0] = Float64(i + 1)
    var y1: List[Scalar[DType.float64]] = [2.0, 4.0, 6.0, 8.0]

    var lr1 = LinearRegression(fit_intercept=True)
    lr1.fit(X1, y1)

    var lr2 = lr1.copy()
    assert_true(lr2.is_fitted)
    assert_true(abs(lr2.coef_[0] - 2.0) < 1e-4)

    var X2 = Matrix[DType.float64](4, 1, 0)
    for i in range(4):
        X2[i, 0] = Float64(i + 1)
    var y2: List[Scalar[DType.float64]] = [100.0, 200.0, 300.0, 400.0]
    lr2.fit(X2, y2)

    assert_true(abs(lr1.coef_[0] - 2.0) < 1e-4)
    assert_true(abs(lr2.coef_[0] - 100.0) < 1e-3)


def test_estimator_copy_mutation_isolation_ridge() raises:
    var X = Matrix[DType.float64](4, 1, 0)
    for i in range(4):
        X[i, 0] = Float64(i + 1)
    var y: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0, 4.0]

    var r1 = Ridge(alpha=1.0)
    r1.fit(X, y)

    var r2 = r1.copy()
    r2.alpha = 5000.0
    r2.fit(X, y)

    assert_equal(r1.alpha, 1.0)
    assert_equal(r2.alpha, 5000.0)
    assert_true(r1.coef_[0] > r2.coef_[0])


def test_estimator_copy_mutation_isolation_logistic_regression() raises:
    var X = Matrix[DType.float64](6, 1, 0)
    for i in range(6):
        X[i, 0] = Float64(i)
    var y: List[Scalar[DType.int32]] = [0, 0, 0, 1, 1, 1]

    var clf1 = LogisticRegression(C=1.0, max_iter=100)
    clf1.fit(X, y)

    var clf2 = clf1.copy()
    assert_equal(clf2.C, 1.0)
    assert_equal(len(clf2.classes_), 2)

    var X3 = Matrix[DType.float64](9, 1, 0)
    for i in range(9):
        X3[i, 0] = Float64(i)
    var y3: List[Scalar[DType.int32]] = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    clf2.fit(X3, y3)

    assert_equal(len(clf1.classes_), 2)
    assert_equal(len(clf2.classes_), 3)


def test_estimator_copy_pipeline_isolation() raises:
    var scaler = StandardScaler(with_mean=True, with_std=True)
    var ridge = Ridge(alpha=1.0)
    var pipe1 = PipelineRegressor(scaler^, ridge^)

    var pipe2 = pipe1.copy()
    pipe2.regressor.alpha = 100.0

    assert_equal(pipe1.regressor.alpha, 1.0)
    assert_equal(pipe2.regressor.alpha, 100.0)


def test_cross_val_score_regression_metrics_consistency() raises:
    var X = Matrix[DType.float64](12, 1, 0)
    var y = List[Scalar[DType.float64]](capacity=12)
    for i in range(12):
        X[i, 0] = Float64(i + 1)
        y.append(Float64(2 * (i + 1)))

    var ridge = Ridge(alpha=0.001)

    var r2_scores = cross_val_score(ridge, X, y, cv=3, scoring="r2")
    var mse_scores = cross_val_score(ridge, X, y, cv=3, scoring="mse")
    var neg_mse = cross_val_score(
        ridge, X, y, cv=3, scoring="neg_mean_squared_error"
    )
    var rmse_scores = cross_val_score(ridge, X, y, cv=3, scoring="rmse")
    var neg_rmse = cross_val_score(
        ridge, X, y, cv=3, scoring="neg_root_mean_squared_error"
    )
    var mae_scores = cross_val_score(ridge, X, y, cv=3, scoring="mae")
    var neg_mae = cross_val_score(
        ridge, X, y, cv=3, scoring="neg_mean_absolute_error"
    )

    for f in range(3):
        assert_true(r2_scores[f] > 0.99)
        assert_equal(neg_mse[f], -mse_scores[f])
        assert_equal(neg_rmse[f], -rmse_scores[f])
        assert_equal(neg_mae[f], -mae_scores[f])


def test_cross_val_score_classification_metrics_consistency() raises:
    var X = Matrix[DType.float64](12, 1, 0)
    var y = List[Scalar[DType.int32]](capacity=12)
    for i in range(12):
        X[i, 0] = Float64(i)
        y.append(Int32(0 if i < 6 else 1))

    var logreg = LogisticRegression(max_iter=150)

    var acc = cross_val_score(logreg, X, y, cv=3, scoring="accuracy")
    var f1 = cross_val_score(logreg, X, y, cv=3, scoring="f1")
    var prec = cross_val_score(logreg, X, y, cv=3, scoring="precision")
    var rec = cross_val_score(logreg, X, y, cv=3, scoring="recall")

    for f in range(3):
        assert_true(acc[f] >= 0.5)
        assert_true(f1[f] >= 0.0 and f1[f] <= 1.0)
        assert_true(prec[f] >= 0.0 and prec[f] <= 1.0)
        assert_true(rec[f] >= 0.0 and rec[f] <= 1.0)


def test_cross_val_score_preserves_initial_estimator_state() raises:
    var X = Matrix[DType.float64](6, 1, 0)
    var y: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    for i in range(6):
        X[i, 0] = Float64(i + 1)

    var lr = LinearRegression()
    assert_false(lr.is_fitted)

    var _ = cross_val_score(lr, X, y, cv=3, scoring="r2")
    assert_false(lr.is_fitted)
    assert_equal(len(lr.coef_), 0)


def test_cross_val_score_error_handling() raises:
    var X = Matrix[DType.float64](6, 1, 0)
    var y: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    var lr = LinearRegression()

    var caught_metric = False
    try:
        var _ = cross_val_score(lr, X, y, cv=3, scoring="bogus_metric")
    except:
        caught_metric = True
    assert_true(caught_metric)

    var empty_splits = List[Split]()
    var caught_splits = False
    try:
        var _ = cross_val_score(lr, X, y, empty_splits)
    except:
        caught_splits = True
    assert_true(caught_splits)


def test_grid_search_regressor_sweep_and_best_selection() raises:
    var X = Matrix[DType.float64](10, 1, 0)
    var y = List[Scalar[DType.float64]](capacity=10)
    for i in range(10):
        X[i, 0] = Float64(i + 1)
        y.append(Float64(2 * (i + 1)))

    var candidates = List[Ridge[DType.float64]]()
    candidates.append(Ridge[DType.float64](alpha=1000.0))
    candidates.append(Ridge[DType.float64](alpha=0.001))
    candidates.append(Ridge[DType.float64](alpha=500.0))

    var grid = GridSearchRegressor[Ridge[DType.float64]](
        candidates^, cv=3, scoring="r2", refit=True
    )
    grid.fit(X, y)

    assert_true(grid.is_fitted)
    assert_equal(grid.best_index_, 1)
    assert_true(grid.best_score_ > 0.99)
    assert_equal(len(grid.cv_results_mean_), 3)
    assert_equal(len(grid.cv_results_std_), 3)

    var preds = grid.predict(X)
    assert_equal(len(preds), 10)
    assert_true(abs(preds[0] - 2.0) < 0.1)


def test_grid_search_regressor_candidate_isolation() raises:
    var X = Matrix[DType.float64](8, 1, 0)
    var y = List[Scalar[DType.float64]](capacity=8)
    for i in range(8):
        X[i, 0] = Float64(i + 1)
        y.append(Float64(i + 1))

    var c1 = Ridge[DType.float64](alpha=0.1)
    var c2 = Ridge[DType.float64](alpha=10.0)
    var candidates = List[Ridge[DType.float64]]()
    candidates.append(c1^)
    candidates.append(c2^)

    var grid = GridSearchRegressor[Ridge[DType.float64]](candidates^, cv=2)
    grid.fit(X, y)

    assert_false(grid.candidates[0].is_fitted)
    assert_false(grid.candidates[1].is_fitted)


def test_grid_search_regressor_no_refit_raises() raises:
    var X = Matrix[DType.float64](6, 1, 0)
    var y: List[Scalar[DType.float64]] = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    for i in range(6):
        X[i, 0] = Float64(i + 1)

    var candidates = List[Ridge[DType.float64]]()
    candidates.append(Ridge[DType.float64](alpha=0.1))

    var grid = GridSearchRegressor[Ridge[DType.float64]](
        candidates^, cv=3, refit=False
    )
    grid.fit(X, y)
    assert_true(grid.is_fitted)

    var caught_no_refit = False
    try:
        var _ = grid.predict(X)
    except:
        caught_no_refit = True
    assert_true(caught_no_refit)


def test_grid_search_regressor_solvers_and_dtype_flexibility() raises:
    var X = Matrix[DType.float64](8, 2, 0)
    var y = List[Scalar[DType.float64]](capacity=8)
    for i in range(8):
        X[i, 0] = Float64(i + 1)
        X[i, 1] = Float64((i + 1) * (i + 1))
        y.append(Float64(3 * (i + 1) + 2 * (i + 1) * (i + 1) + 1))

    var candidates = List[LinearRegression[DType.float64]]()
    candidates.append(LinearRegression[DType.float64](solver="lstsq"))
    candidates.append(LinearRegression[DType.float64](solver="qr"))
    candidates.append(LinearRegression[DType.float64](solver="cholesky"))

    var grid = GridSearchRegressor[LinearRegression[DType.float64]](
        candidates^, cv=4, scoring="r2", refit=True
    )
    grid.fit(X, y)
    assert_true(grid.is_fitted)
    assert_true(grid.best_score_ > 0.99)

    var X_f32 = Matrix[DType.float32](8, 2, 0)
    for i in range(8):
        X_f32[i, 0] = Float32(i + 1)
        X_f32[i, 1] = Float32((i + 1) * (i + 1))

    var preds_f32 = grid.predict(X_f32)
    assert_equal(len(preds_f32), 8)
    assert_true(abs(preds_f32[0] - 6.0) < 0.1)


def test_grid_search_classifier_binary_and_candidate_isolation() raises:
    var X = Matrix[DType.float64](12, 1, 0)
    var y = List[Scalar[DType.int32]](capacity=12)
    for i in range(12):
        X[i, 0] = Float64(i)
        y.append(Int32(0 if i < 6 else 1))

    var candidates = List[LogisticRegression[DType.float64]]()
    candidates.append(LogisticRegression[DType.float64](C=0.001, max_iter=50))
    candidates.append(LogisticRegression[DType.float64](C=10.0, max_iter=150))

    var grid = GridSearchClassifier[LogisticRegression[DType.float64]](
        candidates^, cv=3, scoring="accuracy", refit=True
    )
    grid.fit(X, y)

    assert_true(grid.is_fitted)
    assert_equal(grid.best_index_, 1)
    assert_true(grid.best_score_ >= 0.8)
    assert_false(grid.candidates[0].is_fitted)
    assert_false(grid.candidates[1].is_fitted)

    var class_preds = grid.predict(X)
    assert_equal(len(class_preds), 12)

    var proba = grid.predict_proba(X)
    assert_equal(proba.rows, 12)
    assert_equal(proba.cols, 2)
    for r in range(12):
        var row_sum = proba[r, 0] + proba[r, 1]
        assert_true(abs(row_sum - 1.0) < 1e-4)


def test_grid_search_classifier_multiclass_and_proba() raises:
    var X = Matrix[DType.float64](15, 2, 0)
    var y = List[Scalar[DType.int32]](capacity=15)
    for i in range(15):
        X[i, 0] = Float64(i)
        X[i, 1] = Float64(i * 2)
        if i < 5:
            y.append(0)
        elif i < 10:
            y.append(1)
        else:
            y.append(2)

    var candidates = List[LogisticRegression[DType.float64]]()
    candidates.append(LogisticRegression[DType.float64](C=1.0, max_iter=200))
    candidates.append(LogisticRegression[DType.float64](C=10.0, max_iter=200))

    var grid = GridSearchClassifier[LogisticRegression[DType.float64]](
        candidates^, cv=3, scoring="accuracy", refit=True
    )
    grid.fit(X, y)
    assert_true(grid.is_fitted)

    var preds = grid.predict(X)
    assert_equal(len(preds), 15)

    var proba = grid.predict_proba(X)
    assert_equal(proba.rows, 15)
    assert_equal(proba.cols, 3)

    for r in range(15):
        var row_sum = proba[r, 0] + proba[r, 1] + proba[r, 2]
        assert_true(abs(row_sum - 1.0) < 1e-4)


def test_grid_search_classifier_no_refit_raises() raises:
    var X = Matrix[DType.float64](6, 1, 0)
    var y: List[Scalar[DType.int32]] = [0, 0, 0, 1, 1, 1]
    for i in range(6):
        X[i, 0] = Float64(i)

    var candidates = List[LogisticRegression[DType.float64]]()
    candidates.append(LogisticRegression[DType.float64](C=1.0, max_iter=50))

    var grid = GridSearchClassifier[LogisticRegression[DType.float64]](
        candidates^, cv=3, refit=False
    )
    grid.fit(X, y)
    assert_true(grid.is_fitted)

    var caught_pred = False
    try:
        var _ = grid.predict(X)
    except:
        caught_pred = True
    assert_true(caught_pred)

    var caught_proba = False
    try:
        var _ = grid.predict_proba(X)
    except:
        caught_proba = True
    assert_true(caught_proba)


def test_grid_search_classifier_error_handling() raises:
    var empty_cands = List[LogisticRegression[DType.float64]]()
    var caught_empty = False
    try:
        var _ = GridSearchClassifier[LogisticRegression[DType.float64]](
            empty_cands^
        )
    except:
        caught_empty = True
    assert_true(caught_empty)


def test_grid_search_pipeline_regression_integration() raises:
    var X = Matrix[DType.float64](10, 2, 0)
    var y = List[Scalar[DType.float64]](capacity=10)
    for i in range(10):
        X[i, 0] = Float64(i * 10)
        X[i, 1] = Float64(i * 5)
        y.append(Float64(i + 1))

    var candidates = List[
        PipelineRegressor[
            StandardScaler[DType.float64],
            Ridge[DType.float64],
            DType.float64,
        ]
    ]()

    var s1 = StandardScaler[DType.float64](with_mean=True, with_std=True)
    var r1 = Ridge[DType.float64](alpha=0.01)
    candidates.append(PipelineRegressor(s1^, r1^))

    var s2 = StandardScaler[DType.float64](with_mean=False, with_std=False)
    var r2 = Ridge[DType.float64](alpha=100.0)
    candidates.append(PipelineRegressor(s2^, r2^))

    var grid = GridSearchRegressor[
        PipelineRegressor[
            StandardScaler[DType.float64],
            Ridge[DType.float64],
            DType.float64,
        ]
    ](candidates^, cv=3, scoring="r2", refit=True)

    grid.fit(X, y)
    assert_true(grid.is_fitted)
    assert_equal(grid.best_index_, 0)
    assert_true(grid.best_score_ > 0.95)


def test_grid_search_pipeline_classification_integration() raises:
    var X = Matrix[DType.float64](12, 2, 0)
    var y = List[Scalar[DType.int32]](capacity=12)
    for i in range(12):
        X[i, 0] = Float64(i * 10)
        X[i, 1] = Float64(i * 2)
        y.append(Int32(0 if i < 6 else 1))

    var candidates = List[
        PipelineClassifier[
            StandardScaler[DType.float64],
            LogisticRegression[DType.float64],
            DType.int32,
        ]
    ]()

    var s1 = StandardScaler[DType.float64](with_mean=True, with_std=True)
    var c1 = LogisticRegression[DType.float64](C=10.0, max_iter=150)
    candidates.append(PipelineClassifier(s1^, c1^))

    var s2 = StandardScaler[DType.float64](with_mean=False, with_std=False)
    var c2 = LogisticRegression[DType.float64](C=0.001, max_iter=50)
    candidates.append(PipelineClassifier(s2^, c2^))

    var grid = GridSearchClassifier[
        PipelineClassifier[
            StandardScaler[DType.float64],
            LogisticRegression[DType.float64],
            DType.int32,
        ]
    ](candidates^, cv=3, scoring="accuracy", refit=True)

    grid.fit(X, y)
    assert_true(grid.is_fitted)
    assert_equal(grid.best_index_, 0)
    assert_true(grid.best_score_ >= 0.8)


def test_full_train_test_split_and_gridsearch_workflow() raises:
    var X = Matrix[DType.float64](20, 2, 0)
    var y = List[Scalar[DType.float64]](capacity=20)
    for i in range(20):
        X[i, 0] = Float64(i + 1)
        X[i, 1] = Float64((i + 1) * 3)
        y.append(Float64(2 * (i + 1) + 5))

    var split = train_test_split(X, y, test_size=0.25, shuffle=True, seed=42)

    var candidates = List[Ridge[DType.float64]]()
    candidates.append(Ridge[DType.float64](alpha=100.0))
    candidates.append(Ridge[DType.float64](alpha=0.001))

    var grid = GridSearchRegressor[Ridge[DType.float64]](
        candidates^, cv=3, scoring="r2", refit=True
    )
    grid.fit(split.train.records, split.train.targets)
    assert_equal(grid.best_index_, 1)

    var test_preds = grid.predict(split.test.records)
    assert_equal(len(test_preds), 5)
    for i in range(5):
        var expected = split.test.targets[i]
        assert_true(abs(test_preds[i] - expected) < 0.5)


def _assert_contiguous(indices: List[Int], start: Int, end: Int) raises:
    assert_equal(len(indices), end - start)
    for i in range(end - start):
        assert_equal(indices[i], start + i)


def _assert_forward_only(splits: List[Split], n_samples: Int) raises:
    for f in range(len(splits)):
        var n_train = len(splits[f].train_indices)
        var n_val = len(splits[f].val_indices)
        assert_true(n_train > 0)
        assert_true(n_val > 0)

        for i in range(n_train):
            assert_true(splits[f].train_indices[i] >= 0)
            assert_true(splits[f].train_indices[i] < n_samples)
        for i in range(n_val):
            assert_true(splits[f].val_indices[i] >= 0)
            assert_true(splits[f].val_indices[i] < n_samples)

        for i in range(1, n_train):
            assert_true(
                splits[f].train_indices[i] > splits[f].train_indices[i - 1]
            )
        for i in range(1, n_val):
            assert_true(splits[f].val_indices[i] > splits[f].val_indices[i - 1])

        assert_true(
            splits[f].train_indices[n_train - 1] < splits[f].val_indices[0]
        )

    for f in range(1, len(splits)):
        assert_true(splits[f].val_indices[0] > splits[f - 1].val_indices[0])


def _assert_split_shape(
    splits: List[Split], n_samples: Int, n_train: Int, n_test: Int
) raises:
    for f in range(len(splits)):
        assert_equal(len(splits[f].train_indices), n_train)
        assert_equal(len(splits[f].val_indices), n_test)

        var seen = List[Bool](capacity=n_samples)
        for _ in range(n_samples):
            seen.append(False)

        for i in range(len(splits[f].train_indices)):
            var t = splits[f].train_indices[i]
            assert_true(t >= 0 and t < n_samples)
            assert_false(seen[t])
            seen[t] = True

        for i in range(len(splits[f].val_indices)):
            var v = splits[f].val_indices[i]
            assert_true(v >= 0 and v < n_samples)
            assert_false(seen[v])
            seen[v] = True


def test_time_series_split_expanding_window_defaults() raises:
    var tss = TimeSeriesSplit(n_splits=3)
    var splits = tss.split(10)

    assert_equal(len(splits), 3)
    _assert_contiguous(splits[0].train_indices, 0, 4)
    _assert_contiguous(splits[0].val_indices, 4, 6)
    _assert_contiguous(splits[1].train_indices, 0, 6)
    _assert_contiguous(splits[1].val_indices, 6, 8)
    _assert_contiguous(splits[2].train_indices, 0, 8)
    _assert_contiguous(splits[2].val_indices, 8, 10)


def test_time_series_split_default_five_splits() raises:
    var tss = TimeSeriesSplit()
    assert_equal(tss.get_n_splits(), 5)

    var splits = tss.split(12)
    assert_equal(len(splits), 5)
    for f in range(5):
        _assert_contiguous(splits[f].train_indices, 0, 2 + f * 2)
        _assert_contiguous(splits[f].val_indices, 2 + f * 2, 4 + f * 2)


def test_time_series_split_training_set_grows_monotonically() raises:
    var tss = TimeSeriesSplit(n_splits=4)
    var splits = tss.split(25)

    _assert_forward_only(splits, 25)
    for f in range(1, len(splits)):
        assert_true(
            len(splits[f].train_indices) > len(splits[f - 1].train_indices)
        )
        assert_equal(splits[f].train_indices[0], 0)


def test_time_series_split_uneven_division_absorbs_remainder_upfront() raises:
    var tss = TimeSeriesSplit(n_splits=3)
    var splits = tss.split(11)

    assert_equal(len(splits), 3)
    _assert_contiguous(splits[0].train_indices, 0, 5)
    _assert_contiguous(splits[0].val_indices, 5, 7)
    _assert_contiguous(splits[1].val_indices, 7, 9)
    _assert_contiguous(splits[2].val_indices, 9, 11)

    assert_equal(splits[2].val_indices[1], 10)


def test_time_series_split_equal_test_block_sizes() raises:
    var tss = TimeSeriesSplit(n_splits=4)
    var splits = tss.split(23)

    var expected = len(splits[0].val_indices)
    for f in range(len(splits)):
        assert_equal(len(splits[f].val_indices), expected)

    var n_last = len(splits[len(splits) - 1].val_indices)
    assert_equal(splits[len(splits) - 1].val_indices[n_last - 1], 22)


def test_time_series_split_gap_excludes_leading_samples() raises:
    var tss = TimeSeriesSplit(n_splits=3, gap=2)
    var splits = tss.split(12)

    assert_equal(len(splits), 3)
    _assert_contiguous(splits[0].train_indices, 0, 1)
    _assert_contiguous(splits[0].val_indices, 3, 6)
    _assert_contiguous(splits[1].train_indices, 0, 4)
    _assert_contiguous(splits[1].val_indices, 6, 9)
    _assert_contiguous(splits[2].train_indices, 0, 7)
    _assert_contiguous(splits[2].val_indices, 9, 12)

    for f in range(3):
        var n_train = len(splits[f].train_indices)
        assert_equal(
            splits[f].val_indices[0] - splits[f].train_indices[n_train - 1], 3
        )


def test_time_series_split_zero_gap_is_adjacent() raises:
    var tss = TimeSeriesSplit(n_splits=3, gap=0)
    var splits = tss.split(12)

    for f in range(3):
        var n_train = len(splits[f].train_indices)
        assert_equal(
            splits[f].val_indices[0], splits[f].train_indices[n_train - 1] + 1
        )


def test_time_series_split_max_train_size_slides_window() raises:
    var tss = TimeSeriesSplit(n_splits=3, max_train_size=3)
    var splits = tss.split(12)

    assert_equal(len(splits), 3)
    _assert_contiguous(splits[0].train_indices, 0, 3)
    _assert_contiguous(splits[0].val_indices, 3, 6)
    _assert_contiguous(splits[1].train_indices, 3, 6)
    _assert_contiguous(splits[1].val_indices, 6, 9)
    _assert_contiguous(splits[2].train_indices, 6, 9)
    _assert_contiguous(splits[2].val_indices, 9, 12)


def test_time_series_split_max_train_size_larger_than_data_is_noop() raises:
    var capped = TimeSeriesSplit(n_splits=3, max_train_size=100)
    var uncapped = TimeSeriesSplit(n_splits=3)

    var a = capped.split(10)
    var b = uncapped.split(10)

    assert_equal(len(a), len(b))
    for f in range(len(a)):
        assert_equal(len(a[f].train_indices), len(b[f].train_indices))
        for i in range(len(a[f].train_indices)):
            assert_equal(a[f].train_indices[i], b[f].train_indices[i])


def test_time_series_split_max_train_size_exact_boundary() raises:
    var tss = TimeSeriesSplit(n_splits=3, max_train_size=4)
    var splits = tss.split(10)

    _assert_contiguous(splits[0].train_indices, 0, 4)
    _assert_contiguous(splits[1].train_indices, 2, 6)
    _assert_contiguous(splits[2].train_indices, 4, 8)


def test_time_series_split_max_train_size_one() raises:
    var tss = TimeSeriesSplit(n_splits=3, max_train_size=1)
    var splits = tss.split(12)

    for f in range(3):
        assert_equal(len(splits[f].train_indices), 1)
        assert_equal(splits[f].train_indices[0], splits[f].val_indices[0] - 1)


def test_time_series_split_explicit_test_size() raises:
    var tss = TimeSeriesSplit(n_splits=2, test_size=3)
    var splits = tss.split(12)

    assert_equal(len(splits), 2)
    _assert_contiguous(splits[0].train_indices, 0, 6)
    _assert_contiguous(splits[0].val_indices, 6, 9)
    _assert_contiguous(splits[1].train_indices, 0, 9)
    _assert_contiguous(splits[1].val_indices, 9, 12)


def test_time_series_split_test_size_one() raises:
    var tss = TimeSeriesSplit(n_splits=4, test_size=1)
    var splits = tss.split(10)

    assert_equal(len(splits), 4)
    for f in range(4):
        assert_equal(len(splits[f].val_indices), 1)
        assert_equal(splits[f].val_indices[0], 6 + f)
        _assert_contiguous(splits[f].train_indices, 0, 6 + f)


def test_time_series_split_gap_and_max_train_size_combined() raises:
    var tss = TimeSeriesSplit(n_splits=3, gap=2, max_train_size=4)
    var splits = tss.split(20)

    assert_equal(len(splits), 3)
    _assert_contiguous(splits[0].train_indices, 0, 3)
    _assert_contiguous(splits[0].val_indices, 5, 10)
    _assert_contiguous(splits[1].train_indices, 4, 8)
    _assert_contiguous(splits[1].val_indices, 10, 15)
    _assert_contiguous(splits[2].train_indices, 9, 13)
    _assert_contiguous(splits[2].val_indices, 15, 20)

    _assert_forward_only(splits, 20)


def test_time_series_split_minimal_two_split_boundary() raises:
    var tss = TimeSeriesSplit(n_splits=2)
    var splits = tss.split(3)

    assert_equal(len(splits), 2)
    _assert_contiguous(splits[0].train_indices, 0, 1)
    _assert_contiguous(splits[0].val_indices, 1, 2)
    _assert_contiguous(splits[1].train_indices, 0, 2)
    _assert_contiguous(splits[1].val_indices, 2, 3)


def test_time_series_split_no_leakage_and_disjoint_folds() raises:
    var tss = TimeSeriesSplit(n_splits=5)
    var splits = tss.split(1000)

    assert_equal(len(splits), 5)
    _assert_forward_only(splits, 1000)

    var tested = List[Bool](capacity=1000)
    for _ in range(1000):
        tested.append(False)

    for f in range(5):
        for i in range(len(splits[f].val_indices)):
            var idx = splits[f].val_indices[i]
            assert_false(tested[idx])
            tested[idx] = True

    var n_last = len(splits[4].val_indices)
    assert_equal(splits[4].val_indices[n_last - 1], 999)

    for f in range(5):
        var in_train = List[Bool](capacity=1000)
        for _ in range(1000):
            in_train.append(False)
        for i in range(len(splits[f].train_indices)):
            in_train[splits[f].train_indices[i]] = True
        for i in range(len(splits[f].val_indices)):
            assert_false(in_train[splits[f].val_indices[i]])


def test_time_series_split_is_deterministic_and_stateless() raises:
    var tss = TimeSeriesSplit(n_splits=3)

    var first = tss.split(12)
    var other = tss.split(20)
    var again = tss.split(12)

    assert_equal(len(other), 3)
    assert_equal(len(first), len(again))
    for f in range(len(first)):
        assert_equal(len(first[f].val_indices), len(again[f].val_indices))
        for i in range(len(first[f].val_indices)):
            assert_equal(first[f].val_indices[i], again[f].val_indices[i])
        for i in range(len(first[f].train_indices)):
            assert_equal(first[f].train_indices[i], again[f].train_indices[i])

    assert_equal(tss.test_size, 0)
    assert_equal(tss.get_n_splits(), 3)


def test_time_series_split_matrix_overload_matches_count() raises:
    var X = Matrix[DType.float64](12, 3, 1.0)
    var tss = TimeSeriesSplit(n_splits=3)

    var from_matrix = tss.split(X)
    var from_count = tss.split(12)

    assert_equal(len(from_matrix), len(from_count))
    for f in range(len(from_matrix)):
        assert_equal(
            len(from_matrix[f].val_indices), len(from_count[f].val_indices)
        )
        assert_equal(
            len(from_matrix[f].train_indices), len(from_count[f].train_indices)
        )
        for i in range(len(from_count[f].val_indices)):
            assert_equal(
                from_matrix[f].val_indices[i], from_count[f].val_indices[i]
            )
        for i in range(len(from_count[f].train_indices)):
            assert_equal(
                from_matrix[f].train_indices[i], from_count[f].train_indices[i]
            )


def test_time_series_split_matrix_overload_dtype_flexibility() raises:
    var X32 = Matrix[DType.float32](12, 2, 1.0)
    var Xi = Matrix[DType.int32](12, 2, 1)
    var tss = TimeSeriesSplit(n_splits=3)

    assert_equal(len(tss.split(X32)), 3)
    assert_equal(len(tss.split(Xi)), 3)


def test_time_series_split_matrix_overload_rejects_empty() raises:
    var tss = TimeSeriesSplit(n_splits=3)

    var caught_rows = False
    try:
        var empty = Matrix[DType.float64](0, 3, 0)
        var _ = tss.split(empty)
    except:
        caught_rows = True
    assert_true(caught_rows)

    var caught_cols = False
    try:
        var no_cols = Matrix[DType.float64](12, 0, 0)
        var _ = tss.split(no_cols)
    except:
        caught_cols = True
    assert_true(caught_cols)


def test_time_series_split_constructor_error_handling() raises:
    var caught_one = False
    try:
        var _ = TimeSeriesSplit(n_splits=1)
    except:
        caught_one = True
    assert_true(caught_one)

    var caught_zero = False
    try:
        var _ = TimeSeriesSplit(n_splits=0)
    except:
        caught_zero = True
    assert_true(caught_zero)

    var caught_neg_splits = False
    try:
        var _ = TimeSeriesSplit(n_splits=-3)
    except:
        caught_neg_splits = True
    assert_true(caught_neg_splits)

    var caught_gap = False
    try:
        var _ = TimeSeriesSplit(n_splits=3, gap=-1)
    except:
        caught_gap = True
    assert_true(caught_gap)

    var caught_max = False
    try:
        var _ = TimeSeriesSplit(n_splits=3, max_train_size=-1)
    except:
        caught_max = True
    assert_true(caught_max)

    var caught_test = False
    try:
        var _ = TimeSeriesSplit(n_splits=3, test_size=-1)
    except:
        caught_test = True
    assert_true(caught_test)


def test_time_series_split_rejects_nonpositive_sample_counts() raises:
    var tss = TimeSeriesSplit(n_splits=3)

    var caught_zero = False
    try:
        var _ = tss.split(0)
    except:
        caught_zero = True
    assert_true(caught_zero)

    var caught_neg = False
    try:
        var _ = tss.split(-5)
    except:
        caught_neg = True
    assert_true(caught_neg)


def test_time_series_split_rejects_too_few_samples_for_folds() raises:
    var tss = TimeSeriesSplit(n_splits=5)

    var caught_equal = False
    try:
        var _ = tss.split(5)
    except:
        caught_equal = True
    assert_true(caught_equal)

    var caught_fewer = False
    try:
        var _ = tss.split(3)
    except:
        caught_fewer = True
    assert_true(caught_fewer)

    assert_equal(len(tss.split(6)), 5)


def test_time_series_split_rejects_oversized_test_size() raises:
    var tss = TimeSeriesSplit(n_splits=3, test_size=5)

    var caught = False
    try:
        var _ = tss.split(12)
    except:
        caught = True
    assert_true(caught)

    assert_equal(len(tss.split(16)), 3)


def test_time_series_split_rejects_oversized_gap() raises:
    var tss = TimeSeriesSplit(n_splits=3, gap=10)

    var caught = False
    try:
        var _ = tss.split(12)
    except:
        caught = True
    assert_true(caught)

    var wide = TimeSeriesSplit(n_splits=2, gap=3)
    var splits = wide.split(12)
    assert_equal(len(splits), 2)
    assert_true(len(splits[0].train_indices) > 0)


def test_time_series_split_cross_val_score_integration() raises:
    var X = Matrix[DType.float64](20, 1, 0)
    var y = List[Scalar[DType.float64]](capacity=20)
    for i in range(20):
        X[i, 0] = Float64(i)
        y.append(Float64(2 * i + 5))

    var tss = TimeSeriesSplit(n_splits=3)
    var splits = tss.split(X)
    var model = LinearRegression[DType.float64]()

    var scores = cross_val_score(model, X, y, splits, scoring="r2")
    assert_equal(len(scores), 3)
    for f in range(3):
        assert_true(scores[f] > 0.99)


def test_shuffle_split_default_parameters() raises:
    var ss = ShuffleSplit()
    assert_equal(ss.get_n_splits(), 10)

    var splits = ss.split(100)
    assert_equal(len(splits), 10)
    _assert_split_shape(splits, 100, 90, 10)


def test_shuffle_split_sizes_and_within_split_disjointness() raises:
    var ss = ShuffleSplit(n_splits=3, test_size=0.3)
    var splits = ss.split(10)

    assert_equal(len(splits), 3)
    _assert_split_shape(splits, 10, 7, 3)


def test_shuffle_split_covers_all_samples_when_train_size_unset() raises:
    var ss = ShuffleSplit(n_splits=4, test_size=0.25)
    var splits = ss.split(20)

    for f in range(4):
        assert_equal(
            len(splits[f].train_indices) + len(splits[f].val_indices), 20
        )

        var seen = List[Bool](capacity=20)
        for _ in range(20):
            seen.append(False)
        for i in range(len(splits[f].train_indices)):
            seen[splits[f].train_indices[i]] = True
        for i in range(len(splits[f].val_indices)):
            seen[splits[f].val_indices[i]] = True
        for i in range(20):
            assert_true(seen[i])


def test_shuffle_split_ceil_rounding_for_test_size() raises:
    var half = ShuffleSplit(n_splits=1, test_size=0.1)
    assert_equal(len(half.split(5)[0].val_indices), 1)

    var quarter = ShuffleSplit(n_splits=1, test_size=0.25)
    assert_equal(len(quarter.split(10)[0].val_indices), 3)

    var tenth = ShuffleSplit(n_splits=1, test_size=0.1)
    assert_equal(len(tenth.split(12)[0].val_indices), 2)

    var exact = ShuffleSplit(n_splits=1, test_size=0.2)
    assert_equal(len(exact.split(10)[0].val_indices), 2)


def test_shuffle_split_floor_rounding_for_train_size() raises:
    var ss = ShuffleSplit(n_splits=2, test_size=0.2, train_size=0.35)
    var splits = ss.split(10)

    assert_equal(len(splits), 2)
    for f in range(2):
        assert_equal(len(splits[f].val_indices), 2)
        assert_equal(len(splits[f].train_indices), 3)


def test_shuffle_split_train_size_leaves_samples_unused() raises:
    var ss = ShuffleSplit(n_splits=3, test_size=0.2, train_size=0.3)
    var splits = ss.split(10)

    _assert_split_shape(splits, 10, 3, 2)
    for f in range(3):
        assert_true(
            len(splits[f].train_indices) + len(splits[f].val_indices) < 10
        )


def test_shuffle_split_test_sets_may_overlap_across_draws() raises:
    var ss = ShuffleSplit(n_splits=5, test_size=0.5)
    var splits = ss.split(20)

    var counts = List[Int](capacity=20)
    for _ in range(20):
        counts.append(0)

    for f in range(5):
        for i in range(len(splits[f].val_indices)):
            counts[splits[f].val_indices[i]] += 1

    var overlapping = 0
    for i in range(20):
        if counts[i] > 1:
            overlapping += 1
    assert_true(overlapping > 0)


def test_shuffle_split_draws_are_independent() raises:
    var ss = ShuffleSplit(n_splits=3, test_size=0.3)
    var splits = ss.split(20)

    var differs = False
    for i in range(len(splits[0].val_indices)):
        if splits[0].val_indices[i] != splits[1].val_indices[i]:
            differs = True
    assert_true(differs)


def test_shuffle_split_reproducible_with_same_seed() raises:
    var a = ShuffleSplit(n_splits=4, test_size=0.25, random_state=7)
    var b = ShuffleSplit(n_splits=4, test_size=0.25, random_state=7)

    var sa = a.split(16)
    var sb = b.split(16)

    for f in range(4):
        assert_equal(len(sa[f].val_indices), len(sb[f].val_indices))
        for i in range(len(sa[f].val_indices)):
            assert_equal(sa[f].val_indices[i], sb[f].val_indices[i])
        for i in range(len(sa[f].train_indices)):
            assert_equal(sa[f].train_indices[i], sb[f].train_indices[i])


def test_shuffle_split_different_seeds_produce_different_splits() raises:
    var a = ShuffleSplit(n_splits=3, test_size=0.3, random_state=1)
    var b = ShuffleSplit(n_splits=3, test_size=0.3, random_state=999)

    var sa = a.split(30)
    var sb = b.split(30)

    var differs = False
    for f in range(3):
        for i in range(len(sa[f].val_indices)):
            if sa[f].val_indices[i] != sb[f].val_indices[i]:
                differs = True
    assert_true(differs)


def test_shuffle_split_single_split_allowed() raises:
    var ss = ShuffleSplit(n_splits=1, test_size=0.2)
    assert_equal(ss.get_n_splits(), 1)

    var splits = ss.split(10)
    assert_equal(len(splits), 1)
    _assert_split_shape(splits, 10, 8, 2)


def test_shuffle_split_large_scale_invariants() raises:
    var ss = ShuffleSplit(n_splits=5, test_size=0.2)
    var splits = ss.split(1000)

    assert_equal(len(splits), 5)
    _assert_split_shape(splits, 1000, 800, 200)


def test_shuffle_split_is_stateless_across_calls() raises:
    var ss = ShuffleSplit(n_splits=3, test_size=0.25)

    var first = ss.split(20)
    var other = ss.split(40)
    var again = ss.split(20)

    assert_equal(len(other[0].val_indices), 10)
    for f in range(3):
        assert_equal(len(first[f].val_indices), len(again[f].val_indices))
        for i in range(len(first[f].val_indices)):
            assert_equal(first[f].val_indices[i], again[f].val_indices[i])

    assert_equal(ss.test_size, 0.25)
    assert_equal(ss.train_size, 0.0)


def test_shuffle_split_matrix_overload_matches_count() raises:
    var X = Matrix[DType.float64](20, 3, 1.0)
    var ss = ShuffleSplit(n_splits=3, test_size=0.25)

    var from_matrix = ss.split(X)
    var from_count = ss.split(20)

    assert_equal(len(from_matrix), len(from_count))
    for f in range(len(from_matrix)):
        assert_equal(
            len(from_matrix[f].val_indices), len(from_count[f].val_indices)
        )
        assert_equal(
            len(from_matrix[f].train_indices), len(from_count[f].train_indices)
        )
        for i in range(len(from_count[f].val_indices)):
            assert_equal(
                from_matrix[f].val_indices[i], from_count[f].val_indices[i]
            )
        for i in range(len(from_count[f].train_indices)):
            assert_equal(
                from_matrix[f].train_indices[i], from_count[f].train_indices[i]
            )


def test_shuffle_split_matrix_overload_dtype_flexibility() raises:
    var X32 = Matrix[DType.float32](20, 2, 1.0)
    var Xi = Matrix[DType.int32](20, 2, 1)
    var ss = ShuffleSplit(n_splits=3, test_size=0.25)

    assert_equal(len(ss.split(X32)), 3)
    assert_equal(len(ss.split(Xi)), 3)


def test_shuffle_split_matrix_overload_rejects_empty() raises:
    var ss = ShuffleSplit(n_splits=3, test_size=0.25)

    var caught_rows = False
    try:
        var empty = Matrix[DType.float64](0, 3, 0)
        var _ = ss.split(empty)
    except:
        caught_rows = True
    assert_true(caught_rows)

    var caught_cols = False
    try:
        var no_cols = Matrix[DType.float64](20, 0, 0)
        var _ = ss.split(no_cols)
    except:
        caught_cols = True
    assert_true(caught_cols)


def test_shuffle_split_constructor_error_handling() raises:
    var caught_zero_splits = False
    try:
        var _ = ShuffleSplit(n_splits=0)
    except:
        caught_zero_splits = True
    assert_true(caught_zero_splits)

    var caught_neg_splits = False
    try:
        var _ = ShuffleSplit(n_splits=-2)
    except:
        caught_neg_splits = True
    assert_true(caught_neg_splits)

    var caught_test_zero = False
    try:
        var _ = ShuffleSplit(test_size=0.0)
    except:
        caught_test_zero = True
    assert_true(caught_test_zero)

    var caught_test_one = False
    try:
        var _ = ShuffleSplit(test_size=1.0)
    except:
        caught_test_one = True
    assert_true(caught_test_one)

    var caught_test_neg = False
    try:
        var _ = ShuffleSplit(test_size=-0.1)
    except:
        caught_test_neg = True
    assert_true(caught_test_neg)

    var caught_train_neg = False
    try:
        var _ = ShuffleSplit(test_size=0.2, train_size=-0.1)
    except:
        caught_train_neg = True
    assert_true(caught_train_neg)

    var caught_train_one = False
    try:
        var _ = ShuffleSplit(test_size=0.2, train_size=1.0)
    except:
        caught_train_one = True
    assert_true(caught_train_one)

    var caught_sum = False
    try:
        var _ = ShuffleSplit(test_size=0.6, train_size=0.7)
    except:
        caught_sum = True
    assert_true(caught_sum)

    var ok = ShuffleSplit(n_splits=1, test_size=0.5, train_size=0.5)
    assert_equal(ok.get_n_splits(), 1)


def test_shuffle_split_rejects_nonpositive_sample_counts() raises:
    var ss = ShuffleSplit(n_splits=3, test_size=0.25)

    var caught_zero = False
    try:
        var _ = ss.split(0)
    except:
        caught_zero = True
    assert_true(caught_zero)

    var caught_neg = False
    try:
        var _ = ss.split(-5)
    except:
        caught_neg = True
    assert_true(caught_neg)


def test_shuffle_split_rejects_empty_train_set() raises:
    var ss = ShuffleSplit(n_splits=2, test_size=0.9)

    var caught = False
    try:
        var _ = ss.split(5)
    except:
        caught = True
    assert_true(caught)

    var tiny = ShuffleSplit(n_splits=1, test_size=0.6)
    var caught_tiny = False
    try:
        var _ = tiny.split(1)
    except:
        caught_tiny = True
    assert_true(caught_tiny)

    assert_equal(len(ss.split(20)[0].train_indices), 2)


def test_shuffle_split_rejects_rounded_total_exceeding_samples() raises:
    var ss = ShuffleSplit(n_splits=1, test_size=0.28, train_size=0.72)

    var caught = False
    try:
        var _ = ss.split(25)
    except:
        caught = True
    assert_true(caught)

    var splits = ss.split(10)
    assert_equal(len(splits[0].val_indices), 3)
    assert_equal(len(splits[0].train_indices), 7)


def test_shuffle_split_cross_val_score_integration() raises:
    var X = Matrix[DType.float64](40, 1, 0)
    var y = List[Scalar[DType.float64]](capacity=40)
    for i in range(40):
        X[i, 0] = Float64(i)
        y.append(Float64(3 * i - 4))

    var ss = ShuffleSplit(n_splits=3, test_size=0.25)
    var splits = ss.split(X)
    var model = LinearRegression[DType.float64]()

    var scores = cross_val_score(model, X, y, splits, scoring="r2")
    assert_equal(len(scores), 3)
    for f in range(3):
        assert_true(scores[f] > 0.99)


def _count_label[
    target_dtype: DType
](
    y: List[Scalar[target_dtype]],
    indices: List[Int],
    label: Scalar[target_dtype],
) -> Int:
    var total = 0
    for i in range(len(indices)):
        if y[indices[i]] == label:
            total += 1
    return total


def _assert_sorted_unique(indices: List[Int]) raises:
    for i in range(1, len(indices)):
        assert_true(indices[i - 1] < indices[i])


def _binary_labels(n: Int, n_first: Int) -> List[Scalar[DType.int32]]:
    var y = List[Scalar[DType.int32]](capacity=n)
    for i in range(n):
        y.append(Int32(1 if i < n_first else 0))
    return y^


def test_stratified_shuffle_split_default_parameters() raises:
    var sss = StratifiedShuffleSplit()
    assert_equal(sss.get_n_splits(), 10)

    var y = _binary_labels(100, 30)
    var splits = sss.split(y)

    assert_equal(len(splits), 10)
    _assert_split_shape(splits, 100, 90, 10)

    for s in range(10):
        assert_equal(_count_label(y, splits[s].val_indices, Int32(1)), 3)
        assert_equal(_count_label(y, splits[s].val_indices, Int32(0)), 7)
        assert_equal(_count_label(y, splits[s].train_indices, Int32(1)), 27)
        assert_equal(_count_label(y, splits[s].train_indices, Int32(0)), 63)


def test_stratified_shuffle_split_preserves_class_ratio() raises:
    var y = _binary_labels(20, 4)
    var sss = StratifiedShuffleSplit(n_splits=3, test_size=0.25)
    var splits = sss.split(y)

    assert_equal(len(splits), 3)
    _assert_split_shape(splits, 20, 15, 5)

    for s in range(3):
        assert_equal(_count_label(y, splits[s].val_indices, Int32(1)), 1)
        assert_equal(_count_label(y, splits[s].val_indices, Int32(0)), 4)
        assert_equal(_count_label(y, splits[s].train_indices, Int32(1)), 3)
        assert_equal(_count_label(y, splits[s].train_indices, Int32(0)), 12)


def test_stratified_shuffle_split_balanced_two_class() raises:
    var y = _binary_labels(100, 50)
    var sss = StratifiedShuffleSplit(n_splits=4, test_size=0.2)
    var splits = sss.split(y)

    _assert_split_shape(splits, 100, 80, 20)
    for s in range(4):
        assert_equal(_count_label(y, splits[s].val_indices, Int32(1)), 10)
        assert_equal(_count_label(y, splits[s].val_indices, Int32(0)), 10)
        assert_equal(_count_label(y, splits[s].train_indices, Int32(1)), 40)
        assert_equal(_count_label(y, splits[s].train_indices, Int32(0)), 40)


def test_stratified_shuffle_split_three_classes_one_per_draw() raises:
    var y = List[Scalar[DType.int32]](capacity=12)
    for i in range(12):
        y.append(Int32(i % 3))

    var sss = StratifiedShuffleSplit(n_splits=2, test_size=0.25)
    var splits = sss.split(y)

    _assert_split_shape(splits, 12, 9, 3)
    for s in range(2):
        for c in range(3):
            assert_equal(_count_label(y, splits[s].val_indices, Int32(c)), 1)
            assert_equal(_count_label(y, splits[s].train_indices, Int32(c)), 3)


def test_stratified_shuffle_split_severe_class_imbalance() raises:
    var y = _binary_labels(100, 10)
    var sss = StratifiedShuffleSplit(n_splits=3, test_size=0.1)
    var splits = sss.split(y)

    _assert_split_shape(splits, 100, 90, 10)
    for s in range(3):
        assert_equal(_count_label(y, splits[s].val_indices, Int32(1)), 1)
        assert_equal(_count_label(y, splits[s].train_indices, Int32(1)), 9)


def test_stratified_shuffle_split_indices_sorted_and_disjoint() raises:
    var y = _binary_labels(30, 12)
    var sss = StratifiedShuffleSplit(n_splits=4, test_size=0.3)
    var splits = sss.split(y)

    for s in range(4):
        _assert_sorted_unique(splits[s].train_indices)
        _assert_sorted_unique(splits[s].val_indices)

    _assert_split_shape(splits, 30, 21, 9)


def test_stratified_shuffle_split_no_class_overflow_when_sizes_fill_data() raises:
    var y = _binary_labels(10, 3)
    var sss = StratifiedShuffleSplit(n_splits=2, test_size=0.5, train_size=0.5)
    var splits = sss.split(y)

    _assert_split_shape(splits, 10, 5, 5)
    for s in range(2):
        assert_equal(_count_label(y, splits[s].train_indices, Int32(1)), 2)
        assert_equal(_count_label(y, splits[s].val_indices, Int32(1)), 1)
        assert_equal(_count_label(y, splits[s].train_indices, Int32(0)), 3)
        assert_equal(_count_label(y, splits[s].val_indices, Int32(0)), 4)


def test_stratified_shuffle_split_covers_all_samples_when_train_size_unset() raises:
    var y = _binary_labels(20, 8)
    var sss = StratifiedShuffleSplit(n_splits=3, test_size=0.25)
    var splits = sss.split(y)

    for s in range(3):
        assert_equal(
            len(splits[s].train_indices) + len(splits[s].val_indices), 20
        )

        var seen = List[Bool](capacity=20)
        for _ in range(20):
            seen.append(False)
        for i in range(len(splits[s].train_indices)):
            seen[splits[s].train_indices[i]] = True
        for i in range(len(splits[s].val_indices)):
            seen[splits[s].val_indices[i]] = True
        for i in range(20):
            assert_true(seen[i])


def test_stratified_shuffle_split_train_size_leaves_samples_unused() raises:
    var y = _binary_labels(10, 5)
    var sss = StratifiedShuffleSplit(n_splits=3, test_size=0.2, train_size=0.35)
    var splits = sss.split(y)

    _assert_split_shape(splits, 10, 3, 2)
    for s in range(3):
        assert_true(
            len(splits[s].train_indices) + len(splits[s].val_indices) < 10
        )


def test_stratified_shuffle_split_ceil_rounding_for_test_size() raises:
    var y12 = _binary_labels(12, 6)
    var tenth = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
    assert_equal(len(tenth.split(y12)[0].val_indices), 2)

    var y10 = _binary_labels(10, 5)
    var quarter = StratifiedShuffleSplit(n_splits=1, test_size=0.25)
    assert_equal(len(quarter.split(y10)[0].val_indices), 3)

    var exact = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    assert_equal(len(exact.split(y10)[0].val_indices), 2)

    var y20 = _binary_labels(20, 10)
    var odd = StratifiedShuffleSplit(n_splits=1, test_size=0.11)
    assert_equal(len(odd.split(y20)[0].val_indices), 3)


def test_stratified_shuffle_split_floor_rounding_for_train_size() raises:
    var y = _binary_labels(10, 5)
    var sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, train_size=0.35)
    var splits = sss.split(y)

    assert_equal(len(splits), 2)
    for s in range(2):
        assert_equal(len(splits[s].val_indices), 2)
        assert_equal(len(splits[s].train_indices), 3)


def test_stratified_shuffle_split_test_sets_may_overlap_across_draws() raises:
    var y = _binary_labels(40, 20)
    var sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5)
    var splits = sss.split(y)

    var counts = List[Int](capacity=40)
    for _ in range(40):
        counts.append(0)

    for s in range(5):
        for i in range(len(splits[s].val_indices)):
            counts[splits[s].val_indices[i]] += 1

    var overlapping = 0
    for i in range(40):
        if counts[i] > 1:
            overlapping += 1
    assert_true(overlapping > 0)


def test_stratified_shuffle_split_draws_are_independent() raises:
    var y = _binary_labels(40, 16)
    var sss = StratifiedShuffleSplit(n_splits=3, test_size=0.3)
    var splits = sss.split(y)

    var differs = False
    for i in range(len(splits[0].val_indices)):
        if splits[0].val_indices[i] != splits[1].val_indices[i]:
            differs = True
    assert_true(differs)


def test_stratified_shuffle_split_reproducible_with_same_seed() raises:
    var y = _binary_labels(24, 8)
    var a = StratifiedShuffleSplit(n_splits=4, test_size=0.25, random_state=7)
    var b = StratifiedShuffleSplit(n_splits=4, test_size=0.25, random_state=7)

    var sa = a.split(y)
    var sb = b.split(y)

    for s in range(4):
        assert_equal(len(sa[s].val_indices), len(sb[s].val_indices))
        for i in range(len(sa[s].val_indices)):
            assert_equal(sa[s].val_indices[i], sb[s].val_indices[i])
        for i in range(len(sa[s].train_indices)):
            assert_equal(sa[s].train_indices[i], sb[s].train_indices[i])


def test_stratified_shuffle_split_different_seeds_produce_different_splits() raises:
    var y = _binary_labels(30, 12)
    var a = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=1)
    var b = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=999)

    var sa = a.split(y)
    var sb = b.split(y)

    var differs = False
    for s in range(3):
        for i in range(len(sa[s].val_indices)):
            if sa[s].val_indices[i] != sb[s].val_indices[i]:
                differs = True
    assert_true(differs)


def test_stratified_shuffle_split_single_split_allowed() raises:
    var y = _binary_labels(10, 4)
    var sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    assert_equal(sss.get_n_splits(), 1)

    var splits = sss.split(y)
    assert_equal(len(splits), 1)
    _assert_split_shape(splits, 10, 8, 2)


def test_stratified_shuffle_split_single_class_dataset() raises:
    var y = List[Scalar[DType.int32]](capacity=10)
    for _ in range(10):
        y.append(Int32(7))

    var sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2)
    var splits = sss.split(y)

    _assert_split_shape(splits, 10, 8, 2)
    for s in range(2):
        assert_equal(_count_label(y, splits[s].val_indices, Int32(7)), 2)


def test_stratified_shuffle_split_float_labels() raises:
    var y = List[Scalar[DType.float64]](capacity=20)
    for i in range(20):
        y.append(1.0 if i < 8 else 0.0)

    var sss = StratifiedShuffleSplit(n_splits=3, test_size=0.25)
    var splits = sss.split(y)

    _assert_split_shape(splits, 20, 15, 5)
    for s in range(3):
        assert_equal(_count_label(y, splits[s].val_indices, Float64(1.0)), 2)
        assert_equal(_count_label(y, splits[s].val_indices, Float64(0.0)), 3)


def test_stratified_shuffle_split_is_stateless_across_calls() raises:
    var y20 = _binary_labels(20, 8)
    var y40 = _binary_labels(40, 16)
    var sss = StratifiedShuffleSplit(n_splits=3, test_size=0.25)

    var first = sss.split(y20)
    var other = sss.split(y40)
    var again = sss.split(y20)

    assert_equal(len(other[0].val_indices), 10)
    for s in range(3):
        assert_equal(len(first[s].val_indices), len(again[s].val_indices))
        for i in range(len(first[s].val_indices)):
            assert_equal(first[s].val_indices[i], again[s].val_indices[i])

    assert_equal(sss.test_size, 0.25)
    assert_equal(sss.train_size, 0.0)
    assert_equal(sss.n_splits, 3)


def test_stratified_shuffle_split_large_scale_invariants() raises:
    var y = _binary_labels(1000, 200)
    var sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
    var splits = sss.split(y)

    assert_equal(len(splits), 5)
    _assert_split_shape(splits, 1000, 800, 200)
    for s in range(5):
        assert_equal(_count_label(y, splits[s].val_indices, Int32(1)), 40)
        assert_equal(_count_label(y, splits[s].train_indices, Int32(1)), 160)


def test_stratified_shuffle_split_constructor_error_handling() raises:
    var caught_zero_splits = False
    try:
        var _ = StratifiedShuffleSplit(n_splits=0)
    except:
        caught_zero_splits = True
    assert_true(caught_zero_splits)

    var caught_neg_splits = False
    try:
        var _ = StratifiedShuffleSplit(n_splits=-2)
    except:
        caught_neg_splits = True
    assert_true(caught_neg_splits)

    var caught_test_zero = False
    try:
        var _ = StratifiedShuffleSplit(test_size=0.0)
    except:
        caught_test_zero = True
    assert_true(caught_test_zero)

    var caught_test_one = False
    try:
        var _ = StratifiedShuffleSplit(test_size=1.0)
    except:
        caught_test_one = True
    assert_true(caught_test_one)

    var caught_test_neg = False
    try:
        var _ = StratifiedShuffleSplit(test_size=-0.1)
    except:
        caught_test_neg = True
    assert_true(caught_test_neg)

    var caught_train_neg = False
    try:
        var _ = StratifiedShuffleSplit(test_size=0.2, train_size=-0.1)
    except:
        caught_train_neg = True
    assert_true(caught_train_neg)

    var caught_train_one = False
    try:
        var _ = StratifiedShuffleSplit(test_size=0.2, train_size=1.0)
    except:
        caught_train_one = True
    assert_true(caught_train_one)

    var caught_sum = False
    try:
        var _ = StratifiedShuffleSplit(test_size=0.6, train_size=0.7)
    except:
        caught_sum = True
    assert_true(caught_sum)

    var ok = StratifiedShuffleSplit(n_splits=1, test_size=0.5, train_size=0.5)
    assert_equal(ok.get_n_splits(), 1)


def test_stratified_shuffle_split_rejects_empty_labels() raises:
    var sss = StratifiedShuffleSplit(n_splits=2, test_size=0.25)
    var empty = List[Scalar[DType.int32]]()

    var caught = False
    try:
        var _ = sss.split(empty)
    except:
        caught = True
    assert_true(caught)


def test_stratified_shuffle_split_rejects_test_smaller_than_class_count() raises:
    var y = List[Scalar[DType.int32]](capacity=10)
    for i in range(10):
        y.append(Int32(i % 4))

    var sss = StratifiedShuffleSplit(n_splits=2, test_size=0.1)
    var caught = False
    try:
        var _ = sss.split(y)
    except:
        caught = True
    assert_true(caught)


def test_stratified_shuffle_split_rejects_train_smaller_than_class_count() raises:
    var y = List[Scalar[DType.int32]](capacity=10)
    for i in range(10):
        y.append(Int32(i % 4))

    var sss = StratifiedShuffleSplit(n_splits=2, test_size=0.7, train_size=0.2)
    var caught = False
    try:
        var _ = sss.split(y)
    except:
        caught = True
    assert_true(caught)


def test_stratified_shuffle_split_rejects_empty_train_set() raises:
    var y = _binary_labels(5, 2)
    var sss = StratifiedShuffleSplit(n_splits=1, test_size=0.9)

    var caught = False
    try:
        var _ = sss.split(y)
    except:
        caught = True
    assert_true(caught)


def test_stratified_shuffle_split_rejects_rounded_total_exceeding_samples() raises:
    var sss = StratifiedShuffleSplit(
        n_splits=1, test_size=0.28, train_size=0.72
    )

    var y25 = _binary_labels(25, 10)
    var caught = False
    try:
        var _ = sss.split(y25)
    except:
        caught = True
    assert_true(caught)

    var y10 = _binary_labels(10, 5)
    var splits = sss.split(y10)
    assert_equal(len(splits[0].val_indices), 3)
    assert_equal(len(splits[0].train_indices), 7)


def test_stratified_shuffle_split_matrix_overload_matches_labels_only() raises:
    var X = Matrix[DType.float64](20, 3, 1.0)
    var y = _binary_labels(20, 8)
    var sss = StratifiedShuffleSplit(n_splits=3, test_size=0.25)

    var from_matrix = sss.split(X, y)
    var from_labels = sss.split(y)

    assert_equal(len(from_matrix), len(from_labels))
    for s in range(len(from_matrix)):
        for i in range(len(from_labels[s].val_indices)):
            assert_equal(
                from_matrix[s].val_indices[i], from_labels[s].val_indices[i]
            )
        for i in range(len(from_labels[s].train_indices)):
            assert_equal(
                from_matrix[s].train_indices[i],
                from_labels[s].train_indices[i],
            )


def test_stratified_shuffle_split_matrix_overload_dtype_flexibility() raises:
    var y = _binary_labels(20, 8)
    var sss = StratifiedShuffleSplit(n_splits=3, test_size=0.25)

    var X32 = Matrix[DType.float32](20, 2, 1.0)
    var Xi = Matrix[DType.int32](20, 2, 1)

    assert_equal(len(sss.split(X32, y)), 3)
    assert_equal(len(sss.split(Xi, y)), 3)


def test_stratified_shuffle_split_matrix_overload_rejects_length_mismatch() raises:
    var X = Matrix[DType.float64](20, 3, 1.0)
    var y = _binary_labels(19, 8)
    var sss = StratifiedShuffleSplit(n_splits=2, test_size=0.25)

    var caught = False
    try:
        var _ = sss.split(X, y)
    except:
        caught = True
    assert_true(caught)


def test_stratified_shuffle_split_matrix_overload_rejects_empty() raises:
    var sss = StratifiedShuffleSplit(n_splits=2, test_size=0.25)

    var caught_rows = False
    try:
        var empty_X = Matrix[DType.float64](0, 3, 0)
        var empty_y = List[Scalar[DType.int32]]()
        var _ = sss.split(empty_X, empty_y)
    except:
        caught_rows = True
    assert_true(caught_rows)

    var caught_cols = False
    try:
        var no_cols = Matrix[DType.float64](20, 0, 0)
        var y = _binary_labels(20, 8)
        var _ = sss.split(no_cols, y)
    except:
        caught_cols = True
    assert_true(caught_cols)


def test_stratified_shuffle_split_cross_val_score_integration() raises:
    var X = Matrix[DType.float64](40, 1, 0)
    var y = List[Scalar[DType.int32]](capacity=40)
    for i in range(40):
        X[i, 0] = (Float64(i) - 19.5) * 0.2
        y.append(Int32(0 if i < 20 else 1))

    var sss = StratifiedShuffleSplit(n_splits=3, test_size=0.25)
    var splits = sss.split(X, y)
    var model = LogisticRegression[DType.float64](max_iter=200)

    var scores = cross_val_score(model, X, y, splits, scoring="accuracy")
    assert_equal(len(scores), 3)
    for s in range(3):
        assert_true(scores[s] > 0.9)


def _linear_dataset(
    n: Int,
) -> Tuple[Matrix[DType.float64], List[Scalar[DType.float64]]]:
    var X = Matrix[DType.float64](n, 1, 0)
    var y = List[Scalar[DType.float64]](capacity=n)
    for i in range(n):
        X[i, 0] = Float64(i)
        y.append(Float64(3 * i - 4))
    return (X^, y^)


def _separable_dataset(
    n: Int,
) -> Tuple[Matrix[DType.float64], List[Scalar[DType.int32]]]:
    var X = Matrix[DType.float64](n, 1, 0)
    var y = List[Scalar[DType.int32]](capacity=n)
    var half = n // 2
    for i in range(n):
        if i < half:
            X[i, 0] = -2.0 + Float64(i) * 0.05
            y.append(Int32(0))
        else:
            X[i, 0] = 2.0 + Float64(i - half) * 0.05
            y.append(Int32(1))
    return (X^, y^)


def _manual_splits(
    train_sets: List[List[Int]], val_sets: List[List[Int]]
) -> List[Split]:
    var splits = List[Split]()
    for s in range(len(train_sets)):
        splits.append(Split(train_sets[s].copy(), val_sets[s].copy()))
    return splits^


def test_cross_val_predict_regression_one_prediction_per_row() raises:
    var data = _linear_dataset(20)
    var model = LinearRegression[DType.float64]()

    var preds = cross_val_predict(model, data[0], data[1], cv=5)

    assert_equal(len(preds), 20)
    for i in range(20):
        assert_true(abs(preds[i] - data[1][i]) < 1e-6)


def test_cross_val_predict_regression_matches_manual_fold_fits() raises:
    var X = Matrix[DType.float64](12, 1, 0)
    var y = List[Scalar[DType.float64]](capacity=12)
    for i in range(12):
        X[i, 0] = Float64(i)
        y.append(Float64(i * i))

    var kf = KFold(n_splits=3)
    var splits = kf.split(12)
    var model = LinearRegression[DType.float64]()

    var preds = cross_val_predict(model, X, y, splits)
    assert_equal(len(preds), 12)

    for s in range(3):
        var X_train = take_rows(X, splits[s].train_indices)
        var y_train = take_elements(y, splits[s].train_indices)
        var X_val = take_rows(X, splits[s].val_indices)

        var fold_model = LinearRegression[DType.float64]()
        fold_model.fit(X_train, y_train)
        var fold_preds = fold_model.predict(X_val)

        for i in range(len(splits[s].val_indices)):
            assert_equal(preds[splits[s].val_indices[i]], fold_preds[i])


def test_cross_val_predict_regression_preserves_row_order() raises:
    var data = _linear_dataset(20)
    var model = LinearRegression[DType.float64]()

    var preds = cross_val_predict(model, data[0], data[1], cv=5)

    for i in range(19):
        assert_true(preds[i] < preds[i + 1])


def test_cross_val_predict_regression_cv_overload_matches_kfold() raises:
    var data = _linear_dataset(16)
    var model = LinearRegression[DType.float64]()

    var kf = KFold(n_splits=4)
    var splits = kf.split(16)

    var from_cv = cross_val_predict(model, data[0], data[1], cv=4)
    var from_splits = cross_val_predict(model, data[0], data[1], splits)

    assert_equal(len(from_cv), len(from_splits))
    for i in range(16):
        assert_equal(from_cv[i], from_splits[i])


def test_cross_val_predict_regression_accepts_stratified_kfold_splits() raises:
    var X = Matrix[DType.float64](20, 1, 0)
    var y = List[Scalar[DType.float64]](capacity=20)
    var labels = List[Scalar[DType.int32]](capacity=20)
    for i in range(20):
        X[i, 0] = Float64(i)
        y.append(Float64(2 * i + 1))
        labels.append(Int32(i % 2))

    var skf = StratifiedKFold(n_splits=4)
    var splits = skf.split(labels)
    var model = LinearRegression[DType.float64]()

    var preds = cross_val_predict(model, X, y, splits)
    assert_equal(len(preds), 20)
    for i in range(20):
        assert_true(abs(preds[i] - y[i]) < 1e-6)


def test_cross_val_predict_regression_dtype_flexibility() raises:
    var X = Matrix[DType.float32](12, 1, 0)
    var y = List[Scalar[DType.float32]](capacity=12)
    for i in range(12):
        X[i, 0] = Float32(i)
        y.append(Float32(2 * i + 3))

    var model = LinearRegression[DType.float32]()
    var preds = cross_val_predict(model, X, y, cv=3)

    assert_equal(len(preds), 12)
    for i in range(12):
        assert_true(abs(preds[i] - y[i]) < 1e-2)


def test_cross_val_predict_classification_one_label_per_row() raises:
    var data = _separable_dataset(20)
    var model = LogisticRegression[DType.float64](max_iter=200)

    var preds = cross_val_predict(model, data[0], data[1], cv=4)

    assert_equal(len(preds), 20)
    for i in range(20):
        assert_true(preds[i] == 0 or preds[i] == 1)


def test_cross_val_predict_classification_accurate_on_separable_data() raises:
    var data = _separable_dataset(40)
    var model = LogisticRegression[DType.float64](max_iter=200)

    var preds = cross_val_predict(model, data[0], data[1], cv=4)

    var correct = 0
    for i in range(40):
        if preds[i] == Int(data[1][i]):
            correct += 1
    assert_true(Float64(correct) / 40.0 > 0.9)


def test_cross_val_predict_classification_cv_overload_matches_stratified() raises:
    var data = _separable_dataset(24)
    var model = LogisticRegression[DType.float64](max_iter=200)

    var skf = StratifiedKFold(n_splits=3)
    var splits = skf.split(data[0], data[1])

    var from_cv = cross_val_predict(model, data[0], data[1], cv=3)
    var from_splits = cross_val_predict(model, data[0], data[1], splits)

    assert_equal(len(from_cv), len(from_splits))
    for i in range(24):
        assert_equal(from_cv[i], from_splits[i])


def test_cross_val_predict_classification_matches_manual_fold_fits() raises:
    var data = _separable_dataset(18)
    var skf = StratifiedKFold(n_splits=3)
    var splits = skf.split(data[1])
    var model = LogisticRegression[DType.float64](max_iter=200)

    var preds = cross_val_predict(model, data[0], data[1], splits)

    for s in range(3):
        var X_train = take_rows(data[0], splits[s].train_indices)
        var y_train = take_elements(data[1], splits[s].train_indices)
        var X_val = take_rows(data[0], splits[s].val_indices)

        var fold_model = LogisticRegression[DType.float64](max_iter=200)
        fold_model.fit(X_train, y_train)
        var fold_preds = fold_model.predict(X_val)

        for i in range(len(splits[s].val_indices)):
            assert_equal(preds[splits[s].val_indices[i]], fold_preds[i])


def test_cross_val_predict_rejects_overlapping_validation_folds() raises:
    var data = _linear_dataset(4)
    var model = LinearRegression[DType.float64]()

    var train_sets = List[List[Int]]()
    var val_sets = List[List[Int]]()
    train_sets.append([2, 3])
    val_sets.append([0, 1])
    train_sets.append([0, 3])
    val_sets.append([1, 2])
    var splits = _manual_splits(train_sets, val_sets)

    var caught = False
    try:
        var _ = cross_val_predict(model, data[0], data[1], splits)
    except:
        caught = True
    assert_true(caught)


def test_cross_val_predict_rejects_uncovered_samples() raises:
    var data = _linear_dataset(4)
    var model = LinearRegression[DType.float64]()

    var train_sets = List[List[Int]]()
    var val_sets = List[List[Int]]()
    train_sets.append([2, 3])
    val_sets.append([0, 1])
    train_sets.append([0, 1])
    val_sets.append([2])
    var splits = _manual_splits(train_sets, val_sets)

    var caught = False
    try:
        var _ = cross_val_predict(model, data[0], data[1], splits)
    except:
        caught = True
    assert_true(caught)


def test_cross_val_predict_rejects_out_of_bounds_validation_index() raises:
    var data = _linear_dataset(4)
    var model = LinearRegression[DType.float64]()

    var train_sets = List[List[Int]]()
    var val_sets = List[List[Int]]()
    train_sets.append([2, 3])
    val_sets.append([0, 99])
    var splits = _manual_splits(train_sets, val_sets)

    var caught = False
    try:
        var _ = cross_val_predict(model, data[0], data[1], splits)
    except:
        caught = True
    assert_true(caught)


def test_cross_val_predict_rejects_empty_split_list() raises:
    var data = _linear_dataset(6)
    var model = LinearRegression[DType.float64]()
    var splits = List[Split]()

    var caught = False
    try:
        var _ = cross_val_predict(model, data[0], data[1], splits)
    except:
        caught = True
    assert_true(caught)


def test_cross_val_predict_rejects_empty_train_fold() raises:
    var data = _linear_dataset(4)
    var model = LinearRegression[DType.float64]()

    var train_sets = List[List[Int]]()
    var val_sets = List[List[Int]]()
    train_sets.append(List[Int]())
    val_sets.append([0, 1, 2, 3])
    var splits = _manual_splits(train_sets, val_sets)

    var caught = False
    try:
        var _ = cross_val_predict(model, data[0], data[1], splits)
    except:
        caught = True
    assert_true(caught)


def test_cross_val_predict_rejects_empty_validation_fold() raises:
    var data = _linear_dataset(4)
    var model = LinearRegression[DType.float64]()

    var train_sets = List[List[Int]]()
    var val_sets = List[List[Int]]()
    train_sets.append([0, 1, 2, 3])
    val_sets.append(List[Int]())
    var splits = _manual_splits(train_sets, val_sets)

    var caught = False
    try:
        var _ = cross_val_predict(model, data[0], data[1], splits)
    except:
        caught = True
    assert_true(caught)


def test_cross_val_predict_rejects_shuffle_split_folds() raises:
    var data = _linear_dataset(20)
    var model = LinearRegression[DType.float64]()

    var ss = ShuffleSplit(n_splits=3, test_size=0.3)
    var splits = ss.split(20)

    var caught = False
    try:
        var _ = cross_val_predict(model, data[0], data[1], splits)
    except:
        caught = True
    assert_true(caught)


def test_cross_val_predict_rejects_time_series_split_folds() raises:
    var data = _linear_dataset(20)
    var model = LinearRegression[DType.float64]()

    var tss = TimeSeriesSplit(n_splits=3)
    var splits = tss.split(20)

    var caught = False
    try:
        var _ = cross_val_predict(model, data[0], data[1], splits)
    except:
        caught = True
    assert_true(caught)


def test_cross_val_predict_rejects_mismatched_x_and_y() raises:
    var X = Matrix[DType.float64](20, 1, 0)
    var y = List[Scalar[DType.float64]](capacity=19)
    for i in range(19):
        y.append(Float64(i))

    var model = LinearRegression[DType.float64]()
    var caught = False
    try:
        var _ = cross_val_predict(model, X, y, cv=4)
    except:
        caught = True
    assert_true(caught)


def test_cross_val_predict_classifier_rejects_overlapping_folds() raises:
    var data = _separable_dataset(20)
    var model = LogisticRegression[DType.float64](max_iter=100)

    var ss = ShuffleSplit(n_splits=3, test_size=0.3)
    var splits = ss.split(20)

    var caught = False
    try:
        var _ = cross_val_predict(model, data[0], data[1], splits)
    except:
        caught = True
    assert_true(caught)


def test_cross_val_predict_classifier_rejects_empty_split_list() raises:
    var data = _separable_dataset(20)
    var model = LogisticRegression[DType.float64](max_iter=100)
    var splits = List[Split]()

    var caught = False
    try:
        var _ = cross_val_predict(model, data[0], data[1], splits)
    except:
        caught = True
    assert_true(caught)


def test_cross_val_predict_leaves_estimator_unfitted() raises:
    var data = _linear_dataset(12)
    var model = LinearRegression[DType.float64]()

    var _ = cross_val_predict(model, data[0], data[1], cv=3)

    var caught = False
    try:
        var probe = Matrix[DType.float64](1, 1, 0)
        var _ = model.predict(probe)
    except:
        caught = True
    assert_true(caught)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
