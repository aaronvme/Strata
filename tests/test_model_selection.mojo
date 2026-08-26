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
    Split,
    cross_val_score,
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
