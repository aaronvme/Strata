from std.math import nan, inf
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_raises,
    assert_almost_equal,
)
from strata.core.matrix import Matrix
from strata.core.dataset import Dataset
from strata.base.estimator import (
    fit as fit_ds,
    predict as predict_ds,
)
from strata.cluster.minibatch_kmeans import MiniBatchKMeans
from strata.exceptions.errors import (
    NotFittedError,
    DimensionMismatchError,
    InvalidParameterError,
)


def test_minibatch_kmeans_basic_two_clusters() raises:
    var X = Matrix[DType.float64](6, 2, 0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 0.1
    X[1, 1] = -0.1
    X[2, 0] = -0.1
    X[2, 1] = 0.1

    X[3, 0] = 10.0
    X[3, 1] = 10.0
    X[4, 0] = 10.1
    X[4, 1] = 9.9
    X[5, 0] = 9.9
    X[5, 1] = 10.1

    var mbk = MiniBatchKMeans[DType.float64](
        n_clusters=2, batch_size=4, random_state=42
    )
    mbk.fit(X)

    assert_true(mbk.is_fitted)
    assert_equal(mbk.cluster_centers_.rows, 2)
    assert_equal(mbk.cluster_centers_.cols, 2)
    assert_equal(len(mbk.labels_), 6)

    # First three points belong to same cluster, last three to other
    assert_equal(mbk.labels_[0], mbk.labels_[1])
    assert_equal(mbk.labels_[0], mbk.labels_[2])
    assert_equal(mbk.labels_[3], mbk.labels_[4])
    assert_equal(mbk.labels_[3], mbk.labels_[5])
    assert_true(mbk.labels_[0] != mbk.labels_[3])


def test_minibatch_kmeans_streaming_partial_fit() raises:
    var mbk = MiniBatchKMeans[DType.float64](n_clusters=2, random_state=42)

    # Batch 1 around (0, 0)
    var b1 = Matrix[DType.float64](3, 2, 0)
    b1[0, 0] = 0.0
    b1[0, 1] = 0.0
    b1[1, 0] = 0.1
    b1[1, 1] = 0.1
    b1[2, 0] = 10.0
    b1[2, 1] = 10.0

    mbk.partial_fit(b1)
    assert_true(mbk.is_fitted)
    assert_equal(mbk.n_steps_, 1)
    assert_equal(len(mbk.counts_), 2)

    # Batch 2 around (10, 10)
    var b2 = Matrix[DType.float64](2, 2, 0)
    b2[0, 0] = 10.1
    b2[0, 1] = 9.9
    b2[1, 0] = 0.2
    b2[1, 1] = -0.1

    mbk.partial_fit(b2)
    assert_equal(mbk.n_steps_, 2)

    var preds = mbk.predict(b1)
    assert_equal(len(preds), 3)


def test_minibatch_kmeans_fit_predict() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 0.2
    X[1, 1] = 0.1
    X[2, 0] = 5.0
    X[2, 1] = 5.0
    X[3, 0] = 5.1
    X[3, 1] = 4.9

    var mbk = MiniBatchKMeans[DType.float64](n_clusters=2, random_state=123)
    var fit_preds = mbk.fit_predict(X)
    var preds = mbk.predict(X)

    assert_equal(len(fit_preds), 4)
    assert_equal(len(preds), 4)
    for i in range(4):
        assert_equal(fit_preds[i], preds[i])


def test_minibatch_kmeans_transform_and_fit_transform() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 0.0
    X[1, 1] = 0.0
    X[2, 0] = 10.0
    X[2, 1] = 10.0
    X[3, 0] = 10.0
    X[3, 1] = 10.0

    var mbk = MiniBatchKMeans[DType.float64](n_clusters=2, random_state=42)
    var X_trans = mbk.fit_transform(X)

    assert_equal(X_trans.rows, 4)
    assert_equal(X_trans.cols, 2)


def test_minibatch_kmeans_score() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 0.1
    X[1, 1] = 0.0
    X[2, 0] = 10.0
    X[2, 1] = 10.0
    X[3, 0] = 10.0
    X[3, 1] = 10.2

    var mbk = MiniBatchKMeans[DType.float64](n_clusters=2, random_state=42)
    mbk.fit(X)

    var s = mbk.score(X)
    assert_almost_equal(Float64(s), -Float64(mbk.inertia_), atol=1e-7)


def test_minibatch_kmeans_dataset_integration() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 1.0
    X[1, 0] = 1.1
    X[1, 1] = 1.1
    X[2, 0] = 9.0
    X[2, 1] = 9.0
    X[3, 0] = 9.1
    X[3, 1] = 9.1

    var y = List[Scalar[DType.int32]](capacity=4)
    y.append(0)
    y.append(0)
    y.append(1)
    y.append(1)

    var f_names = List[String](capacity=2)
    f_names.append("f1")
    f_names.append("f2")

    var t_names = List[String](capacity=2)
    t_names.append("c0")
    t_names.append("c1")

    var ds = Dataset[DType.float64, DType.int32](X^, y^, f_names^, t_names^)

    var mbk = MiniBatchKMeans[DType.float64](n_clusters=2, random_state=42)
    fit_ds(mbk, ds)
    assert_true(mbk.is_fitted)

    var preds = predict_ds(mbk, ds)
    assert_equal(len(preds), 4)


def test_minibatch_kmeans_copy_constructor() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 1.0
    X[1, 0] = 1.0
    X[1, 1] = 1.2
    X[2, 0] = 5.0
    X[2, 1] = 5.0
    X[3, 0] = 5.2
    X[3, 1] = 5.0

    var mbk1 = MiniBatchKMeans[DType.float64](n_clusters=2, random_state=42)
    mbk1.fit(X)

    var mbk2 = mbk1.copy()
    assert_true(mbk2.is_fitted)
    assert_equal(mbk2.cluster_centers_.rows, 2)
    assert_almost_equal(
        Float64(mbk1.inertia_), Float64(mbk2.inertia_), atol=1e-7
    )

    # Mutate mbk1
    var X_new = Matrix[DType.float64](2, 2, 100.0)
    mbk1.fit(X_new)

    # mbk2 should remain isolated
    assert_almost_equal(Float64(mbk2.cluster_centers_[0, 0]), 1.0, atol=1.0)


def test_minibatch_kmeans_reproducibility() raises:
    var X = Matrix[DType.float64](8, 2, 0)
    for r in range(8):
        X[r, 0] = Float64(r * 2)
        X[r, 1] = Float64(r * 3)

    var mbk1 = MiniBatchKMeans[DType.float64](n_clusters=3, random_state=999)
    mbk1.fit(X)

    var mbk2 = MiniBatchKMeans[DType.float64](n_clusters=3, random_state=999)
    mbk2.fit(X)

    assert_almost_equal(
        Float64(mbk1.inertia_), Float64(mbk2.inertia_), atol=1e-7
    )
    for r in range(3):
        for c in range(2):
            assert_almost_equal(
                Float64(mbk1.cluster_centers_[r, c]),
                Float64(mbk2.cluster_centers_[r, c]),
                atol=1e-7,
            )


def test_minibatch_kmeans_float32_native() raises:
    var X = Matrix[DType.float32](4, 2, 0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 0.1
    X[1, 1] = 0.1
    X[2, 0] = 20.0
    X[2, 1] = 20.0
    X[3, 0] = 20.1
    X[3, 1] = 20.1

    var mbk = MiniBatchKMeans[DType.float32](n_clusters=2, random_state=42)
    mbk.fit(X)
    assert_true(mbk.is_fitted)

    var preds = mbk.predict(X)
    assert_equal(len(preds), 4)


def test_minibatch_kmeans_batch_size_larger_than_data() raises:
    var X = Matrix[DType.float64](4, 2, 1.0)
    var mbk = MiniBatchKMeans[DType.float64](
        n_clusters=2, batch_size=1000, random_state=42
    )
    mbk.fit(X)
    assert_true(mbk.is_fitted)


def test_minibatch_kmeans_single_cluster() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 2.0
    X[0, 1] = 4.0
    X[1, 0] = 4.0
    X[1, 1] = 6.0
    X[2, 0] = 6.0
    X[2, 1] = 8.0
    X[3, 0] = 8.0
    X[3, 1] = 10.0

    var mbk = MiniBatchKMeans[DType.float64](n_clusters=1, random_state=42)
    mbk.fit(X)

    assert_almost_equal(Float64(mbk.cluster_centers_[0, 0]), 5.0, atol=0.5)
    assert_almost_equal(Float64(mbk.cluster_centers_[0, 1]), 7.0, atol=0.5)


def test_minibatch_kmeans_all_identical_samples() raises:
    var X = Matrix[DType.float64](8, 2, 42.0)
    var mbk = MiniBatchKMeans[DType.float64](n_clusters=3, random_state=42)
    mbk.fit(X)

    assert_true(mbk.is_fitted)
    assert_almost_equal(Float64(mbk.inertia_), 0.0, atol=1e-7)


def test_minibatch_kmeans_nan_and_inf_inputs_rejected() raises:
    var X_nan = Matrix[DType.float64](4, 2, 1.0)
    X_nan[1, 1] = nan[DType.float64]()

    var mbk = MiniBatchKMeans[DType.float64](n_clusters=2)
    with assert_raises():
        mbk.fit(X_nan)
    with assert_raises():
        mbk.partial_fit(X_nan)


def test_minibatch_kmeans_error_handling() raises:
    var mbk = MiniBatchKMeans[DType.float64](n_clusters=2)
    var X = Matrix[DType.float64](4, 2, 1.0)

    # Not fitted errors
    with assert_raises():
        _ = mbk.predict(X)
    with assert_raises():
        _ = mbk.transform(X)
    with assert_raises():
        _ = mbk.score(X)

    # Invalid parameter: n_clusters <= 0
    var mbk_bad_k = MiniBatchKMeans[DType.float64](n_clusters=0)
    with assert_raises():
        mbk_bad_k.fit(X)

    # Invalid parameter: n_clusters > n_samples
    var mbk_too_many_k = MiniBatchKMeans[DType.float64](n_clusters=10)
    with assert_raises():
        mbk_too_many_k.fit(X)

    # Invalid parameter: batch_size <= 0
    var mbk_bad_bs = MiniBatchKMeans[DType.float64](n_clusters=2, batch_size=0)
    with assert_raises():
        mbk_bad_bs.fit(X)

    # Invalid parameter: invalid init
    var mbk_bad_init = MiniBatchKMeans[DType.float64](
        n_clusters=2, init="spectral"
    )
    with assert_raises():
        mbk_bad_init.fit(X)

    # Dimension mismatch on predict
    mbk.fit(X)
    var X_wrong_dim = Matrix[DType.float64](4, 3, 1.0)
    with assert_raises():
        _ = mbk.predict(X_wrong_dim)
    with assert_raises():
        _ = mbk.transform(X_wrong_dim)
    with assert_raises():
        _ = mbk.score(X_wrong_dim)
    with assert_raises():
        mbk.partial_fit(X_wrong_dim)


def test_minibatch_kmeans_multi_cluster_and_higher_dimensions() raises:
    var N = 60
    var D = 6
    var K = 3
    var X = Matrix[DType.float64](N, D, 0)

    for i in range(N):
        var cluster_id = i // 20
        var center_val = Float64(cluster_id * 100)
        for j in range(D):
            X[i, j] = center_val + Float64(i % 20) * 0.1

    var mbk = MiniBatchKMeans[DType.float64](
        n_clusters=K, batch_size=20, random_state=42
    )
    mbk.fit(X)

    assert_true(mbk.is_fitted)
    assert_equal(mbk.cluster_centers_.rows, 3)
    assert_equal(mbk.cluster_centers_.cols, 6)

    # Check consistency of cluster labels
    for c in range(3):
        var first_label = mbk.labels_[c * 20]
        for i in range(c * 20, (c + 1) * 20):
            assert_equal(mbk.labels_[i], first_label)


def test_minibatch_kmeans_colinear_data() raises:
    var X = Matrix[DType.float64](6, 3, 0)
    for r in range(6):
        var t = Float64(r * 10)
        X[r, 0] = t
        X[r, 1] = 2.0 * t
        X[r, 2] = 3.0 * t

    var mbk = MiniBatchKMeans[DType.float64](n_clusters=2, random_state=42)
    mbk.fit(X)

    assert_true(mbk.is_fitted)
    for c in range(2):
        var c0 = Float64(mbk.cluster_centers_[c, 0])
        var c1 = Float64(mbk.cluster_centers_[c, 1])
        var c2 = Float64(mbk.cluster_centers_[c, 2])
        assert_almost_equal(c1, 2.0 * c0, atol=1e-5)
        assert_almost_equal(c2, 3.0 * c0, atol=1e-5)


def test_minibatch_kmeans_partial_fit_multiple_epochs() raises:
    var mbk = MiniBatchKMeans[DType.float64](n_clusters=2, random_state=42)
    var batch = Matrix[DType.float64](4, 2, 0)
    batch[0, 0] = 0.0
    batch[0, 1] = 0.0
    batch[1, 0] = 0.1
    batch[1, 1] = 0.1
    batch[2, 0] = 10.0
    batch[2, 1] = 10.0
    batch[3, 0] = 10.1
    batch[3, 1] = 9.9

    for _ in range(10):
        mbk.partial_fit(batch)

    assert_equal(mbk.n_steps_, 10)
    var preds = mbk.predict(batch)
    assert_equal(preds[0], preds[1])
    assert_equal(preds[2], preds[3])
    assert_true(preds[0] != preds[2])


def test_minibatch_kmeans_unbalanced_clusters() raises:
    var X = Matrix[DType.float64](22, 2, 0)
    for r in range(20):
        X[r, 0] = 0.0 + Float64(r) * 0.01
        X[r, 1] = 0.0 - Float64(r) * 0.01

    X[20, 0] = 100.0
    X[20, 1] = 100.0
    X[21, 0] = 100.1
    X[21, 1] = 99.9

    var mbk = MiniBatchKMeans[DType.float64](
        n_clusters=2, batch_size=10, random_state=42
    )
    mbk.fit(X)

    assert_true(mbk.is_fitted)
    var c_small = mbk.labels_[20]
    var c_large = mbk.labels_[0]
    assert_true(c_small != c_large)
    assert_equal(mbk.labels_[21], c_small)


def test_minibatch_kmeans_high_dimensional_space() raises:
    var N = 30
    var D = 50
    var K = 3
    var X = Matrix[DType.float64](N, D, 0)

    for i in range(N):
        var cluster_id = i // 10
        var center_val = Float64(cluster_id * 50)
        for j in range(D):
            X[i, j] = center_val + Float64(i % 10) * 0.05

    var mbk = MiniBatchKMeans[DType.float64](
        n_clusters=K, batch_size=10, random_state=42
    )
    mbk.fit(X)

    assert_true(mbk.is_fitted)
    assert_equal(mbk.cluster_centers_.rows, 3)
    assert_equal(mbk.cluster_centers_.cols, 50)


def test_minibatch_kmeans_refit_resets_state() raises:
    var X1 = Matrix[DType.float64](10, 4, 1.0)
    var mbk = MiniBatchKMeans[DType.float64](n_clusters=2, random_state=42)
    mbk.fit(X1)
    assert_equal(mbk.cluster_centers_.rows, 2)
    assert_equal(mbk.cluster_centers_.cols, 4)

    var X2 = Matrix[DType.float64](6, 2, 5.0)
    mbk.fit(X2)
    assert_equal(mbk.cluster_centers_.rows, 2)
    assert_equal(mbk.cluster_centers_.cols, 2)
    assert_equal(len(mbk.labels_), 6)


def test_minibatch_kmeans_bfloat16_precision() raises:
    var X = Matrix[DType.bfloat16](4, 2, 0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 0.1
    X[1, 1] = 0.1
    X[2, 0] = 20.0
    X[2, 1] = 20.0
    X[3, 0] = 20.1
    X[3, 1] = 20.1

    var mbk = MiniBatchKMeans[DType.bfloat16](n_clusters=2, random_state=42)
    mbk.fit(X)
    assert_true(mbk.is_fitted)

    var preds = mbk.predict(X)
    assert_equal(len(preds), 4)


def test_minibatch_kmeans_float16_precision() raises:
    var X = Matrix[DType.float16](4, 2, 0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 0.1
    X[1, 1] = 0.1
    X[2, 0] = 20.0
    X[2, 1] = 20.0
    X[3, 0] = 20.1
    X[3, 1] = 20.1

    var mbk = MiniBatchKMeans[DType.float16](n_clusters=2, random_state=42)
    mbk.fit(X)
    assert_true(mbk.is_fitted)

    var preds = mbk.predict(X)
    assert_equal(len(preds), 4)


def test_minibatch_kmeans_batch_size_sweeps() raises:
    var X = Matrix[DType.float64](20, 2, 0)
    for r in range(10):
        X[r, 0] = Float64(r) * 0.1
        X[r, 1] = Float64(r) * 0.1
    for r in range(10, 20):
        X[r, 0] = 20.0 + Float64(r - 10) * 0.1
        X[r, 1] = 20.0 + Float64(r - 10) * 0.1

    # Sweep batch sizes 1, 2, 5, 10, 20
    var b_sizes = List[Int](capacity=5)
    b_sizes.append(1)
    b_sizes.append(2)
    b_sizes.append(5)
    b_sizes.append(10)
    b_sizes.append(20)
    for i in range(len(b_sizes)):
        var bs = b_sizes[i]
        var mbk = MiniBatchKMeans[DType.float64](
            n_clusters=2, batch_size=bs, random_state=42
        )
        mbk.fit(X)
        assert_true(mbk.is_fitted)
        assert_equal(mbk.cluster_centers_.rows, 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
