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
from strata.cluster.kmeans import KMeans
from strata.exceptions.errors import (
    NotFittedError,
    DimensionMismatchError,
    InvalidParameterError,
)


def test_kmeans_basic_two_clusters() raises:
    var X = Matrix[DType.float64](6, 2, 0)
    # Cluster 0 around (0, 0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 0.1
    X[1, 1] = -0.1
    X[2, 0] = -0.1
    X[2, 1] = 0.1

    # Cluster 1 around (10, 10)
    X[3, 0] = 10.0
    X[3, 1] = 10.0
    X[4, 0] = 10.1
    X[4, 1] = 9.9
    X[5, 0] = 9.9
    X[5, 1] = 10.1

    var km = KMeans[DType.float64](n_clusters=2, random_state=42)
    km.fit(X)

    assert_true(km.is_fitted)
    assert_equal(km.cluster_centers_.rows, 2)
    assert_equal(km.cluster_centers_.cols, 2)
    assert_equal(len(km.labels_), 6)

    # First three points must belong to the same cluster, last three to the other
    assert_equal(km.labels_[0], km.labels_[1])
    assert_equal(km.labels_[0], km.labels_[2])
    assert_equal(km.labels_[3], km.labels_[4])
    assert_equal(km.labels_[3], km.labels_[5])
    assert_true(km.labels_[0] != km.labels_[3])

    # Centroid 0 should be near (0, 0) and Centroid 1 near (10, 10) or vice versa
    var c0 = km.labels_[0]
    var c1 = km.labels_[3]
    assert_almost_equal(Float64(km.cluster_centers_[c0, 0]), 0.0, atol=0.2)
    assert_almost_equal(Float64(km.cluster_centers_[c0, 1]), 0.0, atol=0.2)
    assert_almost_equal(Float64(km.cluster_centers_[c1, 0]), 10.0, atol=0.2)
    assert_almost_equal(Float64(km.cluster_centers_[c1, 1]), 10.0, atol=0.2)

    assert_true(km.inertia_ >= 0)
    assert_almost_equal(Float64(km.inertia_), 0.08, atol=0.05)


def test_kmeans_predict_and_fit_predict() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 0.2
    X[1, 1] = 0.1
    X[2, 0] = 5.0
    X[2, 1] = 5.0
    X[3, 0] = 5.1
    X[3, 1] = 4.9

    var km = KMeans[DType.float64](n_clusters=2, random_state=123)
    var fit_pred_labels = km.fit_predict(X)
    var pred_labels = km.predict(X)

    assert_equal(len(fit_pred_labels), 4)
    assert_equal(len(pred_labels), 4)
    for i in range(4):
        assert_equal(fit_pred_labels[i], pred_labels[i])


def test_kmeans_random_init() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 1.0
    X[1, 0] = 1.1
    X[1, 1] = 0.9
    X[2, 0] = 8.0
    X[2, 1] = 8.0
    X[3, 0] = 8.2
    X[3, 1] = 7.9

    var km = KMeans[DType.float64](
        n_clusters=2, init="random", random_state=42
    )
    km.fit(X)
    assert_true(km.is_fitted)
    assert_equal(km.cluster_centers_.rows, 2)


def test_kmeans_transform_and_fit_transform() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 0.0
    X[1, 1] = 0.0
    X[2, 0] = 10.0
    X[2, 1] = 10.0
    X[3, 0] = 10.0
    X[3, 1] = 10.0

    var km = KMeans[DType.float64](n_clusters=2, random_state=42)
    var X_trans = km.fit_transform(X)

    assert_equal(X_trans.rows, 4)
    assert_equal(X_trans.cols, 2)

    # For sample 0, distance to center at (0,0) should be 0
    var c0 = km.labels_[0]
    assert_almost_equal(Float64(X_trans[0, c0]), 0.0, atol=1e-5)


def test_kmeans_score() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 0.1
    X[1, 1] = 0.0
    X[2, 0] = 10.0
    X[2, 1] = 10.0
    X[3, 0] = 10.0
    X[3, 1] = 10.2

    var km = KMeans[DType.float64](n_clusters=2, random_state=42)
    km.fit(X)

    var s = km.score(X)
    assert_almost_equal(Float64(s), -Float64(km.inertia_), atol=1e-7)


def test_kmeans_dataset_overload() raises:
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

    var ds = Dataset[DType.float64, DType.int32](
        X^, y^, f_names^, t_names^
    )

    var km = KMeans[DType.float64](n_clusters=2, random_state=42)
    fit_ds(km, ds)
    assert_true(km.is_fitted)

    var preds = predict_ds(km, ds)
    assert_equal(len(preds), 4)

    var trans = km.transform(ds.records)
    assert_equal(trans.rows, 4)
    assert_equal(trans.cols, 2)


def test_kmeans_copy_constructor() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 1.0
    X[1, 0] = 1.0
    X[1, 1] = 1.2
    X[2, 0] = 5.0
    X[2, 1] = 5.0
    X[3, 0] = 5.2
    X[3, 1] = 5.0

    var km1 = KMeans[DType.float64](n_clusters=2, random_state=42)
    km1.fit(X)

    var km2 = km1.copy()
    assert_true(km2.is_fitted)
    assert_equal(km2.cluster_centers_.rows, 2)
    assert_almost_equal(Float64(km1.inertia_), Float64(km2.inertia_), atol=1e-7)

    # Mutate km1 by refitting
    var X_new = Matrix[DType.float64](2, 2, 100.0)
    km1.fit(X_new)

    # km2 should remain isolated
    assert_almost_equal(
        Float64(km2.cluster_centers_[0, 0]), 1.0, atol=0.5
    )


def test_kmeans_reproducibility() raises:
    var X = Matrix[DType.float64](8, 2, 0)
    for r in range(8):
        X[r, 0] = Float64(r * 2)
        X[r, 1] = Float64(r * 3)

    var km1 = KMeans[DType.float64](n_clusters=3, random_state=999)
    km1.fit(X)

    var km2 = KMeans[DType.float64](n_clusters=3, random_state=999)
    km2.fit(X)

    assert_almost_equal(Float64(km1.inertia_), Float64(km2.inertia_), atol=1e-7)
    for r in range(3):
        for c in range(2):
            assert_almost_equal(
                Float64(km1.cluster_centers_[r, c]),
                Float64(km2.cluster_centers_[r, c]),
                atol=1e-7,
            )


def test_kmeans_single_cluster() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 2.0
    X[0, 1] = 4.0
    X[1, 0] = 4.0
    X[1, 1] = 6.0
    X[2, 0] = 6.0
    X[2, 1] = 8.0
    X[3, 0] = 8.0
    X[3, 1] = 10.0

    var km = KMeans[DType.float64](n_clusters=1, random_state=42)
    km.fit(X)

    assert_almost_equal(Float64(km.cluster_centers_[0, 0]), 5.0, atol=1e-7)
    assert_almost_equal(Float64(km.cluster_centers_[0, 1]), 7.0, atol=1e-7)


def test_kmeans_k_equals_n() raises:
    var X = Matrix[DType.float64](3, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 2.0
    X[1, 0] = 10.0
    X[1, 1] = 20.0
    X[2, 0] = 100.0
    X[2, 1] = 200.0

    var km = KMeans[DType.float64](n_clusters=3, random_state=42)
    km.fit(X)

    assert_almost_equal(Float64(km.inertia_), 0.0, atol=1e-7)


def test_kmeans_float32_native() raises:
    var X = Matrix[DType.float32](4, 2, 0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 0.1
    X[1, 1] = 0.1
    X[2, 0] = 20.0
    X[2, 1] = 20.0
    X[3, 0] = 20.1
    X[3, 1] = 20.1

    var km = KMeans[DType.float32](n_clusters=2, random_state=42)
    km.fit(X)
    assert_true(km.is_fitted)

    var preds = km.predict(X)
    assert_equal(len(preds), 4)


def test_kmeans_error_handling() raises:
    var km = KMeans[DType.float64](n_clusters=2)
    var X = Matrix[DType.float64](4, 2, 1.0)

    # Not fitted errors
    with assert_raises():
        _ = km.predict(X)
    with assert_raises():
        _ = km.transform(X)
    with assert_raises():
        _ = km.score(X)

    # Invalid parameter: n_clusters <= 0
    var km_bad_k = KMeans[DType.float64](n_clusters=0)
    with assert_raises():
        km_bad_k.fit(X)

    # Invalid parameter: n_clusters > n_samples
    var km_too_many_k = KMeans[DType.float64](n_clusters=10)
    with assert_raises():
        km_too_many_k.fit(X)

    # Invalid parameter: invalid init
    var km_bad_init = KMeans[DType.float64](n_clusters=2, init="spectral")
    with assert_raises():
        km_bad_init.fit(X)

    # Invalid parameter: invalid algorithm
    var km_bad_algo = KMeans[DType.float64](n_clusters=2, algorithm="elkan")
    with assert_raises():
        km_bad_algo.fit(X)

    # Dimension mismatch on predict
    km.fit(X)
    var X_wrong_dim = Matrix[DType.float64](4, 3, 1.0)
    with assert_raises():
        _ = km.predict(X_wrong_dim)
    with assert_raises():
        _ = km.transform(X_wrong_dim)
    with assert_raises():
        _ = km.score(X_wrong_dim)


def test_kmeans_multi_cluster_and_higher_dimensions() raises:
    # 4 clusters in 8 dimensions
    var N = 40
    var D = 8
    var K = 4
    var X = Matrix[DType.float64](N, D, 0)

    for i in range(N):
        var cluster_id = i // 10
        var center_val = Float64(cluster_id * 50)
        for j in range(D):
            X[i, j] = center_val + Float64(i % 10) * 0.1

    var km = KMeans[DType.float64](n_clusters=K, random_state=42)
    km.fit(X)

    assert_true(km.is_fitted)
    assert_equal(km.cluster_centers_.rows, 4)
    assert_equal(km.cluster_centers_.cols, 8)

    # Check that points in each block of 10 share the same cluster label
    for c in range(4):
        var first_label = km.labels_[c * 10]
        for i in range(c * 10, (c + 1) * 10):
            assert_equal(km.labels_[i], first_label)


def test_kmeans_max_iter_one() raises:
    var X = Matrix[DType.float64](6, 2, 0)
    for r in range(6):
        X[r, 0] = Float64(r)
        X[r, 1] = Float64(r)

    var km = KMeans[DType.float64](n_clusters=2, max_iter=1, random_state=42)
    km.fit(X)
    assert_true(km.is_fitted)
    assert_equal(km.n_iter_, 1)


def test_kmeans_colinear_data() raises:
    # Points along the line y = 2x, z = 3x
    var X = Matrix[DType.float64](6, 3, 0)
    for r in range(6):
        var t = Float64(r * 10)
        X[r, 0] = t
        X[r, 1] = 2.0 * t
        X[r, 2] = 3.0 * t

    var km = KMeans[DType.float64](n_clusters=2, random_state=42)
    km.fit(X)

    assert_true(km.is_fitted)
    assert_equal(km.cluster_centers_.rows, 2)
    assert_equal(km.cluster_centers_.cols, 3)

    # For any centroid on line (t, 2t, 3t), c1 == 2*c0 and c2 == 3*c0
    for c in range(2):
        var c0 = Float64(km.cluster_centers_[c, 0])
        var c1 = Float64(km.cluster_centers_[c, 1])
        var c2 = Float64(km.cluster_centers_[c, 2])
        assert_almost_equal(c1, 2.0 * c0, atol=1e-5)
        assert_almost_equal(c2, 3.0 * c0, atol=1e-5)


def test_kmeans_all_identical_samples() raises:
    # All 8 samples are exactly identical (42.0, 42.0)
    var X = Matrix[DType.float64](8, 2, 42.0)
    var km = KMeans[DType.float64](n_clusters=3, random_state=42)
    km.fit(X)

    assert_true(km.is_fitted)
    assert_almost_equal(Float64(km.inertia_), 0.0, atol=1e-7)
    for c in range(3):
        assert_almost_equal(Float64(km.cluster_centers_[c, 0]), 42.0, atol=1e-7)
        assert_almost_equal(Float64(km.cluster_centers_[c, 1]), 42.0, atol=1e-7)


def test_kmeans_single_sample_single_cluster() raises:
    # Boundary: N=1, D=4, K=1
    var X = Matrix[DType.float64](1, 4, 0)
    X[0, 0] = 1.0
    X[0, 1] = 2.0
    X[0, 2] = 3.0
    X[0, 3] = 4.0

    var km = KMeans[DType.float64](n_clusters=1, random_state=42)
    km.fit(X)

    assert_true(km.is_fitted)
    assert_almost_equal(Float64(km.inertia_), 0.0, atol=1e-7)
    assert_equal(km.labels_[0], 0)
    assert_almost_equal(Float64(km.cluster_centers_[0, 0]), 1.0, atol=1e-7)
    assert_almost_equal(Float64(km.cluster_centers_[0, 1]), 2.0, atol=1e-7)
    assert_almost_equal(Float64(km.cluster_centers_[0, 2]), 3.0, atol=1e-7)
    assert_almost_equal(Float64(km.cluster_centers_[0, 3]), 4.0, atol=1e-7)


def test_kmeans_large_scale_stress() raises:
    # Stress test: N=200, D=6, K=4
    var N = 200
    var D = 6
    var K = 4
    var X = Matrix[DType.float64](N, D, 0)

    for i in range(N):
        var cluster_id = i % K
        var offset = Float64(cluster_id * 100)
        for j in range(D):
            X[i, j] = offset + Float64(i // K) * 0.01

    var km = KMeans[DType.float64](n_clusters=K, random_state=42)
    km.fit(X)

    assert_true(km.is_fitted)
    assert_equal(km.cluster_centers_.rows, K)
    assert_equal(km.cluster_centers_.cols, D)
    assert_equal(len(km.labels_), N)


def test_kmeans_nan_and_inf_inputs_rejected() raises:
    var X_nan = Matrix[DType.float64](4, 2, 1.0)
    X_nan[1, 1] = nan[DType.float64]()

    var km = KMeans[DType.float64](n_clusters=2)
    with assert_raises():
        km.fit(X_nan)

    var X_inf = Matrix[DType.float64](4, 2, 1.0)
    X_inf[2, 0] = inf[DType.float64]()
    with assert_raises():
        km.fit(X_inf)


def test_kmeans_zero_tolerance_exact_convergence() raises:
    var X = Matrix[DType.float64](6, 2, 0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 0.1
    X[1, 1] = 0.0
    X[2, 0] = 0.0
    X[2, 1] = 0.1
    X[3, 0] = 10.0
    X[3, 1] = 10.0
    X[4, 0] = 10.1
    X[4, 1] = 10.0
    X[5, 0] = 10.0
    X[5, 1] = 10.1

    var km = KMeans[DType.float64](n_clusters=2, tol=0.0, random_state=42)
    km.fit(X)
    assert_true(km.is_fitted)
    assert_true(km.n_iter_ >= 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


