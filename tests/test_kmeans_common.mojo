from std.math import sqrt
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_raises,
    assert_almost_equal,
)
from strata.core.matrix import Matrix
from strata.cluster._common import (
    _squared_euclidean_distance,
    _euclidean_distance,
    _compute_distances,
    _assign_nearest_and_inertia,
    _init_centroids_random,
    _init_centroids_kmeans_plus_plus,
)
from strata.exceptions.errors import InvalidParameterError


def test_distance_metrics() raises:
    var X = Matrix[DType.float64](2, 3, 0)
    X[0, 0] = 1.0
    X[0, 1] = 2.0
    X[0, 2] = 3.0

    X[1, 0] = 4.0
    X[1, 1] = 6.0
    X[1, 2] = 8.0

    var C = Matrix[DType.float64](1, 3, 0)
    C[0, 0] = 1.0
    C[0, 1] = 2.0
    C[0, 2] = 3.0

    var dist_sq_0 = _squared_euclidean_distance(X, 0, C, 0)
    assert_almost_equal(Float64(dist_sq_0), 0.0, atol=1e-7)

    # For row 1: diffs are (3, 4, 5) -> 3^2 + 4^2 + 5^2 = 9 + 16 + 25 = 50.0
    var dist_sq_1 = _squared_euclidean_distance(X, 1, C, 0)
    assert_almost_equal(Float64(dist_sq_1), 50.0, atol=1e-7)

    var dist_1 = _euclidean_distance(X, 1, C, 0)
    assert_almost_equal(Float64(dist_1), 7.0710678118654755, atol=1e-7)


def test_compute_distances_matrix() raises:
    var X = Matrix[DType.float64](3, 2, 0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 3.0
    X[1, 1] = 0.0
    X[2, 0] = 0.0
    X[2, 1] = 4.0

    var C = Matrix[DType.float64](2, 2, 0)
    C[0, 0] = 0.0
    C[0, 1] = 0.0
    C[1, 0] = 3.0
    C[1, 1] = 4.0

    var dists = _compute_distances(X, C)
    assert_equal(dists.rows, 3)
    assert_equal(dists.cols, 2)

    assert_almost_equal(Float64(dists[0, 0]), 0.0, atol=1e-7)
    assert_almost_equal(Float64(dists[0, 1]), 5.0, atol=1e-7)

    assert_almost_equal(Float64(dists[1, 0]), 3.0, atol=1e-7)
    assert_almost_equal(Float64(dists[1, 1]), 4.0, atol=1e-7)

    assert_almost_equal(Float64(dists[2, 0]), 4.0, atol=1e-7)
    assert_almost_equal(Float64(dists[2, 1]), 3.0, atol=1e-7)


def test_assign_nearest_and_inertia() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 0.1
    X[1, 1] = 0.1
    X[2, 0] = 10.0
    X[2, 1] = 10.0
    X[3, 0] = 10.2
    X[3, 1] = 10.1

    var C = Matrix[DType.float64](2, 2, 0)
    C[0, 0] = 0.0
    C[0, 1] = 0.0
    C[1, 0] = 10.0
    C[1, 1] = 10.0

    var labels = List[Int]()
    var min_sq_dists = List[Scalar[DType.float64]]()

    var inertia = _assign_nearest_and_inertia(X, C, labels, min_sq_dists)

    assert_equal(len(labels), 4)
    assert_equal(labels[0], 0)
    assert_equal(labels[1], 0)
    assert_equal(labels[2], 1)
    assert_equal(labels[3], 1)

    # dist_sq[0] = 0.0
    # dist_sq[1] = 0.1^2 + 0.1^2 = 0.02
    # dist_sq[2] = 0.0
    # dist_sq[3] = 0.2^2 + 0.1^2 = 0.04 + 0.01 = 0.05
    # total inertia = 0.07
    assert_almost_equal(Float64(inertia), 0.07, atol=1e-7)


def test_init_centroids_random() raises:
    var X = Matrix[DType.float64](10, 3, 0)
    for r in range(10):
        for c in range(3):
            X[r, c] = Float64(r * 10 + c)

    var centers = _init_centroids_random(X, 3, seed=123)
    assert_equal(centers.rows, 3)
    assert_equal(centers.cols, 3)

    # Reproducibility with same seed
    var centers2 = _init_centroids_random(X, 3, seed=123)
    for r in range(3):
        for c in range(3):
            assert_almost_equal(
                Float64(centers[r, c]), Float64(centers2[r, c]), atol=1e-7
            )

    # Should raise if n_clusters > n_samples
    with assert_raises():
        _ = _init_centroids_random(X, 15, seed=123)


def test_init_centroids_kmeans_plus_plus() raises:
    var X = Matrix[DType.float64](6, 2, 0)
    # Cluster 1 around (0, 0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 0.1
    X[1, 1] = 0.2
    # Cluster 2 around (10, 10)
    X[2, 0] = 10.0
    X[2, 1] = 10.0
    X[3, 0] = 10.1
    X[3, 1] = 9.9
    # Cluster 3 around (50, 50)
    X[4, 0] = 50.0
    X[4, 1] = 50.0
    X[5, 0] = 50.2
    X[5, 1] = 49.8

    var centers = _init_centroids_kmeans_plus_plus(X, 3, seed=42)
    assert_equal(centers.rows, 3)
    assert_equal(centers.cols, 2)

    # Single cluster edge case
    var centers_1 = _init_centroids_kmeans_plus_plus(X, 1, seed=42)
    assert_equal(centers_1.rows, 1)
    assert_equal(centers_1.cols, 2)

    # Reproducibility with same seed
    var centers_rep = _init_centroids_kmeans_plus_plus(X, 3, seed=42)
    for r in range(3):
        for c in range(2):
            assert_almost_equal(
                Float64(centers[r, c]), Float64(centers_rep[r, c]), atol=1e-7
            )

    # Out of bounds error
    with assert_raises():
        _ = _init_centroids_kmeans_plus_plus(X, 10, seed=42)


def test_float32_support() raises:
    var X = Matrix[DType.float32](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 1.0
    X[1, 0] = 1.5
    X[1, 1] = 1.5
    X[2, 0] = 20.0
    X[2, 1] = 20.0
    X[3, 0] = 20.5
    X[3, 1] = 20.5

    var centers = _init_centroids_kmeans_plus_plus(X, 2, seed=99)
    assert_equal(centers.rows, 2)
    assert_equal(centers.cols, 2)

    var labels = List[Int]()
    var min_sq_dists = List[Scalar[DType.float32]]()
    var inertia = _assign_nearest_and_inertia(X, centers, labels, min_sq_dists)
    assert_true(inertia >= 0)


def test_init_centroids_kmeans_plus_plus_all_identical_points() raises:
    var X = Matrix[DType.float64](5, 2, 3.14)
    var centers = _init_centroids_kmeans_plus_plus(X, 3, seed=7)
    assert_equal(centers.rows, 3)
    assert_equal(centers.cols, 2)
    for r in range(3):
        assert_almost_equal(Float64(centers[r, 0]), 3.14, atol=1e-7)
        assert_almost_equal(Float64(centers[r, 1]), 3.14, atol=1e-7)


def test_init_centroids_kmeans_plus_plus_k_equals_n() raises:
    var X = Matrix[DType.float64](3, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 2.0
    X[1, 0] = 3.0
    X[1, 1] = 4.0
    X[2, 0] = 5.0
    X[2, 1] = 6.0

    var centers = _init_centroids_kmeans_plus_plus(X, 3, seed=1)
    assert_equal(centers.rows, 3)
    assert_equal(centers.cols, 2)


def test_distance_computation_multi_dimension() raises:
    var X = Matrix[DType.float64](2, 10, 1.0)
    var C = Matrix[DType.float64](1, 10, 2.0)
    # Distance in 10D: 10 * (1 - 2)^2 = 10
    var dist_sq = _squared_euclidean_distance(X, 0, C, 0)
    assert_almost_equal(Float64(dist_sq), 10.0, atol=1e-7)
    var dist = _euclidean_distance(X, 0, C, 0)
    assert_almost_equal(Float64(dist), sqrt(10.0), atol=1e-7)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


