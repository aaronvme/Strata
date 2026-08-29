from std.math import sqrt
from ..base.estimator import Clusterer, Transformer
from ..core.matrix import Matrix
from ..core.dataset import Dataset
from ..utils.validation import (
    check_array,
    check_floating_dtype,
    check_is_fitted,
)
from ..exceptions.errors import (
    InvalidParameterError,
    DimensionMismatchError,
    NotFittedError,
)
from ._common import (
    _squared_euclidean_distance,
    _euclidean_distance,
    _compute_distances,
    _assign_nearest_and_inertia,
    _init_centroids_random,
    _init_centroids_kmeans_plus_plus,
)


struct KMeans[compute_dtype: DType = DType.float64](
    Clusterer, Copyable, Movable
):
    """K-Means clustering using Lloyd's or Elkan's algorithm.

    Clusters $N$ observations into $K$ disjoint geometric partitions by
    minimizing within-cluster inertia (sum-of-squared Euclidean distances):

    $$
    \\arg\\min_{C} \\sum_{i=1}^{N} \\min_{\\mu_j \\in C} \\|x_i - \\mu_j\\|_2^2
    $$


    Parameters:
        compute_dtype: Computational precision data type. Default DType.float64.

    Args:
        n_clusters: The number of clusters to form as well as the number of centroids to generate. Default 8.
        init: Method for initialization ('k-means++', 'random'). Default 'k-means++'.
        n_init: Number of times the k-means algorithm will be run with different centroid seeds. Default 10.
        max_iter: Maximum number of iterations of the k-means algorithm for a single run. Default 300.
        tol: Relative tolerance with regards to Frobenius norm of the difference in cluster centers. Default 1e-4.
        algorithm: K-means algorithm to use ('lloyd'). Default 'lloyd'.
        random_state: PRNG seed for centroid initialization. Default 42.

    Attributes:
        cluster_centers_: Coordinates of cluster centers matrix of shape $(K, D)$.
        labels_: Labels of each point vector of length $N$.
        inertia_: Sum of squared distances of samples to their closest cluster center.
        n_iter_: Number of iterations run.
        n_features_in_: Number of features seen during fit.
        is_fitted: Boolean flag indicating if estimator has been fitted.

    Examples:
        ```mojo
        from strata.cluster import KMeans
        from strata.core import Matrix

        var kmeans = KMeans[DType.float64](n_clusters=3, init="k-means++")
        kmeans.fit(X_data)
        var labels = kmeans.predict(X_data)
        var distances = kmeans.transform(X_data)
        ```
    """

    var is_fitted: Bool
    var n_clusters: Int
    var init: String
    var n_init: Int
    var max_iter: Int
    var tol: Float64
    var random_state: Int

    var algorithm: String
    var cluster_centers_: Matrix[Self.compute_dtype]
    var labels_: List[Int]
    var inertia_: Scalar[Self.compute_dtype]
    var n_iter_: Int
    var n_features_in_: Int

    def __init__(
        out self,
        n_clusters: Int = 8,
        init: String = "k-means++",
        n_init: Int = 10,
        max_iter: Int = 300,
        tol: Float64 = 1e-4,
        algorithm: String = "lloyd",
        random_state: Int = 42,
    ):
        """Initialize the KMeans estimator.

        Args:
            n_clusters: Number of clusters to form. Default 8.
            init: Centroid initialization strategy ('k-means++', 'random'). Default 'k-means++'.
            n_init: Number of initializations to run. Default 10.
            max_iter: Maximum iterations per run. Default 300.
            tol: Convergence tolerance threshold. Default 1e-4.
            algorithm: Algorithm variant ('lloyd'). Default 'lloyd'.
            random_state: PRNG seed for deterministic centroid placement. Default 42.
        """

        check_floating_dtype[Self.compute_dtype, "KMeans"]()
        self.is_fitted = False
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.algorithm = algorithm
        self.random_state = random_state

        self.cluster_centers_ = Matrix[Self.compute_dtype](0, 0, 0)
        self.labels_ = List[Int]()
        self.inertia_ = 0
        self.n_iter_ = 0
        self.n_features_in_ = 0

    def __init__(out self, *, copy: Self):
        self.is_fitted = copy.is_fitted
        self.n_clusters = copy.n_clusters
        self.init = copy.init
        self.n_init = copy.n_init
        self.max_iter = copy.max_iter
        self.tol = copy.tol
        self.algorithm = copy.algorithm
        self.random_state = copy.random_state

        self.cluster_centers_ = copy.cluster_centers_.copy()
        self.labels_ = copy.labels_.copy()
        self.inertia_ = copy.inertia_
        self.n_iter_ = copy.n_iter_
        self.n_features_in_ = copy.n_features_in_

    def fit[in_dtype: DType](mut self, X: Matrix[in_dtype]) raises:
        check_array[in_dtype](X)
        var n = X.rows
        var d = X.cols

        if self.n_clusters <= 0:
            raise InvalidParameterError.error(
                "n_clusters",
                "n_clusters="
                + String(self.n_clusters)
                + " must be greater than 0",
            )
        if self.n_clusters > n:
            raise InvalidParameterError.error(
                "n_clusters",
                "n_clusters ("
                + String(self.n_clusters)
                + ") cannot exceed n_samples ("
                + String(n)
                + ")",
            )
        if self.n_init <= 0:
            raise InvalidParameterError.error(
                "n_init", "n_init=" + String(self.n_init) + " must be > 0"
            )
        if self.max_iter <= 0:
            raise InvalidParameterError.error(
                "max_iter",
                "max_iter=" + String(self.max_iter) + " must be > 0",
            )
        if self.tol < 0.0:
            raise InvalidParameterError.error(
                "tol", "tol=" + String(self.tol) + " must be >= 0.0"
            )
        if self.init != "k-means++" and self.init != "random":
            raise InvalidParameterError.error(
                "init",
                "init='"
                + self.init
                + "' is unsupported. Expected 'k-means++' or 'random'.",
            )
        if self.algorithm != "lloyd" and self.algorithm != "auto":
            raise InvalidParameterError.error(
                "algorithm",
                "algorithm='"
                + self.algorithm
                + "' is unsupported. Expected 'lloyd' or 'auto'.",
            )

        var X_comp = X.cast[Self.compute_dtype]()
        var k = self.n_clusters

        var col_means = X_comp.mean_along_axis_0()
        var total_var: Scalar[Self.compute_dtype] = 0
        for j in range(d):
            var col_mean = col_means[j]
            var col_var: Scalar[Self.compute_dtype] = 0
            for i in range(n):
                var diff = X_comp[i, j] - col_mean
                col_var += diff * diff
            total_var += col_var / Scalar[Self.compute_dtype](n)
        var mean_var = total_var / Scalar[Self.compute_dtype](d)
        if mean_var == 0:
            mean_var = 1.0
        var effective_tol = Scalar[Self.compute_dtype](self.tol) * mean_var

        var best_inertia: Scalar[Self.compute_dtype] = 0
        var best_centers = Matrix[Self.compute_dtype](k, d, 0)
        var best_labels = List[Int](capacity=n)
        for _ in range(n):
            best_labels.append(0)
        var best_n_iter = 0

        var labels = List[Int](capacity=n)
        for _ in range(n):
            labels.append(0)
        var min_sq_dists = List[Scalar[Self.compute_dtype]](capacity=n)
        for _ in range(n):
            min_sq_dists.append(0)
        var cluster_counts = List[Int](capacity=k)
        for _ in range(k):
            cluster_counts.append(0)
        var new_centers = Matrix[Self.compute_dtype](k, d, 0)

        for run in range(self.n_init):
            var run_seed = self.random_state + run * 31

            var current_centers: Matrix[Self.compute_dtype]
            if self.init == "k-means++":
                current_centers = _init_centroids_kmeans_plus_plus(
                    X_comp, k, run_seed
                )
            else:
                current_centers = _init_centroids_random(X_comp, k, run_seed)

            var iter_count = 0

            for iter_idx in range(self.max_iter):
                iter_count = iter_idx + 1
                _ = _assign_nearest_and_inertia(
                    X_comp, current_centers, labels, min_sq_dists
                )

                for c in range(k):
                    cluster_counts[c] = 0
                    for j in range(d):
                        new_centers[c, j] = 0

                for i in range(n):
                    var c = labels[i]
                    cluster_counts[c] += 1
                    for j in range(d):
                        new_centers[c, j] += X_comp[i, j]

                for c in range(k):
                    var count = cluster_counts[c]
                    if count > 0:
                        var count_scalar = Scalar[Self.compute_dtype](count)
                        for j in range(d):
                            new_centers[c, j] /= count_scalar
                    else:
                        var max_dist_sample = 0
                        var max_dist: Scalar[Self.compute_dtype] = -1
                        for i in range(n):
                            if min_sq_dists[i] > max_dist:
                                max_dist = min_sq_dists[i]
                                max_dist_sample = i
                        for j in range(d):
                            new_centers[c, j] = X_comp[max_dist_sample, j]
                        min_sq_dists[max_dist_sample] = 0

                var shift: Scalar[Self.compute_dtype] = 0
                for c in range(k):
                    for j in range(d):
                        var diff = new_centers[c, j] - current_centers[c, j]
                        shift += diff * diff
                        current_centers[c, j] = new_centers[c, j]

                if shift <= effective_tol:
                    break

            var final_inertia = _assign_nearest_and_inertia(
                X_comp, current_centers, labels, min_sq_dists
            )

            if run == 0 or final_inertia < best_inertia:
                best_inertia = final_inertia
                best_centers = current_centers.copy()
                for i in range(n):
                    best_labels[i] = labels[i]
                best_n_iter = iter_count

        self.cluster_centers_ = best_centers^
        self.labels_ = best_labels^
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        self.n_features_in_ = d
        self.is_fitted = True

    def predict[in_dtype: DType](self, X: Matrix[in_dtype]) raises -> List[Int]:
        check_is_fitted("KMeans", self.is_fitted)
        check_array[in_dtype](X)
        if X.cols != self.n_features_in_:
            raise DimensionMismatchError.error(
                "X.cols == " + String(self.n_features_in_),
                "X.cols == " + String(X.cols),
                "KMeans.predict",
            )

        var X_comp = X.cast[Self.compute_dtype]()
        var labels = List[Int](capacity=X.rows)
        for _ in range(X.rows):
            labels.append(0)
        var min_sq_dists = List[Scalar[Self.compute_dtype]](capacity=X.rows)
        for _ in range(X.rows):
            min_sq_dists.append(0)

        _ = _assign_nearest_and_inertia(
            X_comp, self.cluster_centers_, labels, min_sq_dists
        )
        return labels^

    def fit_predict[
        in_dtype: DType
    ](mut self, X: Matrix[in_dtype]) raises -> List[Int]:
        self.fit[in_dtype](X)
        return self.labels_.copy()

    def transform[
        in_dtype: DType
    ](self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        check_is_fitted("KMeans", self.is_fitted)
        check_array[in_dtype](X)
        if X.cols != self.n_features_in_:
            raise DimensionMismatchError.error(
                "X.cols == " + String(self.n_features_in_),
                "X.cols == " + String(X.cols),
                "KMeans.transform",
            )

        var X_comp = X.cast[Self.compute_dtype]()
        var dists_comp = _compute_distances(X_comp, self.cluster_centers_)
        return dists_comp.cast[in_dtype]()

    def fit_transform[
        in_dtype: DType
    ](mut self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        self.fit[in_dtype](X)
        return self.transform[in_dtype](X)

    def score[
        in_dtype: DType
    ](self, X: Matrix[in_dtype]) raises -> Scalar[Self.compute_dtype]:
        check_is_fitted("KMeans", self.is_fitted)
        check_array[in_dtype](X)
        if X.cols != self.n_features_in_:
            raise DimensionMismatchError.error(
                "X.cols == " + String(self.n_features_in_),
                "X.cols == " + String(X.cols),
                "KMeans.score",
            )

        var X_comp = X.cast[Self.compute_dtype]()
        var labels = List[Int](capacity=X.rows)
        for _ in range(X.rows):
            labels.append(0)
        var min_sq_dists = List[Scalar[Self.compute_dtype]](capacity=X.rows)
        for _ in range(X.rows):
            min_sq_dists.append(0)

        var total_inertia = _assign_nearest_and_inertia(
            X_comp, self.cluster_centers_, labels, min_sq_dists
        )
        return -total_inertia
