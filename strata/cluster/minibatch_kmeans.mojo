from std.math import sqrt
from ..base.estimator import Clusterer
from ..core.matrix import Matrix
from ..utils.random import PRNG
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


struct MiniBatchKMeans[compute_dtype: DType = DType.float64](
    Clusterer, Copyable, Movable
):
    var is_fitted: Bool
    var n_clusters: Int
    var init: String
    var max_iter: Int
    var batch_size: Int
    var tol: Float64
    var max_no_improvement: Int
    var n_init: Int
    var reassignment_ratio: Float64
    var random_state: Int

    var cluster_centers_: Matrix[Self.compute_dtype]
    var labels_: List[Int]
    var inertia_: Scalar[Self.compute_dtype]
    var n_iter_: Int
    var n_steps_: Int
    var n_features_in_: Int
    var counts_: List[Int]

    def __init__(
        out self,
        n_clusters: Int = 8,
        init: String = "k-means++",
        max_iter: Int = 100,
        batch_size: Int = 1024,
        tol: Float64 = 0.0,
        max_no_improvement: Int = 10,
        n_init: Int = 3,
        reassignment_ratio: Float64 = 0.01,
        random_state: Int = 42,
    ):
        check_floating_dtype[Self.compute_dtype, "MiniBatchKMeans"]()
        self.is_fitted = False
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.tol = tol
        self.max_no_improvement = max_no_improvement
        self.n_init = n_init
        self.reassignment_ratio = reassignment_ratio
        self.random_state = random_state

        self.cluster_centers_ = Matrix[Self.compute_dtype](0, 0, 0)
        self.labels_ = List[Int]()
        self.inertia_ = 0
        self.n_iter_ = 0
        self.n_steps_ = 0
        self.n_features_in_ = 0
        self.counts_ = List[Int]()

    def __init__(out self, *, copy: Self):
        self.is_fitted = copy.is_fitted
        self.n_clusters = copy.n_clusters
        self.init = copy.init
        self.max_iter = copy.max_iter
        self.batch_size = copy.batch_size
        self.tol = copy.tol
        self.max_no_improvement = copy.max_no_improvement
        self.n_init = copy.n_init
        self.reassignment_ratio = copy.reassignment_ratio
        self.random_state = copy.random_state

        self.cluster_centers_ = copy.cluster_centers_.copy()
        self.labels_ = copy.labels_.copy()
        self.inertia_ = copy.inertia_
        self.n_iter_ = copy.n_iter_
        self.n_steps_ = copy.n_steps_
        self.n_features_in_ = copy.n_features_in_
        self.counts_ = copy.counts_.copy()

    def _validate_params(self, n_samples: Int) raises:
        if self.n_clusters <= 0:
            raise InvalidParameterError.error(
                "n_clusters",
                "n_clusters="
                + String(self.n_clusters)
                + " must be greater than 0",
            )
        if self.n_clusters > n_samples:
            raise InvalidParameterError.error(
                "n_clusters",
                "n_clusters ("
                + String(self.n_clusters)
                + ") cannot exceed n_samples ("
                + String(n_samples)
                + ")",
            )
        if self.max_iter <= 0:
            raise InvalidParameterError.error(
                "max_iter",
                "max_iter=" + String(self.max_iter) + " must be > 0",
            )
        if self.batch_size <= 0:
            raise InvalidParameterError.error(
                "batch_size",
                "batch_size=" + String(self.batch_size) + " must be > 0",
            )
        if self.tol < 0.0:
            raise InvalidParameterError.error(
                "tol", "tol=" + String(self.tol) + " must be >= 0.0"
            )
        if self.max_no_improvement < 0:
            raise InvalidParameterError.error(
                "max_no_improvement",
                "max_no_improvement="
                + String(self.max_no_improvement)
                + " must be >= 0",
            )
        if self.n_init <= 0:
            raise InvalidParameterError.error(
                "n_init", "n_init=" + String(self.n_init) + " must be > 0"
            )
        if self.reassignment_ratio < 0.0 or self.reassignment_ratio > 1.0:
            raise InvalidParameterError.error(
                "reassignment_ratio",
                "reassignment_ratio="
                + String(self.reassignment_ratio)
                + " must be in [0.0, 1.0]",
            )
        if self.init != "k-means++" and self.init != "random":
            raise InvalidParameterError.error(
                "init",
                "init='"
                + self.init
                + "' is unsupported. Expected 'k-means++' or 'random'.",
            )

    def partial_fit[in_dtype: DType](mut self, X: Matrix[in_dtype]) raises:
        check_array[in_dtype](X)
        var n = X.rows
        var d = X.cols

        if not self.is_fitted:
            self._validate_params(n)
            var X_comp = X.cast[Self.compute_dtype]()
            var k = self.n_clusters

            if self.init == "k-means++":
                self.cluster_centers_ = _init_centroids_kmeans_plus_plus(
                    X_comp, k, self.random_state
                )
            else:
                self.cluster_centers_ = _init_centroids_random(
                    X_comp, k, self.random_state
                )

            self.counts_ = List[Int](capacity=k)
            for _ in range(k):
                self.counts_.append(0)
            self.n_features_in_ = d
            self.n_steps_ = 0
            self.n_iter_ = 0
            self.is_fitted = True
        else:
            if d != self.n_features_in_:
                raise DimensionMismatchError.error(
                    "X.cols == " + String(self.n_features_in_),
                    "X.cols == " + String(d),
                    "MiniBatchKMeans.partial_fit",
                )

        var X_comp = X.cast[Self.compute_dtype]()

        var batch_labels = List[Int](capacity=n)
        for _ in range(n):
            batch_labels.append(0)
        var batch_dists = List[Scalar[Self.compute_dtype]](capacity=n)
        for _ in range(n):
            batch_dists.append(0)

        _ = _assign_nearest_and_inertia(
            X_comp, self.cluster_centers_, batch_labels, batch_dists
        )

        for i in range(n):
            var c = batch_labels[i]
            self.counts_[c] += 1
            var eta = 1.0 / Scalar[Self.compute_dtype](self.counts_[c])
            for j in range(d):
                var diff = X_comp[i, j] - self.cluster_centers_[c, j]
                self.cluster_centers_[c, j] += eta * diff

        self.inertia_ = _assign_nearest_and_inertia(
            X_comp, self.cluster_centers_, batch_labels, batch_dists
        )
        self.labels_ = batch_labels^
        self.n_steps_ += 1

    def fit[in_dtype: DType](mut self, X: Matrix[in_dtype]) raises:
        check_array[in_dtype](X)
        var n = X.rows
        var d = X.cols

        self._validate_params(n)

        var X_comp = X.cast[Self.compute_dtype]()
        var k = self.n_clusters
        var actual_batch_size = min(self.batch_size, n)

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
        var best_counts = List[Int](capacity=k)
        for _ in range(k):
            best_counts.append(0)
        var best_n_iter = 0

        var X_batch = Matrix[Self.compute_dtype](actual_batch_size, d, 0)
        var old_centers = Matrix[Self.compute_dtype](k, d, 0)
        var batch_labels = List[Int](capacity=actual_batch_size)
        for _ in range(actual_batch_size):
            batch_labels.append(0)
        var batch_dists = List[Scalar[Self.compute_dtype]](
            capacity=actual_batch_size
        )
        for _ in range(actual_batch_size):
            batch_dists.append(0)

        var full_labels = List[Int](capacity=n)
        for _ in range(n):
            full_labels.append(0)
        var full_dists = List[Scalar[Self.compute_dtype]](capacity=n)
        for _ in range(n):
            full_dists.append(0)

        var alpha = Scalar[Self.compute_dtype](
            min(1.0, Float64(actual_batch_size) * 2.0 / Float64(n))
        )

        for run in range(self.n_init):
            var run_seed = self.random_state + run * 31

            var current_centers: Matrix[Self.compute_dtype]
            if self.init == "k-means++":
                current_centers = _init_centroids_kmeans_plus_plus(
                    X_comp, k, run_seed
                )
            else:
                current_centers = _init_centroids_random(X_comp, k, run_seed)

            var counts = List[Int](capacity=k)
            for _ in range(k):
                counts.append(0)

            var rng = PRNG(run_seed + 100)
            var no_improvement_count = 0
            var ewa_inertia: Scalar[Self.compute_dtype] = -1.0
            var best_ewa_inertia: Scalar[Self.compute_dtype] = -1.0
            var iter_count = 0

            for iter_idx in range(self.max_iter):
                iter_count = iter_idx + 1

                for c in range(k):
                    for j in range(d):
                        old_centers[c, j] = current_centers[c, j]

                for r in range(actual_batch_size):
                    var src_row = rng.next_int(n)
                    for j in range(d):
                        X_batch[r, j] = X_comp[src_row, j]

                var batch_inertia = _assign_nearest_and_inertia(
                    X_batch, current_centers, batch_labels, batch_dists
                )

                if ewa_inertia < 0:
                    ewa_inertia = batch_inertia
                else:
                    ewa_inertia = (
                        alpha * batch_inertia + (1.0 - alpha) * ewa_inertia
                    )

                for i in range(actual_batch_size):
                    var c = batch_labels[i]
                    counts[c] += 1
                    var eta = 1.0 / Scalar[Self.compute_dtype](counts[c])
                    for j in range(d):
                        var diff = X_batch[i, j] - current_centers[c, j]
                        current_centers[c, j] += eta * diff

                if self.reassignment_ratio > 0.0:
                    var total_samples: Int = 0
                    for c in range(k):
                        total_samples += counts[c]
                    var min_count = Int(
                        Float64(total_samples)
                        / Float64(k)
                        * self.reassignment_ratio
                    )
                    for c in range(k):
                        if counts[c] < min_count:
                            var max_d_idx = -1
                            var max_d_val: Scalar[Self.compute_dtype] = 0
                            for b in range(actual_batch_size):
                                if batch_dists[b] > max_d_val:
                                    max_d_val = batch_dists[b]
                                    max_d_idx = b
                            if max_d_idx >= 0:
                                for j in range(d):
                                    current_centers[c, j] = X_batch[
                                        max_d_idx, j
                                    ]
                                batch_dists[max_d_idx] = 0
                            else:
                                var rand_b = rng.next_int(actual_batch_size)
                                for j in range(d):
                                    current_centers[c, j] = X_batch[rand_b, j]
                            counts[c] = 1

                var center_shift: Scalar[Self.compute_dtype] = 0
                for c in range(k):
                    for j in range(d):
                        var diff = current_centers[c, j] - old_centers[c, j]
                        center_shift += diff * diff

                if (
                    best_ewa_inertia < 0
                    or ewa_inertia < best_ewa_inertia - effective_tol
                ):
                    best_ewa_inertia = ewa_inertia
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if (
                    self.max_no_improvement > 0
                    and no_improvement_count >= self.max_no_improvement
                ):
                    break

                var normalized_shift = center_shift / Scalar[
                    Self.compute_dtype
                ](k)
                if self.tol > 0.0 and normalized_shift <= effective_tol:
                    break

            var final_inertia = _assign_nearest_and_inertia(
                X_comp, current_centers, full_labels, full_dists
            )

            if run == 0 or final_inertia < best_inertia:
                best_inertia = final_inertia
                best_centers = current_centers.copy()
                for c in range(k):
                    best_counts[c] = counts[c]
                best_n_iter = iter_count

        self.cluster_centers_ = best_centers^
        self.counts_ = best_counts^
        self.inertia_ = _assign_nearest_and_inertia(
            X_comp, self.cluster_centers_, full_labels, full_dists
        )
        self.labels_ = full_labels^
        self.n_iter_ = best_n_iter
        self.n_steps_ = best_n_iter
        self.n_features_in_ = d
        self.is_fitted = True

    def predict[in_dtype: DType](self, X: Matrix[in_dtype]) raises -> List[Int]:
        check_is_fitted("MiniBatchKMeans", self.is_fitted)
        check_array[in_dtype](X)
        if X.cols != self.n_features_in_:
            raise DimensionMismatchError.error(
                "X.cols == " + String(self.n_features_in_),
                "X.cols == " + String(X.cols),
                "MiniBatchKMeans.predict",
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
        check_is_fitted("MiniBatchKMeans", self.is_fitted)
        check_array[in_dtype](X)
        if X.cols != self.n_features_in_:
            raise DimensionMismatchError.error(
                "X.cols == " + String(self.n_features_in_),
                "X.cols == " + String(X.cols),
                "MiniBatchKMeans.transform",
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
        check_is_fitted("MiniBatchKMeans", self.is_fitted)
        check_array[in_dtype](X)
        if X.cols != self.n_features_in_:
            raise DimensionMismatchError.error(
                "X.cols == " + String(self.n_features_in_),
                "X.cols == " + String(X.cols),
                "MiniBatchKMeans.score",
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
