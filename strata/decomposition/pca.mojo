from std.math import sqrt
from ..base.estimator import Transformer
from ..core.matrix import Matrix
from ..core.dataset import Dataset
from ..core.linalg import svd, gemm
from ..utils.validation import check_is_fitted, check_array
from ..exceptions.errors import (
    NotFittedError,
    DimensionMismatchError,
    InvalidParameterError,
)


struct PCA[compute_dtype: DType = DType.float64](
    Copyable, Movable, Transformer
):
    """Principal Component Analysis (PCA).

    Linear dimensionality reduction using Singular Value Decomposition of the
    centered data matrix to project it to a lower dimensional subspace:

    $$
    X_{\\text{projected}} = (X - \\mu) V_k
    $$


    Parameters:
        n_components: Number of components to keep. If 0, all components are kept. Default 0.
        whiten: When True, components vectors are divided by the singular values to ensure uncorrelated outputs with unit component-wise variances. Default False.

    Attributes:
        components_: Principal axes in feature space, representing directions of maximum variance matrix of shape $(K, D)$.
        explained_variance_: Variance explained by each selected component vector of length $K$.
        explained_variance_ratio_: Percentage of variance explained by each component.
        singular_values_: Singular values corresponding to each of the selected components.
        mean_: Per-feature empirical mean estimated from the training set.
        n_components_: Estimated number of components.
        n_features_in_: Number of features seen during fit.
        is_fitted: Boolean flag indicating if estimator has been fitted.

    Examples:
        ```mojo
        from strata.decomposition import PCA
        from strata.core import Matrix

        var pca = PCA[DType.float64](n_components=2)
        pca.fit(X_train)
        var X_proj = pca.transform(X_train)
        ```
    """

    var is_fitted: Bool
    var n_components: Int
    var whiten: Bool
    var n_samples_seen_: Int
    var n_features_in_: Int
    var n_components_: Int
    var mean_: List[Scalar[Self.compute_dtype]]
    var components_: Matrix[Self.compute_dtype]
    var explained_variance_: List[Scalar[Self.compute_dtype]]
    var explained_variance_ratio_: List[Scalar[Self.compute_dtype]]
    var singular_values_: List[Scalar[Self.compute_dtype]]

    def __init__(
        out self,
        n_components: Int = 0,
        whiten: Bool = False,
    ):
        """Initialize the PCA transformer.

        Args:
            n_components: Number of principal components to project onto. Default 0 (all).
            whiten: Whether to scale component projections by variance. Default False.
        """

        self.is_fitted = False
        self.n_components = n_components
        self.whiten = whiten
        self.n_samples_seen_ = 0
        self.n_features_in_ = 0
        self.n_components_ = 0
        self.mean_ = List[Scalar[Self.compute_dtype]]()
        self.components_ = Matrix[Self.compute_dtype](0, 0, 0)
        self.explained_variance_ = List[Scalar[Self.compute_dtype]]()
        self.explained_variance_ratio_ = List[Scalar[Self.compute_dtype]]()
        self.singular_values_ = List[Scalar[Self.compute_dtype]]()

    def __init__(out self, *, copy: Self):
        self.is_fitted = copy.is_fitted
        self.n_components = copy.n_components
        self.whiten = copy.whiten
        self.n_samples_seen_ = copy.n_samples_seen_
        self.n_features_in_ = copy.n_features_in_
        self.n_components_ = copy.n_components_
        self.mean_ = copy.mean_.copy()
        self.components_ = copy.components_.copy()
        self.explained_variance_ = copy.explained_variance_.copy()
        self.explained_variance_ratio_ = copy.explained_variance_ratio_.copy()
        self.singular_values_ = copy.singular_values_.copy()

    def fit[in_dtype: DType](mut self, X: Matrix[in_dtype]) raises:
        """Fits the PCA model on matrix X."""
        check_array[in_dtype](X)
        var N = X.rows
        var D = X.cols

        if N < 1 or D < 1:
            raise InvalidParameterError.error(
                "X", "n_samples and n_features must be >= 1"
            )

        var max_components = min(N, D)
        if self.n_components < 0 or self.n_components > max_components:
            raise InvalidParameterError.error(
                "n_components",
                "n_components="
                + String(self.n_components)
                + " must be between 0 and min(n_samples, n_features)="
                + String(max_components),
            )

        var k = self.n_components if self.n_components > 0 else max_components
        var X_comp = X.cast[Self.compute_dtype]()
        var mean_vec = X_comp.mean_along_axis_0()

        var X_centered = Matrix[Self.compute_dtype](N, D, 0)
        for r in range(N):
            for c in range(D):
                X_centered[r, c] = X_comp[r, c] - mean_vec[c]

        var svd_res = svd[Self.compute_dtype](X_centered, full_matrices=False)
        var K_svd = len(svd_res.S)

        var components = Matrix[Self.compute_dtype](k, D, 0)
        for i in range(k):
            var max_abs_val: Scalar[Self.compute_dtype] = 0
            var max_col_idx = 0
            for c in range(D):
                var val = svd_res.Vt[i, c]
                var abs_val = abs(val)
                if abs_val > max_abs_val:
                    max_abs_val = abs_val
                    max_col_idx = c

            var flip_sign: Scalar[Self.compute_dtype] = (
                -1.0 if svd_res.Vt[i, max_col_idx] < 0 else 1.0
            )
            for c in range(D):
                components[i, c] = svd_res.Vt[i, c] * flip_sign

        var total_variance: Scalar[Self.compute_dtype] = 0
        var denom: Scalar[Self.compute_dtype] = (
            Scalar[Self.compute_dtype](N - 1) if N > 1 else 1.0
        )
        for i in range(K_svd):
            var s = svd_res.S[i]
            total_variance += (s * s) / denom

        var singular_values = List[Scalar[Self.compute_dtype]](capacity=k)
        var explained_variance = List[Scalar[Self.compute_dtype]](capacity=k)
        var explained_variance_ratio = List[Scalar[Self.compute_dtype]](
            capacity=k
        )

        for i in range(k):
            var s = svd_res.S[i]
            var var_i = (s * s) / denom
            singular_values.append(s)
            explained_variance.append(var_i)
            if total_variance > 0:
                explained_variance_ratio.append(var_i / total_variance)
            else:
                explained_variance_ratio.append(0.0)

        self.mean_ = mean_vec^
        self.components_ = components^
        self.singular_values_ = singular_values^
        self.explained_variance_ = explained_variance^
        self.explained_variance_ratio_ = explained_variance_ratio^
        self.n_samples_seen_ = N
        self.n_features_in_ = D
        self.n_components_ = k
        self.is_fitted = True

    def fit[
        feat_dtype: DType,
        target_dtype: DType,
    ](mut self, dataset: Dataset[feat_dtype, target_dtype]) raises:
        """Fits PCA using a Dataset container."""
        self.fit[feat_dtype](dataset.records)

    def transform[
        in_dtype: DType
    ](self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        """Projects matrix X onto the principal components."""
        check_is_fitted("PCA", self.is_fitted)
        check_array[in_dtype](X)

        if X.cols != self.n_features_in_:
            raise DimensionMismatchError.error(
                "X.cols == " + String(self.n_features_in_),
                "X.cols == " + String(X.cols),
                "PCA.transform",
            )

        var N = X.rows
        var D = X.cols
        var k = self.n_components_
        var X_comp = X.cast[Self.compute_dtype]()

        var X_centered = Matrix[Self.compute_dtype](N, D, 0)
        for r in range(N):
            for c in range(D):
                X_centered[r, c] = X_comp[r, c] - self.mean_[c]

        var comp_t = self.components_.transpose()
        var X_proj = gemm(X_centered, comp_t)

        if self.whiten:
            for j in range(k):
                var scale = sqrt(self.explained_variance_[j])
                if scale > 0:
                    for r in range(N):
                        X_proj[r, j] /= scale

        return X_proj.cast[in_dtype]()

    def fit_transform[
        in_dtype: DType
    ](mut self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        """Fits PCA to X and returns the projected data."""
        self.fit[in_dtype](X)
        return self.transform[in_dtype](X)

    def transform[
        feat_dtype: DType,
        target_dtype: DType,
    ](self, dataset: Dataset[feat_dtype, target_dtype]) raises -> Dataset[
        feat_dtype, target_dtype
    ]:
        """Transforms a Dataset container with PCA."""
        var proj_records = self.transform[feat_dtype](dataset.records)
        return Dataset[feat_dtype, target_dtype](
            proj_records^,
            dataset.targets.copy(),
            List[String](),
            dataset.target_names.copy(),
        )

    def fit_transform[
        feat_dtype: DType,
        target_dtype: DType,
    ](mut self, dataset: Dataset[feat_dtype, target_dtype]) raises -> Dataset[
        feat_dtype, target_dtype
    ]:
        """Fits PCA and transforms a Dataset container."""
        self.fit[feat_dtype, target_dtype](dataset)
        return self.transform[feat_dtype, target_dtype](dataset)

    def inverse_transform[
        in_dtype: DType
    ](self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        """Transforms data back to its original space."""
        check_is_fitted("PCA", self.is_fitted)
        check_array[in_dtype](X)

        if X.cols != self.n_components_:
            raise DimensionMismatchError.error(
                "X.cols == " + String(self.n_components_),
                "X.cols == " + String(X.cols),
                "PCA.inverse_transform",
            )

        var N = X.rows
        var k = self.n_components_
        var D = self.n_features_in_
        var X_comp = X.cast[Self.compute_dtype]()

        if self.whiten:
            for j in range(k):
                var scale = sqrt(self.explained_variance_[j])
                if scale > 0:
                    for r in range(N):
                        X_comp[r, j] *= scale
                else:
                    for r in range(N):
                        X_comp[r, j] = 0

        var X_orig = gemm(X_comp, self.components_)

        for r in range(N):
            for c in range(D):
                X_orig[r, c] += self.mean_[c]

        return X_orig.cast[in_dtype]()
