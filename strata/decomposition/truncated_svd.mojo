from std.math import sqrt
from ..base.estimator import Transformer
from ..core.matrix import Matrix
from ..core.csr_matrix import CSRMatrix
from ..core.dataset import Dataset
from ..core.linalg import svd, gemm
from ..core.sparse_ops import spmm
from ..utils.validation import check_is_fitted, check_array, check_sparse
from ..exceptions.errors import (
    NotFittedError,
    DimensionMismatchError,
    InvalidParameterError,
)


struct TruncatedSVD[compute_dtype: DType = DType.float64](
    Copyable, Movable, Transformer
):
    """Dimensionality reduction using truncated SVD."""

    var is_fitted: Bool
    var n_components: Int
    var n_samples_seen_: Int
    var n_features_in_: Int
    var n_components_: Int
    var components_: Matrix[Self.compute_dtype]
    var explained_variance_: List[Scalar[Self.compute_dtype]]
    var explained_variance_ratio_: List[Scalar[Self.compute_dtype]]
    var singular_values_: List[Scalar[Self.compute_dtype]]

    def __init__(
        out self,
        n_components: Int = 2,
    ):
        self.is_fitted = False
        self.n_components = n_components
        self.n_samples_seen_ = 0
        self.n_features_in_ = 0
        self.n_components_ = 0
        self.components_ = Matrix[Self.compute_dtype](0, 0, 0)
        self.explained_variance_ = List[Scalar[Self.compute_dtype]]()
        self.explained_variance_ratio_ = List[Scalar[Self.compute_dtype]]()
        self.singular_values_ = List[Scalar[Self.compute_dtype]]()

    def __init__(out self, *, copy: Self):
        self.is_fitted = copy.is_fitted
        self.n_components = copy.n_components
        self.n_samples_seen_ = copy.n_samples_seen_
        self.n_features_in_ = copy.n_features_in_
        self.n_components_ = copy.n_components_
        self.components_ = copy.components_.copy()
        self.explained_variance_ = copy.explained_variance_.copy()
        self.explained_variance_ratio_ = copy.explained_variance_ratio_.copy()
        self.singular_values_ = copy.singular_values_.copy()

    def fit[in_dtype: DType](mut self, X: Matrix[in_dtype]) raises:
        """Fits TruncatedSVD on dense matrix X."""
        check_array[in_dtype](X)
        var N = X.rows
        var D = X.cols

        if N < 1 or D < 1:
            raise InvalidParameterError.error(
                "X", "n_samples and n_features must be >= 1"
            )

        var max_components = min(N, D)
        if self.n_components < 1 or self.n_components > max_components:
            raise InvalidParameterError.error(
                "n_components",
                "n_components="
                + String(self.n_components)
                + " must be between 1 and min(n_samples, n_features)="
                + String(max_components),
            )

        var k = self.n_components
        var X_comp = X.cast[Self.compute_dtype]()
        var svd_res = svd[Self.compute_dtype](X_comp, full_matrices=False)

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

        var comp_t = components.transpose()
        var X_proj = gemm(X_comp, comp_t)

        var denom: Scalar[Self.compute_dtype] = (
            Scalar[Self.compute_dtype](N - 1) if N > 1 else 1.0
        )

        var means = X_comp.mean_along_axis_0()
        var total_variance: Scalar[Self.compute_dtype] = 0
        for c in range(D):
            var col_mean = means[c]
            var col_sq_diff: Scalar[Self.compute_dtype] = 0
            for r in range(N):
                var diff = X_comp[r, c] - col_mean
                col_sq_diff += diff * diff
            total_variance += col_sq_diff / denom

        var singular_values = List[Scalar[Self.compute_dtype]](capacity=k)
        var explained_variance = List[Scalar[Self.compute_dtype]](capacity=k)
        var explained_variance_ratio = List[Scalar[Self.compute_dtype]](
            capacity=k
        )

        for j in range(k):
            singular_values.append(svd_res.S[j])

            var col_sum: Scalar[Self.compute_dtype] = 0
            for r in range(N):
                col_sum += X_proj[r, j]
            var z_mean = col_sum / Scalar[Self.compute_dtype](N)

            var var_sum: Scalar[Self.compute_dtype] = 0
            for r in range(N):
                var diff = X_proj[r, j] - z_mean
                var_sum += diff * diff
            var var_j = var_sum / denom
            explained_variance.append(var_j)

            if total_variance > 0:
                explained_variance_ratio.append(var_j / total_variance)
            else:
                explained_variance_ratio.append(0.0)

        self.components_ = components^
        self.singular_values_ = singular_values^
        self.explained_variance_ = explained_variance^
        self.explained_variance_ratio_ = explained_variance_ratio^
        self.n_samples_seen_ = N
        self.n_features_in_ = D
        self.n_components_ = k
        self.is_fitted = True

    def fit[in_dtype: DType](mut self, X: CSRMatrix[in_dtype]) raises:
        """Fits TruncatedSVD on sparse matrix X."""
        check_sparse[in_dtype](
            X.rows,
            X.cols,
            X.data,
            X.indices,
            X.indptr,
            is_csr=True,
            allow_empty=False,
            caller="TruncatedSVD.fit",
        )
        var N = X.rows
        var D = X.cols

        if N < 1 or D < 1:
            raise InvalidParameterError.error(
                "X", "n_samples and n_features must be >= 1"
            )

        var max_components = min(N, D)
        if self.n_components < 1 or self.n_components > max_components:
            raise InvalidParameterError.error(
                "n_components",
                "n_components="
                + String(self.n_components)
                + " must be between 1 and min(n_samples, n_features)="
                + String(max_components),
            )

        var k = self.n_components
        var X_comp = X.cast[Self.compute_dtype]()
        var dense_X = X_comp.to_dense()
        var svd_res = svd[Self.compute_dtype](dense_X, full_matrices=False)

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

        var comp_t = components.transpose()
        var X_proj = spmm[Self.compute_dtype](X_comp, comp_t)

        var denom: Scalar[Self.compute_dtype] = (
            Scalar[Self.compute_dtype](N - 1) if N > 1 else 1.0
        )

        var col_sums = List[Scalar[Self.compute_dtype]](capacity=D)
        var col_sq_sums = List[Scalar[Self.compute_dtype]](capacity=D)
        for _ in range(D):
            col_sums.append(0)
            col_sq_sums.append(0)

        for i in range(len(X_comp.data)):
            var c = X_comp.indices[i]
            var val = X_comp.data[i]
            col_sums[c] += val
            col_sq_sums[c] += val * val

        var total_variance: Scalar[Self.compute_dtype] = 0
        for c in range(D):
            var col_mean = col_sums[c] / Scalar[Self.compute_dtype](N)
            var var_c = (
                col_sq_sums[c]
                - Scalar[Self.compute_dtype](N) * col_mean * col_mean
            ) / denom
            if var_c > 0:
                total_variance += var_c

        var singular_values = List[Scalar[Self.compute_dtype]](capacity=k)
        var explained_variance = List[Scalar[Self.compute_dtype]](capacity=k)
        var explained_variance_ratio = List[Scalar[Self.compute_dtype]](
            capacity=k
        )

        for j in range(k):
            singular_values.append(svd_res.S[j])

            var col_sum: Scalar[Self.compute_dtype] = 0
            for r in range(N):
                col_sum += X_proj[r, j]
            var z_mean = col_sum / Scalar[Self.compute_dtype](N)

            var var_sum: Scalar[Self.compute_dtype] = 0
            for r in range(N):
                var diff = X_proj[r, j] - z_mean
                var_sum += diff * diff
            var var_j = var_sum / denom
            explained_variance.append(var_j)

            if total_variance > 0:
                explained_variance_ratio.append(var_j / total_variance)
            else:
                explained_variance_ratio.append(0.0)

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
        """Fits TruncatedSVD using a Dataset container."""
        self.fit[feat_dtype](dataset.records)

    def transform[
        in_dtype: DType
    ](self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        """Projects dense matrix X onto the truncated components."""
        check_is_fitted("TruncatedSVD", self.is_fitted)
        check_array[in_dtype](X)

        if X.cols != self.n_features_in_:
            raise DimensionMismatchError.error(
                "X.cols == " + String(self.n_features_in_),
                "X.cols == " + String(X.cols),
                "TruncatedSVD.transform",
            )

        var X_comp = X.cast[Self.compute_dtype]()
        var comp_t = self.components_.transpose()
        var X_proj = gemm(X_comp, comp_t)

        return X_proj.cast[in_dtype]()

    def transform[
        in_dtype: DType
    ](self, X: CSRMatrix[in_dtype]) raises -> Matrix[in_dtype]:
        """Projects sparse matrix X onto the truncated components without dense conversion.
        """
        check_is_fitted("TruncatedSVD", self.is_fitted)
        check_sparse[in_dtype](
            X.rows,
            X.cols,
            X.data,
            X.indices,
            X.indptr,
            is_csr=True,
            allow_empty=False,
            caller="TruncatedSVD.transform",
        )

        if X.cols != self.n_features_in_:
            raise DimensionMismatchError.error(
                "X.cols == " + String(self.n_features_in_),
                "X.cols == " + String(X.cols),
                "TruncatedSVD.transform",
            )

        var X_comp = X.cast[Self.compute_dtype]()
        var comp_t = self.components_.transpose()
        var X_proj = spmm[Self.compute_dtype](X_comp, comp_t)

        return X_proj.cast[in_dtype]()

    def fit_transform[
        in_dtype: DType
    ](mut self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        """Fits TruncatedSVD to X and returns the projected data."""
        self.fit[in_dtype](X)
        return self.transform[in_dtype](X)

    def fit_transform[
        in_dtype: DType
    ](mut self, X: CSRMatrix[in_dtype]) raises -> Matrix[in_dtype]:
        """Fits TruncatedSVD to sparse X and returns the projected data."""
        self.fit[in_dtype](X)
        return self.transform[in_dtype](X)

    def transform[
        feat_dtype: DType,
        target_dtype: DType,
    ](self, dataset: Dataset[feat_dtype, target_dtype]) raises -> Dataset[
        feat_dtype, target_dtype
    ]:
        """Transforms a Dataset container with TruncatedSVD."""
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
        """Fits TruncatedSVD and transforms a Dataset container."""
        self.fit[feat_dtype, target_dtype](dataset)
        return self.transform[feat_dtype, target_dtype](dataset)

    def inverse_transform[
        in_dtype: DType
    ](self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        """Transforms data back to its original space."""
        check_is_fitted("TruncatedSVD", self.is_fitted)
        check_array[in_dtype](X)

        if X.cols != self.n_components_:
            raise DimensionMismatchError.error(
                "X.cols == " + String(self.n_components_),
                "X.cols == " + String(X.cols),
                "TruncatedSVD.inverse_transform",
            )

        var X_comp = X.cast[Self.compute_dtype]()
        var X_orig = gemm(X_comp, self.components_)

        return X_orig.cast[in_dtype]()
