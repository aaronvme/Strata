from std.math import sqrt
from ..base.estimator import Transformer
from ..core.matrix import Matrix
from ..core.dataset import Dataset
from ..utils.validation import check_is_fitted, check_array
from ..exceptions.errors import NotFittedError, DataConversionError


struct StandardScaler[compute_dtype: DType = DType.float64](
    Copyable, Movable, Transformer
):
    """Standardizes features by removing the mean and scaling to unit variance.
    """

    var is_fitted: Bool
    var fit_dtype: DType
    var with_mean: Bool
    var with_std: Bool
    var mean_: List[Scalar[Self.compute_dtype]]
    var scale_: List[Scalar[Self.compute_dtype]]

    def __init__(out self, with_mean: Bool = True, with_std: Bool = True):
        self.is_fitted = False
        self.fit_dtype = DType.float64
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = List[Scalar[Self.compute_dtype]]()
        self.scale_ = List[Scalar[Self.compute_dtype]]()

    def fit[in_dtype: DType](mut self, X: Matrix[in_dtype]) raises:
        check_array[in_dtype](X)
        var n_rows = X.rows
        var n_cols = X.cols

        self.mean_ = List[Scalar[Self.compute_dtype]](capacity=n_cols)
        self.scale_ = List[Scalar[Self.compute_dtype]](capacity=n_cols)

        for c in range(n_cols):
            var col_sum: Scalar[Self.compute_dtype] = 0
            for r in range(n_rows):
                col_sum += Scalar[Self.compute_dtype](X[r, c])
            var mean_val = col_sum / Scalar[Self.compute_dtype](n_rows)
            self.mean_.append(mean_val)

            var var_sum: Scalar[Self.compute_dtype] = 0
            for r in range(n_rows):
                var diff = Scalar[Self.compute_dtype](X[r, c]) - mean_val
                var_sum += diff * diff
            var std_val = sqrt(var_sum / Scalar[Self.compute_dtype](n_rows))
            if std_val == 0:
                std_val = 1
            self.scale_.append(std_val)

        self.fit_dtype = in_dtype
        self.is_fitted = True

    def fit[
        feat_dtype: DType,
        target_dtype: DType,
    ](mut self, dataset: Dataset[feat_dtype, target_dtype]) raises:
        self.fit[feat_dtype](dataset.records)

    def transform[
        in_dtype: DType
    ](self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        check_is_fitted("StandardScaler", self.is_fitted)
        if in_dtype != self.fit_dtype:
            raise DataConversionError.error(
                "StandardScaler.transform received Matrix["
                + String(in_dtype)
                + "] but was fitted on Matrix["
                + String(self.fit_dtype)
                + "]"
            )
        check_array[in_dtype](X)

        var res = Matrix[in_dtype](X.rows, X.cols, 0)
        for r in range(X.rows):
            for c in range(X.cols):
                var val = Scalar[Self.compute_dtype](X[r, c])
                if self.with_mean:
                    val -= self.mean_[c]
                if self.with_std:
                    val /= self.scale_[c]
                res[r, c] = Scalar[in_dtype](val)
        return res^

    def transform[
        feat_dtype: DType,
        target_dtype: DType,
    ](self, dataset: Dataset[feat_dtype, target_dtype]) raises -> Dataset[
        feat_dtype, target_dtype
    ]:
        var scaled_records = self.transform[feat_dtype](dataset.records)
        return Dataset[feat_dtype, target_dtype](
            scaled_records^,
            dataset.targets.copy(),
            dataset.feature_names.copy(),
            dataset.target_names.copy(),
        )

    def fit_transform[
        in_dtype: DType
    ](mut self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        self.fit[in_dtype](X)
        return self.transform[in_dtype](X)

    def fit_transform[
        feat_dtype: DType,
        target_dtype: DType,
    ](mut self, dataset: Dataset[feat_dtype, target_dtype]) raises -> Dataset[
        feat_dtype, target_dtype
    ]:
        self.fit[feat_dtype, target_dtype](dataset)
        return self.transform[feat_dtype, target_dtype](dataset)
