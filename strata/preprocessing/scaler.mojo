from std.math import sqrt
from ..base.estimator import Transformer
from ..core.matrix import Matrix
from ..core.dataset import Dataset
from ..utils.validation import check_is_fitted, check_array
from ..exceptions.errors import NotFittedError

struct StandardScaler[dtype: DType = DType.float64](
    Copyable, Movable, Transformer
):
    var is_fitted: Bool
    var with_mean: Bool
    var with_std: Bool
    var mean_: List[Scalar[Self.dtype]]
    var scale_: List[Scalar[Self.dtype]]

    def __init__(out self, with_mean: Bool = True, with_std: Bool = True):
        self.is_fitted = False
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = List[Scalar[Self.dtype]]()
        self.scale_ = List[Scalar[Self.dtype]]()

    def fit(mut self, X: Matrix[Self.dtype]) raises:
        check_array[Self.dtype](X)
        var n_rows = X.rows
        var n_cols = X.cols

        self.mean_ = List[Scalar[Self.dtype]](capacity=n_cols)
        self.scale_ = List[Scalar[Self.dtype]](capacity=n_cols)

        for c in range(n_cols):
            var col_sum: Scalar[Self.dtype] = 0
            for r in range(n_rows):
                col_sum += X[r, c]
            var mean_val = col_sum / Scalar[Self.dtype](n_rows)
            self.mean_.append(mean_val)

            var var_sum: Scalar[Self.dtype] = 0
            for r in range(n_rows):
                var diff = X[r, c] - mean_val
                var_sum += diff * diff
            var std_val = sqrt(var_sum / Scalar[Self.dtype](n_rows))
            if std_val == 0:
                std_val = 1
            self.scale_.append(std_val)

        self.is_fitted = True

    def fit[target_dtype: DType](
        mut self, dataset: Dataset[Self.dtype, target_dtype]
    ) raises:
        self.fit(dataset.records)

    def transform(self, X: Matrix[Self.dtype]) raises -> Matrix[Self.dtype]:
        check_is_fitted("StandardScaler", self.is_fitted)
        check_array[Self.dtype](X)

        var res = Matrix[Self.dtype](X.rows, X.cols, 0)
        for r in range(X.rows):
            for c in range(X.cols):
                var val = X[r, c]
                if self.with_mean:
                    val -= self.mean_[c]
                if self.with_std:
                    val /= self.scale_[c]
                res[r, c] = val
        return res^

    def transform[target_dtype: DType](
        self, dataset: Dataset[Self.dtype, target_dtype]
    ) raises -> Dataset[Self.dtype, target_dtype]:
        var scaled_records = self.transform(dataset.records)
        return Dataset[Self.dtype, target_dtype](
            scaled_records^,
            dataset.targets.copy(),
            dataset.feature_names.copy(),
            dataset.target_names.copy(),
        )

    def fit_transform(
        mut self, X: Matrix[Self.dtype]
    ) raises -> Matrix[Self.dtype]:
        self.fit(X)
        return self.transform(X)

    def fit_transform[target_dtype: DType](
        mut self, dataset: Dataset[Self.dtype, target_dtype]
    ) raises -> Dataset[Self.dtype, target_dtype]:
        self.fit(dataset)
        return self.transform(dataset)
