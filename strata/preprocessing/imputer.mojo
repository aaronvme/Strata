from std.math import isinf, isnan, nan
from ..core.matrix import Matrix
from ..core.dataset import Dataset
from ..utils.validation import (
    check_is_fitted,
    check_array,
    check_floating_dtype,
)
from ..exceptions.errors import (
    NotFittedError,
    DataConversionError,
    DimensionMismatchError,
    InvalidParameterError,
)


struct SimpleImputer[compute_dtype: DType = DType.float64](Copyable, Movable):
    """Replace missing values with a statistic learned from each column.

    Missing entries are located by comparing against missing_values, which
    defaults to NaN. Each column is filled independently using the chosen
    strategy, so the output keeps the shape of the input.

    Parameters:
        compute_dtype: Computational precision data type. Default DType.float64.

    Args:
        strategy: Replacement statistic ('mean', 'median', 'most_frequent', 'constant'). Default 'mean'.
        missing_values: The placeholder that marks a missing entry. Default NaN.
        fill_value: Replacement used when strategy is 'constant'. Default 0.0.

    Attributes:
        statistics_: Replacement value of each feature determined during fitting.
        n_features_in_: Number of features seen during fit.
        is_fitted: Boolean flag indicating if estimator has been fitted.

    Examples:
        ```mojo
        from strata.preprocessing import SimpleImputer
        from strata.core import Matrix

        var imputer = SimpleImputer[DType.float64](strategy="median")
        imputer.fit(X_train)
        var X_filled = imputer.transform(X_train)
        ```
    """

    var is_fitted: Bool
    var fit_dtype: DType
    var strategy: String
    var missing_values: Scalar[Self.compute_dtype]
    var fill_value: Scalar[Self.compute_dtype]
    var statistics_: List[Scalar[Self.compute_dtype]]
    var n_features_in_: Int

    def __init__(
        out self,
        strategy: String = "mean",
        missing_values: Scalar[Self.compute_dtype] = nan[Self.compute_dtype](),
        fill_value: Scalar[Self.compute_dtype] = 0.0,
    ) raises:
        """Initialize the SimpleImputer.

        Args:
            strategy: Replacement statistic ('mean', 'median', 'most_frequent', 'constant'). Default 'mean'.
            missing_values: The placeholder that marks a missing entry. Default NaN.
            fill_value: Replacement used when strategy is 'constant'. Default 0.0.

        Raises:
            InvalidParameterError: If strategy is unrecognized.
        """

        check_floating_dtype[Self.compute_dtype, "SimpleImputer"]()
        if (
            strategy != "mean"
            and strategy != "median"
            and strategy != "most_frequent"
            and strategy != "constant"
        ):
            raise InvalidParameterError.error(
                "strategy",
                "expected 'mean', 'median', 'most_frequent' or 'constant', got"
                " '"
                + strategy
                + "'",
            )
        self.is_fitted = False
        self.fit_dtype = DType.float64
        self.strategy = strategy
        self.missing_values = missing_values
        self.fill_value = fill_value
        self.statistics_ = List[Scalar[Self.compute_dtype]]()
        self.n_features_in_ = 0

    def __init__(out self, *, copy: Self):
        """Copies an existing SimpleImputer instance."""
        self.is_fitted = copy.is_fitted
        self.fit_dtype = copy.fit_dtype
        self.strategy = copy.strategy
        self.missing_values = copy.missing_values
        self.fill_value = copy.fill_value
        self.statistics_ = copy.statistics_.copy()
        self.n_features_in_ = copy.n_features_in_

    def _is_missing(self, value: Scalar[Self.compute_dtype]) -> Bool:
        if isnan(self.missing_values):
            return isnan(value)
        return value == self.missing_values

    def _column_statistic(
        self, var values: List[Scalar[Self.compute_dtype]], col: Int
    ) raises -> Scalar[Self.compute_dtype]:
        """Replacement value for one column, given its non-missing entries."""
        if self.strategy == "constant":
            return self.fill_value

        var n = len(values)
        if n == 0:
            raise InvalidParameterError.error(
                "X",
                "SimpleImputer cannot take the "
                + self.strategy
                + " of column "
                + String(col)
                + " because every entry is missing",
            )

        if self.strategy == "mean":
            var total: Float64 = 0.0
            for i in range(n):
                total += Float64(values[i])
            return Scalar[Self.compute_dtype](total / Float64(n))

        sort(values)

        if self.strategy == "median":
            var mid = n // 2
            if n % 2 == 1:
                return values[mid]
            return Scalar[Self.compute_dtype](
                (Float64(values[mid - 1]) + Float64(values[mid])) / 2.0
            )

        var best = values[0]
        var best_count = 0
        var start = 0
        while start < n:
            var stop = start
            while stop + 1 < n and values[stop + 1] == values[start]:
                stop += 1
            var count = stop - start + 1
            if count > best_count:
                best_count = count
                best = values[start]
            start = stop + 1
        return best

    def _check_input[
        in_dtype: DType
    ](self, X: Matrix[in_dtype], caller: String) raises:
        check_array[in_dtype](X, force_all_finite=False)

        comptime if in_dtype.is_floating_point():
            var nan_is_missing = isnan(self.missing_values)
            for r in range(X.rows):
                for c in range(X.cols):
                    var value = Float64(X[r, c])
                    if isinf(value):
                        raise InvalidParameterError.error(
                            "X", caller + " does not accept infinite values"
                        )
                    if isnan(value) and not nan_is_missing:
                        raise InvalidParameterError.error(
                            "X",
                            caller
                            + " does not accept NaN values unless"
                            " missing_values is NaN",
                        )

    def fit[in_dtype: DType](mut self, X: Matrix[in_dtype]) raises:
        """Learns the replacement value of each feature column.

        Args:
            X: Matrix of features, possibly holding missing entries.

        Raises:
            InvalidParameterError: If a column is entirely missing and the strategy needs data.
        """
        self._check_input[in_dtype](X, "SimpleImputer.fit")

        var n_cols = X.cols
        var statistics = List[Scalar[Self.compute_dtype]](capacity=n_cols)
        for c in range(n_cols):
            var observed = List[Scalar[Self.compute_dtype]](capacity=X.rows)
            for r in range(X.rows):
                var value = Scalar[Self.compute_dtype](X[r, c])
                if not self._is_missing(value):
                    observed.append(value)
            statistics.append(self._column_statistic(observed^, c))

        self.statistics_ = statistics^
        self.n_features_in_ = n_cols
        self.fit_dtype = in_dtype
        self.is_fitted = True

    def fit[
        feat_dtype: DType,
        target_dtype: DType,
    ](mut self, dataset: Dataset[feat_dtype, target_dtype]) raises:
        """Learns the replacement values from the feature matrix of a Dataset.
        """
        self.fit[feat_dtype](dataset.records)
