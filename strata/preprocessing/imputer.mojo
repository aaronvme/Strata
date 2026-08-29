from std.math import isnan, nan
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
