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


struct PolynomialFeatures[compute_dtype: DType = DType.float64](
    Copyable, Movable
):
    """Expand features into polynomial and interaction terms.

    Every output column is a product of the input features raised to a set of
    exponents summing to at most degree. Two features at degree 2 expand to
    1, x0, x1, x0^2, x0 x1 and x1^2, so the output is wider than the input.

    Parameters:
        compute_dtype: Computational precision data type. Default DType.float64.

    Args:
        degree: Largest total degree of the generated terms. Default 2.
        interaction_only: Keep only products of distinct features. Default False.
        include_bias: Emit the leading all-ones bias column. Default True.

    Attributes:
        powers_: Exponent of each input feature in each output column.
        n_features_in_: Number of features seen during fit.
        is_fitted: Boolean flag indicating if estimator has been fitted.

    Examples:
        ```mojo
        from strata.preprocessing import PolynomialFeatures
        from strata.core import Matrix

        var poly = PolynomialFeatures[DType.float64](degree=2)
        var X_poly = poly.fit_transform(X_train)
        ```
    """

    var is_fitted: Bool
    var fit_dtype: DType
    var degree: Int
    var interaction_only: Bool
    var include_bias: Bool
    var powers_: List[List[Int]]
    var n_features_in_: Int

    def __init__(
        out self,
        degree: Int = 2,
        interaction_only: Bool = False,
        include_bias: Bool = True,
    ) raises:
        """Initialize the PolynomialFeatures transformer.

        Args:
            degree: Largest total degree of the generated terms. Default 2.
            interaction_only: Keep only products of distinct features. Default False.
            include_bias: Emit the leading all-ones bias column. Default True.

        Raises:
            InvalidParameterError: If degree is negative, or the settings produce no columns.
        """

        check_floating_dtype[Self.compute_dtype, "PolynomialFeatures"]()
        if degree < 0:
            raise InvalidParameterError.error(
                "degree",
                "expected a non-negative degree, got " + String(degree),
            )
        if degree == 0 and not include_bias:
            raise InvalidParameterError.error(
                "degree",
                "degree 0 with include_bias False would produce no columns",
            )
        self.is_fitted = False
        self.fit_dtype = DType.float64
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.powers_ = List[List[Int]]()
        self.n_features_in_ = 0

    def __init__(out self, *, copy: Self):
        """Copies an existing PolynomialFeatures instance."""
        self.is_fitted = copy.is_fitted
        self.fit_dtype = copy.fit_dtype
        self.degree = copy.degree
        self.interaction_only = copy.interaction_only
        self.include_bias = copy.include_bias
        self.powers_ = copy.powers_.copy()
        self.n_features_in_ = copy.n_features_in_

    def n_output_features(self) -> Int:
        """Number of columns produced by transform."""
        return len(self.powers_)
