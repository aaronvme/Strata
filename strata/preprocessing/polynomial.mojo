from ..base.estimator import Transformer
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


def _build_powers(
    n_features: Int, degree: Int, interaction_only: Bool, include_bias: Bool
) -> List[List[Int]]:
    var powers = List[List[Int]]()
    var start = 0 if include_bias else 1

    for d in range(start, degree + 1):
        if d > 0 and (n_features == 0 or (interaction_only and d > n_features)):
            continue

        var indices = List[Int](capacity=d)
        for j in range(d):
            indices.append(j if interaction_only else 0)

        while True:
            var exponents = List[Int](length=n_features, fill=0)
            for j in range(d):
                exponents[indices[j]] += 1
            powers.append(exponents^)

            var i = d - 1
            if interaction_only:
                while i >= 0 and indices[i] == i + n_features - d:
                    i -= 1
            else:
                while i >= 0 and indices[i] == n_features - 1:
                    i -= 1
            if i < 0:
                break

            indices[i] += 1
            for j in range(i + 1, d):
                if interaction_only:
                    indices[j] = indices[j - 1] + 1
                else:
                    indices[j] = indices[i]

    return powers^


struct PolynomialFeatures[compute_dtype: DType = DType.float64](
    Copyable, Movable, Transformer
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

    def fit[in_dtype: DType](mut self, X: Matrix[in_dtype]) raises:
        """Records the input width and builds the exponent table.

        Args:
            X: Matrix of features with one column per feature.
        """
        check_array[in_dtype](X)

        self.powers_ = _build_powers(
            X.cols, self.degree, self.interaction_only, self.include_bias
        )
        self.n_features_in_ = X.cols
        self.fit_dtype = in_dtype
        self.is_fitted = True

    def fit[
        feat_dtype: DType,
        target_dtype: DType,
    ](mut self, dataset: Dataset[feat_dtype, target_dtype]) raises:
        """Records the input width from the feature matrix of a Dataset."""
        self.fit[feat_dtype](dataset.records)

    def transform[
        in_dtype: DType
    ](self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        """Expands each row into its polynomial and interaction terms.

        Args:
            X: Matrix of features with one column per fitted feature.

        Returns:
            A matrix with one row per sample and one column per generated term.
        """
        check_is_fitted("PolynomialFeatures", self.is_fitted)
        if in_dtype != self.fit_dtype:
            raise DataConversionError.error(
                "PolynomialFeatures.transform received Matrix["
                + String(in_dtype)
                + "] but was fitted on Matrix["
                + String(self.fit_dtype)
                + "]"
            )
        check_array[in_dtype](X)
        if X.cols != self.n_features_in_:
            raise DimensionMismatchError.error(
                "X.cols == " + String(self.n_features_in_),
                "X.cols == " + String(X.cols),
                "PolynomialFeatures.transform",
            )

        var res = Matrix[in_dtype](X.rows, self.n_output_features(), 0)
        for r in range(X.rows):
            for j in range(len(self.powers_)):
                var term = Scalar[Self.compute_dtype](1)
                for i in range(self.n_features_in_):
                    var value = Scalar[Self.compute_dtype](X[r, i])
                    for _ in range(self.powers_[j][i]):
                        term *= value
                res[r, j] = Scalar[in_dtype](term)

        return res^

    def fit_transform[
        in_dtype: DType
    ](mut self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        """Builds the exponent table for X and returns its expanded terms."""
        self.fit[in_dtype](X)
        return self.transform[in_dtype](X)
