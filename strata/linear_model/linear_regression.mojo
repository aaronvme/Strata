from ..core.matrix import Matrix
from ..base.estimator import Regressor


struct LinearRegression[
    compute_dtype: DType = DType.float64,
](Movable, Regressor):
    """Ordinary Least Squares Linear Regression.

    Fits a linear model with coefficients w = (w_1, ..., w_p) to minimize
    the residual sum of squares between observed targets and predictions.

    Parameters:
        compute_dtype: Floating point precision for internal computations (default: Float64).
    """

    var is_fitted: Bool
    var fit_intercept: Bool
    var solver: String
    var coef_: List[Scalar[Self.compute_dtype]]
    var intercept_: Scalar[Self.compute_dtype]

    def __init__(
        out self,
        fit_intercept: Bool = True,
        solver: String = "lstsq",
    ):
        """Initializes the linear regression estimator.

        Args:
            fit_intercept: Whether to calculate the intercept for this model.
            solver: Solver algorithm ('lstsq', 'qr', 'cholesky', 'solve').
        """
        self.is_fitted = False
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.coef_ = List[Scalar[Self.compute_dtype]]()
        self.intercept_ = 0
