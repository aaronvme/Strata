from ..core.matrix import Matrix
from ..core.linalg import (
    gemm,
    dense_dot_vec,
    svd,
    solve,
    solve_cholesky,
)
from ..base.estimator import Regressor
from ..utils.validation import (
    check_X_y,
    check_array,
    check_floating_dtype,
    check_is_fitted,
)
from ..exceptions.errors import InvalidParameterError, DimensionMismatchError


struct Ridge[
    compute_dtype: DType = DType.float64,
](Movable, Regressor):
    """Ridge regression with L2 regularization.

    Minimizes the penalized objective:
        ||y - Xw||^2_2 + alpha * ||w||^2_2

    Parameters:
        compute_dtype: Floating point precision for internal computations (default: Float64).
    """

    var is_fitted: Bool
    var alpha: Scalar[Self.compute_dtype]
    var fit_intercept: Bool
    var solver: String
    var coef_: List[Scalar[Self.compute_dtype]]
    var intercept_: Scalar[Self.compute_dtype]

    def __init__(
        out self,
        alpha: Scalar[Self.compute_dtype] = 1.0,
        fit_intercept: Bool = True,
        solver: String = "auto",
    ) raises:
        """Initializes the Ridge regression estimator.

        Args:
            alpha: Regularization strength (must be non-negative).
            fit_intercept: Whether to calculate the intercept for this model.
            solver: Solver algorithm ('auto', 'cholesky', 'svd', 'solve').
        """
        check_floating_dtype[Self.compute_dtype, "Ridge"]()
        if alpha < 0:
            raise InvalidParameterError.error(
                "alpha",
                "alpha must be non-negative, got " + String(alpha),
            )
        self.is_fitted = False
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.coef_ = List[Scalar[Self.compute_dtype]]()
        self.intercept_ = 0

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]]) raises:
        pass

    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Scalar[feat_dtype]]:
        return List[Scalar[feat_dtype]]()
