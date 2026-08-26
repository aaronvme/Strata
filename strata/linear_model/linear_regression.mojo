from ..core.matrix import Matrix
from ..core.dataset import Dataset
from ..core.linalg import (
    gemm,
    dense_dot_vec,
    qr,
    lstsq,
    solve,
    solve_cholesky,
)
from ..base.estimator import Regressor
from ..utils.validation import check_X_y, check_floating_dtype
from ..exceptions.errors import InvalidParameterError


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
        check_floating_dtype[Self.compute_dtype, "LinearRegression"]()
        self.is_fitted = False
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.coef_ = List[Scalar[Self.compute_dtype]]()
        self.intercept_ = 0

    def fit[
        feat_dtype: DType, in_target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[in_target_dtype]]) raises:
        """Fits the linear model from training data (X, y).

        Args:
            X: Training feature matrix (N x D).
            y: Target values (length N).
        """
        check_X_y(X, y)

        var N = X.rows
        var D = X.cols

        # Promote upfront to compute_dtype once
        var X_comp: Matrix[Self.compute_dtype]
        comptime if feat_dtype == Self.compute_dtype:
            X_comp = X.copy()
        else:
            X_comp = X.cast[Self.compute_dtype]()

        var y_comp = List[Scalar[Self.compute_dtype]](capacity=N)
        for i in range(N):
            y_comp.append(Scalar[Self.compute_dtype](y[i]))

        if self.fit_intercept:
            # 1. Compute means along axis 0
            var X_means = X_comp.mean_along_axis_0()
            var y_sum: Scalar[Self.compute_dtype] = 0
            for i in range(N):
                y_sum += y_comp[i]
            var y_mean = y_sum / Scalar[Self.compute_dtype](N)

            # 2. Center X and y
            var X_centered = Matrix[Self.compute_dtype](N, D, 0)
            for r in range(N):
                for c in range(D):
                    X_centered[r, c] = X_comp[r, c] - X_means[c]

            var y_centered = List[Scalar[Self.compute_dtype]](capacity=N)
            for i in range(N):
                y_centered.append(y_comp[i] - y_mean)

            # 3. Solve for beta on centered data
            var beta = self._solve(X_centered, y_centered)

            # 4. Calculate intercept: y_mean - sum(beta_j * X_means_j)
            var beta_dot_mean: Scalar[Self.compute_dtype] = 0
            for j in range(D):
                beta_dot_mean += beta[j] * X_means[j]

            self.intercept_ = y_mean - beta_dot_mean
            self.coef_ = beta^
        else:
            var beta = self._solve(X_comp, y_comp)
            self.intercept_ = 0
            self.coef_ = beta^

        self.is_fitted = True

    def fit[
        feat_dtype: DType, in_target_dtype: DType
    ](mut self, dataset: Dataset[feat_dtype, in_target_dtype]) raises:
        """Fits the linear model from a Dataset container."""
        self.fit[feat_dtype, in_target_dtype](dataset.records, dataset.targets)

    def _solve(
        self,
        X: Matrix[Self.compute_dtype],
        y: List[Scalar[Self.compute_dtype]],
    ) raises -> List[Scalar[Self.compute_dtype]]:
        """Internal solver dispatch for least-squares optimization."""
        if self.solver == "lstsq" or self.solver == "svd":
            return lstsq(X, y)
        elif self.solver == "qr":
            var qr_res = qr(X)
            var Qt = qr_res.Q.transpose()
            var Qt_y = dense_dot_vec(Qt, y)
            return solve(qr_res.R, Qt_y)
        elif self.solver == "cholesky":
            var Xt = X.transpose()
            var XtX = gemm(Xt, X)
            var Xty = dense_dot_vec(Xt, y)
            return solve_cholesky(XtX, Xty)
        elif self.solver == "solve" or self.solver == "lu":
            var Xt = X.transpose()
            var XtX = gemm(Xt, X)
            var Xty = dense_dot_vec(Xt, y)
            return solve(XtX, Xty)
        else:
            raise InvalidParameterError.error(
                "solver",
                "Unsupported solver '"
                + self.solver
                + "'. Expected 'lstsq', 'qr', 'cholesky', or 'solve'.",
            )
