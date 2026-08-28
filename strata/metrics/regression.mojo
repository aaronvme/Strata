from std.math import sqrt
from ..utils.validation import check_consistent_length, check_finite
from ..exceptions.errors import InvalidParameterError


def _check_regression_targets[
    true_dtype: DType, pred_dtype: DType
](
    y_true: List[Scalar[true_dtype]],
    y_pred: List[Scalar[pred_dtype]],
    caller: String,
) raises:
    check_consistent_length(y_true, y_pred)
    if len(y_true) == 0:
        raise InvalidParameterError.error(
            "y_true", caller + " requires at least one sample"
        )
    check_finite(y_true, "y_true", caller)
    check_finite(y_pred, "y_pred", caller)


def mean_squared_error[
    true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64
](
    y_true: List[Scalar[true_dtype]], y_pred: List[Scalar[pred_dtype]]
) raises -> Float64:
    """Compute Mean Squared Error (MSE) regression loss.

    $$
    \\text{MSE}(y, \\hat{y}) = \\frac{1}{N} \\sum_{i=1}^{N} (y_i - \\hat{y}_i)^2
    $$

    Args:
        y_true: Ground truth target values of length $N$.
        y_pred: Estimated target values of length $N$.

    Returns:
        Float64: Non-negative floating point mean squared error.

    Raises:
        InvalidParameterError: If inputs are empty or contain NaN/infinite values.
        DimensionMismatchError: If `len(y_true) != len(y_pred)`.
    """
    _check_regression_targets(y_true, y_pred, "mean_squared_error")

    var total: Float64 = 0.0
    for i in range(len(y_true)):
        var diff = Float64(y_true[i]) - Float64(y_pred[i])
        total += diff * diff
    return total / Float64(len(y_true))


def root_mean_squared_error[
    true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64
](
    y_true: List[Scalar[true_dtype]], y_pred: List[Scalar[pred_dtype]]
) raises -> Float64:
    """Compute Root Mean Squared Error (RMSE) regression loss.

    $$
    \\text{RMSE}(y, \\hat{y}) = \\sqrt{\\frac{1}{N} \\sum_{i=1}^{N} (y_i - \\hat{y}_i)^2}
    $$

    Args:
        y_true: Ground truth target values of length $N$.
        y_pred: Estimated target values of length $N$.

    Returns:
        Float64: Non-negative square root of mean squared error.

    Raises:
        InvalidParameterError: If inputs are empty or contain NaN/infinite values.
    """
    return sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error[
    true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64
](
    y_true: List[Scalar[true_dtype]], y_pred: List[Scalar[pred_dtype]]
) raises -> Float64:
    """Compute Mean Absolute Error (MAE) regression loss.

    $$
    \\text{MAE}(y, \\hat{y}) = \\frac{1}{N} \\sum_{i=1}^{N} |y_i - \\hat{y}_i|
    $$

    Args:
        y_true: Ground truth target values of length $N$.
        y_pred: Estimated target values of length $N$.

    Returns:
        Float64: Non-negative floating point mean absolute error.

    Raises:
        InvalidParameterError: If inputs are empty or contain NaN/infinite values.
    """
    _check_regression_targets(y_true, y_pred, "mean_absolute_error")

    var total: Float64 = 0.0
    for i in range(len(y_true)):
        total += abs(Float64(y_true[i]) - Float64(y_pred[i]))
    return total / Float64(len(y_true))


def r2_score[
    true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64
](
    y_true: List[Scalar[true_dtype]], y_pred: List[Scalar[pred_dtype]]
) raises -> Float64:
    """Compute $R^2$ (coefficient of determination) regression score function.

    $$
    R^2(y, \\hat{y}) = 1 - \\frac{\\sum_{i=1}^N (y_i - \\hat{y}_i)^2}{\\sum_{i=1}^N (y_i - \\bar{y})^2}
    $$


    Args:
        y_true: Ground truth target values of length $N$.
        y_pred: Estimated target values of length $N$.

    Returns:
        Float64: $R^2$ score (best possible score is 1.0, can be negative for arbitrarily worse models).

    Raises:
        InvalidParameterError: If $N < 2$ or inputs contain non-finite values.
    """
    _check_regression_targets(y_true, y_pred, "r2_score")

    var n = len(y_true)
    if n < 2:
        raise InvalidParameterError.error(
            "y_true", "r2_score requires at least 2 samples"
        )

    var mean: Float64 = 0.0
    for i in range(n):
        mean += Float64(y_true[i])
    mean /= Float64(n)

    var ss_res: Float64 = 0.0
    var ss_tot: Float64 = 0.0
    for i in range(n):
        var res = Float64(y_true[i]) - Float64(y_pred[i])
        ss_res += res * res
        var tot = Float64(y_true[i]) - mean
        ss_tot += tot * tot

    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot

