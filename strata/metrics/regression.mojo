from std.math import sqrt
from ..utils.validation import check_consistent_length
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


def mean_squared_error[
    true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64
](
    y_true: List[Scalar[true_dtype]], y_pred: List[Scalar[pred_dtype]]
) raises -> Float64:
    """Mean squared error regression loss."""
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
    """Root mean squared error regression loss."""
    return sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error[
    true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64
](
    y_true: List[Scalar[true_dtype]], y_pred: List[Scalar[pred_dtype]]
) raises -> Float64:
    """Mean absolute error regression loss."""
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
    """Coefficient of determination: 1 - SS_res / SS_tot."""
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
