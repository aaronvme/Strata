from std.math import isnan
from ..core.matrix import Matrix
from ..utils.validation import check_consistent_length, check_finite
from ..exceptions.errors import InvalidParameterError


def _insertion_point(labels: List[Float64], value: Float64) -> Int:
    var lo = 0
    var hi = len(labels)
    while lo < hi:
        var mid = (lo + hi) // 2
        if labels[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _search_sorted(labels: List[Float64], value: Float64) -> Int:
    var i = _insertion_point(labels, value)
    if i < len(labels) and labels[i] == value:
        return i
    return -1


def _insert_label(mut labels: List[Float64], value: Float64):
    var i = _insertion_point(labels, value)
    if i < len(labels) and labels[i] == value:
        return
    labels.insert(i, value)


def unique_labels[
    true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64
](
    y_true: List[Scalar[true_dtype]], y_pred: List[Scalar[pred_dtype]]
) raises -> List[Float64]:
    """Sorted list of the distinct labels appearing in y_true or y_pred."""
    check_finite(y_true, "y_true", "unique_labels")
    check_finite(y_pred, "y_pred", "unique_labels")

    var labels = List[Float64]()
    for i in range(len(y_true)):
        _insert_label(labels, Float64(y_true[i]))
    for i in range(len(y_pred)):
        _insert_label(labels, Float64(y_pred[i]))
    return labels^


def _check_classification_targets[
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


def accuracy_score[
    true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64
](
    y_true: List[Scalar[true_dtype]],
    y_pred: List[Scalar[pred_dtype]],
    normalize: Bool = True,
) raises -> Float64:
    """Fraction (or count, if normalize is False) of correctly classified samples.
    """
    _check_classification_targets(y_true, y_pred, "accuracy_score")

    var correct: Int = 0
    for i in range(len(y_true)):
        if Float64(y_true[i]) == Float64(y_pred[i]):
            correct += 1

    if not normalize:
        return Float64(correct)
    return Float64(correct) / Float64(len(y_true))


def confusion_matrix[
    true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64
](
    y_true: List[Scalar[true_dtype]], y_pred: List[Scalar[pred_dtype]]
) raises -> Matrix[DType.int64]:
    """Confusion matrix C where C[i, j] counts samples of label i predicted as label j.

    Rows and columns are indexed by the sorted distinct labels.
    """
    _check_classification_targets(y_true, y_pred, "confusion_matrix")

    var labels = unique_labels(y_true, y_pred)
    var cm = Matrix[DType.int64](len(labels), len(labels), 0)
    for i in range(len(y_true)):
        var r = _search_sorted(labels, Float64(y_true[i]))
        var c = _search_sorted(labels, Float64(y_pred[i]))
        cm[r, c] = cm[r, c] + 1
    return cm^


def _numerator(tp: Float64, which: Int) -> Float64:
    if which == 2:
        return 2.0 * tp
    return tp


def _denominator(
    tp: Float64, fp: Float64, false_neg: Float64, which: Int
) -> Float64:
    if which == 0:
        return tp + fp
    if which == 1:
        return tp + false_neg
    return 2.0 * tp + fp + false_neg


def _averaged_score[
    true_dtype: DType, pred_dtype: DType
](
    y_true: List[Scalar[true_dtype]],
    y_pred: List[Scalar[pred_dtype]],
    average: String,
    pos_label: Float64,
    zero_division: Float64,
    which: Int,
    caller: String,
) raises -> Float64:
    _check_classification_targets(y_true, y_pred, caller)

    var labels = unique_labels(y_true, y_pred)
    var k = len(labels)
    var n = len(y_true)

    var tp = List[Float64](length=k, fill=0.0)
    var fp = List[Float64](length=k, fill=0.0)
    var false_neg = List[Float64](length=k, fill=0.0)
    var support = List[Float64](length=k, fill=0.0)

    for i in range(n):
        var t = _search_sorted(labels, Float64(y_true[i]))
        var p = _search_sorted(labels, Float64(y_pred[i]))
        support[t] += 1.0
        if t == p:
            tp[t] += 1.0
        else:
            fp[p] += 1.0
            false_neg[t] += 1.0

    if average == "binary":
        if k > 2:
            raise InvalidParameterError.error(
                "average",
                "'binary' is only supported for binary targets, but "
                + String(k)
                + " labels were found. Use 'micro', 'macro' or 'weighted'.",
            )
        var idx = _search_sorted(labels, pos_label)
        if idx < 0:
            if k >= 2:
                raise InvalidParameterError.error(
                    "pos_label",
                    String(pos_label) + " is not present in y_true or y_pred",
                )
            return zero_division
        var den = _denominator(tp[idx], fp[idx], false_neg[idx], which)
        if den == 0.0:
            return zero_division
        return _numerator(tp[idx], which) / den

    if average == "micro":
        var tp_sum: Float64 = 0.0
        var fp_sum: Float64 = 0.0
        var fn_sum: Float64 = 0.0
        for i in range(k):
            tp_sum += tp[i]
            fp_sum += fp[i]
            fn_sum += false_neg[i]
        var micro_den = _denominator(tp_sum, fp_sum, fn_sum, which)
        if micro_den == 0.0:
            return zero_division
        return _numerator(tp_sum, which) / micro_den

    if average != "macro" and average != "weighted":
        raise InvalidParameterError.error(
            "average",
            "'"
            + average
            + "' is not supported. Use 'binary', 'micro', 'macro' or"
            " 'weighted'.",
        )

    var weighted = average == "weighted"
    var total: Float64 = 0.0
    var weight_total: Float64 = 0.0
    for i in range(k):
        var weight = support[i] if weighted else 1.0
        var den = _denominator(tp[i], fp[i], false_neg[i], which)
        if den == 0.0:
            # A NaN fallback drops ill-defined labels from the average
            if isnan(zero_division):
                continue
            total += zero_division * weight
        else:
            total += (_numerator(tp[i], which) / den) * weight
        weight_total += weight

    if weight_total == 0.0:
        return zero_division
    return total / weight_total


def precision_score[
    true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64
](
    y_true: List[Scalar[true_dtype]],
    y_pred: List[Scalar[pred_dtype]],
    average: String = "binary",
    pos_label: Float64 = 1.0,
    zero_division: Float64 = 0.0,
) raises -> Float64:
    """Compute classification precision score.

    $$
    \\text{Precision} = \\frac{TP}{TP + FP}
    $$

    Args:
        y_true: Ground truth target labels.
        y_pred: Estimated target labels.
        average: Averaging strategy ('binary', 'micro', 'macro', 'weighted'). Default 'binary'.
        pos_label: Label of the positive class when average='binary'. Default 1.0.
        zero_division: Value returned when there is zero division (e.g. 0.0). Default 0.0.

    Returns:
        Float64: Precision score ratio.

    Raises:
        InvalidParameterError: If inputs are empty or average strategy is unrecognized.
    """
    return _averaged_score(
        y_true, y_pred, average, pos_label, zero_division, 0, "precision_score"
    )


def recall_score[
    true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64
](
    y_true: List[Scalar[true_dtype]],
    y_pred: List[Scalar[pred_dtype]],
    average: String = "binary",
    pos_label: Float64 = 1.0,
    zero_division: Float64 = 0.0,
) raises -> Float64:
    """Compute classification recall (sensitivity) score.

    $$
    \\text{Recall} = \\frac{TP}{TP + FN}
    $$

    Args:
        y_true: Ground truth target labels.
        y_pred: Estimated target labels.
        average: Averaging strategy ('binary', 'micro', 'macro', 'weighted'). Default 'binary'.
        pos_label: Label of the positive class when average='binary'. Default 1.0.
        zero_division: Value returned when there is zero division. Default 0.0.

    Returns:
        Float64: Recall score ratio.

    Raises:
        InvalidParameterError: If inputs are empty or average strategy is unrecognized.
    """
    return _averaged_score(
        y_true, y_pred, average, pos_label, zero_division, 1, "recall_score"
    )


def f1_score[
    true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64
](
    y_true: List[Scalar[true_dtype]],
    y_pred: List[Scalar[pred_dtype]],
    average: String = "binary",
    pos_label: Float64 = 1.0,
    zero_division: Float64 = 0.0,
) raises -> Float64:
    """Compute classification F1 score (harmonic mean of precision and recall).

    $$
    F_1 = 2 \\cdot \\frac{\\text{Precision} \\cdot \\text{Recall}}{\\text{Precision} + \\text{Recall}} = \\frac{2 TP}{2 TP + FP + FN}
    $$


    Args:
        y_true: Ground truth target labels.
        y_pred: Estimated target labels.
        average: Averaging strategy ('binary', 'micro', 'macro', 'weighted'). Default 'binary'.
        pos_label: Label of the positive class when average='binary'. Default 1.0.
        zero_division: Value returned when there is zero division. Default 0.0.

    Returns:
        Float64: F1 score between 0.0 and 1.0.

    Raises:
        InvalidParameterError: If inputs are empty or average strategy is unrecognized.
    """
    return _averaged_score(
        y_true, y_pred, average, pos_label, zero_division, 2, "f1_score"
    )

