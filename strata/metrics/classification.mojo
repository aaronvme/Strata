from ..core.matrix import Matrix
from ..utils.validation import check_consistent_length
from ..exceptions.errors import InvalidParameterError


def _index_of(labels: List[Float64], value: Float64) -> Int:
    for i in range(len(labels)):
        if labels[i] == value:
            return i
    return -1


def _check_finite_labels[
    dtype: DType
](values: List[Scalar[dtype]], name: String, caller: String) raises:
    for i in range(len(values)):
        var v = Float64(values[i])
        if v != v:
            raise InvalidParameterError.error(
                name, caller + " does not accept NaN labels"
            )


def unique_labels[
    true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64
](
    y_true: List[Scalar[true_dtype]], y_pred: List[Scalar[pred_dtype]]
) raises -> List[Float64]:
    """Sorted list of the distinct labels appearing in y_true or y_pred."""
    _check_finite_labels(y_true, "y_true", "unique_labels")
    _check_finite_labels(y_pred, "y_pred", "unique_labels")

    var labels = List[Float64]()
    for i in range(len(y_true)):
        var v = Float64(y_true[i])
        if _index_of(labels, v) < 0:
            labels.append(v)
    for i in range(len(y_pred)):
        var v = Float64(y_pred[i])
        if _index_of(labels, v) < 0:
            labels.append(v)

    for i in range(1, len(labels)):
        var key = labels[i]
        var j = i - 1
        while j >= 0 and labels[j] > key:
            labels[j + 1] = labels[j]
            j -= 1
        labels[j + 1] = key
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
    _check_finite_labels(y_true, "y_true", caller)
    _check_finite_labels(y_pred, "y_pred", caller)


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
    var cm = Matrix[DType.int64](len(labels), len(labels))
    for i in range(len(y_true)):
        var r = _index_of(labels, Float64(y_true[i]))
        var c = _index_of(labels, Float64(y_pred[i]))
        cm[r, c] = cm[r, c] + 1
    return cm^


def _ratio(numerator: Float64, denominator: Float64) -> Float64:
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _score(tp: Float64, fp: Float64, false_neg: Float64, which: Int) -> Float64:
    if which == 0:
        return _ratio(tp, tp + fp)
    if which == 1:
        return _ratio(tp, tp + false_neg)
    return _ratio(2.0 * tp, 2.0 * tp + fp + false_neg)


def _averaged_score[
    true_dtype: DType, pred_dtype: DType
](
    y_true: List[Scalar[true_dtype]],
    y_pred: List[Scalar[pred_dtype]],
    average: String,
    pos_label: Float64,
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
        var t = _index_of(labels, Float64(y_true[i]))
        var p = _index_of(labels, Float64(y_pred[i]))
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
        var idx = _index_of(labels, pos_label)
        if idx < 0:
            raise InvalidParameterError.error(
                "pos_label",
                String(pos_label) + " is not present in y_true or y_pred",
            )
        return _score(tp[idx], fp[idx], false_neg[idx], which)

    if average == "micro":
        var tp_sum: Float64 = 0.0
        var fp_sum: Float64 = 0.0
        var fn_sum: Float64 = 0.0
        for i in range(k):
            tp_sum += tp[i]
            fp_sum += fp[i]
            fn_sum += false_neg[i]
        return _score(tp_sum, fp_sum, fn_sum, which)

    if average == "macro":
        var total: Float64 = 0.0
        for i in range(k):
            total += _score(tp[i], fp[i], false_neg[i], which)
        return _ratio(total, Float64(k))

    if average == "weighted":
        var total: Float64 = 0.0
        for i in range(k):
            total += _score(tp[i], fp[i], false_neg[i], which) * support[i]
        return _ratio(total, Float64(n))

    raise InvalidParameterError.error(
        "average",
        "'"
        + average
        + "' is not supported. Use 'binary', 'micro', 'macro' or 'weighted'.",
    )


def precision_score[
    true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64
](
    y_true: List[Scalar[true_dtype]],
    y_pred: List[Scalar[pred_dtype]],
    average: String = "binary",
    pos_label: Float64 = 1.0,
) raises -> Float64:
    """Precision: tp / (tp + fp), 0.0 when the denominator is zero."""
    return _averaged_score(
        y_true, y_pred, average, pos_label, 0, "precision_score"
    )


def recall_score[
    true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64
](
    y_true: List[Scalar[true_dtype]],
    y_pred: List[Scalar[pred_dtype]],
    average: String = "binary",
    pos_label: Float64 = 1.0,
) raises -> Float64:
    """Recall: tp / (tp + fn), 0.0 when the denominator is zero."""
    return _averaged_score(
        y_true, y_pred, average, pos_label, 1, "recall_score"
    )


def f1_score[
    true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64
](
    y_true: List[Scalar[true_dtype]],
    y_pred: List[Scalar[pred_dtype]],
    average: String = "binary",
    pos_label: Float64 = 1.0,
) raises -> Float64:
    """F1: harmonic mean of precision and recall, 2tp / (2tp + fp + fn)."""
    return _averaged_score(y_true, y_pred, average, pos_label, 2, "f1_score")
