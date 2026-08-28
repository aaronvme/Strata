from ..core.matrix import Matrix
from ..base.estimator import Regressor, Classifier
from ..metrics.regression import (
    r2_score,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
)
from ..metrics.classification import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from ..utils.validation import check_X_y
from ..exceptions.errors import InvalidParameterError
from .kfold import KFold, Split
from .stratified_kfold import StratifiedKFold
from .subsampling import take_rows, take_elements


def _regression_metric_code(scoring: String) raises -> Int:
    """Maps a regression metric name to an internal dispatch code.

    Args:
        scoring: Name of a supported regression metric.

    Returns:
        Dispatch code accepted by _apply_regression_metric.
    """
    if scoring == "r2":
        return 0
    elif scoring == "mse":
        return 1
    elif scoring == "neg_mean_squared_error":
        return 2
    elif scoring == "rmse":
        return 3
    elif scoring == "neg_root_mean_squared_error":
        return 4
    elif scoring == "mae":
        return 5
    elif scoring == "neg_mean_absolute_error":
        return 6
    raise InvalidParameterError.error(
        "scoring",
        "Unsupported regression scoring metric: '" + scoring + "'",
    )


def _apply_regression_metric[
    target_dtype: DType, pred_dtype: DType
](
    metric_code: Int,
    y_true: List[Scalar[target_dtype]],
    preds: List[Scalar[pred_dtype]],
) raises -> Float64:
    """Evaluates a regression metric selected by dispatch code.

    Args:
        metric_code: Code returned by _regression_metric_code.
        y_true: Observed target values.
        preds: Predicted target values.

    Returns:
        The metric value, negated for the neg_ variants.
    """
    if metric_code == 0:
        return r2_score(y_true, preds)
    elif metric_code == 1:
        return mean_squared_error(y_true, preds)
    elif metric_code == 2:
        return -mean_squared_error(y_true, preds)
    elif metric_code == 3:
        return root_mean_squared_error(y_true, preds)
    elif metric_code == 4:
        return -root_mean_squared_error(y_true, preds)
    elif metric_code == 5:
        return mean_absolute_error(y_true, preds)
    return -mean_absolute_error(y_true, preds)


def _classification_metric_code(scoring: String) raises -> Int:
    """Maps a classification metric name to an internal dispatch code.

    Args:
        scoring: Name of a supported classification metric.

    Returns:
        Dispatch code accepted by _apply_classification_metric.
    """
    if scoring == "accuracy":
        return 0
    elif scoring == "f1":
        return 1
    elif scoring == "precision":
        return 2
    elif scoring == "recall":
        return 3
    raise InvalidParameterError.error(
        "scoring",
        "Unsupported classification scoring metric: '" + scoring + "'",
    )


def _apply_classification_metric[
    target_dtype: DType
](
    metric_code: Int,
    y_true: List[Scalar[target_dtype]],
    preds: List[Int],
) raises -> Float64:
    """Evaluates a classification metric selected by dispatch code.

    Args:
        metric_code: Code returned by _classification_metric_code.
        y_true: Observed class labels.
        preds: Predicted class labels.

    Returns:
        The metric value.
    """
    var y_int = List[Int](capacity=len(y_true))
    for i in range(len(y_true)):
        y_int.append(Int(y_true[i]))

    if metric_code == 0:
        return accuracy_score(y_int, preds)
    elif metric_code == 1:
        return f1_score(y_int, preds)
    elif metric_code == 2:
        return precision_score(y_int, preds)
    return recall_score(y_int, preds)


def cross_val_score[
    ModelType: Regressor,
    feat_dtype: DType = DType.float64,
    target_dtype: DType = DType.float64,
](
    estimator: ModelType,
    X: Matrix[feat_dtype],
    y: List[Scalar[target_dtype]],
    cv: Int = 5,
    scoring: String = "r2",
) raises -> List[Float64]:
    """Evaluates regression scores by cross-validation across K folds."""
    var kf = KFold(n_splits=cv)
    var splits = kf.split(X.rows)
    return cross_val_score[ModelType, feat_dtype, target_dtype](
        estimator, X, y, splits, scoring=scoring
    )


def cross_val_score[
    ModelType: Regressor,
    feat_dtype: DType = DType.float64,
    target_dtype: DType = DType.float64,
](
    estimator: ModelType,
    X: Matrix[feat_dtype],
    y: List[Scalar[target_dtype]],
    splits: List[Split],
    scoring: String = "r2",
) raises -> List[Float64]:
    """Evaluates regression scores by cross-validation on pre-defined splits."""
    check_X_y(X, y)
    var n_splits = len(splits)
    if n_splits == 0:
        raise InvalidParameterError.error(
            "splits", "Number of cross-validation splits must be greater than 0"
        )

    # Validate metric name upfront before running optimization folds
    var metric_code = _regression_metric_code(scoring)

    var scores = List[Float64](capacity=n_splits)

    for s in range(n_splits):
        if len(splits[s].train_indices) == 0:
            raise InvalidParameterError.error(
                "splits",
                "Fold " + String(s) + " contains empty training indices",
            )
        if len(splits[s].val_indices) == 0:
            raise InvalidParameterError.error(
                "splits",
                "Fold " + String(s) + " contains empty validation indices",
            )

        var X_train = take_rows(X, splits[s].train_indices)
        var y_train = take_elements(y, splits[s].train_indices)
        var X_val = take_rows(X, splits[s].val_indices)
        var y_val = take_elements(y, splits[s].val_indices)

        var model = estimator.copy()
        model.fit(X_train, y_train)
        var preds = model.predict(X_val)

        scores.append(_apply_regression_metric(metric_code, y_val, preds))

    return scores^


def cross_val_score[
    ModelType: Classifier,
    feat_dtype: DType = DType.float64,
    target_dtype: DType = DType.int32,
](
    estimator: ModelType,
    X: Matrix[feat_dtype],
    y: List[Scalar[target_dtype]],
    cv: Int = 5,
    scoring: String = "accuracy",
) raises -> List[Float64]:
    """Evaluates classification scores by stratified cross-validation across K folds.
    """
    var skf = StratifiedKFold(n_splits=cv)
    var splits = skf.split[feat_dtype, target_dtype](X, y)
    return cross_val_score[ModelType, feat_dtype, target_dtype](
        estimator, X, y, splits, scoring=scoring
    )


def cross_val_score[
    ModelType: Classifier,
    feat_dtype: DType = DType.float64,
    target_dtype: DType = DType.int32,
](
    estimator: ModelType,
    X: Matrix[feat_dtype],
    y: List[Scalar[target_dtype]],
    splits: List[Split],
    scoring: String = "accuracy",
) raises -> List[Float64]:
    """Evaluates classification scores by cross-validation on pre-defined splits.
    """
    check_X_y(X, y)
    var n_splits = len(splits)
    if n_splits == 0:
        raise InvalidParameterError.error(
            "splits", "Number of cross-validation splits must be greater than 0"
        )

    # Validate metric name upfront before running optimization folds
    var metric_code = _classification_metric_code(scoring)

    var scores = List[Float64](capacity=n_splits)

    for s in range(n_splits):
        if len(splits[s].train_indices) == 0:
            raise InvalidParameterError.error(
                "splits",
                "Fold " + String(s) + " contains empty training indices",
            )
        if len(splits[s].val_indices) == 0:
            raise InvalidParameterError.error(
                "splits",
                "Fold " + String(s) + " contains empty validation indices",
            )

        var X_train = take_rows(X, splits[s].train_indices)
        var y_train = take_elements(y, splits[s].train_indices)
        var X_val = take_rows(X, splits[s].val_indices)
        var y_val = take_elements(y, splits[s].val_indices)

        var model = estimator.copy()
        model.fit(X_train, y_train)
        var preds = model.predict(X_val)

        scores.append(_apply_classification_metric(metric_code, y_val, preds))

    return scores^


def _validate_partition(splits: List[Split], n_samples: Int) raises:
    """Verifies that validation folds cover every sample exactly once.

    Args:
        splits: Cross-validation folds to inspect.
        n_samples: Total number of samples in the dataset.
    """
    var covered = List[Bool](capacity=n_samples)
    for _ in range(n_samples):
        covered.append(False)

    for s in range(len(splits)):
        if len(splits[s].train_indices) == 0:
            raise InvalidParameterError.error(
                "splits",
                "Fold " + String(s) + " contains empty training indices",
            )
        if len(splits[s].val_indices) == 0:
            raise InvalidParameterError.error(
                "splits",
                "Fold " + String(s) + " contains empty validation indices",
            )
        for i in range(len(splits[s].val_indices)):
            var v = splits[s].val_indices[i]
            if v < 0 or v >= n_samples:
                raise InvalidParameterError.error(
                    "splits",
                    "Validation index "
                    + String(v)
                    + " is out of bounds for a dataset with "
                    + String(n_samples)
                    + " samples",
                )
            if covered[v]:
                raise InvalidParameterError.error(
                    "splits",
                    "cross_val_predict requires splits that partition the"
                    " data, but sample "
                    + String(v)
                    + " appears in more than one validation fold",
                )
            covered[v] = True

    for i in range(n_samples):
        if not covered[i]:
            raise InvalidParameterError.error(
                "splits",
                "cross_val_predict requires splits that partition the data,"
                " but sample "
                + String(i)
                + " appears in no validation fold",
            )


def cross_val_predict[
    ModelType: Regressor,
    feat_dtype: DType = DType.float64,
    target_dtype: DType = DType.float64,
](
    estimator: ModelType,
    X: Matrix[feat_dtype],
    y: List[Scalar[target_dtype]],
    cv: Int = 5,
) raises -> List[Scalar[feat_dtype]]:
    """Generates out-of-fold regression predictions across K folds."""
    var kf = KFold(n_splits=cv)
    var splits = kf.split(X.rows)
    return cross_val_predict[ModelType, feat_dtype, target_dtype](
        estimator, X, y, splits
    )


def cross_val_predict[
    ModelType: Regressor,
    feat_dtype: DType = DType.float64,
    target_dtype: DType = DType.float64,
](
    estimator: ModelType,
    X: Matrix[feat_dtype],
    y: List[Scalar[target_dtype]],
    splits: List[Split],
) raises -> List[Scalar[feat_dtype]]:
    """Generates out-of-fold regression predictions on pre-defined splits."""
    check_X_y(X, y)
    var n_splits = len(splits)
    if n_splits == 0:
        raise InvalidParameterError.error(
            "splits", "Number of cross-validation splits must be greater than 0"
        )

    var n_samples = X.rows
    _validate_partition(splits, n_samples)

    var preds = List[Scalar[feat_dtype]](capacity=n_samples)
    for _ in range(n_samples):
        preds.append(Scalar[feat_dtype](0))

    for s in range(n_splits):
        var X_train = take_rows(X, splits[s].train_indices)
        var y_train = take_elements(y, splits[s].train_indices)
        var X_val = take_rows(X, splits[s].val_indices)

        var model = estimator.copy()
        model.fit(X_train, y_train)
        var fold_preds = model.predict(X_val)

        for i in range(len(splits[s].val_indices)):
            preds[splits[s].val_indices[i]] = fold_preds[i]

    return preds^


def cross_val_predict[
    ModelType: Classifier,
    feat_dtype: DType = DType.float64,
    target_dtype: DType = DType.int32,
](
    estimator: ModelType,
    X: Matrix[feat_dtype],
    y: List[Scalar[target_dtype]],
    cv: Int = 5,
) raises -> List[Int]:
    """Generates out-of-fold class label predictions across K stratified folds.
    """
    var skf = StratifiedKFold(n_splits=cv)
    var splits = skf.split[feat_dtype, target_dtype](X, y)
    return cross_val_predict[ModelType, feat_dtype, target_dtype](
        estimator, X, y, splits
    )


def cross_val_predict[
    ModelType: Classifier,
    feat_dtype: DType = DType.float64,
    target_dtype: DType = DType.int32,
](
    estimator: ModelType,
    X: Matrix[feat_dtype],
    y: List[Scalar[target_dtype]],
    splits: List[Split],
) raises -> List[Int]:
    """Generates out-of-fold class label predictions on pre-defined splits."""
    check_X_y(X, y)
    var n_splits = len(splits)
    if n_splits == 0:
        raise InvalidParameterError.error(
            "splits", "Number of cross-validation splits must be greater than 0"
        )

    var n_samples = X.rows
    _validate_partition(splits, n_samples)

    var preds = List[Int](capacity=n_samples)
    for _ in range(n_samples):
        preds.append(0)

    for s in range(n_splits):
        var X_train = take_rows(X, splits[s].train_indices)
        var y_train = take_elements(y, splits[s].train_indices)
        var X_val = take_rows(X, splits[s].val_indices)

        var model = estimator.copy()
        model.fit(X_train, y_train)
        var fold_preds = model.predict(X_val)

        for i in range(len(splits[s].val_indices)):
            preds[splits[s].val_indices[i]] = fold_preds[i]

    return preds^


struct CrossValidateResult(Movable):
    """Per-fold scores for one or more metrics from a cross-validation run.

    Scores are stored as parallel lists: metrics[m] names the metric whose
    per-fold values are held in test_scores[m] and, when requested,
    train_scores[m].
    """

    var metrics: List[String]
    var test_scores: List[List[Float64]]
    var train_scores: List[List[Float64]]

    def __init__(
        out self,
        var metrics: List[String],
        var test_scores: List[List[Float64]],
        var train_scores: List[List[Float64]],
    ):
        """Initializes a cross-validation result.

        Args:
            metrics: Names of the evaluated metrics.
            test_scores: Validation-fold scores for each metric.
            train_scores: Training-fold scores for each metric, or an empty
                list when training scores were not requested.
        """
        self.metrics = metrics^
        self.test_scores = test_scores^
        self.train_scores = train_scores^

    def metric_index(self, metric: String) raises -> Int:
        """Returns the position of a metric name within this result.

        Args:
            metric: Name of a metric evaluated during cross-validation.

        Returns:
            Index into the metrics, test_scores, and train_scores lists.
        """
        for m in range(len(self.metrics)):
            if self.metrics[m] == metric:
                return m
        raise InvalidParameterError.error(
            "metric",
            "Metric '"
            + metric
            + "' was not evaluated in this cross-validation run",
        )

    def test_scores_for(self, metric: String) raises -> List[Float64]:
        """Returns the per-fold validation scores for a named metric.

        Args:
            metric: Name of a metric evaluated during cross-validation.

        Returns:
            One score per fold, in fold order.
        """
        return self.test_scores[self.metric_index(metric)].copy()

    def train_scores_for(self, metric: String) raises -> List[Float64]:
        """Returns the per-fold training scores for a named metric.

        Args:
            metric: Name of a metric evaluated during cross-validation.

        Returns:
            One score per fold, in fold order.
        """
        if len(self.train_scores) == 0:
            raise InvalidParameterError.error(
                "train_scores",
                (
                    "Training scores were not recorded for this"
                    " cross-validation run"
                ),
            )
        return self.train_scores[self.metric_index(metric)].copy()
