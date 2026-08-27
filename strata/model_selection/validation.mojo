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
    var metric_code: Int
    if scoring == "r2":
        metric_code = 0
    elif scoring == "mse":
        metric_code = 1
    elif scoring == "neg_mean_squared_error":
        metric_code = 2
    elif scoring == "rmse":
        metric_code = 3
    elif scoring == "neg_root_mean_squared_error":
        metric_code = 4
    elif scoring == "mae":
        metric_code = 5
    elif scoring == "neg_mean_absolute_error":
        metric_code = 6
    else:
        raise InvalidParameterError.error(
            "scoring",
            "Unsupported regression scoring metric: '" + scoring + "'",
        )

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

        var score: Float64
        if metric_code == 0:
            score = r2_score(y_val, preds)
        elif metric_code == 1:
            score = mean_squared_error(y_val, preds)
        elif metric_code == 2:
            score = -mean_squared_error(y_val, preds)
        elif metric_code == 3:
            score = root_mean_squared_error(y_val, preds)
        elif metric_code == 4:
            score = -root_mean_squared_error(y_val, preds)
        elif metric_code == 5:
            score = mean_absolute_error(y_val, preds)
        else:
            score = -mean_absolute_error(y_val, preds)

        scores.append(score)

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
    var metric_code: Int
    if scoring == "accuracy":
        metric_code = 0
    elif scoring == "f1":
        metric_code = 1
    elif scoring == "precision":
        metric_code = 2
    elif scoring == "recall":
        metric_code = 3
    else:
        raise InvalidParameterError.error(
            "scoring",
            "Unsupported classification scoring metric: '" + scoring + "'",
        )

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

        var y_val_int = List[Int](capacity=len(y_val))
        for i in range(len(y_val)):
            y_val_int.append(Int(y_val[i]))

        var score: Float64
        if metric_code == 0:
            score = accuracy_score(y_val_int, preds)
        elif metric_code == 1:
            score = f1_score(y_val_int, preds)
        elif metric_code == 2:
            score = precision_score(y_val_int, preds)
        else:
            score = recall_score(y_val_int, preds)

        scores.append(score)

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
