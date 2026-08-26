from std.math import sqrt
from ..core.matrix import Matrix
from ..base.estimator import Regressor, Classifier
from ..utils.validation import check_X_y, check_is_fitted
from ..exceptions.errors import InvalidParameterError, NotFittedError
from .kfold import KFold, Split
from .stratified_kfold import StratifiedKFold
from .validation import cross_val_score


struct GridSearchRegressor[
    ModelType: Regressor,
    feat_dtype: DType = DType.float64,
    target_dtype: DType = DType.float64,
](Copyable, Movable, Regressor):
    """Exhaustive hyperparameter grid search for regression models."""

    var candidates: List[Self.ModelType]
    var cv: Int
    var scoring: String
    var refit: Bool
    var is_fitted: Bool
    var best_index_: Int
    var best_score_: Float64
    var best_estimator_: Self.ModelType
    var cv_results_mean_: List[Float64]
    var cv_results_std_: List[Float64]

    def __init__(
        out self,
        var candidates: List[Self.ModelType],
        cv: Int = 5,
        scoring: String = "r2",
        refit: Bool = True,
    ) raises:
        """Initializes the grid search regressor cross-validator.

        Args:
            candidates: List of candidate model configurations to evaluate.
            cv: Number of cross-validation folds.
            scoring: Scoring metric name.
            refit: Whether to refit the best model on the complete dataset.
        """
        if len(candidates) == 0:
            raise InvalidParameterError.error(
                "candidates",
                "GridSearchRegressor requires at least one candidate estimator",
            )
        if cv < 2:
            raise InvalidParameterError.error(
                "cv", "cv must be at least 2, got " + String(cv)
            )
        self.candidates = candidates^
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        self.is_fitted = False
        self.best_index_ = -1
        self.best_score_ = -1e30
        self.best_estimator_ = self.candidates[0].copy()
        self.cv_results_mean_ = List[Float64]()
        self.cv_results_std_ = List[Float64]()

    def __init__(out self, *, copy: Self):
        """Copies an existing GridSearchRegressor instance."""
        self.candidates = copy.candidates.copy()
        self.cv = copy.cv
        self.scoring = copy.scoring
        self.refit = copy.refit
        self.is_fitted = copy.is_fitted
        self.best_index_ = copy.best_index_
        self.best_score_ = copy.best_score_
        self.best_estimator_ = copy.best_estimator_.copy()
        self.cv_results_mean_ = copy.cv_results_mean_.copy()
        self.cv_results_std_ = copy.cv_results_std_.copy()

    def fit[
        in_feat_dtype: DType, in_target_dtype: DType
    ](
        mut self, X: Matrix[in_feat_dtype], y: List[Scalar[in_target_dtype]]
    ) raises:
        """Runs cross-validation across all candidate models and fits the best one.

        Args:
            X: Feature training matrix.
            y: Target values.
        """
        check_X_y(X, y)
        var X_cast = X.cast[Self.feat_dtype]()
        var y_cast = List[Scalar[Self.target_dtype]](capacity=len(y))
        for i in range(len(y)):
            y_cast.append(Scalar[Self.target_dtype](y[i]))

        var kf = KFold(n_splits=self.cv)
        var splits = kf.split(X_cast.rows)

        var n_candidates = len(self.candidates)
        self.cv_results_mean_ = List[Float64](capacity=n_candidates)
        self.cv_results_std_ = List[Float64](capacity=n_candidates)

        var best_idx = -1
        var best_score = -1e30

        for i in range(n_candidates):
            # Pass fresh clone to cross_val_score to leave pristine candidate untouched
            var candidate_eval = self.candidates[i].copy()
            var fold_scores = cross_val_score[
                Self.ModelType, Self.feat_dtype, Self.target_dtype
            ](
                candidate_eval,
                X_cast,
                y_cast,
                splits,
                scoring=self.scoring,
            )
            var n_folds = len(fold_scores)
            var mean_score: Float64 = 0.0
            for f in range(n_folds):
                mean_score += fold_scores[f]
            mean_score /= Float64(n_folds)

            var var_sum: Float64 = 0.0
            for f in range(n_folds):
                var diff = fold_scores[f] - mean_score
                var_sum += diff * diff
            var std_score = sqrt(var_sum / Float64(n_folds))

            self.cv_results_mean_.append(mean_score)
            self.cv_results_std_.append(std_score)

            if mean_score > best_score or best_idx == -1:
                best_score = mean_score
                best_idx = i

        self.best_index_ = best_idx
        self.best_score_ = best_score
        self.best_estimator_ = self.candidates[best_idx].copy()

        if self.refit:
            self.best_estimator_.fit(X_cast, y_cast)

        self.is_fitted = True

    def predict[
        in_feat_dtype: DType
    ](self, X: Matrix[in_feat_dtype]) raises -> List[Scalar[in_feat_dtype]]:
        """Predicts targets using the best discovered model configuration."""
        check_is_fitted("GridSearchRegressor", self.is_fitted)
        if not self.refit:
            raise NotFittedError.error(
                "This GridSearchRegressor instance was initialized with"
                " refit=False. "
                + "Predictions are unavailable without refit."
            )
        var X_cast = X.cast[Self.feat_dtype]()
        var preds_comp = self.best_estimator_.predict(X_cast)
        var preds = List[Scalar[in_feat_dtype]](capacity=len(preds_comp))
        for i in range(len(preds_comp)):
            preds.append(Scalar[in_feat_dtype](preds_comp[i]))
        return preds^


struct GridSearchClassifier[
    ModelType: Classifier,
    feat_dtype: DType = DType.float64,
    target_dtype: DType = DType.int32,
](Classifier, Copyable, Movable):
    """Exhaustive hyperparameter grid search for classification models."""

    var candidates: List[Self.ModelType]
    var cv: Int
    var scoring: String
    var refit: Bool
    var is_fitted: Bool
    var best_index_: Int
    var best_score_: Float64
    var best_estimator_: Self.ModelType
    var cv_results_mean_: List[Float64]
    var cv_results_std_: List[Float64]

    def __init__(
        out self,
        var candidates: List[Self.ModelType],
        cv: Int = 5,
        scoring: String = "accuracy",
        refit: Bool = True,
    ) raises:
        """Initializes the grid search classifier cross-validator.

        Args:
            candidates: List of candidate model configurations to evaluate.
            cv: Number of cross-validation folds.
            scoring: Scoring metric name.
            refit: Whether to refit the best model on the complete dataset.
        """
        if len(candidates) == 0:
            raise InvalidParameterError.error(
                "candidates",
                (
                    "GridSearchClassifier requires at least one candidate"
                    " estimator"
                ),
            )
        if cv < 2:
            raise InvalidParameterError.error(
                "cv", "cv must be at least 2, got " + String(cv)
            )
        self.candidates = candidates^
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        self.is_fitted = False
        self.best_index_ = -1
        self.best_score_ = -1e30
        self.best_estimator_ = self.candidates[0].copy()
        self.cv_results_mean_ = List[Float64]()
        self.cv_results_std_ = List[Float64]()

    def __init__(out self, *, copy: Self):
        """Copies an existing GridSearchClassifier instance."""
        self.candidates = copy.candidates.copy()
        self.cv = copy.cv
        self.scoring = copy.scoring
        self.refit = copy.refit
        self.is_fitted = copy.is_fitted
        self.best_index_ = copy.best_index_
        self.best_score_ = copy.best_score_
        self.best_estimator_ = copy.best_estimator_.copy()
        self.cv_results_mean_ = copy.cv_results_mean_.copy()
        self.cv_results_std_ = copy.cv_results_std_.copy()

    def fit[
        in_feat_dtype: DType, in_target_dtype: DType
    ](
        mut self, X: Matrix[in_feat_dtype], y: List[Scalar[in_target_dtype]]
    ) raises:
        """Runs stratified cross-validation across all candidate models and fits the best one.

        Args:
            X: Feature training matrix.
            y: Discrete class labels.
        """
        check_X_y(X, y)
        var X_cast = X.cast[Self.feat_dtype]()
        var y_cast = List[Scalar[Self.target_dtype]](capacity=len(y))
        for i in range(len(y)):
            y_cast.append(Scalar[Self.target_dtype](y[i]))

        var skf = StratifiedKFold(n_splits=self.cv)
        var splits = skf.split(X_cast, y_cast)

        var n_candidates = len(self.candidates)
        self.cv_results_mean_ = List[Float64](capacity=n_candidates)
        self.cv_results_std_ = List[Float64](capacity=n_candidates)

        var best_idx = -1
        var best_score = -1e30

        for i in range(n_candidates):
            # Pass fresh clone to cross_val_score to leave pristine candidate untouched
            var candidate_eval = self.candidates[i].copy()
            var fold_scores = cross_val_score[
                Self.ModelType, Self.feat_dtype, Self.target_dtype
            ](
                candidate_eval,
                X_cast,
                y_cast,
                splits,
                scoring=self.scoring,
            )
            var n_folds = len(fold_scores)
            var mean_score: Float64 = 0.0
            for f in range(n_folds):
                mean_score += fold_scores[f]
            mean_score /= Float64(n_folds)

            var var_sum: Float64 = 0.0
            for f in range(n_folds):
                var diff = fold_scores[f] - mean_score
                var_sum += diff * diff
            var std_score = sqrt(var_sum / Float64(n_folds))

            self.cv_results_mean_.append(mean_score)
            self.cv_results_std_.append(std_score)

            if mean_score > best_score or best_idx == -1:
                best_score = mean_score
                best_idx = i

        self.best_index_ = best_idx
        self.best_score_ = best_score
        self.best_estimator_ = self.candidates[best_idx].copy()

        if self.refit:
            self.best_estimator_.fit(X_cast, y_cast)

        self.is_fitted = True

    def predict[
        in_feat_dtype: DType
    ](self, X: Matrix[in_feat_dtype]) raises -> List[Int]:
        """Predicts class labels using the best discovered model configuration.
        """
        check_is_fitted("GridSearchClassifier", self.is_fitted)
        if not self.refit:
            raise NotFittedError.error(
                "This GridSearchClassifier instance was initialized with"
                " refit=False. "
                + "Predictions are unavailable without refit."
            )
        var X_cast = X.cast[Self.feat_dtype]()
        return self.best_estimator_.predict(X_cast)

    def predict_proba[
        in_feat_dtype: DType
    ](self, X: Matrix[in_feat_dtype]) raises -> Matrix[in_feat_dtype]:
        """Predicts class probabilities using the best discovered model configuration.
        """
        check_is_fitted("GridSearchClassifier", self.is_fitted)
        if not self.refit:
            raise NotFittedError.error(
                "This GridSearchClassifier instance was initialized with"
                " refit=False. "
                + "Probability predictions are unavailable without refit."
            )
        var X_cast = X.cast[Self.feat_dtype]()
        var proba_comp = self.best_estimator_.predict_proba(X_cast)
        return proba_comp.cast[in_feat_dtype]()
