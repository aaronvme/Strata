from std.math import sqrt

from ..core.matrix import Matrix
from ..base.estimator import Regressor, Classifier
from ..utils.random import permutation
from ..utils.validation import check_X_y, check_is_fitted
from ..exceptions.errors import InvalidParameterError, NotFittedError
from .kfold import KFold, Split
from .stratified_kfold import StratifiedKFold
from .validation import cross_val_score


struct RandomizedSearchRegressor[
    ModelType: Regressor,
    feat_dtype: DType = DType.float64,
    target_dtype: DType = DType.float64,
](Copyable, Movable):
    """Randomized hyperparameter search for regression models.

    Evaluates a random subset of the supplied candidate configurations rather
    than the full grid, trading exhaustive coverage for a fixed search budget.
    """

    var candidates: List[Self.ModelType]
    var n_iter: Int
    var cv: Int
    var scoring: String
    var refit: Bool
    var random_state: Int
    var is_fitted: Bool
    var best_index_: Int
    var best_score_: Float64
    var best_estimator_: Self.ModelType
    var sampled_indices_: List[Int]
    var cv_results_mean_: List[Float64]
    var cv_results_std_: List[Float64]

    def __init__(
        out self,
        var candidates: List[Self.ModelType],
        n_iter: Int = 10,
        cv: Int = 5,
        scoring: String = "r2",
        refit: Bool = True,
        random_state: Int = 42,
    ) raises:
        """Initializes the randomized search regressor cross-validator.

        Args:
            candidates: Pool of candidate model configurations to sample from.
            n_iter: Number of candidates to evaluate, clamped to the pool size.
            cv: Number of cross-validation folds.
            scoring: Scoring metric name.
            refit: Whether to refit the best model on the complete dataset.
            random_state: Seed controlling which candidates are sampled.
        """
        if len(candidates) == 0:
            raise InvalidParameterError.error(
                "candidates",
                (
                    "RandomizedSearchRegressor requires at least one candidate"
                    " estimator"
                ),
            )
        if n_iter < 1:
            raise InvalidParameterError.error(
                "n_iter", "n_iter must be at least 1, got " + String(n_iter)
            )
        if cv < 2:
            raise InvalidParameterError.error(
                "cv", "cv must be at least 2, got " + String(cv)
            )
        self.candidates = candidates^
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        self.random_state = random_state
        self.is_fitted = False
        self.best_index_ = -1
        self.best_score_ = -1e30
        self.best_estimator_ = self.candidates[0].copy()
        self.sampled_indices_ = List[Int]()
        self.cv_results_mean_ = List[Float64]()
        self.cv_results_std_ = List[Float64]()

    def __init__(out self, *, copy: Self):
        """Copies an existing RandomizedSearchRegressor instance."""
        self.candidates = copy.candidates.copy()
        self.n_iter = copy.n_iter
        self.cv = copy.cv
        self.scoring = copy.scoring
        self.refit = copy.refit
        self.random_state = copy.random_state
        self.is_fitted = copy.is_fitted
        self.best_index_ = copy.best_index_
        self.best_score_ = copy.best_score_
        self.best_estimator_ = copy.best_estimator_.copy()
        self.sampled_indices_ = copy.sampled_indices_.copy()
        self.cv_results_mean_ = copy.cv_results_mean_.copy()
        self.cv_results_std_ = copy.cv_results_std_.copy()

    def fit[
        in_feat_dtype: DType, in_target_dtype: DType
    ](
        mut self, X: Matrix[in_feat_dtype], y: List[Scalar[in_target_dtype]]
    ) raises:
        """Cross-validates a random subset of candidates and fits the best one.

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
        var n_sampled = self.n_iter
        if n_sampled > n_candidates:
            n_sampled = n_candidates

        var order = permutation(n_candidates, seed=self.random_state)

        self.sampled_indices_ = List[Int](capacity=n_sampled)
        for i in range(n_sampled):
            self.sampled_indices_.append(order[i])

        self.cv_results_mean_ = List[Float64](capacity=n_sampled)
        self.cv_results_std_ = List[Float64](capacity=n_sampled)

        var best_idx = -1
        var best_score = -1e30

        for i in range(n_sampled):
            var candidate_idx = self.sampled_indices_[i]
            var candidate_eval = self.candidates[candidate_idx].copy()
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
                best_idx = candidate_idx

        self.best_index_ = best_idx
        self.best_score_ = best_score
        self.best_estimator_ = self.candidates[best_idx].copy()

        if self.refit:
            self.best_estimator_.fit(X_cast, y_cast)

        self.is_fitted = True
