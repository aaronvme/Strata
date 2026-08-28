from ..core.matrix import Matrix
from ..base.estimator import Regressor
from ..utils.validation import (
    check_X_y,
    check_array,
    check_floating_dtype,
    check_is_fitted,
)
from ..utils.random import PRNG
from ..exceptions.errors import InvalidParameterError, DimensionMismatchError
from ..tree.regressor import DecisionTreeRegressor
from ._forest import (
    generate_bootstrap_indices,
    get_oob_mask,
    compute_tree_feature_importances,
    normalize_feature_importances,
)


struct RandomForestRegressor[
    compute_dtype: DType = DType.float64,
](Copyable, Movable, Regressor):
    """Random Forest Regressor.

    An ensemble of decision trees trained via bootstrap aggregation (bagging).
    Predictions are computed as the arithmetic mean of individual tree predictions.

    Parameters
    ----------
    n_estimators : Int
        Number of trees in the forest. Default 100.
    criterion : String
        Impurity criterion for each tree: 'squared_error', 'friedman_mse',
        or 'absolute_error'. Default 'squared_error'.
    max_depth : Int
        Maximum tree depth. -1 means unlimited. Default -1.
    min_samples_split : Int
        Minimum samples required to split an internal node. Default 2.
    min_samples_leaf : Int
        Minimum samples required to be a leaf. Default 1.
    min_impurity_decrease : Float64
        A node is split only if the impurity decrease is >= this value.
        Default 0.0.
    max_features : String
        Number of features to consider per split: 'all', 'sqrt', 'log2'.
        Default 'sqrt'.
    max_features_count : Int
        Exact number of features per split. Overrides max_features when > 0.
        Default -1 (disabled).
    max_features_ratio : Float64
        Fraction of features per split. Overrides max_features when > 0.0.
        Default 0.0 (disabled).
    bootstrap : Bool
        Whether to use bootstrap sampling. If False, each tree trains on
        the full dataset. Default True.
    max_samples_ratio : Float64
        Fraction of samples drawn per tree (with replacement). Only used when
        bootstrap=True and max_samples_count <= 0. Default 1.0.
    max_samples_count : Int
        Exact number of samples drawn per tree. Overrides max_samples_ratio
        when > 0. Default -1 (disabled).
    oob_score : Bool
        Whether to estimate out-of-bag R² score after fitting.
        Only valid when bootstrap=True. Default False.
    random_state : Int
        Seed for the PRNG. Default 42.
    """

    var is_fitted: Bool
    var n_estimators: Int
    var criterion: String
    var max_depth: Int
    var min_samples_split: Int
    var min_samples_leaf: Int
    var min_impurity_decrease: Float64
    var max_features: String
    var max_features_count: Int
    var max_features_ratio: Float64
    var bootstrap: Bool
    var max_samples_ratio: Float64
    var max_samples_count: Int
    var oob_score: Bool
    var random_state: Int

    # Fitted attributes
    var n_features_in_: Int
    var estimators_: List[DecisionTreeRegressor[Self.compute_dtype]]
    var feature_importances_: List[Float64]
    var oob_score_: Float64
    var rng: PRNG

    def __init__(
        out self,
        n_estimators: Int = 100,
        criterion: String = "squared_error",
        max_depth: Int = -1,
        min_samples_split: Int = 2,
        min_samples_leaf: Int = 1,
        min_impurity_decrease: Float64 = 0.0,
        max_features: String = "sqrt",
        max_features_count: Int = -1,
        max_features_ratio: Float64 = 0.0,
        bootstrap: Bool = True,
        max_samples_ratio: Float64 = 1.0,
        max_samples_count: Int = -1,
        oob_score: Bool = False,
        random_state: Int = 42,
    ) raises:
        check_floating_dtype[Self.compute_dtype, "RandomForestRegressor"]()

        if n_estimators < 1:
            raise InvalidParameterError.error(
                "n_estimators",
                "n_estimators must be >= 1, got " + String(n_estimators),
            )

        if (
            criterion != "squared_error"
            and criterion != "friedman_mse"
            and criterion != "absolute_error"
        ):
            raise InvalidParameterError.error(
                "criterion",
                "RandomForestRegressor requires criterion in ['squared_error',"
                " 'friedman_mse', 'absolute_error'], got '"
                + criterion
                + "'",
            )

        if max_depth < 1 and max_depth != -1:
            raise InvalidParameterError.error(
                "max_depth",
                "max_depth must be >= 1 or -1 (unlimited), got "
                + String(max_depth),
            )

        if min_samples_split < 2:
            raise InvalidParameterError.error(
                "min_samples_split",
                "min_samples_split must be >= 2, got "
                + String(min_samples_split),
            )

        if min_samples_leaf < 1:
            raise InvalidParameterError.error(
                "min_samples_leaf",
                "min_samples_leaf must be >= 1, got "
                + String(min_samples_leaf),
            )

        if min_impurity_decrease < 0.0:
            raise InvalidParameterError.error(
                "min_impurity_decrease",
                "min_impurity_decrease must be >= 0.0, got "
                + String(min_impurity_decrease),
            )

        if (
            max_features_count == -1
            and max_features_ratio == 0.0
            and max_features != "all"
            and max_features != "sqrt"
            and max_features != "log2"
        ):
            raise InvalidParameterError.error(
                "max_features",
                "max_features must be 'all', 'sqrt', or 'log2', got '"
                + max_features
                + "'",
            )

        if max_features_count != -1 and max_features_count < 1:
            raise InvalidParameterError.error(
                "max_features_count",
                "max_features_count must be >= 1, got "
                + String(max_features_count),
            )

        if max_features_ratio != 0.0 and (
            max_features_ratio <= 0.0 or max_features_ratio > 1.0
        ):
            raise InvalidParameterError.error(
                "max_features_ratio",
                "max_features_ratio must be in (0.0, 1.0], got "
                + String(max_features_ratio),
            )

        if max_samples_ratio <= 0.0 or max_samples_ratio > 1.0:
            raise InvalidParameterError.error(
                "max_samples_ratio",
                "max_samples_ratio must be in (0.0, 1.0], got "
                + String(max_samples_ratio),
            )

        if max_samples_count != -1 and max_samples_count < 1:
            raise InvalidParameterError.error(
                "max_samples_count",
                "max_samples_count must be >= 1, got "
                + String(max_samples_count),
            )

        if oob_score and not bootstrap:
            raise InvalidParameterError.error(
                "oob_score",
                "oob_score=True requires bootstrap=True",
            )

        self.is_fitted = False
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.max_features_count = max_features_count
        self.max_features_ratio = max_features_ratio
        self.bootstrap = bootstrap
        self.max_samples_ratio = max_samples_ratio
        self.max_samples_count = max_samples_count
        self.oob_score = oob_score
        self.random_state = random_state

        self.n_features_in_ = 0
        self.estimators_ = List[DecisionTreeRegressor[Self.compute_dtype]]()
        self.feature_importances_ = List[Float64]()
        self.oob_score_ = 0.0
        self.rng = PRNG(random_state if random_state >= 0 else 42)

    def __init__(out self, *, copy: Self):
        self.is_fitted = copy.is_fitted
        self.n_estimators = copy.n_estimators
        self.criterion = copy.criterion
        self.max_depth = copy.max_depth
        self.min_samples_split = copy.min_samples_split
        self.min_samples_leaf = copy.min_samples_leaf
        self.min_impurity_decrease = copy.min_impurity_decrease
        self.max_features = copy.max_features
        self.max_features_count = copy.max_features_count
        self.max_features_ratio = copy.max_features_ratio
        self.bootstrap = copy.bootstrap
        self.max_samples_ratio = copy.max_samples_ratio
        self.max_samples_count = copy.max_samples_count
        self.oob_score = copy.oob_score
        self.random_state = copy.random_state
        self.n_features_in_ = copy.n_features_in_
        self.estimators_ = List[DecisionTreeRegressor[Self.compute_dtype]]()
        for i in range(len(copy.estimators_)):
            self.estimators_.append(
                DecisionTreeRegressor[Self.compute_dtype](
                    copy=copy.estimators_[i]
                )
            )
        self.feature_importances_ = copy.feature_importances_.copy()
        self.oob_score_ = copy.oob_score_
        self.rng = PRNG(copy.random_state if copy.random_state >= 0 else 42)

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](
        mut self,
        X: Matrix[feat_dtype],
        y: List[Scalar[target_dtype]],
    ) raises:
        """Fits the random forest on (X, y).

        Each tree is trained on a bootstrap sample (or the full dataset when
        bootstrap=False). After fitting, `feature_importances_` holds averaged
        MDI importances and, when oob_score=True, `oob_score_` holds the OOB R² score.
        """
        check_X_y(X, y)
        self.n_features_in_ = X.cols

        var n_samples = X.rows
        self.estimators_ = List[DecisionTreeRegressor[Self.compute_dtype]](
            capacity=self.n_estimators
        )
        self.rng = PRNG(self.random_state if self.random_state >= 0 else 42)

        # OOB accumulation buffers: sum of predictions and count per sample.
        var oob_preds = List[Float64](capacity=n_samples)
        var oob_counts = List[Int](capacity=n_samples)
        for _ in range(n_samples):
            oob_preds.append(0.0)
            oob_counts.append(0)

        # Accumulate raw (un-normalized) MDI importances across all trees.
        var raw_importances = List[Float64](capacity=self.n_features_in_)
        for _ in range(self.n_features_in_):
            raw_importances.append(0.0)

        for _ in range(self.n_estimators):
            # Each tree gets its own independent PRNG streams seeded from the parent.
            # Splitting into two separate streams prevents tree-seed and
            # bootstrap-index generation from interleaving on the same state.
            var tree_seed = Int(self.rng.next_u64() % UInt64(2147483647))
            var boot_rng = PRNG(Int(self.rng.next_u64() % UInt64(2147483647)))

            var tree = DecisionTreeRegressor[Self.compute_dtype](
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=self.max_features,
                max_features_count=self.max_features_count,
                max_features_ratio=self.max_features_ratio,
                random_state=tree_seed,
            )

            if self.bootstrap:
                var sample_indices = generate_bootstrap_indices(
                    n_samples,
                    self.max_samples_count,
                    self.max_samples_ratio,
                    boot_rng,
                )

                # Build bootstrap X and y views (index-filtered copies).
                var k = len(sample_indices)
                var X_boot = Matrix[feat_dtype](k, X.cols)
                var y_boot = List[Scalar[target_dtype]](capacity=k)
                for i in range(k):
                    var src = sample_indices[i]
                    for j in range(X.cols):
                        X_boot[i, j] = X[src, j]
                    y_boot.append(y[src])

                tree.fit[feat_dtype, target_dtype](X_boot, y_boot)

                if self.oob_score:
                    var oob_mask = get_oob_mask(n_samples, sample_indices)
                    # Collect all OOB row indices to perform a single batched inference pass per tree.
                    var oob_row_indices = List[Int]()
                    for i in range(n_samples):
                        if oob_mask[i]:
                            oob_row_indices.append(i)

                    var n_oob = len(oob_row_indices)
                    if n_oob > 0:
                        var X_oob = Matrix[feat_dtype](n_oob, X.cols)
                        for i in range(n_oob):
                            var src = oob_row_indices[i]
                            for j in range(X.cols):
                                X_oob[i, j] = X[src, j]

                        var oob_preds_tree = tree.predict[feat_dtype](X_oob)
                        for i in range(n_oob):
                            var orig = oob_row_indices[i]
                            oob_preds[orig] += oob_preds_tree[i].cast[
                                DType.float64
                            ]()
                            oob_counts[orig] += 1
            else:
                # No bootstrapping: train on full dataset.
                tree.fit[feat_dtype, target_dtype](X, y)

            # Accumulate MDI importances from this tree.
            var tree_imp = compute_tree_feature_importances(
                tree.tree_, self.n_features_in_, n_samples
            )
            for j in range(self.n_features_in_):
                raw_importances[j] += tree_imp[j]

            self.estimators_.append(tree^)

        # L1-normalize directly — dividing by n_estimators before normalization
        # is mathematically redundant since the sum cancels in the ratio.
        self.feature_importances_ = normalize_feature_importances(
            raw_importances, self.n_features_in_
        )

        # Compute OOB R² if requested.
        if self.oob_score and self.bootstrap:
            var ss_res: Float64 = 0.0
            var ss_tot: Float64 = 0.0
            var y_mean: Float64 = 0.0
            var oob_n: Int = 0

            for i in range(n_samples):
                if oob_counts[i] > 0:
                    y_mean += y[i].cast[DType.float64]()
                    oob_n += 1
            if oob_n > 0:
                y_mean /= Float64(oob_n)

            for i in range(n_samples):
                if oob_counts[i] > 0:
                    var pred = oob_preds[i] / Float64(oob_counts[i])
                    var diff = y[i].cast[DType.float64]() - pred
                    var mean_diff = y[i].cast[DType.float64]() - y_mean
                    ss_res += diff * diff
                    ss_tot += mean_diff * mean_diff

            self.oob_score_ = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        self.is_fitted = True

    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Scalar[feat_dtype]]:
        """Predicts regression targets as the arithmetic mean across all tree predictions."""
        check_is_fitted("RandomForestRegressor", self.is_fitted)
        check_array(X)
        if X.cols != self.n_features_in_:
            raise DimensionMismatchError.error(
                String(self.n_features_in_) + " features",
                String(X.cols) + " features",
                "predict",
            )

        var n_samples = X.rows
        var sums = List[Float64](capacity=n_samples)
        for _ in range(n_samples):
            sums.append(0.0)

        for t in range(len(self.estimators_)):
            var preds = self.estimators_[t].predict[feat_dtype](X)
            for i in range(n_samples):
                sums[i] += preds[i].cast[DType.float64]()

        var n_trees = Float64(len(self.estimators_))
        var out = List[Scalar[feat_dtype]](capacity=n_samples)
        for i in range(n_samples):
            out.append((sums[i] / n_trees).cast[feat_dtype]())
        return out^

    def get_n_estimators(self) -> Int:
        """Returns the number of fitted trees."""
        return len(self.estimators_)

    def get_feature_importances(self) raises -> List[Float64]:
        """Returns normalized MDI feature importances (sums to 1.0)."""
        check_is_fitted("RandomForestRegressor", self.is_fitted)
        return self.feature_importances_.copy()

    def get_oob_score(self) raises -> Float64:
        """Returns out-of-bag R² score. Requires oob_score=True and bootstrap=True."""
        check_is_fitted("RandomForestRegressor", self.is_fitted)
        if not self.oob_score:
            raise InvalidParameterError.error(
                "oob_score",
                "oob_score was not enabled at construction time",
            )
        return self.oob_score_
