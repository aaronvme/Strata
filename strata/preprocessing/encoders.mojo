from ..base.estimator import Transformer
from ..core.matrix import Matrix
from ..core.dataset import Dataset
from ..utils.validation import (
    check_is_fitted,
    check_array,
    check_floating_dtype,
)
from ..exceptions.errors import (
    NotFittedError,
    DataConversionError,
    DimensionMismatchError,
    InvalidParameterError,
)


def _unique_sorted_column[
    compute_dtype: DType, in_dtype: DType
](X: Matrix[in_dtype], col: Int) -> List[Scalar[compute_dtype]]:
    var values = List[Scalar[compute_dtype]](capacity=X.rows)
    for r in range(X.rows):
        values.append(Scalar[compute_dtype](X[r, col]))
    sort(values)

    var uniques = List[Scalar[compute_dtype]]()
    for i in range(len(values)):
        if i == 0 or values[i] != values[i - 1]:
            uniques.append(values[i])
    return uniques^


def _index_of[
    dtype: DType
](categories: List[Scalar[dtype]], value: Scalar[dtype]) -> Int:
    var lo = 0
    var hi = len(categories)
    while lo < hi:
        var mid = (lo + hi) // 2
        if categories[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    if lo < len(categories) and categories[lo] == value:
        return lo
    return -1


struct OneHotEncoder[compute_dtype: DType = DType.float64](
    Copyable, Movable, Transformer
):
    """Encode categorical features as a one-hot numeric array.

    The input to this transformer should be a 2D matrix of integer or float
    categorical features. The features are encoded using a one-hot (also known as
    'one-of-K' or 'dummy') encoding scheme.

    Parameters:
        compute_dtype: Computational precision data type. Default DType.float64.

    Args:
        drop: Category dropping strategy ('none', 'first', 'if_binary'). Default 'none'.
        handle_unknown: Behavior for unseen categories during transform ('error', 'ignore'). Default 'error'.

    Attributes:
        categories_: Categories of each feature determined during fitting.
        drop_idx_: Indices of dropped categories for each feature.
        n_features_in_: Number of features seen during fit.
        is_fitted: Boolean flag indicating if estimator has been fitted.

    Examples:
        ```mojo
        from strata.preprocessing import OneHotEncoder
        from strata.core import Matrix

        var encoder = OneHotEncoder[DType.float64](drop="if_binary")
        encoder.fit(X_cat)
        var X_encoded = encoder.transform(X_cat)
        ```
    """

    var is_fitted: Bool
    var fit_dtype: DType
    var handle_unknown: String
    var drop: String
    var categories_: List[List[Scalar[Self.compute_dtype]]]
    var drop_idx_: List[Int]
    var n_features_in_: Int

    def __init__(
        out self, drop: String = "none", handle_unknown: String = "error"
    ) raises:
        """Initialize the OneHotEncoder.

        Args:
            drop: Category dropping strategy ('none', 'first', 'if_binary'). Default 'none'.
            handle_unknown: Behavior for unseen categories ('error', 'ignore'). Default 'error'.

        Raises:
            InvalidParameterError: If drop or handle_unknown strategies are unrecognized.
        """

        check_floating_dtype[Self.compute_dtype, "OneHotEncoder"]()
        if drop != "none" and drop != "first" and drop != "if_binary":
            raise InvalidParameterError.error(
                "drop",
                "expected 'none', 'first' or 'if_binary', got '" + drop + "'",
            )
        if handle_unknown != "error" and handle_unknown != "ignore":
            raise InvalidParameterError.error(
                "handle_unknown",
                "expected 'error' or 'ignore', got '" + handle_unknown + "'",
            )
        self.is_fitted = False
        self.fit_dtype = DType.float64
        self.handle_unknown = handle_unknown
        self.drop = drop
        self.categories_ = List[List[Scalar[Self.compute_dtype]]]()
        self.drop_idx_ = List[Int]()
        self.n_features_in_ = 0

    def __init__(out self, *, copy: Self):
        """Copies an existing OneHotEncoder instance."""
        self.is_fitted = copy.is_fitted
        self.fit_dtype = copy.fit_dtype
        self.handle_unknown = copy.handle_unknown
        self.drop = copy.drop
        self.categories_ = copy.categories_.copy()
        self.drop_idx_ = copy.drop_idx_.copy()
        self.n_features_in_ = copy.n_features_in_

    def n_features_out(self) -> Int:
        """Number of indicator columns produced by transform."""
        var total = 0
        for f in range(len(self.categories_)):
            total += len(self.categories_[f])
            if self.drop_idx_[f] >= 0:
                total -= 1
        return total

    def _column_offsets(self) -> List[Int]:
        var offsets = List[Int](capacity=len(self.categories_) + 1)
        var running = 0
        for f in range(len(self.categories_)):
            offsets.append(running)
            running += len(self.categories_[f])
            if self.drop_idx_[f] >= 0:
                running -= 1
        offsets.append(running)
        return offsets^

    def fit[in_dtype: DType](mut self, X: Matrix[in_dtype]) raises:
        check_array[in_dtype](X)
        var n_cols = X.cols

        self.categories_ = List[List[Scalar[Self.compute_dtype]]](
            capacity=n_cols
        )
        self.drop_idx_ = List[Int](capacity=n_cols)

        for c in range(n_cols):
            var uniques = _unique_sorted_column[Self.compute_dtype, in_dtype](
                X, c
            )

            if self.drop == "first" and len(uniques) > 0:
                self.drop_idx_.append(0)
            elif self.drop == "if_binary" and len(uniques) == 2:
                self.drop_idx_.append(0)
            else:
                self.drop_idx_.append(-1)

            self.categories_.append(uniques^)

        self.n_features_in_ = n_cols
        self.fit_dtype = in_dtype
        self.is_fitted = True

    def fit[
        feat_dtype: DType,
        target_dtype: DType,
    ](mut self, dataset: Dataset[feat_dtype, target_dtype]) raises:
        self.fit[feat_dtype](dataset.records)

    def transform[
        in_dtype: DType
    ](self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        check_is_fitted("OneHotEncoder", self.is_fitted)
        if in_dtype != self.fit_dtype:
            raise DataConversionError.error(
                "OneHotEncoder.transform received Matrix["
                + String(in_dtype)
                + "] but was fitted on Matrix["
                + String(self.fit_dtype)
                + "]"
            )
        check_array[in_dtype](X)
        if X.cols != len(self.categories_):
            raise DimensionMismatchError.error(
                "X.cols == " + String(len(self.categories_)),
                "X.cols == " + String(X.cols),
                "OneHotEncoder.transform",
            )

        var offsets = self._column_offsets()
        var res = Matrix[in_dtype](X.rows, offsets[X.cols], 0)

        for r in range(X.rows):
            for c in range(X.cols):
                var value = Scalar[Self.compute_dtype](X[r, c])
                var idx = _index_of[Self.compute_dtype](
                    self.categories_[c], value
                )
                if idx < 0:
                    if self.handle_unknown == "error":
                        raise InvalidParameterError.error(
                            "X",
                            "OneHotEncoder.transform found unknown category "
                            + String(value)
                            + " in column "
                            + String(c),
                        )
                    continue

                var dropped = self.drop_idx_[c]
                if idx == dropped:
                    continue
                if dropped >= 0 and idx > dropped:
                    idx -= 1
                res[r, offsets[c] + idx] = 1

        return res^

    def get_feature_names_out(
        self, input_features: List[String] = List[String]()
    ) raises -> List[String]:
        """Output column names as '<feature>_<category>' pairs."""
        check_is_fitted("OneHotEncoder", self.is_fitted)
        if len(input_features) != 0 and len(input_features) != len(
            self.categories_
        ):
            raise DimensionMismatchError.error(
                "len(input_features) == " + String(len(self.categories_)),
                "len(input_features) == " + String(len(input_features)),
                "OneHotEncoder.get_feature_names_out",
            )

        var names = List[String](capacity=self.n_features_out())
        for f in range(len(self.categories_)):
            var base: String
            if len(input_features) != 0:
                base = input_features[f]
            else:
                base = "x" + String(f)
            for i in range(len(self.categories_[f])):
                if i == self.drop_idx_[f]:
                    continue
                names.append(base + "_" + String(self.categories_[f][i]))
        return names^

    def transform[
        feat_dtype: DType,
        target_dtype: DType,
    ](self, dataset: Dataset[feat_dtype, target_dtype]) raises -> Dataset[
        feat_dtype, target_dtype
    ]:
        var encoded_records = self.transform[feat_dtype](dataset.records)
        return Dataset[feat_dtype, target_dtype](
            encoded_records^,
            dataset.targets.copy(),
            self.get_feature_names_out(dataset.feature_names),
            dataset.target_names.copy(),
        )

    def fit_transform[
        in_dtype: DType
    ](mut self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        self.fit[in_dtype](X)
        return self.transform[in_dtype](X)

    def fit_transform[
        feat_dtype: DType,
        target_dtype: DType,
    ](mut self, dataset: Dataset[feat_dtype, target_dtype]) raises -> Dataset[
        feat_dtype, target_dtype
    ]:
        self.fit[feat_dtype, target_dtype](dataset)
        return self.transform[feat_dtype, target_dtype](dataset)

    def inverse_transform[
        in_dtype: DType
    ](self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        """Recovers the original categorical values from a one-hot matrix."""
        check_is_fitted("OneHotEncoder", self.is_fitted)
        if in_dtype != self.fit_dtype:
            raise DataConversionError.error(
                "OneHotEncoder.inverse_transform received Matrix["
                + String(in_dtype)
                + "] but was fitted on Matrix["
                + String(self.fit_dtype)
                + "]"
            )
        check_array[in_dtype](X, allow_empty=X.cols == 0)
        var n_features = len(self.categories_)
        var offsets = self._column_offsets()
        if X.cols != offsets[n_features]:
            raise DimensionMismatchError.error(
                "X.cols == " + String(offsets[n_features]),
                "X.cols == " + String(X.cols),
                "OneHotEncoder.inverse_transform",
            )

        var res = Matrix[in_dtype](X.rows, n_features, 0)
        for r in range(X.rows):
            for f in range(n_features):
                var block = offsets[f + 1] - offsets[f]
                var active = -1
                for j in range(block):
                    if X[r, offsets[f] + j] != 0:
                        active = j
                        break

                var dropped = self.drop_idx_[f]
                if active < 0:
                    if dropped < 0:
                        raise InvalidParameterError.error(
                            "X",
                            "OneHotEncoder.inverse_transform found an"
                            " all-zero block for column "
                            + String(f),
                        )
                    res[r, f] = Scalar[in_dtype](self.categories_[f][dropped])
                    continue

                if dropped >= 0 and active >= dropped:
                    active += 1
                res[r, f] = Scalar[in_dtype](self.categories_[f][active])

        return res^


struct OrdinalEncoder[compute_dtype: DType = DType.float64](Copyable, Movable):
    """Encode categorical features as integer codes.

    Each feature column is mapped onto the integers 0 to n_categories - 1,
    following the sorted order of the categories seen during fit. Unlike
    OneHotEncoder the output keeps the shape of the input, producing a single
    code per original column rather than a block of indicator columns.

    Parameters:
        compute_dtype: Computational precision data type. Default DType.float64.

    Args:
        handle_unknown: Behavior for unseen categories during transform ('error', 'use_encoded_value'). Default 'error'.
        unknown_value: Code assigned to unseen categories when handle_unknown is 'use_encoded_value'. Default -1.0.

    Attributes:
        categories_: Categories of each feature determined during fitting.
        n_features_in_: Number of features seen during fit.
        is_fitted: Boolean flag indicating if estimator has been fitted.

    Examples:
        ```mojo
        from strata.preprocessing import OrdinalEncoder
        from strata.core import Matrix

        var encoder = OrdinalEncoder[DType.float64]()
        encoder.fit(X_cat)
        var X_encoded = encoder.transform(X_cat)
        ```
    """

    var is_fitted: Bool
    var fit_dtype: DType
    var handle_unknown: String
    var unknown_value: Scalar[Self.compute_dtype]
    var categories_: List[List[Scalar[Self.compute_dtype]]]
    var n_features_in_: Int

    def __init__(
        out self,
        handle_unknown: String = "error",
        unknown_value: Scalar[Self.compute_dtype] = -1.0,
    ) raises:
        """Initialize the OrdinalEncoder.

        Args:
            handle_unknown: Behavior for unseen categories ('error', 'use_encoded_value'). Default 'error'.
            unknown_value: Code assigned to unseen categories. Default -1.0.

        Raises:
            InvalidParameterError: If handle_unknown is unrecognized.
        """

        check_floating_dtype[Self.compute_dtype, "OrdinalEncoder"]()
        if handle_unknown != "error" and handle_unknown != "use_encoded_value":
            raise InvalidParameterError.error(
                "handle_unknown",
                "expected 'error' or 'use_encoded_value', got '"
                + handle_unknown
                + "'",
            )
        self.is_fitted = False
        self.fit_dtype = DType.float64
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.categories_ = List[List[Scalar[Self.compute_dtype]]]()
        self.n_features_in_ = 0

    def __init__(out self, *, copy: Self):
        """Copies an existing OrdinalEncoder instance."""
        self.is_fitted = copy.is_fitted
        self.fit_dtype = copy.fit_dtype
        self.handle_unknown = copy.handle_unknown
        self.unknown_value = copy.unknown_value
        self.categories_ = copy.categories_.copy()
        self.n_features_in_ = copy.n_features_in_

    def fit[in_dtype: DType](mut self, X: Matrix[in_dtype]) raises:
        """Learns the sorted categories of each feature column.

        Args:
            X: Matrix of categorical features with one column per feature.

        Raises:
            InvalidParameterError: If unknown_value collides with a category code.
        """
        check_array[in_dtype](X)
        var n_cols = X.cols

        var categories = List[List[Scalar[Self.compute_dtype]]](capacity=n_cols)
        for c in range(n_cols):
            var uniques = _unique_sorted_column[Self.compute_dtype, in_dtype](
                X, c
            )
            if self.handle_unknown == "use_encoded_value":
                var n_cats = Scalar[Self.compute_dtype](len(uniques))
                if self.unknown_value >= 0 and self.unknown_value < n_cats:
                    raise InvalidParameterError.error(
                        "unknown_value",
                        String(self.unknown_value)
                        + " is already used as a category code for column "
                        + String(c),
                    )
            categories.append(uniques^)

        self.categories_ = categories^
        self.n_features_in_ = n_cols
        self.fit_dtype = in_dtype
        self.is_fitted = True

    def fit[
        feat_dtype: DType,
        target_dtype: DType,
    ](mut self, dataset: Dataset[feat_dtype, target_dtype]) raises:
        """Learns the categories from the feature matrix of a Dataset."""
        self.fit[feat_dtype](dataset.records)

    def transform[
        in_dtype: DType
    ](self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        """Replaces each categorical value with its integer code.

        Args:
            X: Matrix of categorical features with one column per fitted feature.

        Returns:
            A matrix of the same shape holding the code of each value.

        Raises:
            InvalidParameterError: If an unseen category is found and handle_unknown is 'error'.
        """
        check_is_fitted("OrdinalEncoder", self.is_fitted)
        if in_dtype != self.fit_dtype:
            raise DataConversionError.error(
                "OrdinalEncoder.transform received Matrix["
                + String(in_dtype)
                + "] but was fitted on Matrix["
                + String(self.fit_dtype)
                + "]"
            )
        check_array[in_dtype](X)
        if X.cols != len(self.categories_):
            raise DimensionMismatchError.error(
                "X.cols == " + String(len(self.categories_)),
                "X.cols == " + String(X.cols),
                "OrdinalEncoder.transform",
            )

        var res = Matrix[in_dtype](X.rows, X.cols, 0)
        for r in range(X.rows):
            for c in range(X.cols):
                var value = Scalar[Self.compute_dtype](X[r, c])
                var idx = _index_of[Self.compute_dtype](
                    self.categories_[c], value
                )
                if idx < 0:
                    if self.handle_unknown == "error":
                        raise InvalidParameterError.error(
                            "X",
                            "OrdinalEncoder.transform found unknown category "
                            + String(value)
                            + " in column "
                            + String(c),
                        )
                    res[r, c] = Scalar[in_dtype](self.unknown_value)
                    continue
                res[r, c] = Scalar[in_dtype](idx)

        return res^
