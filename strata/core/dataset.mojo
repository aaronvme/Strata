from ..core.matrix import Matrix
from ..utils.validation import check_consistent_length, check_array
from ..utils.random import permutation
from ..exceptions.errors import InvalidParameterError, DimensionMismatchError


struct DatasetSplit[
    feat_dtype: DType = DType.float64,
    target_dtype: DType = DType.float64,
](Movable):
    """Container holding train and test partitions of a Dataset."""

    var train: Dataset[Self.feat_dtype, Self.target_dtype]
    var test: Dataset[Self.feat_dtype, Self.target_dtype]

    def __init__(
        out self,
        var train: Dataset[Self.feat_dtype, Self.target_dtype],
        var test: Dataset[Self.feat_dtype, Self.target_dtype],
    ):
        """Initialize a DatasetSplit container.

        Args:
            train: Training partition Dataset.
            test: Testing partition Dataset.
        """
        self.train = train^
        self.test = test^


struct Dataset[
    feat_dtype: DType = DType.float64,
    target_dtype: DType = DType.float64,
](Copyable, Movable):
    """Machine learning dataset container pairing a feature matrix with targets.

    Encapsulates 2D feature observations, 1D target labels/values, and optional
    feature/target name metadata.

    Parameters:
        feat_dtype: Data type of the feature matrix (default: Float64).
        target_dtype: Data type of the target values (default: Float64).

    Attributes:
        records: Feature matrix of shape $(N, D)$.
        targets: Target label/value vector of length $N$.
        feature_names: List of feature names of length $D$.
        target_names: List of class or target variable names.

    Examples:
        ```mojo
        from strata.core import Matrix, Dataset

        var X = Matrix[DType.float64](100, 4, fill=1.0)
        var y = List[Scalar[DType.float64]](capacity=100)
        var ds = Dataset(X^, y^)
        var split = ds.split_with_ratio(ratio=0.2)
        ```
    """

    var records: Matrix[Self.feat_dtype]
    var targets: List[Scalar[Self.target_dtype]]
    var feature_names: List[String]
    var target_names: List[String]

    def __init__(
        out self,
        var records: Matrix[Self.feat_dtype],
        var targets: List[Scalar[Self.target_dtype]],
    ) raises:
        """Initialize a Dataset from records and targets.

        Args:
            records: Feature matrix of shape $(N, D)$.
            targets: Target vector of length $N$.

        Raises:
            DimensionMismatchError: If records.rows != len(targets).
        """
        check_consistent_length(records, targets)
        check_array[Self.feat_dtype](records)
        self.records = records^
        self.targets = targets^
        self.feature_names = List[String]()
        self.target_names = List[String]()

    def __init__(
        out self,
        var records: Matrix[Self.feat_dtype],
        var targets: List[Scalar[Self.target_dtype]],
        var feature_names: List[String],
        var target_names: List[String],
    ) raises:
        check_consistent_length(records, targets)
        check_array[Self.feat_dtype](records)
        if len(feature_names) > 0 and len(feature_names) != records.cols:
            raise DimensionMismatchError.error(
                "len(feature_names) == records.cols ("
                + String(records.cols)
                + ")",
                "len(feature_names) == " + String(len(feature_names)),
                "Dataset.__init__",
            )
        self.records = records^
        self.targets = targets^
        self.feature_names = feature_names^
        self.target_names = target_names^

    def __init__(out self, *, copy: Self):
        self.records = copy.records.copy()
        self.targets = copy.targets.copy()
        self.feature_names = copy.feature_names.copy()
        self.target_names = copy.target_names.copy()

    def __init__(out self, *, deinit move: Self):
        self.records = move.records^
        self.targets = move.targets^
        self.feature_names = move.feature_names^
        self.target_names = move.target_names^

    def n_samples(self) -> Int:
        return self.records.rows

    def n_features(self) -> Int:
        return self.records.cols

    def split_with_ratio(
        self,
        ratio: Float64 = 0.25,
        shuffle: Bool = True,
        seed: Int = 42,
    ) raises -> DatasetSplit[Self.feat_dtype, Self.target_dtype]:
        if ratio <= 0.0 or ratio >= 1.0:
            raise InvalidParameterError.error(
                "ratio",
                "must be between 0.0 and 1.0 (got " + String(ratio) + ")",
            )

        var total = self.n_samples()
        var n_test = Int(Float64(total) * ratio)
        if total >= 2 and n_test == 0:
            n_test = 1
        var n_train = total - n_test

        var indices: List[Int]
        if shuffle:
            indices = permutation(total, seed=seed)
        else:
            indices = List[Int](range(total))

        var train_records = Matrix[Self.feat_dtype](
            n_train, self.n_features(), 0
        )
        var train_targets = List[Scalar[Self.target_dtype]](capacity=n_train)
        for i in range(n_train):
            var r = indices[i]
            for c in range(self.n_features()):
                train_records[i, c] = self.records[r, c]
            train_targets.append(self.targets[r])

        var test_records = Matrix[Self.feat_dtype](n_test, self.n_features(), 0)
        var test_targets = List[Scalar[Self.target_dtype]](capacity=n_test)
        for i in range(n_test):
            var r = indices[n_train + i]
            for c in range(self.n_features()):
                test_records[i, c] = self.records[r, c]
            test_targets.append(self.targets[r])

        var train_fn = self.feature_names.copy()
        var train_tn = self.target_names.copy()
        var train_ds = Dataset[Self.feat_dtype, Self.target_dtype](
            train_records^,
            train_targets^,
            train_fn^,
            train_tn^,
        )

        var test_fn = self.feature_names.copy()
        var test_tn = self.target_names.copy()
        var test_ds = Dataset[Self.feat_dtype, Self.target_dtype](
            test_records^,
            test_targets^,
            test_fn^,
            test_tn^,
        )

        return DatasetSplit[Self.feat_dtype, Self.target_dtype](
            train=train_ds^,
            test=test_ds^,
        )
