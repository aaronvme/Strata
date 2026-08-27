from ..core.matrix import Matrix
from ..utils.validation import check_array
from ..exceptions.errors import InvalidParameterError
from .kfold import Split


struct TimeSeriesSplit(Movable):
    """Time Series cross-validator.

    Provides train/test indices to split time series data samples that are
    observed at fixed time intervals. In each split, test indices must be
    higher than before, so shuffling in cross-validator is inappropriate.
    Successive training sets are supersets of those that come before them.
    """

    var n_splits: Int
    var max_train_size: Int
    var test_size: Int
    var gap: Int

    def __init__(
        out self,
        n_splits: Int = 5,
        max_train_size: Int = 0,
        test_size: Int = 0,
        gap: Int = 0,
    ) raises:
        """Initializes the TimeSeriesSplit cross-validator.

        Args:
            n_splits: Number of splits (must be at least 2).
            max_train_size: Maximum size for a single training set, or 0 for
                an expanding window with no upper bound.
            test_size: Size used for each test set, or 0 to default to
                n_samples // (n_splits + 1).
            gap: Number of samples to exclude between the end of each training
                set and the start of the following test set.
        """
        if n_splits < 2:
            raise InvalidParameterError.error(
                "n_splits",
                "n_splits must be at least 2, got " + String(n_splits),
            )
        if max_train_size < 0:
            raise InvalidParameterError.error(
                "max_train_size",
                "max_train_size must be non-negative, got "
                + String(max_train_size),
            )
        if test_size < 0:
            raise InvalidParameterError.error(
                "test_size",
                "test_size must be non-negative, got " + String(test_size),
            )
        if gap < 0:
            raise InvalidParameterError.error(
                "gap",
                "gap must be non-negative, got " + String(gap),
            )
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    def get_n_splits(self) -> Int:
        """Returns the number of splitting iterations in the cross-validator."""
        return self.n_splits

    def split(self, n_samples: Int) raises -> List[Split]:
        """Generates indices to split time-ordered data into train and test sets.

        Args:
            n_samples: Total number of samples in the dataset.

        Returns:
            List of Split objects containing train and validation indices.
        """
        if n_samples <= 0:
            raise InvalidParameterError.error(
                "n_samples",
                "n_samples must be strictly positive, got " + String(n_samples),
            )

        var n_folds = self.n_splits + 1
        if n_folds > n_samples:
            raise InvalidParameterError.error(
                "n_splits",
                "Cannot have number of folds n_folds="
                + String(n_folds)
                + " greater than the number of samples n_samples="
                + String(n_samples),
            )

        var fold_test_size = self.test_size
        if fold_test_size == 0:
            fold_test_size = n_samples // n_folds

        if n_samples - self.gap - fold_test_size * self.n_splits <= 0:
            raise InvalidParameterError.error(
                "n_splits",
                "Too many splits n_splits="
                + String(self.n_splits)
                + " for number of samples n_samples="
                + String(n_samples)
                + " with test_size="
                + String(fold_test_size)
                + " and gap="
                + String(self.gap),
            )

        var first_test_start = n_samples - self.n_splits * fold_test_size
        var splits = List[Split](capacity=self.n_splits)

        for f in range(self.n_splits):
            var test_start = first_test_start + f * fold_test_size
            var train_end = test_start - self.gap
            var train_start = 0
            if self.max_train_size > 0 and self.max_train_size < train_end:
                train_start = train_end - self.max_train_size

            var train_indices = List[Int](capacity=train_end - train_start)
            for i in range(train_start, train_end):
                train_indices.append(i)

            var val_indices = List[Int](capacity=fold_test_size)
            for i in range(test_start, test_start + fold_test_size):
                val_indices.append(i)

            splits.append(Split(train_indices^, val_indices^))

        return splits^

    def split[dtype: DType](self, X: Matrix[dtype]) raises -> List[Split]:
        """Generates indices to split matrix records into train and test sets.
        """
        check_array(X)
        return self.split(X.rows)
