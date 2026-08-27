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
