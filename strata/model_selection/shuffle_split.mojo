from ..core.matrix import Matrix
from ..utils.random import permutation
from ..utils.validation import check_array
from ..exceptions.errors import InvalidParameterError
from .kfold import Split


struct ShuffleSplit(Movable):
    """Random permutation cross-validator.

    Yields indices to split data into training and test sets. Each split is an
    independent random permutation of the samples, so successive test sets may
    overlap. Sizes are expressed as proportions of the total sample count.
    """

    var n_splits: Int
    var test_size: Float64
    var train_size: Float64
    var random_state: Int

    def __init__(
        out self,
        n_splits: Int = 10,
        test_size: Float64 = 0.1,
        train_size: Float64 = 0.0,
        random_state: Int = 42,
    ) raises:
        """Initializes the ShuffleSplit cross-validator.

        Args:
            n_splits: Number of re-shuffling and splitting iterations.
            test_size: Proportion of samples held out for testing, in (0, 1).
            train_size: Proportion of samples used for training, in (0, 1), or
                0.0 to use the complement of test_size.
            random_state: Base seed controlling the permutation of each split.
        """
        if n_splits < 1:
            raise InvalidParameterError.error(
                "n_splits",
                "n_splits must be at least 1, got " + String(n_splits),
            )
        if test_size <= 0.0 or test_size >= 1.0:
            raise InvalidParameterError.error(
                "test_size",
                "test_size must be strictly between 0 and 1, got "
                + String(test_size),
            )
        if train_size < 0.0 or train_size >= 1.0:
            raise InvalidParameterError.error(
                "train_size",
                "train_size must be 0.0 or strictly between 0 and 1, got "
                + String(train_size),
            )
        if train_size > 0.0 and test_size + train_size > 1.0:
            raise InvalidParameterError.error(
                "train_size",
                "test_size + train_size must not exceed 1, got "
                + String(test_size + train_size),
            )
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
