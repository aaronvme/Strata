from std.math import ceil, floor

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

    def split(self, n_samples: Int) raises -> List[Split]:
        """Generates randomly permuted train and test indices for each split.

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

        var n_test = Int(ceil(self.test_size * Float64(n_samples)))
        var n_train: Int
        if self.train_size == 0.0:
            n_train = n_samples - n_test
        else:
            n_train = Int(floor(self.train_size * Float64(n_samples)))

        if n_train + n_test > n_samples:
            raise InvalidParameterError.error(
                "train_size",
                "The sum of train_size and test_size resolves to "
                + String(n_train + n_test)
                + " samples, which exceeds n_samples="
                + String(n_samples),
            )
        if n_train == 0:
            raise InvalidParameterError.error(
                "train_size",
                "With n_samples="
                + String(n_samples)
                + ", test_size="
                + String(self.test_size)
                + " and train_size="
                + String(self.train_size)
                + ", the resulting train set is empty",
            )

        var splits = List[Split](capacity=self.n_splits)

        for s in range(self.n_splits):
            var indices = permutation(n_samples, seed=self.random_state + s)

            var val_indices = List[Int](capacity=n_test)
            for i in range(n_test):
                val_indices.append(indices[i])

            var train_indices = List[Int](capacity=n_train)
            for i in range(n_test, n_test + n_train):
                train_indices.append(indices[i])

            splits.append(Split(train_indices^, val_indices^))

        return splits^
