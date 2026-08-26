from ..core.matrix import Matrix
from ..utils.random import permutation
from ..utils.validation import check_array
from ..exceptions.errors import InvalidParameterError


struct Split(Movable):
    """Pair of train and validation sample indices for a cross-validation fold."""

    var train_indices: List[Int]
    var val_indices: List[Int]

    def __init__(
        out self, var train_indices: List[Int], var val_indices: List[Int]
    ):
        self.train_indices = train_indices^
        self.val_indices = val_indices^

    def copy(self) -> Self:
        """Returns a deep copy of this split pair."""
        return Self(self.train_indices.copy(), self.val_indices.copy())


struct KFold(Movable):
    """K-Fold cross-validator.

    Provides train/test indices to split data into train/test sets.
    Splits dataset into k consecutive folds (with optional shuffling).
    """

    var n_splits: Int
    var shuffle: Bool
    var random_state: Int

    def __init__(
        out self,
        n_splits: Int = 5,
        shuffle: Bool = False,
        random_state: Int = 42,
    ) raises:
        """Initializes the KFold splitter.

        Args:
            n_splits: Number of folds (must be at least 2).
            shuffle: Whether to shuffle the data before splitting into batches.
            random_state: Random state seed when shuffle is True.
        """
        if n_splits < 2:
            raise InvalidParameterError.error(
                "n_splits",
                "n_splits must be at least 2, got " + String(n_splits),
            )
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self) -> Int:
        """Returns the number of splitting iterations in the cross-validator."""
        return self.n_splits

    def split(self, n_samples: Int) raises -> List[Split]:
        """Generates indices to split data into training and test sets.

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
        if self.n_splits > n_samples:
            raise InvalidParameterError.error(
                "n_splits",
                "Cannot have number of splits n_splits="
                + String(self.n_splits)
                + " greater than the number of samples n_samples="
                + String(n_samples),
            )

        var indices: List[Int]
        if self.shuffle:
            indices = permutation(n_samples, seed=self.random_state)
        else:
            indices = List[Int](capacity=n_samples)
            for i in range(n_samples):
                indices.append(i)

        var base_size = n_samples // self.n_splits
        var remainder = n_samples % self.n_splits

        var fold_sizes = List[Int](capacity=self.n_splits)
        for i in range(self.n_splits):
            fold_sizes.append(base_size + (1 if i < remainder else 0))

        var splits = List[Split](capacity=self.n_splits)
        var current = 0

        for i in range(self.n_splits):
            var val_size = fold_sizes[i]
            var val_indices = List[Int](capacity=val_size)
            var train_indices = List[Int](capacity=n_samples - val_size)

            for idx in range(current):
                train_indices.append(indices[idx])
            for idx in range(current, current + val_size):
                val_indices.append(indices[idx])
            for idx in range(current + val_size, n_samples):
                train_indices.append(indices[idx])

            splits.append(Split(train_indices^, val_indices^))
            current += val_size

        return splits^

    def split[dtype: DType](self, X: Matrix[dtype]) raises -> List[Split]:
        """Generates indices to split matrix records into train and test sets."""
        check_array(X)
        return self.split(X.rows)
