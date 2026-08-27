from std.math import ceil, floor

from ..core.matrix import Matrix
from ..utils.random import permutation
from ..utils.validation import check_X_y
from ..exceptions.errors import InvalidParameterError
from .kfold import Split


def _allocate_quota(counts: List[Int], n_draws: Int) -> List[Int]:
    """Distributes n_draws across classes in proportion to their counts.

    Args:
        counts: Number of available samples in each class.
        n_draws: Total number of samples to allocate.

    Returns:
        Per-class allocation summing to n_draws, with each entry bounded by the
        matching entry of counts.
    """
    var num_classes = len(counts)
    var total = 0
    for c in range(num_classes):
        total += counts[c]

    var quota = List[Int](capacity=num_classes)
    var remainders = List[Float64](capacity=num_classes)
    var assigned = 0

    for c in range(num_classes):
        var exact = Float64(counts[c]) * Float64(n_draws) / Float64(total)
        var base = Int(floor(exact))
        quota.append(base)
        remainders.append(exact - Float64(base))
        assigned += base

    var need = n_draws - assigned
    while need > 0:
        var best = -1
        for c in range(num_classes):
            if quota[c] >= counts[c]:
                continue
            if best == -1 or remainders[c] > remainders[best]:
                best = c
        if best == -1:
            break
        quota[best] += 1
        remainders[best] = -1.0
        need -= 1

    return quota^


struct StratifiedShuffleSplit(Movable):
    """Stratified random permutation cross-validator.

    Yields indices to split data into training and test sets. Each split is an
    independent random draw that preserves the percentage of samples for each
    target class, so successive test sets may overlap.
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
        """Initializes the StratifiedShuffleSplit cross-validator.

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

    def get_n_splits(self) -> Int:
        """Returns the number of splitting iterations in the cross-validator."""
        return self.n_splits

    def split[
        target_dtype: DType
    ](self, y: List[Scalar[target_dtype]]) raises -> List[Split]:
        """Generates class-balanced random train and test indices for each split.

        Args:
            y: Target class labels of length N.

        Returns:
            List of Split objects containing train and validation indices.
        """
        var n_samples = len(y)
        if n_samples == 0:
            raise InvalidParameterError.error(
                "y", "Target list y cannot be empty"
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

        var raw_classes = List[Scalar[target_dtype]]()
        var class_indices = List[List[Int]]()

        for i in range(n_samples):
            var label = y[i]
            var found_idx = -1
            for c in range(len(raw_classes)):
                if raw_classes[c] == label:
                    found_idx = c
                    break
            if found_idx == -1:
                raw_classes.append(label)
                var new_list = List[Int]()
                new_list.append(i)
                class_indices.append(new_list^)
            else:
                class_indices[found_idx].append(i)

        var num_classes = len(raw_classes)

        if n_test < num_classes:
            raise InvalidParameterError.error(
                "test_size",
                "The resolved test set size "
                + String(n_test)
                + " is smaller than the number of classes "
                + String(num_classes),
            )
        if n_train < num_classes:
            raise InvalidParameterError.error(
                "train_size",
                "The resolved train set size "
                + String(n_train)
                + " is smaller than the number of classes "
                + String(num_classes),
            )

        var class_counts = List[Int](capacity=num_classes)
        for c in range(num_classes):
            class_counts.append(len(class_indices[c]))

        var train_quota = _allocate_quota(class_counts, n_train)

        var remaining = List[Int](capacity=num_classes)
        for c in range(num_classes):
            remaining.append(class_counts[c] - train_quota[c])

        var test_quota = _allocate_quota(remaining, n_test)

        var splits = List[Split](capacity=self.n_splits)

        for s in range(self.n_splits):
            var train_indices = List[Int](capacity=n_train)
            var val_indices = List[Int](capacity=n_test)

            for c in range(num_classes):
                var order = permutation(
                    class_counts[c],
                    seed=self.random_state + s * num_classes + c,
                )
                var n_c_train = train_quota[c]
                var n_c_test = test_quota[c]
                for i in range(n_c_train):
                    train_indices.append(class_indices[c][order[i]])
                for i in range(n_c_train, n_c_train + n_c_test):
                    val_indices.append(class_indices[c][order[i]])

            sort(train_indices)
            sort(val_indices)

            splits.append(Split(train_indices^, val_indices^))

        return splits^
