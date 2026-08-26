from ..core.matrix import Matrix
from ..utils.random import shuffle
from ..exceptions.errors import InvalidParameterError
from .kfold import Split


struct StratifiedKFold(Movable):
    """Stratified K-Fold cross-validator for classification datasets.

    Splits dataset into k folds such that each fold preserves approximately
    the same percentage of samples for each target class.
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
        """Initializes the StratifiedKFold cross-validator.

        Args:
            n_splits: Number of folds (must be at least 2).
            shuffle: Whether to shuffle each class's samples before splitting.
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
        """Returns the number of splitting iterations."""
        return self.n_splits

    def split[
        target_dtype: DType
    ](self, y: List[Scalar[target_dtype]]) raises -> List[Split]:
        """Generates stratified train and test indices from target labels.

        Args:
            y: Target class labels of length N.

        Returns:
            List of Split instances containing train and validation indices.
        """
        var n_samples = len(y)
        if self.n_splits > n_samples:
            raise InvalidParameterError.error(
                "n_splits",
                "Cannot have number of splits n_splits="
                + String(self.n_splits)
                + " greater than the number of samples n_samples="
                + String(n_samples),
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

        if self.shuffle:
            for c in range(num_classes):
                shuffle(class_indices[c], seed=self.random_state + c)

        var fold_val_indices = List[List[Int]]()
        for _ in range(self.n_splits):
            fold_val_indices.append(List[Int]())

        for c in range(num_classes):
            var cls_n = len(class_indices[c])
            var base_size = cls_n // self.n_splits
            var remainder = cls_n % self.n_splits
            var cur = 0
            for f in range(self.n_splits):
                var count = base_size + (1 if f < remainder else 0)
                for idx in range(cur, cur + count):
                    fold_val_indices[f].append(class_indices[c][idx])
                cur += count

        var splits = List[Split](capacity=self.n_splits)

        for f in range(self.n_splits):
            sort(fold_val_indices[f])

            var val_size = len(fold_val_indices[f])
            var train_indices = List[Int](capacity=n_samples - val_size)

            var is_val = List[Bool](capacity=n_samples)
            for _ in range(n_samples):
                is_val.append(False)
            for i in range(val_size):
                is_val[fold_val_indices[f][i]] = True

            for i in range(n_samples):
                if not is_val[i]:
                    train_indices.append(i)

            splits.append(Split(train_indices^, fold_val_indices[f].copy()))

        return splits^

    def split[
        feat_dtype: DType, target_dtype: DType
    ](
        self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]]
    ) raises -> List[Split]:
        """Generates stratified train and test indices from feature matrix X and targets y."""
        if X.rows != len(y):
            raise InvalidParameterError.error(
                "X.rows",
                "X.rows ("
                + String(X.rows)
                + ") must match len(y) ("
                + String(len(y))
                + ")",
            )
        return self.split[target_dtype](y)
