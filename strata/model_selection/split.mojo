from ..core.matrix import Matrix
from ..core.dataset import Dataset, DatasetSplit
from ..utils.validation import check_consistent_length, check_array
from ..utils.random import permutation
from ..exceptions.errors import InvalidParameterError


def train_test_split[
    feat_dtype: DType = DType.float64,
    target_dtype: DType = DType.float64,
](
    dataset: Dataset[feat_dtype, target_dtype],
    test_size: Float64 = 0.25,
    shuffle: Bool = True,
    seed: Int = 42,
) raises -> DatasetSplit[feat_dtype, target_dtype]:
    """Split a Dataset container into random train and test partitions.

    Args:
        dataset: Input dataset pairing features and target values.
        test_size: Proportion of the dataset to include in the test split ($0 < \\text{test\\_size} < 1$). Default 0.25.
        shuffle: Whether to shuffle the data before splitting. Default True.

        seed: PRNG seed for reproducible pseudo-random shuffling. Default 42.

    Returns:
        DatasetSplit: Container holding `train` and `test` Dataset partitions.

    Raises:
        InvalidParameterError: If test_size is not strictly between 0.0 and 1.0.
    """
    return dataset.split_with_ratio(test_size, shuffle=shuffle, seed=seed)


def train_test_split[
    dtype: DType = DType.float64
](
    X: Matrix[dtype],
    y: List[Scalar[dtype]],
    test_size: Float64 = 0.25,
    shuffle: Bool = True,
    seed: Int = 42,
) raises -> DatasetSplit[dtype, dtype]:
    """Split feature matrix and target list into train and test partitions.

    Args:
        X: Feature matrix of shape $(N, D)$.
        y: Target label vector of length $N$.
        test_size: Proportion of the dataset to include in the test split. Default 0.25.
        shuffle: Whether to shuffle the data before splitting. Default True.
        seed: PRNG seed for reproducible shuffling. Default 42.

    Returns:
        DatasetSplit: Container holding partitioned training and testing datasets.

    Raises:
        InvalidParameterError: If test_size is not strictly between 0.0 and 1.0.
        DimensionMismatchError: If `X.rows != len(y)`.
    """
    var ds = Dataset[dtype, dtype](X.copy(), y.copy())
    return ds.split_with_ratio(test_size, shuffle=shuffle, seed=seed)
