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
    """Splits a Dataset into train and test subsets."""
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
    """Splits matrix records and target lists into a DatasetSplit."""
    var ds = Dataset[dtype, dtype](X.copy(), y.copy())
    return ds.split_with_ratio(test_size, shuffle=shuffle, seed=seed)
