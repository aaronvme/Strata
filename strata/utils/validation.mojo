from ..core.matrix import Matrix
from ..exceptions.errors import NotFittedError, DimensionMismatchError, InvalidParameterError

def check_is_fitted(estimator_name: String, is_fitted: Bool) raises:
    if not is_fitted:
        raise NotFittedError.error(estimator_name)

def check_array[dtype: DType](X: Matrix[dtype], allow_empty: Bool = False) raises:
    if not allow_empty and (X.rows == 0 or X.cols == 0):
        raise DimensionMismatchError.error(
            "non-empty 2D array",
            "array shape (" + String(X.rows) + ", " + String(X.cols) + ")",
            "check_array",
        )

def check_X_y[feat_dtype: DType, target_dtype: DType](
    X: Matrix[feat_dtype],
    y: List[Scalar[target_dtype]],
) raises:
    check_array[feat_dtype](X)
    if X.rows != len(y):
        raise DimensionMismatchError.error(
            "len(y) == " + String(X.rows),
            "len(y) == " + String(len(y)),
            "check_X_y",
        )

def check_consistent_length[T1: Copyable, T2: Copyable](a: List[T1], b: List[T2]) raises:
    if len(a) != len(b):
        raise DimensionMismatchError.error(
            "len(b) == " + String(len(a)),
            "len(b) == " + String(len(b)),
            "check_consistent_length",
        )

def check_consistent_length[dtype: DType, T: Copyable](
    X: Matrix[dtype],
    y: List[T],
) raises:
    if X.rows != len(y):
        raise DimensionMismatchError.error(
            "len(y) == " + String(X.rows),
            "len(y) == " + String(len(y)),
            "check_consistent_length",
        )
