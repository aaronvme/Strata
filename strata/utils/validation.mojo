from std.math import isnan, isinf
from ..core.matrix import Matrix
from ..exceptions.errors import (
    NotFittedError,
    DimensionMismatchError,
    InvalidParameterError,
)


def check_is_fitted(estimator_name: String, is_fitted: Bool) raises:
    if not is_fitted:
        raise NotFittedError.error(estimator_name)


def check_floating_dtype[
    dtype: DType,
    caller: StringLiteral = "Estimator",
]():
    """Asserts at compile time that the specified dtype is a floating-point type.
    """
    comptime assert dtype.is_floating_point(), (
        caller
        + " compute_dtype must be a floating-point type (e.g. Float64 or"
        " Float32)"
    )


def check_array[
    dtype: DType
](
    X: Matrix[dtype],
    allow_empty: Bool = False,
    force_all_finite: Bool = True,
) raises:
    if not allow_empty and (X.rows == 0 or X.cols == 0):
        raise DimensionMismatchError.error(
            "non-empty 2D array",
            "array shape (" + String(X.rows) + ", " + String(X.cols) + ")",
            "check_array",
        )

    if force_all_finite:
        comptime if dtype.is_floating_point():
            var total = X.rows * X.cols
            for i in range(total):
                var val = X.data[i]
                if isnan(val) or isinf(val):
                    raise InvalidParameterError.error(
                        "Input contains NaN or infinity",
                        "check_array",
                    )


def check_X_y[
    feat_dtype: DType, target_dtype: DType
](
    X: Matrix[feat_dtype],
    y: List[Scalar[target_dtype]],
    force_all_finite: Bool = True,
) raises:
    check_array[feat_dtype](X, force_all_finite=force_all_finite)
    if X.rows != len(y):
        raise DimensionMismatchError.error(
            "len(y) == " + String(X.rows),
            "len(y) == " + String(len(y)),
            "check_X_y",
        )
    if force_all_finite:
        comptime if target_dtype.is_floating_point():
            for i in range(len(y)):
                var val = y[i]
                if isnan(val) or isinf(val):
                    raise InvalidParameterError.error(
                        "Target y contains NaN or infinity",
                        "check_X_y",
                    )


def check_consistent_length[
    T1: Copyable, T2: Copyable
](a: List[T1], b: List[T2]) raises:
    if len(a) != len(b):
        raise DimensionMismatchError.error(
            "len(b) == " + String(len(a)),
            "len(b) == " + String(len(b)),
            "check_consistent_length",
        )


def check_consistent_length[
    dtype: DType, T: Copyable
](X: Matrix[dtype], y: List[T],) raises:
    if X.rows != len(y):
        raise DimensionMismatchError.error(
            "len(y) == " + String(X.rows),
            "len(y) == " + String(len(y)),
            "check_consistent_length",
        )


def check_sparse[
    dtype: DType
](
    rows: Int,
    cols: Int,
    data: List[Scalar[dtype]],
    indices: List[Int],
    indptr: List[Int],
    is_csr: Bool = True,
    allow_empty: Bool = True,
    caller: String = "SparseMatrix.__init__",
) raises:
    """Validates CSR/CSC sparse matrix format invariants."""
    if rows < 0 or cols < 0:
        raise DimensionMismatchError.error(
            "rows >= 0 and cols >= 0",
            "rows=" + String(rows) + ", cols=" + String(cols),
            caller,
        )
    if not allow_empty and (rows == 0 or cols == 0):
        raise DimensionMismatchError.error(
            "rows > 0 and cols > 0 (allow_empty=False)",
            "rows=" + String(rows) + ", cols=" + String(cols),
            caller,
        )
    var major_dim = rows if is_csr else cols
    var minor_dim = cols if is_csr else rows

    if len(indptr) != major_dim + 1:
        raise DimensionMismatchError.error(
            "len(indptr) == " + String(major_dim + 1),
            "len(indptr) == " + String(len(indptr)),
            caller,
        )
    if len(indptr) > 0 and indptr[0] != 0:
        raise DimensionMismatchError.error(
            "indptr[0] == 0",
            "indptr[0] == " + String(indptr[0]),
            caller,
        )
    for i in range(major_dim):
        if indptr[i] > indptr[i + 1]:
            raise DimensionMismatchError.error(
                "indptr monotonic non-decreasing",
                "indptr["
                + String(i)
                + "] ("
                + String(indptr[i])
                + ") > indptr["
                + String(i + 1)
                + "] ("
                + String(indptr[i + 1])
                + ")",
                caller,
            )
    if len(data) != len(indices):
        raise DimensionMismatchError.error(
            "len(data) == len(indices)",
            "len(data)="
            + String(len(data))
            + ", len(indices)="
            + String(len(indices)),
            caller,
        )
    if len(indptr) > 0 and indptr[major_dim] != len(data):
        raise DimensionMismatchError.error(
            "indptr[-1] == len(data)",
            "indptr[-1]="
            + String(indptr[major_dim])
            + ", len(data)="
            + String(len(data)),
            caller,
        )
    for idx in range(len(indices)):
        var c = indices[idx]
        if c < 0 or c >= minor_dim:
            raise DimensionMismatchError.error(
                "index in [0, " + String(minor_dim) + ")",
                "index=" + String(c),
                caller,
            )
