from ..core.matrix import Matrix
from ..exceptions.errors import InvalidParameterError


def take_rows[
    dtype: DType
](X: Matrix[dtype], indices: List[Int]) raises -> Matrix[dtype]:
    """Constructs a new Matrix containing the specified row indices from X.

    Args:
        X: Source matrix.
        indices: List of row indices to extract.

    Returns:
        New Matrix containing the selected rows in index order.
    """
    var num_rows = len(indices)
    var num_cols = X.cols
    var out_mat = Matrix[dtype](num_rows, num_cols, 0)
    for r in range(num_rows):
        var src_r = indices[r]
        if src_r < 0 or src_r >= X.rows:
            raise InvalidParameterError.error(
                "indices",
                "Index "
                + String(src_r)
                + " out of bounds for matrix with "
                + String(X.rows)
                + " rows",
            )
        for c in range(num_cols):
            out_mat[r, c] = X[src_r, c]
    return out_mat^


def take_elements[
    dtype: DType
](y: List[Scalar[dtype]], indices: List[Int]) raises -> List[Scalar[dtype]]:
    """Constructs a new List containing the specified element indices from y.

    Args:
        y: Source list.
        indices: List of element indices to extract.

    Returns:
        New List containing the selected elements in index order.
    """
    var out_list = List[Scalar[dtype]](capacity=len(indices))
    for i in range(len(indices)):
        var idx = indices[i]
        if idx < 0 or idx >= len(y):
            raise InvalidParameterError.error(
                "indices",
                "Index "
                + String(idx)
                + " out of bounds for list with "
                + String(len(y))
                + " elements",
            )
        out_list.append(y[idx])
    return out_list^
