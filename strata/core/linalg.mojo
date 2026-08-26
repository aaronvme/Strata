from .matrix import Matrix
from ..exceptions.errors import DimensionMismatchError


def gemm[
    dtype: DType = DType.float64
](A: Matrix[dtype], B: Matrix[dtype]) raises -> Matrix[dtype]:
    """Dense matrix multiplication: C = A @ B."""
    if A.cols != B.rows:
        raise DimensionMismatchError.error(
            "A.cols == B.rows",
            "A("
            + String(A.rows)
            + "x"
            + String(A.cols)
            + ") @ B("
            + String(B.rows)
            + "x"
            + String(B.cols)
            + ")",
            "gemm",
        )

    var M = A.rows
    var K = A.cols
    var N = B.cols

    var C = Matrix[dtype](M, N, 0)
    var row_acc = List[Scalar[dtype]](capacity=N)
    for _ in range(N):
        row_acc.append(0)

    for i in range(M):
        var c_offset = i * N
        for j in range(N):
            row_acc[j] = 0

        for k in range(K):
            var a_ik = A[i, k]
            if a_ik == 0:
                continue

            var b_offset = k * N
            for j in range(N):
                row_acc[j] += a_ik * B.data[b_offset + j]

        for j in range(N):
            C.data[c_offset + j] = row_acc[j]

    return C^


def dense_dot_vec[
    dtype: DType = DType.float64
](
    A: Matrix[dtype],
    x: List[Scalar[dtype]],
    bias: Scalar[dtype] = 0,
) raises -> List[Scalar[dtype]]:
    """Dense matrix-vector product: y = A @ x + bias."""
    if A.cols != len(x):
        raise DimensionMismatchError.error(
            "len(x) == " + String(A.cols),
            "len(x) == " + String(len(x)),
            "dense_dot_vec",
        )

    var res = List[Scalar[dtype]](capacity=A.rows)

    for r in range(A.rows):
        var row_offset = r * A.cols
        var sum_val: Scalar[dtype] = bias
        for c in range(A.cols):
            sum_val += A.data[row_offset + c] * x[c]
        res.append(sum_val)

    return res^
