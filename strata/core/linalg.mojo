from .matrix import Matrix
from ..exceptions.errors import DimensionMismatchError


def gemm[
    a_dtype: DType,
    b_dtype: DType,
    out_dtype: DType = a_dtype,
](A: Matrix[a_dtype], B: Matrix[b_dtype]) raises -> Matrix[out_dtype]:
    """Dense matrix multiplication with mixed precision: C = A @ B."""
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

    var C = Matrix[out_dtype](M, N, 0)

    for i in range(M):
        var c_offset = i * N
        var row_acc = List[Float64](capacity=N)
        for _ in range(N):
            row_acc.append(0.0)

        for k in range(K):
            var a_ik = Float64(A[i, k])
            if a_ik == 0:
                continue

            var b_offset = k * N
            for j in range(N):
                row_acc[j] += a_ik * Float64(B.data[b_offset + j])

        for j in range(N):
            C.data[c_offset + j] = Scalar[out_dtype](row_acc[j])

    return C^


def dense_dot_vec[
    mat_dtype: DType,
    vec_dtype: DType,
    out_dtype: DType = vec_dtype,
](
    A: Matrix[mat_dtype],
    x: List[Scalar[vec_dtype]],
    bias: Scalar[vec_dtype] = 0,
) raises -> List[Scalar[out_dtype]]:
    """Dense matrix-vector product with mixed precision: y = A @ x + bias."""
    if A.cols != len(x):
        raise DimensionMismatchError.error(
            "len(x) == " + String(A.cols),
            "len(x) == " + String(len(x)),
            "dense_dot_vec",
        )

    var res = List[Scalar[out_dtype]](capacity=A.rows)
    var b_val = Float64(bias)

    for r in range(A.rows):
        var row_offset = r * A.cols
        var sum_val: Float64 = b_val
        for c in range(A.cols):
            sum_val += Float64(A.data[row_offset + c]) * Float64(x[c])
        res.append(Scalar[out_dtype](sum_val))

    return res^
