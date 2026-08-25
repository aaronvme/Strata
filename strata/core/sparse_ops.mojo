from .matrix import Matrix
from .csr_matrix import CSRMatrix
from ..exceptions.errors import DimensionMismatchError


def spmv[
    mat_dtype: DType,
    vec_dtype: DType,
    out_dtype: DType = vec_dtype,
](
    A: CSRMatrix[mat_dtype],
    x: List[Scalar[vec_dtype]],
    bias: Scalar[vec_dtype] = 0,
) raises -> List[Scalar[out_dtype]]:
    """Sparse matrix-vector multiplication with mixed precision: y = A @ x + bias.
    """
    if A.cols != len(x):
        raise DimensionMismatchError.error(
            "len(x) == " + String(A.cols),
            "len(x) == " + String(len(x)),
            "spmv",
        )

    var y = List[Scalar[out_dtype]](capacity=A.rows)
    var b_val = Float64(bias)
    for r in range(A.rows):
        var start = A.indptr[r]
        var end = A.indptr[r + 1]
        var sum_val: Float64 = b_val
        for idx in range(start, end):
            var c = A.indices[idx]
            sum_val += Float64(A.data[idx]) * Float64(x[c])
        y.append(Scalar[out_dtype](sum_val))
    return y^


def spvm[
    mat_dtype: DType,
    vec_dtype: DType,
    out_dtype: DType = vec_dtype,
](x: List[Scalar[vec_dtype]], A: CSRMatrix[mat_dtype]) raises -> List[
    Scalar[out_dtype]
]:
    """Vector-sparse matrix multiplication: y^T = x^T @ A."""
    if len(x) != A.rows:
        raise DimensionMismatchError.error(
            "len(x) == " + String(A.rows),
            "len(x) == " + String(len(x)),
            "spvm",
        )

    var y = List[Scalar[out_dtype]](capacity=A.cols)
    for _ in range(A.cols):
        y.append(0)

    for r in range(A.rows):
        var x_r = Float64(x[r])
        if x_r == 0:
            continue

        var start = A.indptr[r]
        var end = A.indptr[r + 1]
        for idx in range(start, end):
            var c = A.indices[idx]
            var prod = x_r * Float64(A.data[idx])
            y[c] = Scalar[out_dtype](Float64(y[c]) + prod)

    return y^


def spmm[
    mat_dtype: DType,
    dense_dtype: DType,
    out_dtype: DType = dense_dtype,
](A: CSRMatrix[mat_dtype], B: Matrix[dense_dtype]) raises -> Matrix[out_dtype]:
    """Sparse-Dense matrix multiplication with mixed precision: C = A @ B."""
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
            "spmm",
        )

    var M = A.rows
    var N = B.cols
    var C = Matrix[out_dtype](M, N, 0)
    var row_acc = List[Float64](capacity=N)
    for _ in range(N):
        row_acc.append(0.0)

    for i in range(M):
        var start = A.indptr[i]
        var end = A.indptr[i + 1]
        var c_offset = i * N
        for j in range(N):
            row_acc[j] = 0.0

        for idx in range(start, end):
            var k = A.indices[idx]
            var a_val = Float64(A.data[idx])
            var b_offset = k * N

            for j in range(N):
                row_acc[j] += a_val * Float64(B.data[b_offset + j])

        for j in range(N):
            C.data[c_offset + j] = Scalar[out_dtype](row_acc[j])

    return C^


def sddmm[
    dtype: DType
](
    A: Matrix[dtype],
    B: Matrix[dtype],
    S: CSRMatrix[dtype],
) raises -> CSRMatrix[
    dtype
]:
    """Sampled Dense-Dense Matrix Multiplication: P = (A @ B) ⊙ S."""
    if A.cols != B.rows:
        raise DimensionMismatchError.error(
            "A.cols == B.rows",
            "A.cols=" + String(A.cols) + ", B.rows=" + String(B.rows),
            "sddmm inner dimension",
        )
    if A.rows != S.rows or B.cols != S.cols:
        raise DimensionMismatchError.error(
            "Product shape (A.rows, B.cols) == S shape",
            "A("
            + String(A.rows)
            + "), B("
            + String(B.cols)
            + "), S("
            + String(S.rows)
            + "x"
            + String(S.cols)
            + ")",
            "sddmm pattern shape",
        )

    var K = A.cols
    var out_data = List[Scalar[dtype]](capacity=len(S.data))

    for r in range(S.rows):
        var start = S.indptr[r]
        var end = S.indptr[r + 1]
        var a_row_offset = r * K

        for idx in range(start, end):
            var c = S.indices[idx]
            var s_val = S.data[idx]
            var dot_prod: Scalar[dtype] = 0

            for k in range(K):
                dot_prod += A.data[a_row_offset + k] * B.data[k * B.cols + c]

            out_data.append(dot_prod * s_val)

    var indptr_copy = S.indptr.copy()
    var indices_copy = S.indices.copy()

    return CSRMatrix[dtype](
        S.rows,
        S.cols,
        out_data^,
        indices_copy^,
        indptr_copy^,
    )


def spgemm[
    dtype: DType
](A: CSRMatrix[dtype], B: CSRMatrix[dtype],) raises -> CSRMatrix[dtype]:
    """Sparse-sparse matrix multiplication: C = A @ B."""
    if A.cols != B.rows:
        raise DimensionMismatchError.error(
            "A.cols == B.rows ("
            + String(A.cols)
            + " == "
            + String(B.rows)
            + ")",
            "A("
            + String(A.rows)
            + "x"
            + String(A.cols)
            + ") @ B("
            + String(B.rows)
            + "x"
            + String(B.cols)
            + ")",
            "spgemm",
        )

    var M = A.rows
    var N = B.cols

    var c_data = List[Scalar[dtype]]()
    var c_indices = List[Int]()
    var c_indptr = List[Int](capacity=M + 1)
    c_indptr.append(0)

    var accumulator = List[Scalar[dtype]](capacity=N)
    var marker = List[Int](capacity=N)
    for _ in range(N):
        accumulator.append(0)
        marker.append(-1)

    for i in range(M):
        var a_start = A.indptr[i]
        var a_end = A.indptr[i + 1]
        var row_nonzero_cols = List[Int]()

        for a_idx in range(a_start, a_end):
            var k = A.indices[a_idx]
            var a_val = A.data[a_idx]

            var b_start = B.indptr[k]
            var b_end = B.indptr[k + 1]

            for b_idx in range(b_start, b_end):
                var j = B.indices[b_idx]
                var b_val = B.data[b_idx]

                if marker[j] != i:
                    marker[j] = i
                    row_nonzero_cols.append(j)
                    accumulator[j] = a_val * b_val
                else:
                    accumulator[j] += a_val * b_val

        for idx in range(len(row_nonzero_cols)):
            var j = row_nonzero_cols[idx]
            var val = accumulator[j]
            if val != 0:
                c_data.append(val)
                c_indices.append(j)
            accumulator[j] = 0

        c_indptr.append(len(c_data))

    return CSRMatrix[dtype](M, N, c_data^, c_indices^, c_indptr^)
