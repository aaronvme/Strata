from std.algorithm import vectorize
from .matrix import Matrix
from .csr_matrix import CSRMatrix
from ..exceptions.errors import DimensionMismatchError

def simd_width_of[dtype: DType]() -> Int:
    comptime if dtype == DType.float32 or dtype == DType.int32 or dtype == DType.uint32:
        return 8
    elif dtype == DType.float64 or dtype == DType.int64 or dtype == DType.uint64:
        return 4
    elif dtype == DType.float16 or dtype == DType.bfloat16 or dtype == DType.int16 or dtype == DType.uint16:
        return 16
    elif dtype == DType.int8 or dtype == DType.uint8 or dtype == DType.bool:
        return 32
    else:
        return 4

def spmv[dtype: DType](
    A: CSRMatrix[dtype],
    x: List[Scalar[dtype]],
) raises -> List[Scalar[dtype]]:
    """Sparse matrix-vector multiplication: y = A @ x."""
    if A.cols != len(x):
        raise DimensionMismatchError.error(
            "len(x) == " + String(A.cols),
            "len(x) == " + String(len(x)),
            "spmv",
        )

    var y = List[Scalar[dtype]](capacity=A.rows)
    for r in range(A.rows):
        var start = A.indptr[r]
        var end = A.indptr[r + 1]
        var sum_val: Scalar[dtype] = 0
        for idx in range(start, end):
            var c = A.indices[idx]
            sum_val += A.data[idx] * x[c]
        y.append(sum_val)
    return y^

def spvm[dtype: DType](
    x: List[Scalar[dtype]],
    A: CSRMatrix[dtype],
) raises -> List[Scalar[dtype]]:
    """Vector-sparse matrix multiplication: y^T = x^T @ A."""
    if len(x) != A.rows:
        raise DimensionMismatchError.error(
            "len(x) == " + String(A.rows),
            "len(x) == " + String(len(x)),
            "spvm",
        )

    var y = List[Scalar[dtype]](capacity=A.cols)
    for _ in range(A.cols):
        y.append(0)

    for r in range(A.rows):
        var x_r = x[r]
        if x_r == 0:
            continue
        var start = A.indptr[r]
        var end = A.indptr[r + 1]
        for idx in range(start, end):
            var c = A.indices[idx]
            y[c] += x_r * A.data[idx]

    return y^

def spmm[dtype: DType](
    A: CSRMatrix[dtype],
    B: Matrix[dtype],
) raises -> Matrix[dtype]:
    """Sparse-dense matrix multiplication: C = A @ B."""
    if A.cols != B.rows:
        raise DimensionMismatchError.error(
            "A.cols == B.rows (" + String(A.cols) + " == " + String(B.rows) + ")",
            "A(" + String(A.rows) + "x" + String(A.cols) + ") @ B(" + String(B.rows) + "x" + String(B.cols) + ")",
            "spmm",
        )

    var M = A.rows
    var N = B.cols
    var C = Matrix[dtype](M, N, 0)
    comptime simd_w = simd_width_of[dtype]()

    for r in range(M):
        var start = A.indptr[r]
        var end = A.indptr[r + 1]
        var c_row_offset = r * N

        for idx in range(start, end):
            var k = A.indices[idx]
            var a_val = A.data[idx]
            var b_row_offset = k * N

            def _simd_add[w: Int](col_idx: Int) {imm B, mut C, imm a_val, imm b_row_offset, imm c_row_offset}:
                var b_vec = SIMD[dtype, w]()
                var c_vec = SIMD[dtype, w]()
                for lane in range(w):
                    b_vec[lane] = B.data[b_row_offset + col_idx + lane]
                    c_vec[lane] = C.data[c_row_offset + col_idx + lane]

                var a_vec = SIMD[dtype, w](a_val)
                var updated = c_vec + a_vec * b_vec
                for lane in range(w):
                    C.data[c_row_offset + col_idx + lane] = updated[lane]

            vectorize[simd_w](N, _simd_add)

    return C^

def sddmm[dtype: DType](
    A: Matrix[dtype],
    B: Matrix[dtype],
    S: CSRMatrix[dtype],
) raises -> CSRMatrix[dtype]:
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
            "A(" + String(A.rows) + "), B(" + String(B.cols) + "), S(" + String(S.rows) + "x" + String(S.cols) + ")",
            "sddmm pattern shape",
        )

    var K = A.cols
    var out_data = List[Scalar[dtype]](capacity=len(S.data))
    comptime simd_w = simd_width_of[dtype]()

    for r in range(S.rows):
        var start = S.indptr[r]
        var end = S.indptr[r + 1]
        var a_row_offset = r * K

        for idx in range(start, end):
            var c = S.indices[idx]
            var s_val = S.data[idx]
            var dot_prod: Scalar[dtype] = 0

            def _simd_dot[w: Int](k_idx: Int) {imm A, imm B, imm a_row_offset, imm c, mut dot_prod}:
                var a_vec = SIMD[dtype, w]()
                var b_vec = SIMD[dtype, w]()
                for lane in range(w):
                    a_vec[lane] = A.data[a_row_offset + k_idx + lane]
                    b_vec[lane] = B.data[(k_idx + lane) * B.cols + c]
                var prod = a_vec * b_vec
                dot_prod += prod.reduce_add()

            vectorize[simd_w](K, _simd_dot)
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

def spgemm[dtype: DType](
    A: CSRMatrix[dtype],
    B: CSRMatrix[dtype],
) raises -> CSRMatrix[dtype]:
    """Sparse-sparse matrix multiplication: C = A @ B."""
    if A.cols != B.rows:
        raise DimensionMismatchError.error(
            "A.cols == B.rows (" + String(A.cols) + " == " + String(B.rows) + ")",
            "A(" + String(A.rows) + "x" + String(A.cols) + ") @ B(" + String(B.rows) + "x" + String(B.cols) + ")",
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
