from std.algorithm import vectorize
from .matrix import Matrix
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


def gemm[
    dtype: DType
](A: Matrix[dtype], B: Matrix[dtype],) raises -> Matrix[dtype]:
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
    comptime simd_w = simd_width_of[dtype]()

    for i in range(M):
        for k in range(K):
            var a_ik = A[i, k]
            if a_ik == 0:
                continue

            var b_offset = k * N
            var c_offset = i * N

            def _simd_fma[
                w: Int
            ](col_idx: Int) {
                imm B, mut C, imm a_ik, imm b_offset, imm c_offset
            }:
                var b_vec = SIMD[dtype, w]()
                var c_vec = SIMD[dtype, w]()
                for lane in range(w):
                    b_vec[lane] = B.data[b_offset + col_idx + lane]
                    c_vec[lane] = C.data[c_offset + col_idx + lane]

                var a_val = SIMD[dtype, w](a_ik)
                var updated = c_vec + a_val * b_vec
                for lane in range(w):
                    C.data[c_offset + col_idx + lane] = updated[lane]

            vectorize[simd_w](N, _simd_fma)

    return C^


def dense_dot_vec[
    dtype: DType
](A: Matrix[dtype], x: List[Scalar[dtype]],) raises -> List[Scalar[dtype]]:
    """Dense matrix-vector product: y = A @ x."""
    if A.cols != len(x):
        raise DimensionMismatchError.error(
            "len(x) == " + String(A.cols),
            "len(x) == " + String(len(x)),
            "dense_dot_vec",
        )

    var res = List[Scalar[dtype]](capacity=A.rows)
    comptime simd_w = simd_width_of[dtype]()

    for r in range(A.rows):
        var row_offset = r * A.cols
        var sum_scalar: Scalar[dtype] = 0

        def _simd_row_dot[
            w: Int
        ](col_idx: Int) {imm A, imm x, imm row_offset, mut sum_scalar}:
            var a_vec = SIMD[dtype, w]()
            var x_vec = SIMD[dtype, w]()
            for lane in range(w):
                a_vec[lane] = A.data[row_offset + col_idx + lane]
                x_vec[lane] = x[col_idx + lane]
            var prod = a_vec * x_vec
            sum_scalar += prod.reduce_add()

        vectorize[simd_w](A.cols, _simd_row_dot)
        res.append(sum_scalar)

    return res^
