from std.ffi import external_call, c_int, c_char, c_double, c_float
from std.math import sqrt
from .matrix import Matrix
from ..exceptions.errors import DimensionMismatchError, InvalidParameterError


struct SVDResult[dtype: DType = DType.float64](Copyable, Movable):
    """Result of Singular Value Decomposition: A = U * diag(S) * Vt."""

    var U: Matrix[Self.dtype]
    var S: List[Scalar[Self.dtype]]
    var Vt: Matrix[Self.dtype]

    def __init__(
        out self,
        var U: Matrix[Self.dtype],
        var S: List[Scalar[Self.dtype]],
        var Vt: Matrix[Self.dtype],
    ):
        self.U = U^
        self.S = S^
        self.Vt = Vt^


struct QRResult[dtype: DType = DType.float64](Copyable, Movable):
    """Result of QR Decomposition: A = Q * R."""

    var Q: Matrix[Self.dtype]
    var R: Matrix[Self.dtype]

    def __init__(
        out self, var Q: Matrix[Self.dtype], var R: Matrix[Self.dtype]
    ):
        self.Q = Q^
        self.R = R^


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


def svd[
    dtype: DType = DType.float64
](A: Matrix[dtype], full_matrices: Bool = False) raises -> SVDResult[dtype]:
    """Computes the Singular Value Decomposition (SVD) of a matrix A = U * S * Vt.

    Uses LAPACK's divide-and-conquer algorithm (dgesdd/sgesdd).
    """
    comptime assert (
        dtype.is_floating_point()
    ), "Floating-point type required for SVD decomposition"

    var M = A.rows
    var N = A.cols
    var K = min(M, N)
    var u_cols = M if full_matrices else K
    var vt_rows = N if full_matrices else K
    var jobz: Int8 = Int8(ord("A")) if full_matrices else Int8(ord("S"))

    var A_copy = A.copy()
    var S = List[Scalar[dtype]](capacity=K)
    for _ in range(K):
        S.append(0)

    var U = Matrix[dtype].zeros(M, u_cols)
    var Vt = Matrix[dtype].zeros(vt_rows, N)

    var info: c_int = 0

    comptime if dtype == DType.float64:
        info = external_call[
            "LAPACKE_dgesdd",
            c_int,
            c_int,
            c_char,
            c_int,
            c_int,
            Pointer[Scalar[dtype], origin_of(A_copy.data)],
            c_int,
            Pointer[Scalar[dtype], origin_of(S)],
            Pointer[Scalar[dtype], origin_of(U.data)],
            c_int,
            Pointer[Scalar[dtype], origin_of(Vt.data)],
            c_int,
        ](
            c_int(101),
            jobz,
            c_int(M),
            c_int(N),
            A_copy.data.unsafe_ptr(),
            c_int(N),
            S.unsafe_ptr(),
            U.data.unsafe_ptr(),
            c_int(u_cols),
            Vt.data.unsafe_ptr(),
            c_int(N),
        )
    elif dtype == DType.float32:
        info = external_call[
            "LAPACKE_sgesdd",
            c_int,
            c_int,
            c_char,
            c_int,
            c_int,
            Pointer[Scalar[dtype], origin_of(A_copy.data)],
            c_int,
            Pointer[Scalar[dtype], origin_of(S)],
            Pointer[Scalar[dtype], origin_of(U.data)],
            c_int,
            Pointer[Scalar[dtype], origin_of(Vt.data)],
            c_int,
        ](
            c_int(101),
            jobz,
            c_int(M),
            c_int(N),
            A_copy.data.unsafe_ptr(),
            c_int(N),
            S.unsafe_ptr(),
            U.data.unsafe_ptr(),
            c_int(u_cols),
            Vt.data.unsafe_ptr(),
            c_int(N),
        )

    if info != 0:
        raise InvalidParameterError.error(
            "LAPACK SVD did not converge (info=" + String(info) + ")",
            "svd",
        )

    return SVDResult[dtype](U^, S^, Vt^)


def qr[
    dtype: DType = DType.float64
](A: Matrix[dtype]) raises -> QRResult[dtype]:
    """Computes the QR Decomposition of matrix A = Q * R using Householder reflectors.
    """
    comptime assert (
        dtype.is_floating_point()
    ), "Floating-point type required for QR decomposition"

    var M = A.rows
    var N = A.cols
    var K = min(M, N)

    var A_qr = A.copy()
    var tau = List[Scalar[dtype]](capacity=K)
    for _ in range(K):
        tau.append(0)

    var info_geqrf: c_int = 0

    comptime if dtype == DType.float64:
        info_geqrf = external_call[
            "LAPACKE_dgeqrf",
            c_int,
            c_int,
            c_int,
            c_int,
            Pointer[Scalar[dtype], origin_of(A_qr.data)],
            c_int,
            Pointer[Scalar[dtype], origin_of(tau)],
        ](
            c_int(101),
            c_int(M),
            c_int(N),
            A_qr.data.unsafe_ptr(),
            c_int(N),
            tau.unsafe_ptr(),
        )
    elif dtype == DType.float32:
        info_geqrf = external_call[
            "LAPACKE_sgeqrf",
            c_int,
            c_int,
            c_int,
            c_int,
            Pointer[Scalar[dtype], origin_of(A_qr.data)],
            c_int,
            Pointer[Scalar[dtype], origin_of(tau)],
        ](
            c_int(101),
            c_int(M),
            c_int(N),
            A_qr.data.unsafe_ptr(),
            c_int(N),
            tau.unsafe_ptr(),
        )

    if info_geqrf != 0:
        raise InvalidParameterError.error(
            "LAPACK QR factorization failed (info=" + String(info_geqrf) + ")",
            "qr",
        )

    # Extract upper triangular R (K x N)
    var R = Matrix[dtype].zeros(K, N)
    for r in range(K):
        for c in range(r, N):
            R[r, c] = A_qr[r, c]

    # Generate orthogonal matrix Q (M x K)
    var Q = Matrix[dtype].zeros(M, K)
    for r in range(M):
        for c in range(K):
            Q[r, c] = A_qr[r, c]

    var info_orgqr: c_int = 0

    comptime if dtype == DType.float64:
        info_orgqr = external_call[
            "LAPACKE_dorgqr",
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            Pointer[Scalar[dtype], origin_of(Q.data)],
            c_int,
            Pointer[Scalar[dtype], origin_of(tau)],
        ](
            c_int(101),
            c_int(M),
            c_int(K),
            c_int(K),
            Q.data.unsafe_ptr(),
            c_int(K),
            tau.unsafe_ptr(),
        )
    elif dtype == DType.float32:
        info_orgqr = external_call[
            "LAPACKE_sorgqr",
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            Pointer[Scalar[dtype], origin_of(Q.data)],
            c_int,
            Pointer[Scalar[dtype], origin_of(tau)],
        ](
            c_int(101),
            c_int(M),
            c_int(K),
            c_int(K),
            Q.data.unsafe_ptr(),
            c_int(K),
            tau.unsafe_ptr(),
        )

    if info_orgqr != 0:
        raise InvalidParameterError.error(
            "LAPACK Q matrix reconstruction failed (info="
            + String(info_orgqr)
            + ")",
            "qr",
        )

    return QRResult[dtype](Q^, R^)


def cholesky[
    dtype: DType = DType.float64
](A: Matrix[dtype], lower: Bool = True) raises -> Matrix[dtype]:
    """Computes the Cholesky factorization of a symmetric positive-definite matrix A = L * L^T.
    """
    comptime assert (
        dtype.is_floating_point()
    ), "Floating-point type required for Cholesky factorization"

    if A.rows != A.cols:
        raise DimensionMismatchError.error(
            "Square matrix (A.rows == A.cols)",
            "A(" + String(A.rows) + "x" + String(A.cols) + ")",
            "cholesky",
        )

    var N = A.rows
    var L = A.copy()
    var uplo_char = Int8(ord("L")) if lower else Int8(ord("U"))

    var info: c_int = 0

    comptime if dtype == DType.float64:
        info = external_call[
            "LAPACKE_dpotrf",
            c_int,
            c_int,
            c_char,
            c_int,
            Pointer[Scalar[dtype], origin_of(L.data)],
            c_int,
        ](
            c_int(101),
            uplo_char,
            c_int(N),
            L.data.unsafe_ptr(),
            c_int(N),
        )
    elif dtype == DType.float32:
        info = external_call[
            "LAPACKE_spotrf",
            c_int,
            c_int,
            c_char,
            c_int,
            Pointer[Scalar[dtype], origin_of(L.data)],
            c_int,
        ](
            c_int(101),
            uplo_char,
            c_int(N),
            L.data.unsafe_ptr(),
            c_int(N),
        )

    if info > 0:
        raise InvalidParameterError.error(
            "Matrix A is not positive-definite (leading minor "
            + String(info)
            + " is not positive)",
            "cholesky",
        )
    elif info < 0:
        raise InvalidParameterError.error(
            "Illegal argument in LAPACK cholesky (info=" + String(info) + ")",
            "cholesky",
        )

    # Zero out non-triangular elements
    if lower:
        for r in range(N):
            for c in range(r + 1, N):
                L[r, c] = 0
    else:
        for r in range(N):
            for c in range(r):
                L[r, c] = 0

    return L^


def lstsq[
    dtype: DType = DType.float64
](
    A: Matrix[dtype],
    b: List[Scalar[dtype]],
    rcond: Float64 = -1.0,
) raises -> List[Scalar[dtype]]:
    """Solves the linear least-squares problem min ||A * x - b||_2 using SVD (dgelss/sgelss).
    """
    comptime assert (
        dtype.is_floating_point()
    ), "Floating-point type required for least squares"

    if A.rows != len(b):
        raise DimensionMismatchError.error(
            "A.rows == len(b)",
            "A.rows=" + String(A.rows) + ", len(b)=" + String(len(b)),
            "lstsq",
        )

    var M = A.rows
    var N = A.cols
    var K = min(M, N)
    var max_mn = max(M, N)

    var A_copy = A.copy()
    var b_buf = List[Scalar[dtype]](capacity=max_mn)
    for i in range(M):
        b_buf.append(b[i])
    for _ in range(M, max_mn):
        b_buf.append(0)

    var s = List[Scalar[dtype]](capacity=K)
    for _ in range(K):
        s.append(0)

    var rank_buf = List[c_int](capacity=1)
    rank_buf.append(0)

    var info: c_int = 0

    comptime if dtype == DType.float64:
        info = external_call[
            "LAPACKE_dgelss",
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            Pointer[Scalar[dtype], origin_of(A_copy.data)],
            c_int,
            Pointer[Scalar[dtype], origin_of(b_buf)],
            c_int,
            Pointer[Scalar[dtype], origin_of(s)],
            c_double,
            Pointer[c_int, origin_of(rank_buf)],
        ](
            c_int(101),
            c_int(M),
            c_int(N),
            c_int(1),
            A_copy.data.unsafe_ptr(),
            c_int(N),
            b_buf.unsafe_ptr(),
            c_int(1),
            s.unsafe_ptr(),
            rcond,
            rank_buf.unsafe_ptr(),
        )
    elif dtype == DType.float32:
        info = external_call[
            "LAPACKE_sgelss",
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            Pointer[Scalar[dtype], origin_of(A_copy.data)],
            c_int,
            Pointer[Scalar[dtype], origin_of(b_buf)],
            c_int,
            Pointer[Scalar[dtype], origin_of(s)],
            c_float,
            Pointer[c_int, origin_of(rank_buf)],
        ](
            c_int(101),
            c_int(M),
            c_int(N),
            c_int(1),
            A_copy.data.unsafe_ptr(),
            c_int(N),
            b_buf.unsafe_ptr(),
            c_int(1),
            s.unsafe_ptr(),
            Float32(rcond),
            rank_buf.unsafe_ptr(),
        )

    if info != 0:
        raise InvalidParameterError.error(
            "LAPACK least squares solver failed (info=" + String(info) + ")",
            "lstsq",
        )

    var x = List[Scalar[dtype]](capacity=N)
    for i in range(N):
        x.append(b_buf[i])

    return x^


def solve[
    dtype: DType = DType.float64
](A: Matrix[dtype], b: List[Scalar[dtype]]) raises -> List[Scalar[dtype]]:
    """Solves a square linear system A * x = b using LU decomposition (dgesv/sgesv).
    """
    comptime assert (
        dtype.is_floating_point()
    ), "Floating-point type required for linear solve"

    if A.rows != A.cols:
        raise DimensionMismatchError.error(
            "Square matrix A (rows == cols)",
            "A(" + String(A.rows) + "x" + String(A.cols) + ")",
            "solve",
        )
    if A.rows != len(b):
        raise DimensionMismatchError.error(
            "A.rows == len(b)",
            "A.rows=" + String(A.rows) + ", len(b)=" + String(len(b)),
            "solve",
        )

    var N = A.rows
    var A_copy = A.copy()
    var b_copy = b.copy()
    var ipiv = List[c_int](capacity=N)
    for _ in range(N):
        ipiv.append(0)

    var info: c_int = 0

    comptime if dtype == DType.float64:
        info = external_call[
            "LAPACKE_dgesv",
            c_int,
            c_int,
            c_int,
            c_int,
            Pointer[Scalar[dtype], origin_of(A_copy.data)],
            c_int,
            Pointer[c_int, origin_of(ipiv)],
            Pointer[Scalar[dtype], origin_of(b_copy)],
            c_int,
        ](
            c_int(101),
            c_int(N),
            c_int(1),
            A_copy.data.unsafe_ptr(),
            c_int(N),
            ipiv.unsafe_ptr(),
            b_copy.unsafe_ptr(),
            c_int(1),
        )
    elif dtype == DType.float32:
        info = external_call[
            "LAPACKE_sgesv",
            c_int,
            c_int,
            c_int,
            c_int,
            Pointer[Scalar[dtype], origin_of(A_copy.data)],
            c_int,
            Pointer[c_int, origin_of(ipiv)],
            Pointer[Scalar[dtype], origin_of(b_copy)],
            c_int,
        ](
            c_int(101),
            c_int(N),
            c_int(1),
            A_copy.data.unsafe_ptr(),
            c_int(N),
            ipiv.unsafe_ptr(),
            b_copy.unsafe_ptr(),
            c_int(1),
        )

    if info > 0:
        raise InvalidParameterError.error(
            "Matrix A is singular and cannot be inverted (minor "
            + String(info)
            + " is exactly zero)",
            "solve",
        )
    elif info < 0:
        raise InvalidParameterError.error(
            "Illegal argument in LAPACK solve (info=" + String(info) + ")",
            "solve",
        )

    return b_copy^


def inv[dtype: DType = DType.float64](A: Matrix[dtype]) raises -> Matrix[dtype]:
    """Computes the multiplicative inverse of a square matrix A using LU decomposition (dgetrf/dgetri).
    """
    comptime assert (
        dtype.is_floating_point()
    ), "Floating-point type required for matrix inversion"

    if A.rows != A.cols:
        raise DimensionMismatchError.error(
            "Square matrix A (rows == cols)",
            "A(" + String(A.rows) + "x" + String(A.cols) + ")",
            "inv",
        )

    var N = A.rows
    var A_inv = A.copy()
    var ipiv = List[c_int](capacity=N)
    for _ in range(N):
        ipiv.append(0)

    var info_getrf: c_int = 0
    var info_getri: c_int = 0

    comptime if dtype == DType.float64:
        info_getrf = external_call[
            "LAPACKE_dgetrf",
            c_int,
            c_int,
            c_int,
            c_int,
            Pointer[Scalar[dtype], origin_of(A_inv.data)],
            c_int,
            Pointer[c_int, origin_of(ipiv)],
        ](
            c_int(101),
            c_int(N),
            c_int(N),
            A_inv.data.unsafe_ptr(),
            c_int(N),
            ipiv.unsafe_ptr(),
        )
        if info_getrf == 0:
            info_getri = external_call[
                "LAPACKE_dgetri",
                c_int,
                c_int,
                c_int,
                Pointer[Scalar[dtype], origin_of(A_inv.data)],
                c_int,
                Pointer[c_int, origin_of(ipiv)],
            ](
                c_int(101),
                c_int(N),
                A_inv.data.unsafe_ptr(),
                c_int(N),
                ipiv.unsafe_ptr(),
            )
    elif dtype == DType.float32:
        info_getrf = external_call[
            "LAPACKE_sgetrf",
            c_int,
            c_int,
            c_int,
            c_int,
            Pointer[Scalar[dtype], origin_of(A_inv.data)],
            c_int,
            Pointer[c_int, origin_of(ipiv)],
        ](
            c_int(101),
            c_int(N),
            c_int(N),
            A_inv.data.unsafe_ptr(),
            c_int(N),
            ipiv.unsafe_ptr(),
        )
        if info_getrf == 0:
            info_getri = external_call[
                "LAPACKE_sgetri",
                c_int,
                c_int,
                c_int,
                Pointer[Scalar[dtype], origin_of(A_inv.data)],
                c_int,
                Pointer[c_int, origin_of(ipiv)],
            ](
                c_int(101),
                c_int(N),
                A_inv.data.unsafe_ptr(),
                c_int(N),
                ipiv.unsafe_ptr(),
            )

    if info_getrf > 0 or info_getri > 0:
        raise InvalidParameterError.error(
            "Matrix A is singular and cannot be inverted",
            "inv",
        )

    return A_inv^


def norm[
    dtype: DType = DType.float64
](A: Matrix[dtype], ord: String = "fro") raises -> Float64:
    """Computes the Frobenius matrix norm: ||A||_F = sqrt(sum(A_ij^2))."""
    if ord != "fro":
        raise InvalidParameterError.error("ord", "Only 'fro' is supported")

    var total: Float64 = 0.0
    var count = A.rows * A.cols
    for i in range(count):
        var v = Float64(A.data[i])
        total += v * v

    return sqrt(total)
