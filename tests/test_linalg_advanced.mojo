from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_raises,
    assert_almost_equal,
)
from strata import (
    Matrix,
    gemm,
    svd,
    qr,
    cholesky,
    lstsq,
    solve,
    inv,
    norm,
    eigh,
    SVDResult,
    QRResult,
    EigResult,
    DimensionMismatchError,
    InvalidParameterError,
)


def test_matrix_elementwise_operators() raises:
    var A = Matrix[DType.float64](2, 2, 0)
    A[0, 0] = 1.0
    A[0, 1] = 2.0
    A[1, 0] = 3.0
    A[1, 1] = 4.0

    var B = Matrix[DType.float64](2, 2, 0)
    B[0, 0] = 5.0
    B[0, 1] = 6.0
    B[1, 0] = 7.0
    B[1, 1] = 8.0

    # 1. Addition A + B
    var C_add = A + B
    assert_equal(C_add[0, 0], 6.0)
    assert_equal(C_add[0, 1], 8.0)
    assert_equal(C_add[1, 0], 10.0)
    assert_equal(C_add[1, 1], 12.0)

    # 2. Subtraction A - B
    var C_sub = A - B
    assert_equal(C_sub[0, 0], -4.0)
    assert_equal(C_sub[0, 1], -4.0)
    assert_equal(C_sub[1, 0], -4.0)
    assert_equal(C_sub[1, 1], -4.0)

    # 3. Element-wise Hadamard Product A * B
    var C_mul = A * B
    assert_equal(C_mul[0, 0], 5.0)  # 1*5
    assert_equal(C_mul[0, 1], 12.0)  # 2*6
    assert_equal(C_mul[1, 0], 21.0)  # 3*7
    assert_equal(C_mul[1, 1], 32.0)  # 4*8

    # 4. Scalar multiplication A * 2.5 and 2.5 * A
    var C_scale = A * 2.5
    assert_equal(C_scale[0, 0], 2.5)
    assert_equal(C_scale[0, 1], 5.0)
    assert_equal(C_scale[1, 0], 7.5)
    assert_equal(C_scale[1, 1], 10.0)

    # 5. Scalar division A / 2.0
    var C_div = A / 2.0
    assert_equal(C_div[0, 0], 0.5)
    assert_equal(C_div[0, 1], 1.0)
    assert_equal(C_div[1, 0], 1.5)
    assert_equal(C_div[1, 1], 2.0)

    # 6. Unary negation -A
    var C_neg = -A
    assert_equal(C_neg[0, 0], -1.0)
    assert_equal(C_neg[0, 1], -2.0)
    assert_equal(C_neg[1, 0], -3.0)
    assert_equal(C_neg[1, 1], -4.0)

    # 7. Shape mismatch validation
    var Bad = Matrix[DType.float64](3, 2, 0)
    with assert_raises():
        _ = A + Bad
    with assert_raises():
        _ = A - Bad
    with assert_raises():
        _ = A * Bad


def test_matrix_algebra_axioms() raises:
    # 3x3 Matrices testing algebraic properties
    var A = Matrix[DType.float64](3, 3, 0)
    A[0, 0] = 1.0
    A[0, 1] = 2.0
    A[0, 2] = 3.0
    A[1, 0] = 4.0
    A[1, 1] = 5.0
    A[1, 2] = 6.0
    A[2, 0] = 7.0
    A[2, 1] = 8.0
    A[2, 2] = 9.0

    var B = Matrix[DType.float64](3, 3, 0)
    B[0, 0] = 9.0
    B[0, 1] = 8.0
    B[0, 2] = 7.0
    B[1, 0] = 6.0
    B[1, 1] = 5.0
    B[1, 2] = 4.0
    B[2, 0] = 3.0
    B[2, 1] = 2.0
    B[2, 2] = 1.0

    var C = Matrix[DType.float64](3, 3, 0)
    C[0, 0] = 2.0
    C[1, 1] = 3.0
    C[2, 2] = 4.0

    # 1. Commutativity: A + B == B + A
    var AB = A + B
    var BA = B + A
    for r in range(3):
        for c in range(3):
            assert_equal(AB[r, c], BA[r, c])

    # 2. Associativity: (A + B) + C == A + (B + C)
    var Left = (A + B) + C
    var Right = A + (B + C)
    for r in range(3):
        for c in range(3):
            assert_equal(Left[r, c], Right[r, c])

    # 3. Distributivity: 3.0 * (A + B) == 3.0*A + 3.0*B
    var DistLeft = (A + B) * 3.0
    var DistRight = (A * 3.0) + (B * 3.0)
    for r in range(3):
        for c in range(3):
            assert_almost_equal(DistLeft[r, c], DistRight[r, c])

    # 4. Self-subtraction: A - A == 0
    var Zero = A - A
    for r in range(3):
        for c in range(3):
            assert_equal(Zero[r, c], 0.0)

    # 5. Double negation: -(-A) == A
    var NegNeg = -(-A)
    for r in range(3):
        for c in range(3):
            assert_equal(NegNeg[r, c], A[r, c])


def test_svd_tall_and_reduced_f64() raises:
    # 3x2 matrix (Tall)
    var A = Matrix[DType.float64](3, 2, 0)
    A[0, 0] = 1.0
    A[0, 1] = 2.0
    A[1, 0] = 3.0
    A[1, 1] = 4.0
    A[2, 0] = 5.0
    A[2, 1] = 6.0

    # 1. Reduced SVD (default full_matrices=False)
    var res = svd(A, full_matrices=False)
    assert_equal(res.U.rows, 3)
    assert_equal(res.U.cols, 2)
    assert_equal(len(res.S), 2)
    assert_equal(res.Vt.rows, 2)
    assert_equal(res.Vt.cols, 2)

    # Singular values are positive and non-increasing
    assert_true(res.S[0] >= res.S[1])
    assert_almost_equal(res.S[0], 9.52551809156511)
    assert_almost_equal(res.S[1], 0.51430058065864)

    # Reconstruction: U * diag(S) * Vt == A
    var Sigma_Vt = Matrix[DType.float64](2, 2, 0)
    for r in range(2):
        for c in range(2):
            Sigma_Vt[r, c] = res.S[r] * res.Vt[r, c]

    var A_reconstructed = gemm(res.U, Sigma_Vt)
    for r in range(3):
        for c in range(2):
            assert_almost_equal(A_reconstructed[r, c], A[r, c], rtol=1e-5)

    # Orthogonality: U^T * U == I_2 and Vt * Vt^T == I_2
    var UtU = gemm(res.U.transpose(), res.U)
    assert_almost_equal(UtU[0, 0], 1.0, rtol=1e-5)
    assert_almost_equal(UtU[1, 1], 1.0, rtol=1e-5)
    assert_almost_equal(UtU[0, 1], 0.0, atol=1e-5)


def test_svd_wide_and_full_matrices() raises:
    # 1. Wide matrix (2x3)
    var A_wide = Matrix[DType.float64](2, 3, 0)
    A_wide[0, 0] = 3.0
    A_wide[0, 1] = 2.0
    A_wide[0, 2] = 2.0
    A_wide[1, 0] = 2.0
    A_wide[1, 1] = 3.0
    A_wide[1, 2] = -2.0

    var res_wide = svd(A_wide, full_matrices=False)
    assert_equal(res_wide.U.rows, 2)
    assert_equal(res_wide.U.cols, 2)
    assert_equal(len(res_wide.S), 2)
    assert_equal(res_wide.Vt.rows, 2)
    assert_equal(res_wide.Vt.cols, 3)

    # Reconstruct wide matrix
    var Sigma_Vt_wide = Matrix[DType.float64](2, 3, 0)
    for r in range(2):
        for c in range(3):
            Sigma_Vt_wide[r, c] = res_wide.S[r] * res_wide.Vt[r, c]
    var A_rec_wide = gemm(res_wide.U, Sigma_Vt_wide)
    for r in range(2):
        for c in range(3):
            assert_almost_equal(A_rec_wide[r, c], A_wide[r, c], rtol=1e-5)

    # 2. Full matrices flag on 3x2 matrix
    var A_tall = Matrix[DType.float64](3, 2, 0)
    A_tall[0, 0] = 1.0
    A_tall[0, 1] = 0.0
    A_tall[1, 0] = 0.0
    A_tall[1, 1] = 1.0
    A_tall[2, 0] = 0.0
    A_tall[2, 1] = 0.0

    var res_full = svd(A_tall, full_matrices=True)
    assert_equal(res_full.U.rows, 3)
    assert_equal(res_full.U.cols, 3)  # Full square U
    assert_equal(res_full.Vt.rows, 2)
    assert_equal(res_full.Vt.cols, 2)

    # U is orthogonal 3x3
    var UtU_full = gemm(res_full.U.transpose(), res_full.U)
    for r in range(3):
        for c in range(3):
            var expected: Float64 = 1.0 if r == c else 0.0
            assert_almost_equal(UtU_full[r, c], expected, atol=1e-5)


def test_svd_single_element_and_diagonal() raises:
    # 1. 1x1 Matrix SVD
    var A_1x1 = Matrix[DType.float64](1, 1, 7.5)
    var res_1x1 = svd(A_1x1)
    assert_almost_equal(res_1x1.S[0], 7.5)
    assert_almost_equal(res_1x1.U[0, 0] * res_1x1.Vt[0, 0], 1.0)

    # 2. 3x3 Diagonal Matrix SVD
    var Diag = Matrix[DType.float64].zeros(3, 3)
    Diag[0, 0] = -10.0
    Diag[1, 1] = 5.0
    Diag[2, 2] = 2.0
    var res_diag = svd(Diag)
    assert_almost_equal(res_diag.S[0], 10.0)
    assert_almost_equal(res_diag.S[1], 5.0)
    assert_almost_equal(res_diag.S[2], 2.0)

    # 3. 3x3 Zero Matrix SVD
    var Zero = Matrix[DType.float64].zeros(3, 3)
    var res_zero = svd(Zero)
    assert_almost_equal(res_zero.S[0], 0.0)
    assert_almost_equal(res_zero.S[1], 0.0)
    assert_almost_equal(res_zero.S[2], 0.0)


def test_svd_float32() raises:
    var A = Matrix[DType.float32](2, 2, 0)
    A[0, 0] = 3.0
    A[0, 1] = 0.0
    A[1, 0] = 0.0
    A[1, 1] = -4.0

    var res = svd[DType.float32](A)
    assert_almost_equal(res.S[0], 4.0, rtol=1e-4)
    assert_almost_equal(res.S[1], 3.0, rtol=1e-4)


def test_qr_square_and_tall() raises:
    # 1. 3x2 Tall Matrix QR
    var A = Matrix[DType.float64](3, 2, 0)
    A[0, 0] = 12.0
    A[0, 1] = -51.0
    A[1, 0] = 6.0
    A[1, 1] = 167.0
    A[2, 0] = -4.0
    A[2, 1] = 24.0

    var res = qr(A)
    assert_equal(res.Q.rows, 3)
    assert_equal(res.Q.cols, 2)
    assert_equal(res.R.rows, 2)
    assert_equal(res.R.cols, 2)

    # R is upper triangular (R[1, 0] == 0)
    assert_almost_equal(res.R[1, 0], 0.0, atol=1e-5)

    # Reconstruction: Q * R == A
    var A_reconstructed = gemm(res.Q, res.R)
    for r in range(3):
        for c in range(2):
            assert_almost_equal(A_reconstructed[r, c], A[r, c], rtol=1e-5)

    # Orthogonality: Q^T * Q == I_2
    var QtQ = gemm(res.Q.transpose(), res.Q)
    assert_almost_equal(QtQ[0, 0], 1.0, rtol=1e-5)
    assert_almost_equal(QtQ[1, 1], 1.0, rtol=1e-5)
    assert_almost_equal(QtQ[0, 1], 0.0, atol=1e-5)

    # 2. 3x3 Square Matrix QR
    var A_sq = Matrix[DType.float64](3, 3, 0)
    A_sq[0, 0] = 1.0
    A_sq[0, 1] = 2.0
    A_sq[0, 2] = 4.0
    A_sq[1, 0] = 3.0
    A_sq[1, 1] = 8.0
    A_sq[1, 2] = 14.0
    A_sq[2, 0] = 2.0
    A_sq[2, 1] = 6.0
    A_sq[2, 2] = 13.0

    var res_sq = qr(A_sq)
    var A_sq_rec = gemm(res_sq.Q, res_sq.R)
    for r in range(3):
        for c in range(3):
            assert_almost_equal(A_sq_rec[r, c], A_sq[r, c], rtol=1e-5)


def test_qr_single_element_and_identity() raises:
    # 1. 1x1 Matrix QR
    var A_1x1 = Matrix[DType.float64](1, 1, 4.0)
    var res_1x1 = qr(A_1x1)
    assert_almost_equal(res_1x1.Q[0, 0] * res_1x1.R[0, 0], 4.0)

    # 2. 3x3 Identity Matrix QR
    var Eye = Matrix[DType.float64].eye(3)
    var res_eye = qr(Eye)
    var Eye_rec = gemm(res_eye.Q, res_eye.R)
    for r in range(3):
        for c in range(3):
            var expected: Float64 = 1.0 if r == c else 0.0
            assert_almost_equal(Eye_rec[r, c], expected, atol=1e-5)

    var QtQ_eye = gemm(res_eye.Q.transpose(), res_eye.Q)
    for r in range(3):
        for c in range(3):
            var expected: Float64 = 1.0 if r == c else 0.0
            assert_almost_equal(QtQ_eye[r, c], expected, atol=1e-5)


def test_qr_float32() raises:
    var A = Matrix[DType.float32](2, 2, 0)
    A[0, 0] = 1.0
    A[0, 1] = 1.0
    A[1, 0] = 0.0
    A[1, 1] = 1.0

    var res = qr[DType.float32](A)
    var A_rec = gemm(res.Q, res.R)
    assert_almost_equal(A_rec[0, 0], 1.0, rtol=1e-4)
    assert_almost_equal(A_rec[0, 1], 1.0, rtol=1e-4)
    assert_almost_equal(A_rec[1, 1], 1.0, rtol=1e-4)


def test_cholesky_decomposition_and_modes() raises:
    # 1. Lower triangular Cholesky: A = L * L^T
    var A = Matrix[DType.float64](3, 3, 0)
    A[0, 0] = 4.0
    A[0, 1] = 12.0
    A[0, 2] = -16.0
    A[1, 0] = 12.0
    A[1, 1] = 37.0
    A[1, 2] = -43.0
    A[2, 0] = -16.0
    A[2, 1] = -43.0
    A[2, 2] = 98.0

    var L = cholesky(A, lower=True)
    assert_equal(L.rows, 3)
    assert_equal(L.cols, 3)

    # Verify L is lower triangular
    assert_almost_equal(L[0, 1], 0.0)
    assert_almost_equal(L[0, 2], 0.0)
    assert_almost_equal(L[1, 2], 0.0)

    # Reconstruction: L * L^T == A
    var LLt = gemm(L, L.transpose())
    for r in range(3):
        for c in range(3):
            assert_almost_equal(LLt[r, c], A[r, c], rtol=1e-5)

    # 2. Upper triangular Cholesky: A = U^T * U
    var U = cholesky(A, lower=False)
    assert_almost_equal(U[1, 0], 0.0)
    assert_almost_equal(U[2, 0], 0.0)
    assert_almost_equal(U[2, 1], 0.0)

    var UtU = gemm(U.transpose(), U)
    for r in range(3):
        for c in range(3):
            assert_almost_equal(UtU[r, c], A[r, c], rtol=1e-5)

    # 3. Consistency: (L_lower)^T == U_upper
    for r in range(3):
        for c in range(3):
            assert_almost_equal(L[r, c], U[c, r], rtol=1e-5)

    # 4. Non-positive definite matrix error
    var Bad = Matrix[DType.float64](2, 2, 0)
    Bad[0, 0] = -1.0
    Bad[1, 1] = -1.0
    with assert_raises():
        _ = cholesky(Bad)

    # 5. Non-square matrix error
    var NonSquare = Matrix[DType.float64](3, 2, 1.0)
    with assert_raises():
        _ = cholesky(NonSquare)


def test_cholesky_scalar_and_float32() raises:
    # 1. 1x1 Matrix Cholesky
    var A_1x1 = Matrix[DType.float64](1, 1, 9.0)
    var L_1x1 = cholesky(A_1x1)
    assert_almost_equal(L_1x1[0, 0], 3.0)

    # 2. Float32 Cholesky
    var A_f32 = Matrix[DType.float32](2, 2, 0)
    A_f32[0, 0] = 4.0
    A_f32[0, 1] = 2.0
    A_f32[1, 0] = 2.0
    A_f32[1, 1] = 5.0
    var L_f32 = cholesky[DType.float32](A_f32)
    assert_almost_equal(L_f32[0, 0], 2.0, rtol=1e-4)
    assert_almost_equal(L_f32[1, 0], 1.0, rtol=1e-4)
    assert_almost_equal(L_f32[1, 1], 2.0, rtol=1e-4)


def test_least_squares_and_multivariate() raises:
    # Multiple regression: y = 2 + 3*x1 - 4*x2
    # Data points:
    # (x1=1, x2=0) -> y = 2 + 3 - 0 = 5
    # (x1=0, x2=1) -> y = 2 + 0 - 4 = -2
    # (x1=1, x2=1) -> y = 2 + 3 - 4 = 1
    # (x1=2, x2=1) -> y = 2 + 6 - 4 = 4
    var A = Matrix[DType.float64](4, 3, 0)
    # Row 0: intercept=1, x1=1, x2=0
    A[0, 0] = 1.0
    A[0, 1] = 1.0
    A[0, 2] = 0.0
    # Row 1: intercept=1, x1=0, x2=1
    A[1, 0] = 1.0
    A[1, 1] = 0.0
    A[1, 2] = 1.0
    # Row 2: intercept=1, x1=1, x2=1
    A[2, 0] = 1.0
    A[2, 1] = 1.0
    A[2, 2] = 1.0
    # Row 3: intercept=1, x1=2, x2=1
    A[3, 0] = 1.0
    A[3, 1] = 2.0
    A[3, 2] = 1.0

    var y: List[Scalar[DType.float64]] = [5.0, -2.0, 1.0, 4.0]

    var beta = lstsq(A, y)
    assert_equal(len(beta), 3)
    assert_almost_equal(beta[0], 2.0, rtol=1e-4)  # Intercept = 2.0
    assert_almost_equal(beta[1], 3.0, rtol=1e-4)  # Beta1 = 3.0
    assert_almost_equal(beta[2], -4.0, rtol=1e-4)  # Beta2 = -4.0

    # Dimension mismatch check
    var Bad_y: List[Scalar[DType.float64]] = [1.0, 2.0]
    with assert_raises():
        _ = lstsq(A, Bad_y)


def test_least_squares_exact_and_underdetermined() raises:
    # 1. Exact square system (lstsq matches solve)
    var A_sq = Matrix[DType.float64](2, 2, 0)
    A_sq[0, 0] = 2.0
    A_sq[0, 1] = 1.0
    A_sq[1, 0] = 1.0
    A_sq[1, 1] = 3.0
    var b: List[Scalar[DType.float64]] = [5.0, 5.0]

    var x_lstsq = lstsq(A_sq, b)
    var x_solve = solve(A_sq, b)
    assert_almost_equal(x_lstsq[0], x_solve[0], rtol=1e-5)
    assert_almost_equal(x_lstsq[1], x_solve[1], rtol=1e-5)

    # 2. 1x1 Least Squares
    var A_1x1 = Matrix[DType.float64](1, 1, 4.0)
    var b_1x1: List[Scalar[DType.float64]] = [12.0]
    var x_1x1 = lstsq(A_1x1, b_1x1)
    assert_almost_equal(x_1x1[0], 3.0)


def test_solve_and_inv_high_dim() raises:
    # 3x3 System:
    # 2*x + 1*y - 1*z = 8
    # -3*x - 1*y + 2*z = -11
    # -2*x + 1*y + 2*z = -3
    # Exact solution: x = 2, y = 3, z = -1
    var A = Matrix[DType.float64](3, 3, 0)
    A[0, 0] = 2.0
    A[0, 1] = 1.0
    A[0, 2] = -1.0
    A[1, 0] = -3.0
    A[1, 1] = -1.0
    A[1, 2] = 2.0
    A[2, 0] = -2.0
    A[2, 1] = 1.0
    A[2, 2] = 2.0

    var b: List[Scalar[DType.float64]] = [8.0, -11.0, -3.0]

    # 1. solve(A, b)
    var x = solve(A, b)
    assert_almost_equal(x[0], 2.0, rtol=1e-5)
    assert_almost_equal(x[1], 3.0, rtol=1e-5)
    assert_almost_equal(x[2], -1.0, rtol=1e-5)

    # 2. inv(A) & Double Inverse Identity ((A^-1)^-1 == A)
    var A_inv = inv(A)
    var I = gemm(A, A_inv)
    for r in range(3):
        for c in range(3):
            var expected: Float64 = 1.0 if r == c else 0.0
            assert_almost_equal(I[r, c], expected, atol=1e-5)

    var A_inv_inv = inv(A_inv)
    for r in range(3):
        for c in range(3):
            assert_almost_equal(A_inv_inv[r, c], A[r, c], rtol=1e-5)

    # 3. Singular Matrix Error Handling
    var Singular = Matrix[DType.float64](2, 2, 0)
    Singular[0, 0] = 1.0
    Singular[0, 1] = 2.0
    Singular[1, 0] = 2.0
    Singular[1, 1] = 4.0  # Linearly dependent row!

    with assert_raises():
        _ = inv(Singular)

    var b_sing: List[Scalar[DType.float64]] = [1.0, 2.0]
    with assert_raises():
        _ = solve(Singular, b_sing)


def test_solve_and_inv_diagonal_and_float32() raises:
    # 1. Diagonal matrix inversion: inv(diag([2, 4, 8])) == diag([0.5, 0.25, 0.125])
    var Diag = Matrix[DType.float64].zeros(3, 3)
    Diag[0, 0] = 2.0
    Diag[1, 1] = 4.0
    Diag[2, 2] = 8.0

    var Diag_inv = inv(Diag)
    assert_almost_equal(Diag_inv[0, 0], 0.5)
    assert_almost_equal(Diag_inv[1, 1], 0.25)
    assert_almost_equal(Diag_inv[2, 2], 0.125)

    # 2. Float32 Solve
    var A_f32 = Matrix[DType.float32](2, 2, 0)
    A_f32[0, 0] = 3.0
    A_f32[0, 1] = 1.0
    A_f32[1, 0] = 1.0
    A_f32[1, 1] = 2.0
    var b_f32: List[Scalar[DType.float32]] = [9.0, 8.0]
    var x_f32 = solve[DType.float32](A_f32, b_f32)
    assert_almost_equal(x_f32[0], 2.0, rtol=1e-4)
    assert_almost_equal(x_f32[1], 3.0, rtol=1e-4)


def test_frobenius_norm_properties() raises:
    # 1. Zero matrix norm == 0
    var Zero = Matrix[DType.float64].zeros(3, 3)
    assert_almost_equal(norm(Zero), 0.0)

    # 2. Identity matrix norm ||I_3||_F == sqrt(3) ~ 1.7320508
    var Eye = Matrix[DType.float64].eye(3)
    assert_almost_equal(norm(Eye), 1.7320508075688772)

    # 3. 3-4-5 Triangle matrix
    var Tri = Matrix[DType.float64](1, 2, 0)
    Tri[0, 0] = 3.0
    Tri[0, 1] = 4.0
    assert_almost_equal(norm(Tri), 5.0)

    # 4. Unsupported norm ord error
    with assert_raises():
        _ = norm(Eye, ord="l1")


def test_eigh_symmetric_decomposition() raises:
    # 1. 2x2 Symmetric Matrix: [[2.0, 1.0], [1.0, 2.0]]
    # Eigenvalues: 1.0, 3.0
    var A = Matrix[DType.float64](2, 2, 0)
    A[0, 0] = 2.0
    A[0, 1] = 1.0
    A[1, 0] = 1.0
    A[1, 1] = 2.0

    var res = eigh(A)
    assert_equal(len(res.eigenvalues), 2)
    assert_almost_equal(res.eigenvalues[0], 1.0, rtol=1e-5)
    assert_almost_equal(res.eigenvalues[1], 3.0, rtol=1e-5)

    # Reconstruction: V * diag(eigenvalues) * V^T == A
    var Lambda_Vt = Matrix[DType.float64](2, 2, 0)
    var Vt = res.eigenvectors.transpose()
    for r in range(2):
        for c in range(2):
            Lambda_Vt[r, c] = res.eigenvalues[r] * Vt[r, c]

    var A_rec = gemm(res.eigenvectors, Lambda_Vt)
    for r in range(2):
        for c in range(2):
            assert_almost_equal(A_rec[r, c], A[r, c], rtol=1e-5)

    # Orthogonality of eigenvectors: V^T * V == I
    var VtV = gemm(Vt, res.eigenvectors)
    assert_almost_equal(VtV[0, 0], 1.0, atol=1e-5)
    assert_almost_equal(VtV[1, 1], 1.0, atol=1e-5)
    assert_almost_equal(VtV[0, 1], 0.0, atol=1e-5)


def test_eigh_float32_and_errors() raises:
    # 1. Float32 Symmetric Matrix
    var A_f32 = Matrix[DType.float32](2, 2, 0)
    A_f32[0, 0] = 5.0
    A_f32[0, 1] = 2.0
    A_f32[1, 0] = 2.0
    A_f32[1, 1] = 2.0

    var res_f32 = eigh[DType.float32](A_f32)
    assert_almost_equal(res_f32.eigenvalues[0], 1.0, rtol=1e-4)
    assert_almost_equal(res_f32.eigenvalues[1], 6.0, rtol=1e-4)

    # 2. Non-square matrix error
    var NonSquare = Matrix[DType.float64](3, 2, 1.0)
    with assert_raises():
        _ = eigh(NonSquare)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
