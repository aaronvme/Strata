from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_raises,
    assert_almost_equal,
)
from strata import Matrix, gemm, dense_dot_vec, DimensionMismatchError


def test_matrix_construction_and_access() raises:
    # Zeros, ones, custom fill
    var z = Matrix[DType.float64].zeros(3, 4)
    assert_equal(z.rows, 3)
    assert_equal(z.cols, 4)
    assert_equal(z.num_elements(), 12)
    assert_equal(z.shape()[0], 3)
    assert_equal(z.shape()[1], 4)
    for r in range(3):
        for c in range(4):
            assert_equal(z[r, c], 0.0)

    var o = Matrix[DType.float32].ones(2, 5)
    assert_equal(o.rows, 2)
    assert_equal(o.cols, 5)
    for r in range(2):
        for c in range(5):
            assert_equal(o[r, c], 1.0)

    var filled = Matrix[DType.int32](3, 3, -7)
    for r in range(3):
        for c in range(3):
            assert_equal(filled[r, c], -7)

    # Set item
    filled[1, 2] = 42
    assert_equal(filled[1, 2], 42)


def test_matrix_row_and_col_extraction() raises:
    var m = Matrix[DType.float64](3, 3, 0)
    var counter: Float64 = 1.0
    for r in range(3):
        for c in range(3):
            m[r, c] = counter
            counter += 1.0

    var r1 = m.row(1)
    assert_equal(len(r1), 3)
    assert_equal(r1[0], 4.0)
    assert_equal(r1[1], 5.0)
    assert_equal(r1[2], 6.0)

    var c2 = m.col(2)
    assert_equal(len(c2), 3)
    assert_equal(c2[0], 3.0)
    assert_equal(c2[1], 6.0)
    assert_equal(c2[2], 9.0)


def test_matrix_transpose() raises:
    # Non-square matrix transpose (2x4 -> 4x2)
    var A = Matrix[DType.float64](2, 4, 0)
    A[0, 0] = 1.0
    A[0, 1] = 2.0
    A[0, 2] = 3.0
    A[0, 3] = 4.0
    A[1, 0] = 5.0
    A[1, 1] = 6.0
    A[1, 2] = 7.0
    A[1, 3] = 8.0

    var At = A.transpose()
    assert_equal(At.rows, 4)
    assert_equal(At.cols, 2)
    assert_equal(At[0, 0], 1.0)
    assert_equal(At[0, 1], 5.0)
    assert_equal(At[3, 0], 4.0)
    assert_equal(At[3, 1], 8.0)

    # Double transpose identity: (A^T)^T == A
    var Att = At.transpose()
    assert_equal(Att.rows, 2)
    assert_equal(Att.cols, 4)
    for r in range(2):
        for c in range(4):
            assert_equal(Att[r, c], A[r, c])


def test_matrix_cast_promotions() raises:
    var m_int = Matrix[DType.int32](2, 2, 0)
    m_int[0, 0] = 10
    m_int[0, 1] = -20
    m_int[1, 0] = 30
    m_int[1, 1] = 40

    # Int32 -> Float64
    var m_f64 = m_int.cast[DType.float64]()
    assert_equal(m_f64.rows, 2)
    assert_equal(m_f64.cols, 2)
    assert_equal(m_f64[0, 0], 10.0)
    assert_equal(m_f64[0, 1], -20.0)

    # Float64 -> Float32
    var m_f32 = m_f64.cast[DType.float32]()
    assert_equal(m_f32[1, 0], 30.0)
    assert_equal(m_f32[1, 1], 40.0)

    # Float32 -> Int64
    var m_i64 = m_f32.cast[DType.int64]()
    assert_equal(m_i64[0, 1], -20)


def test_matrix_mean_and_std_axis_0() raises:
    var m = Matrix[DType.float64](4, 3, 0)
    # Col 0: [10, 20, 30, 40] -> Mean = 25, Std = sqrt(125)
    m[0, 0] = 10.0
    m[1, 0] = 20.0
    m[2, 0] = 30.0
    m[3, 0] = 40.0

    # Col 1: Constant [5, 5, 5, 5] -> Mean = 5, Std fallback = 1.0
    m[0, 1] = 5.0
    m[1, 1] = 5.0
    m[2, 1] = 5.0
    m[3, 1] = 5.0

    # Col 2: [0, 0, 0, 0] -> Mean = 0, Std fallback = 1.0
    m[0, 2] = 0.0
    m[1, 2] = 0.0
    m[2, 2] = 0.0
    m[3, 2] = 0.0

    var means = m.mean_along_axis_0()
    assert_equal(len(means), 3)
    assert_equal(means[0], 25.0)
    assert_equal(means[1], 5.0)
    assert_equal(means[2], 0.0)

    var stds = m.std_along_axis_0(means)
    assert_equal(len(stds), 3)
    assert_equal(stds[1], 1.0)  # Constant column fallback
    assert_equal(stds[2], 1.0)

    # Zero-row matrix guard
    var empty_m = Matrix[DType.float64](0, 3, 0)
    var empty_means = empty_m.mean_along_axis_0()
    assert_equal(len(empty_means), 3)
    var empty_stds = empty_m.std_along_axis_0(empty_means)
    assert_equal(len(empty_stds), 3)


def test_gemm_properties() raises:
    # 1. Standard non-square GEMM: (2x3) @ (3x2) -> (2x2)
    var A = Matrix[DType.float64](2, 3, 0)
    A[0, 0] = 1.0
    A[0, 1] = 2.0
    A[0, 2] = 3.0
    A[1, 0] = 4.0
    A[1, 1] = 5.0
    A[1, 2] = 6.0

    var B = Matrix[DType.float64](3, 2, 0)
    B[0, 0] = 7.0
    B[0, 1] = 8.0
    B[1, 0] = 9.0
    B[1, 1] = 1.0
    B[2, 0] = 2.0
    B[2, 1] = 3.0

    var C = gemm(A, B)
    assert_equal(C.rows, 2)
    assert_equal(C.cols, 2)
    # C[0,0] = 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
    assert_equal(C[0, 0], 31.0)
    # C[0,1] = 1*8 + 2*1 + 3*3 = 8 + 2 + 9 = 19
    assert_equal(C[0, 1], 19.0)
    # C[1,0] = 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85
    assert_equal(C[1, 0], 85.0)
    # C[1,1] = 4*8 + 5*1 + 6*3 = 32 + 5 + 18 = 55
    assert_equal(C[1, 1], 55.0)

    # 2. Identity matrix multiplication: A @ I == A
    var I = Matrix[DType.float64].zeros(3, 3)
    I[0, 0] = 1.0
    I[1, 1] = 1.0
    I[2, 2] = 1.0
    var AI = gemm(A, I)
    for r in range(2):
        for c in range(3):
            assert_equal(AI[r, c], A[r, c])

    # 3. Dimension mismatch error check
    var BadB = Matrix[DType.float64].ones(4, 2)
    with assert_raises():
        _ = gemm(A, BadB)


def test_dense_dot_vec_properties() raises:
    var A = Matrix[DType.float64](2, 3, 0)
    A[0, 0] = 1.0
    A[0, 1] = 2.0
    A[0, 2] = 3.0
    A[1, 0] = 4.0
    A[1, 1] = 5.0
    A[1, 2] = 6.0

    var x: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0]
    var bias: Scalar[DType.float64] = 10.0

    # y = A @ x + bias: [1*1 + 2*2 + 3*3 + 10 = 24, 4*1 + 5*2 + 6*3 + 10 = 42]
    var y = dense_dot_vec(A, x, bias)
    assert_equal(len(y), 2)
    assert_equal(y[0], 24.0)
    assert_equal(y[1], 42.0)

    # Dimension mismatch check
    var bad_x: List[Scalar[DType.float64]] = [1.0, 2.0]
    with assert_raises():
        _ = dense_dot_vec(A, bad_x)


def test_matrix_string_representation() raises:
    var m = Matrix[DType.float64](2, 2, 3.14)
    var s = String(m)
    assert_true(s.byte_length() > 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
