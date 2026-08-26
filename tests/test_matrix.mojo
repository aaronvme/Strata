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


def test_matrix_eye_properties() raises:
    for n in range(1, 6):
        var Eye = Matrix[DType.float64].eye(n)
        assert_equal(Eye.rows, n)
        assert_equal(Eye.cols, n)
        for r in range(n):
            for c in range(n):
                var expected: Float64 = 1.0 if r == c else 0.0
                assert_equal(Eye[r, c], expected)


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


def test_matrix_copy_independence() raises:
    var A = Matrix[DType.float64](2, 2, 10.0)
    var B = A.copy()
    B[0, 0] = 999.0

    assert_equal(A[0, 0], 10.0)  # Original remains unmutated
    assert_equal(B[0, 0], 999.0)


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

    # 2. Identity matrix multiplication: A @ I == A and I @ A == A
    var I3 = Matrix[DType.float64].eye(3)
    var AI = gemm(A, I3)
    for r in range(2):
        for c in range(3):
            assert_equal(AI[r, c], A[r, c])

    var I2 = Matrix[DType.float64].eye(2)
    var IA = gemm(I2, A)
    for r in range(2):
        for c in range(3):
            assert_equal(IA[r, c], A[r, c])

    # 3. Dimension mismatch error check
    var BadB = Matrix[DType.float64].ones(4, 2)
    with assert_raises():
        _ = gemm(A, BadB)


def test_integer_gemm_and_outer_products() raises:
    # 1. Integer GEMM (Int32) fallback loop verification
    var A_i32 = Matrix[DType.int32](2, 2, 0)
    A_i32[0, 0] = 1
    A_i32[0, 1] = 2
    A_i32[1, 0] = 3
    A_i32[1, 1] = 4

    var B_i32 = Matrix[DType.int32](2, 2, 0)
    B_i32[0, 0] = 5
    B_i32[0, 1] = 6
    B_i32[1, 0] = 7
    B_i32[1, 1] = 8

    var C_i32 = gemm(A_i32, B_i32)
    assert_equal(C_i32[0, 0], 19)
    assert_equal(C_i32[0, 1], 22)
    assert_equal(C_i32[1, 0], 43)
    assert_equal(C_i32[1, 1], 50)

    # 2. Outer product: (3x1) @ (1x3) -> (3x3)
    var Col = Matrix[DType.float64](3, 1, 0)
    Col[0, 0] = 1.0
    Col[1, 0] = 2.0
    Col[2, 0] = 3.0

    var Row = Matrix[DType.float64](1, 3, 0)
    Row[0, 0] = 4.0
    Row[0, 1] = 5.0
    Row[0, 2] = 6.0

    var Outer = gemm(Col, Row)
    assert_equal(Outer.rows, 3)
    assert_equal(Outer.cols, 3)
    assert_equal(Outer[0, 0], 4.0)
    assert_equal(Outer[1, 1], 10.0)
    assert_equal(Outer[2, 2], 18.0)

    # 3. Inner product: (1x3) @ (3x1) -> (1x1)
    var Inner = gemm(Row, Col)
    assert_equal(Inner.rows, 1)
    assert_equal(Inner.cols, 1)
    assert_equal(Inner[0, 0], 32.0)  # 4*1 + 5*2 + 6*3 = 4 + 10 + 18 = 32


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

    # Zero bias test
    var y_nobias = dense_dot_vec(A, x, 0.0)
    assert_equal(y_nobias[0], 14.0)
    assert_equal(y_nobias[1], 32.0)

    # Integer dot vec
    var A_i32 = Matrix[DType.int32](2, 2, 3)
    var x_i32: List[Scalar[DType.int32]] = [2, 4]
    var y_i32 = dense_dot_vec(A_i32, x_i32, 5)
    assert_equal(y_i32[0], 23)
    assert_equal(y_i32[1], 23)

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
