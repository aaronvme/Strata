from std.testing import TestSuite, assert_equal, assert_true
from strata import (
    Matrix,
    CSRMatrix,
    CSCMatrix,
    MatrixView,
    gemm,
    dense_dot_vec,
    spmv,
    spvm,
    spmm,
    spgemm,
    sddmm,
)


def test_large_dense_gemm() raises:
    var size = 256
    var A = Matrix[DType.float64].ones(size, size)
    var B = Matrix[DType.float64](size, size, 2.0)

    var C = gemm(A, B)
    assert_equal(C.rows, size)
    assert_equal(C.cols, size)
    assert_equal(C[0, 0], 512.0)
    assert_equal(C[127, 127], 512.0)
    assert_equal(C[255, 255], 512.0)


def test_large_dense_dot_vec() raises:
    var rows = 2000
    var cols = 500
    var A = Matrix[DType.float64].ones(rows, cols)

    var x = List[Scalar[DType.float64]](capacity=cols)
    for _ in range(cols):
        x.append(3.5)

    var y = dense_dot_vec(A, x)
    assert_equal(len(y), rows)
    assert_equal(y[0], 1750.0)
    assert_equal(y[999], 1750.0)
    assert_equal(y[1999], 1750.0)


def test_large_sparse_spmv() raises:
    var dim = 10000
    var data = List[Scalar[DType.float64]](capacity=dim * 3)
    var indices = List[Int](capacity=dim * 3)
    var indptr = List[Int](capacity=dim + 1)
    indptr.append(0)

    for r in range(dim):
        data.append(2.0)
        indices.append(r)
        if r + 1 < dim:
            data.append(1.0)
            indices.append(r + 1)
        indptr.append(len(data))

    var A = CSRMatrix[DType.float64](dim, dim, data^, indices^, indptr^)
    assert_equal(A.rows, dim)
    assert_equal(A.cols, dim)

    var x = List[Scalar[DType.float64]](capacity=dim)
    for _ in range(dim):
        x.append(1.0)

    var y = spmv[DType.float64](A, x)
    assert_equal(len(y), dim)
    assert_equal(y[0], 3.0)
    assert_equal(y[5000], 3.0)
    assert_equal(y[dim - 1], 2.0)


def test_large_sparse_spmm() raises:
    var dim = 1000
    var embed_dim = 128

    var data = List[Scalar[DType.float64]](capacity=dim * 5)
    var indices = List[Int](capacity=dim * 5)
    var indptr = List[Int](capacity=dim + 1)
    indptr.append(0)

    for r in range(dim):
        for offset in range(5):
            var c = (r + offset) % dim
            data.append(2.0)
            indices.append(c)
        indptr.append(len(data))

    var A = CSRMatrix[DType.float64](dim, dim, data^, indices^, indptr^)
    var B = Matrix[DType.float64].ones(dim, embed_dim)

    var C = spmm[DType.float64](A, B)
    assert_equal(C.rows, dim)
    assert_equal(C.cols, embed_dim)
    assert_equal(C[0, 0], 10.0)
    assert_equal(C[500, 64], 10.0)
    assert_equal(C[999, 127], 10.0)


def test_large_matrix_view_slicing() raises:
    var rows = 10000
    var cols = 200
    var X = Matrix[DType.float64](rows, cols, 0)

    for r in range(rows):
        X[r, r % cols] = Float64(r)

    var X_train = X.slice_rows(0, 8000)
    var X_test = X.slice_rows(8000, 10000)

    assert_equal(X_train.rows, 8000)
    assert_equal(X_train.cols, 200)
    assert_equal(X_test.rows, 2000)
    assert_equal(X_test.cols, 200)

    var X_test_sub = X_test.slice_cols(0, 50)
    assert_equal(X_test_sub.rows, 2000)
    assert_equal(X_test_sub.cols, 50)
    assert_equal(X_test_sub[0, 0], 8000.0)

    var window = X.slice_2d(5000, 6000, 50, 100)
    assert_equal(window.rows, 1000)
    assert_equal(window.cols, 50)

    var materialized = window.to_matrix()
    assert_equal(materialized.rows, 1000)
    assert_equal(materialized.cols, 50)


def test_large_matrix_transpose() raises:
    var rows = 1000
    var cols = 20
    var A = Matrix[DType.float64](rows, cols, 0)
    for r in range(rows):
        A[r, 0] = Float64(r)
        A[r, cols - 1] = Float64(r * 2)

    var At = A.transpose()
    assert_equal(At.rows, cols)
    assert_equal(At.cols, rows)
    assert_equal(At[0, 500], 500.0)
    assert_equal(At[cols - 1, 500], 1000.0)


def test_large_matrix_elementwise_add_sub() raises:
    var size = 300
    var A = Matrix[DType.float64](size, size, 5.0)
    var B = Matrix[DType.float64](size, size, 2.0)

    var C_add = A + B
    var C_sub = A - B
    assert_equal(C_add[0, 0], 7.0)
    assert_equal(C_add[size - 1, size - 1], 7.0)
    assert_equal(C_sub[0, 0], 3.0)
    assert_equal(C_sub[size - 1, size - 1], 3.0)


def test_large_matrix_scalar_multiplication() raises:
    var A = Matrix[DType.float64](500, 50, 4.0)
    var B = A * 2.5
    assert_equal(B.rows, 500)
    assert_equal(B.cols, 50)
    assert_equal(B[0, 0], 10.0)
    assert_equal(B[499, 49], 10.0)


def test_large_matrix_column_means() raises:
    var rows = 1000
    var cols = 10
    var A = Matrix[DType.float64](rows, cols, 0)
    for r in range(rows):
        for c in range(cols):
            A[r, c] = Float64(c + 1)

    var means = A.mean_along_axis_0()
    assert_equal(len(means), cols)
    for c in range(cols):
        assert_equal(means[c], Float64(c + 1))


def test_large_matrix_cast_f64_to_f32() raises:
    var rows = 500
    var cols = 50
    var A_f64 = Matrix[DType.float64](rows, cols, 3.14)
    var A_f32 = A_f64.cast[DType.float32]()

    assert_equal(A_f32.rows, rows)
    assert_equal(A_f32.cols, cols)
    assert_equal(A_f32[0, 0], Float32(3.14))


def test_large_matrix_float32_gemm() raises:
    var size = 128
    var A = Matrix[DType.float32].ones(size, size)
    var B = Matrix[DType.float32](size, size, 2.0)
    var C = gemm(A, B)

    assert_equal(C.rows, size)
    assert_equal(C.cols, size)
    assert_equal(C[0, 0], Float32(256.0))


def test_large_sparse_spgemm_chain() raises:
    var dim = 200
    var dense_A = Matrix[DType.float64](dim, dim, 0)
    var dense_B = Matrix[DType.float64](dim, dim, 0)
    for i in range(dim):
        dense_A[i, i] = 2.0
        dense_B[i, i] = 3.0

    var csr_A = CSRMatrix[DType.float64].from_dense(dense_A)
    var csr_B = CSRMatrix[DType.float64].from_dense(dense_B)
    var csr_C = spgemm(csr_A, csr_B)

    assert_equal(csr_C.rows, dim)
    assert_equal(csr_C.cols, dim)
    assert_equal(csr_C.nnz(), dim)
    assert_equal(csr_C.data[0], 6.0)


def test_large_sparse_sddmm_pattern() raises:
    var dim = 100
    var A = Matrix[DType.float64](dim, dim, 1.0)
    var B = Matrix[DType.float64](dim, dim, 2.0)

    # Sparse mask S with diagonal non-zeros
    var dense_S = Matrix[DType.float64].zeros(dim, dim)
    for i in range(dim):
        dense_S[i, i] = 1.0
    var S = CSRMatrix[DType.float64].from_dense(dense_S)

    var P = sddmm(A, B, S)
    assert_equal(P.nnz(), dim)
    var P_dense = P.to_dense()
    assert_equal(P_dense[0, 0], 200.0)
    assert_equal(P_dense[0, 1], 0.0)


def test_large_sparse_spvm_transpose() raises:
    var dim = 500
    var dense = Matrix[DType.float64].eye(dim)
    var csr = CSRMatrix[DType.float64].from_dense(dense)

    var x = List[Scalar[DType.float64]](capacity=dim)
    for i in range(dim):
        x.append(Float64(i))

    var y = spvm(x, csr)
    assert_equal(len(y), dim)
    assert_equal(y[0], 0.0)
    assert_equal(y[499], 499.0)


def test_large_sparse_to_dense_roundtrip() raises:
    var dim = 50
    var dense = Matrix[DType.float64](dim, dim, 0)
    dense[0, 0] = 10.0
    dense[25, 25] = 20.0
    dense[49, 49] = 30.0

    var csr = CSRMatrix[DType.float64].from_dense(dense)
    var dense_back = csr.to_dense()
    assert_equal(dense_back.rows, dim)
    assert_equal(dense_back.cols, dim)
    assert_equal(dense_back[0, 0], 10.0)
    assert_equal(dense_back[25, 25], 20.0)
    assert_equal(dense_back[49, 49], 30.0)


def test_large_matrix_hadamard_product() raises:
    var size = 150
    var A = Matrix[DType.float64](size, size, 3.0)
    var B = Matrix[DType.float64](size, size, 4.0)
    var C = A * B
    assert_equal(C.rows, size)
    assert_equal(C.cols, size)
    assert_equal(C[0, 0], 12.0)
    assert_equal(C[149, 149], 12.0)


def test_large_matrix_zeros_factory() raises:
    var size = 400
    var z = Matrix[DType.float64].zeros(size, size)
    assert_equal(z.rows, size)
    assert_equal(z.cols, size)
    assert_equal(z[0, 0], 0.0)
    assert_equal(z[size - 1, size - 1], 0.0)


def test_large_matrix_ones_factory() raises:
    var size = 300
    var o = Matrix[DType.float64].ones(size, size)
    assert_equal(o.rows, size)
    assert_equal(o.cols, size)
    assert_equal(o[0, 0], 1.0)
    assert_equal(o[size - 1, size - 1], 1.0)


def test_large_matrix_eye_factory() raises:
    var size = 250
    var Eye = Matrix[DType.float64].eye(size)
    assert_equal(Eye.rows, size)
    assert_equal(Eye.cols, size)
    assert_equal(Eye[0, 0], 1.0)
    assert_equal(Eye[100, 100], 1.0)
    assert_equal(Eye[0, 100], 0.0)


def test_large_matrix_copy_independence() raises:
    var size = 200
    var A = Matrix[DType.float64](size, size, 1.0)
    var B = A.copy()
    B[0, 0] = 999.0
    assert_equal(A[0, 0], 1.0)
    assert_equal(B[0, 0], 999.0)


def test_large_sparse_empty_matrix() raises:
    var dim = 5000
    var empty_csr = CSRMatrix[DType.float64].empty(dim, dim)
    assert_equal(empty_csr.rows, dim)
    assert_equal(empty_csr.cols, dim)
    assert_equal(empty_csr.nnz(), 0)


def test_large_dense_dot_vec_with_bias() raises:
    var rows = 1000
    var cols = 100
    var A = Matrix[DType.float64].ones(rows, cols)
    var x = List[Scalar[DType.float64]](capacity=cols)
    for _ in range(cols):
        x.append(1.0)

    var y = dense_dot_vec(A, x, bias=50.0)
    assert_equal(len(y), rows)
    assert_equal(y[0], 150.0)
    assert_equal(y[rows - 1], 150.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
