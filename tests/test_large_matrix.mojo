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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
