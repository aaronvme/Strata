from std.testing import assert_equal, TestSuite
from strata import (
    Matrix,
    CSRMatrix,
    gemm,
    spmv,
    spmm,
)

def test_dense_matrix_gemm() raises:
    var A = Matrix[DType.float32](2, 2, 0)
    A[0, 0] = 1.0
    A[0, 1] = 2.0
    A[1, 0] = 3.0
    A[1, 1] = 4.0

    var B = Matrix[DType.float32](2, 2, 0)
    B[0, 0] = 5.0
    B[0, 1] = 6.0
    B[1, 0] = 7.0
    B[1, 1] = 8.0

    var C = gemm[DType.float32](A, B)
    assert_equal(C[0, 0], 19.0)
    assert_equal(C[0, 1], 22.0)
    assert_equal(C[1, 0], 43.0)
    assert_equal(C[1, 1], 50.0)

def test_sparse_csr_spmv_spmm() raises:
    var dense = Matrix[DType.float64](3, 3, 0)
    dense[0, 0] = 10.0
    dense[1, 2] = 20.0
    dense[2, 1] = 30.0

    var csr = CSRMatrix[DType.float64].from_dense(dense)
    assert_equal(csr.nnz(), 3)

    var x: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0]
    var y = spmv[DType.float64](csr, x)
    assert_equal(y[0], 10.0)
    assert_equal(y[1], 60.0)
    assert_equal(y[2], 60.0)

    var B = Matrix[DType.float64].ones(3, 2)
    var C = spmm[DType.float64](csr, B)
    assert_equal(C[0, 0], 10.0)
    assert_equal(C[1, 0], 20.0)
    assert_equal(C[2, 0], 30.0)

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
