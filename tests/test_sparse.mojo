from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_raises,
    assert_almost_equal,
)
from strata import (
    Matrix,
    CSRMatrix,
    CSCMatrix,
    spmv,
    spvm,
    spmm,
    spgemm,
    sddmm,
    DimensionMismatchError,
)


def test_sparse_validation_invariants() raises:
    # 1. Invalid indptr length for 3x3 matrix (needs len=4, given len=3)
    var data: List[Scalar[DType.float64]] = [1.0, 2.0]
    var indices: List[Int] = [0, 1]
    var bad_indptr_len: List[Int] = [0, 1, 2]

    with assert_raises():
        _ = CSRMatrix(3, 3, data.copy(), indices.copy(), bad_indptr_len.copy())

    with assert_raises():
        _ = CSCMatrix(3, 3, data.copy(), indices.copy(), bad_indptr_len.copy())

    # 2. Non-monotonic indptr: indptr[i] > indptr[i+1]
    var non_monotonic_indptr: List[Int] = [0, 2, 1, 2]
    with assert_raises():
        _ = CSRMatrix(
            3, 3, data.copy(), indices.copy(), non_monotonic_indptr.copy()
        )

    # 3. Indptr doesn't start with 0
    var nonzero_start_indptr: List[Int] = [1, 1, 2, 2]
    with assert_raises():
        _ = CSRMatrix(
            3, 3, data.copy(), indices.copy(), nonzero_start_indptr.copy()
        )

    # 4. Out of bounds column index in indices: index >= cols
    var oob_indices: List[Int] = [0, 5]  # cols=3, so 5 is invalid
    var valid_indptr: List[Int] = [0, 1, 2, 2]
    with assert_raises():
        _ = CSRMatrix(
            3, 3, data.copy(), oob_indices.copy(), valid_indptr.copy()
        )

    # 5. Negative rows or cols
    with assert_raises():
        _ = CSRMatrix(-1, 3, data.copy(), indices.copy(), valid_indptr.copy())


def test_sparse_dense_roundtrips() raises:
    # 3x3 sparse matrix
    var dense = Matrix[DType.float64](3, 3, 0)
    dense[0, 0] = 10.0
    dense[1, 2] = 20.0
    dense[2, 1] = 30.0

    var csr = CSRMatrix[DType.float64].from_dense(dense)
    assert_equal(csr.nnz(), 3)
    assert_equal(csr.rows, 3)
    assert_equal(csr.cols, 3)

    var dense_back = csr.to_dense()
    for r in range(3):
        for c in range(3):
            assert_equal(dense_back[r, c], dense[r, c])

    var csc = CSCMatrix[DType.float64].from_dense(dense)
    assert_equal(csc.nnz(), 3)
    var csc_dense_back = csc.to_dense()
    for r in range(3):
        for c in range(3):
            assert_equal(csc_dense_back[r, c], dense[r, c])


def test_csr_csc_cross_conversions() raises:
    var dense = Matrix[DType.float64](4, 3, 0)
    dense[0, 1] = 1.5
    dense[2, 0] = 3.5
    dense[3, 2] = 7.5

    var csr = CSRMatrix[DType.float64].from_dense(dense)
    var csc = csr.to_csc()
    assert_equal(csc.rows, 4)
    assert_equal(csc.cols, 3)
    assert_equal(csc.nnz(), 3)

    var csr_roundtrip = csc.to_csr()
    assert_equal(csr_roundtrip.nnz(), 3)
    var dense_recovered = csr_roundtrip.to_dense()
    for r in range(4):
        for c in range(3):
            assert_equal(dense_recovered[r, c], dense[r, c])


def test_spmv_and_spvm() raises:
    # CSR Matrix:
    # [ 2.0,  0.0,  1.0 ]
    # [ 0.0,  3.0,  0.0 ]
    var dense = Matrix[DType.float64](2, 3, 0)
    dense[0, 0] = 2.0
    dense[0, 2] = 1.0
    dense[1, 1] = 3.0

    var csr = CSRMatrix[DType.float64].from_dense(dense)

    # 1. SpMV: y = A @ x + bias
    var x: List[Scalar[DType.float64]] = [4.0, 5.0, 6.0]
    var y = spmv(csr, x, bias=10.0)
    # y[0] = 2*4 + 1*6 + 10 = 8 + 6 + 10 = 24
    # y[1] = 3*5 + 10 = 15 + 10 = 25
    assert_equal(len(y), 2)
    assert_equal(y[0], 24.0)
    assert_equal(y[1], 25.0)

    # 2. SpVM: y^T = x^T @ A
    var x_left: List[Scalar[DType.float64]] = [2.0, 3.0]
    var y_left = spvm(x_left, csr)
    # y_left[0] = 2*2 + 3*0 = 4
    # y_left[1] = 2*0 + 3*3 = 9
    # y_left[2] = 2*1 + 3*0 = 2
    assert_equal(len(y_left), 3)
    assert_equal(y_left[0], 4.0)
    assert_equal(y_left[1], 9.0)
    assert_equal(y_left[2], 2.0)


def test_spmm_and_spgemm() raises:
    # A = [ [1, 2], [0, 3] ]
    var dense_A = Matrix[DType.float64](2, 2, 0)
    dense_A[0, 0] = 1.0
    dense_A[0, 1] = 2.0
    dense_A[1, 1] = 3.0
    var csr_A = CSRMatrix[DType.float64].from_dense(dense_A)

    # B = [ [4, 5], [6, 7] ]
    var dense_B = Matrix[DType.float64](2, 2, 0)
    dense_B[0, 0] = 4.0
    dense_B[0, 1] = 5.0
    dense_B[1, 0] = 6.0
    dense_B[1, 1] = 7.0
    var csr_B = CSRMatrix[DType.float64].from_dense(dense_B)

    # 1. SpMM: Sparse A @ Dense B
    var C_dense = spmm(csr_A, dense_B)
    assert_equal(C_dense[0, 0], 16.0)  # 1*4 + 2*6 = 16
    assert_equal(C_dense[0, 1], 19.0)  # 1*5 + 2*7 = 19
    assert_equal(C_dense[1, 0], 18.0)  # 0*4 + 3*6 = 18
    assert_equal(C_dense[1, 1], 21.0)  # 0*5 + 3*7 = 21

    # 2. SpGEMM: Sparse A @ Sparse B
    var C_sparse = spgemm(csr_A, csr_B)
    var C_sparse_dense = C_sparse.to_dense()
    assert_equal(C_sparse_dense[0, 0], 16.0)
    assert_equal(C_sparse_dense[0, 1], 19.0)
    assert_equal(C_sparse_dense[1, 0], 18.0)
    assert_equal(C_sparse_dense[1, 1], 21.0)


def test_sddmm() raises:
    # A = 2x3, B = 3x2, S = 2x2 sparsity mask
    var A = Matrix[DType.float64](2, 3, 1.0)
    var B = Matrix[DType.float64](3, 2, 2.0)
    # A @ B = [[6.0, 6.0], [6.0, 6.0]]

    # Pattern mask S with only diagonal non-zeros
    var dense_S = Matrix[DType.float64].zeros(2, 2)
    dense_S[0, 0] = 1.0
    dense_S[1, 1] = 0.5
    var S = CSRMatrix[DType.float64].from_dense(dense_S)

    # P = (A @ B) ⊙ S
    var P = sddmm(A, B, S)
    assert_equal(P.nnz(), 2)
    var P_dense = P.to_dense()
    assert_equal(P_dense[0, 0], 6.0)  # 6.0 * 1.0
    assert_equal(P_dense[0, 1], 0.0)  # Filtered by mask
    assert_equal(P_dense[1, 0], 0.0)  # Filtered by mask
    assert_equal(P_dense[1, 1], 3.0)  # 6.0 * 0.5


def test_sparse_cast() raises:
    var dense_i32 = Matrix[DType.int32](2, 2, 0)
    dense_i32[0, 0] = 5
    dense_i32[1, 1] = 10

    var csr_i32 = CSRMatrix[DType.int32].from_dense(dense_i32)
    var csr_f64 = csr_i32.cast[DType.float64]()

    assert_equal(csr_f64.nnz(), 2)
    assert_equal(csr_f64.data[0], 5.0)
    assert_equal(csr_f64.data[1], 10.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
