from std.testing import TestSuite, assert_equal, assert_true
from std.python import Python
from strata import Matrix, CSRMatrix


def test_numpy_roundtrip_f64() raises:
    var m = Matrix[DType.float64](2, 4, 0)
    m[0, 0] = 1.5
    m[0, 3] = 4.5
    m[1, 1] = 2.5
    m[1, 2] = 3.5

    var np_arr = m.to_numpy()
    assert_equal(Float64(py=np_arr[0, 0]), 1.5)
    assert_equal(Float64(py=np_arr[0, 3]), 4.5)
    assert_equal(Float64(py=np_arr[1, 1]), 2.5)

    var m_back = Matrix[DType.float64].from_numpy(np_arr)
    assert_equal(m_back.rows, 2)
    assert_equal(m_back.cols, 4)
    assert_equal(m_back[0, 0], 1.5)
    assert_equal(m_back[0, 3], 4.5)
    assert_equal(m_back[1, 1], 2.5)
    assert_equal(m_back[1, 2], 3.5)


def test_numpy_roundtrip_f32() raises:
    var m = Matrix[DType.float32](2, 2, 0)
    m[0, 0] = 10.0
    m[1, 1] = 20.0

    var np_arr = m.to_numpy()
    var m_back = Matrix[DType.float32].from_numpy(np_arr)
    assert_equal(m_back.rows, 2)
    assert_equal(m_back.cols, 2)
    assert_equal(m_back[0, 0], 10.0)
    assert_equal(m_back[1, 1], 20.0)


def test_scipy_csr_roundtrip() raises:
    var data = List[Scalar[DType.float64]](capacity=3)
    data.append(10.0)
    data.append(20.0)
    data.append(30.0)

    var indices = List[Int](capacity=3)
    indices.append(0)
    indices.append(2)
    indices.append(1)

    var indptr = List[Int](capacity=3)
    indptr.append(0)
    indptr.append(2)
    indptr.append(3)

    var csr = CSRMatrix[DType.float64](2, 3, data^, indices^, indptr^)

    var sp_csr = csr.to_scipy()
    var shape = sp_csr.shape
    assert_equal(Int(py=shape[0]), 2)
    assert_equal(Int(py=shape[1]), 3)

    var csr_back = CSRMatrix[DType.float64].from_scipy(sp_csr)
    assert_equal(csr_back.rows, 2)
    assert_equal(csr_back.cols, 3)
    assert_equal(csr_back.nnz(), 3)
    assert_equal(csr_back.data[0], 10.0)
    assert_equal(csr_back.data[1], 20.0)
    assert_equal(csr_back.data[2], 30.0)


def test_numpy_roundtrip_int32() raises:
    var m = Matrix[DType.int32](2, 3, 0)
    m[0, 0] = 10
    m[0, 2] = -30
    m[1, 1] = 50

    var np_arr = m.to_numpy()
    var m_back = Matrix[DType.int32].from_numpy(np_arr)
    assert_equal(m_back.rows, 2)
    assert_equal(m_back.cols, 3)
    assert_equal(m_back[0, 0], 10)
    assert_equal(m_back[0, 2], -30)
    assert_equal(m_back[1, 1], 50)


def test_numpy_roundtrip_int64() raises:
    var m = Matrix[DType.int64](2, 2, 0)
    m[0, 0] = 1000000
    m[1, 1] = -2000000

    var np_arr = m.to_numpy()
    var m_back = Matrix[DType.int64].from_numpy(np_arr)
    assert_equal(m_back[0, 0], 1000000)
    assert_equal(m_back[1, 1], -2000000)


def test_numpy_roundtrip_1x1() raises:
    var m = Matrix[DType.float64](1, 1, 42.0)
    var np_arr = m.to_numpy()
    var m_back = Matrix[DType.float64].from_numpy(np_arr)
    assert_equal(m_back.rows, 1)
    assert_equal(m_back.cols, 1)
    assert_equal(m_back[0, 0], 42.0)


def test_numpy_roundtrip_tall_matrix() raises:
    var N = 50
    var m = Matrix[DType.float64](N, 2, 0)
    for i in range(N):
        m[i, 0] = Float64(i)
        m[i, 1] = Float64(i * 10)

    var np_arr = m.to_numpy()
    var m_back = Matrix[DType.float64].from_numpy(np_arr)
    assert_equal(m_back.rows, N)
    assert_equal(m_back.cols, 2)
    assert_equal(m_back[49, 1], 490.0)


def test_numpy_roundtrip_wide_matrix() raises:
    var D = 50
    var m = Matrix[DType.float64](2, D, 0)
    for j in range(D):
        m[0, j] = Float64(j)
        m[1, j] = Float64(j * 2)

    var np_arr = m.to_numpy()
    var m_back = Matrix[DType.float64].from_numpy(np_arr)
    assert_equal(m_back.rows, 2)
    assert_equal(m_back.cols, D)
    assert_equal(m_back[1, 49], 98.0)


def test_numpy_zeros_matrix() raises:
    var z = Matrix[DType.float64].zeros(4, 4)
    var np_arr = z.to_numpy()
    var z_back = Matrix[DType.float64].from_numpy(np_arr)
    for r in range(4):
        for c in range(4):
            assert_equal(z_back[r, c], 0.0)


def test_numpy_ones_matrix() raises:
    var o = Matrix[DType.float64].ones(3, 3)
    var np_arr = o.to_numpy()
    var o_back = Matrix[DType.float64].from_numpy(np_arr)
    for r in range(3):
        for c in range(3):
            assert_equal(o_back[r, c], 1.0)


def test_numpy_eye_matrix() raises:
    var Eye = Matrix[DType.float64].eye(3)
    var np_arr = Eye.to_numpy()
    var Eye_back = Matrix[DType.float64].from_numpy(np_arr)
    for r in range(3):
        for c in range(3):
            assert_equal(Eye_back[r, c], 1.0 if r == c else 0.0)


def test_numpy_negative_values() raises:
    var m = Matrix[DType.float64](2, 2, 0)
    m[0, 0] = -1.5
    m[0, 1] = -2.5
    m[1, 0] = -3.5
    m[1, 1] = -4.5

    var np_arr = m.to_numpy()
    var m_back = Matrix[DType.float64].from_numpy(np_arr)
    assert_equal(m_back[0, 0], -1.5)
    assert_equal(m_back[1, 1], -4.5)


def test_scipy_csr_empty_matrix() raises:
    var empty_csr = CSRMatrix[DType.float64].empty(4, 4)
    var sp_csr = empty_csr.to_scipy()
    assert_equal(Int(py=sp_csr.nnz), 0)

    var csr_back = CSRMatrix[DType.float64].from_scipy(sp_csr)
    assert_equal(csr_back.rows, 4)
    assert_equal(csr_back.cols, 4)
    assert_equal(csr_back.nnz(), 0)


def test_scipy_csr_diagonal_matrix() raises:
    var dense = Matrix[DType.float64](3, 3, 0)
    dense[0, 0] = 5.0
    dense[1, 1] = 10.0
    dense[2, 2] = 15.0
    var csr = CSRMatrix[DType.float64].from_dense(dense)

    var sp_csr = csr.to_scipy()
    assert_equal(Int(py=sp_csr.nnz), 3)

    var csr_back = CSRMatrix[DType.float64].from_scipy(sp_csr)
    assert_equal(csr_back.nnz(), 3)
    assert_equal(csr_back.data[0], 5.0)
    assert_equal(csr_back.data[1], 10.0)
    assert_equal(csr_back.data[2], 15.0)


def test_scipy_csr_float32_roundtrip() raises:
    var dense = Matrix[DType.float32](2, 2, 0)
    dense[0, 1] = 7.5
    dense[1, 0] = 3.5
    var csr = CSRMatrix[DType.float32].from_dense(dense)

    var sp_csr = csr.to_scipy()
    var csr_back = CSRMatrix[DType.float32].from_scipy(sp_csr)
    assert_equal(csr_back.nnz(), 2)
    assert_equal(csr_back.data[0], 7.5)
    assert_equal(csr_back.data[1], 3.5)


def test_scipy_csr_rectangular_tall() raises:
    var dense = Matrix[DType.float64](20, 3, 0)
    dense[0, 0] = 1.0
    dense[10, 1] = 2.0
    dense[19, 2] = 3.0
    var csr = CSRMatrix[DType.float64].from_dense(dense)

    var sp_csr = csr.to_scipy()
    var csr_back = CSRMatrix[DType.float64].from_scipy(sp_csr)
    assert_equal(csr_back.rows, 20)
    assert_equal(csr_back.cols, 3)
    assert_equal(csr_back.nnz(), 3)


def test_scipy_csr_rectangular_wide() raises:
    var dense = Matrix[DType.float64](3, 20, 0)
    dense[0, 0] = 1.0
    dense[1, 10] = 2.0
    dense[2, 19] = 3.0
    var csr = CSRMatrix[DType.float64].from_dense(dense)

    var sp_csr = csr.to_scipy()
    var csr_back = CSRMatrix[DType.float64].from_scipy(sp_csr)
    assert_equal(csr_back.rows, 3)
    assert_equal(csr_back.cols, 20)
    assert_equal(csr_back.nnz(), 3)


def test_scipy_csr_single_element() raises:
    var dense = Matrix[DType.float64](1, 1, 0)
    dense[0, 0] = 99.0
    var csr = CSRMatrix[DType.float64].from_dense(dense)

    var sp_csr = csr.to_scipy()
    var csr_back = CSRMatrix[DType.float64].from_scipy(sp_csr)
    assert_equal(csr_back.rows, 1)
    assert_equal(csr_back.cols, 1)
    assert_equal(csr_back.nnz(), 1)
    assert_equal(csr_back.data[0], 99.0)


def test_numpy_roundtrip_large_matrix() raises:
    var N = 30
    var m = Matrix[DType.float64](N, N, 0)
    for r in range(N):
        for c in range(N):
            m[r, c] = Float64(r * N + c)

    var np_arr = m.to_numpy()
    var m_back = Matrix[DType.float64].from_numpy(np_arr)
    assert_equal(m_back.rows, N)
    assert_equal(m_back.cols, N)
    assert_equal(m_back[29, 29], Float64(N * N - 1))


def test_scipy_csr_all_dense_entries() raises:
    var dense = Matrix[DType.float64](2, 2, 5.0)
    var csr = CSRMatrix[DType.float64].from_dense(dense)

    var sp_csr = csr.to_scipy()
    assert_equal(Int(py=sp_csr.nnz), 4)

    var csr_back = CSRMatrix[DType.float64].from_scipy(sp_csr)
    assert_equal(csr_back.nnz(), 4)


def test_numpy_matrix_row_slices() raises:
    var m = Matrix[DType.float64](3, 3, 0)
    m[1, 0] = 100.0
    m[1, 1] = 200.0
    m[1, 2] = 300.0

    var np_arr = m.to_numpy()
    var row1 = np_arr[1]
    assert_equal(Float64(py=row1[0]), 100.0)
    assert_equal(Float64(py=row1[1]), 200.0)
    assert_equal(Float64(py=row1[2]), 300.0)


def test_numpy_direct_function_calls() raises:
    from strata.core.interop import matrix_to_numpy, matrix_from_numpy

    var m = Matrix[DType.float64](2, 2, 7.0)
    var py_arr = matrix_to_numpy(m)
    var m_back = matrix_from_numpy[DType.float64](py_arr)
    assert_equal(m_back[0, 0], 7.0)
    assert_equal(m_back[1, 1], 7.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
