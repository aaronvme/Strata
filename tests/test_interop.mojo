from std.testing import TestSuite, assert_equal, assert_true
from std.python import Python
from strata import Matrix, CSRMatrix


def test_numpy_roundtrip() raises:
    var m = Matrix[DType.float64](3, 3, 0)
    m[0, 0] = 1.5
    m[1, 1] = 2.5
    m[2, 2] = 3.5

    var np_arr = m.to_numpy()
    assert_equal(Float64(py=np_arr[0, 0]), 1.5)
    assert_equal(Float64(py=np_arr[1, 1]), 2.5)
    assert_equal(Float64(py=np_arr[2, 2]), 3.5)

    var m_back = Matrix[DType.float64].from_numpy(np_arr)
    assert_equal(m_back.rows, 3)
    assert_equal(m_back.cols, 3)
    assert_equal(m_back[0, 0], 1.5)
    assert_equal(m_back[1, 1], 2.5)
    assert_equal(m_back[2, 2], 3.5)


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
