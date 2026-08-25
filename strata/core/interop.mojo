from std.python import Python, PythonObject
from .matrix import Matrix
from .csr_matrix import CSRMatrix
from ..exceptions.errors import DataConversionError

def matrix_to_numpy[dtype: DType](matrix: Matrix[dtype]) raises -> PythonObject:
    """Converts a Strata Matrix[dtype] to a NumPy 2D array."""
    var np = Python.import_module("numpy")
    var np_dtype = "float64"
    comptime if dtype == DType.float32:
        np_dtype = "float32"
    elif dtype == DType.int32:
        np_dtype = "int32"
    elif dtype == DType.int64:
        np_dtype = "int64"

    var arr = np.zeros(Python.tuple(matrix.rows, matrix.cols), dtype=np_dtype)
    for r in range(matrix.rows):
        for c in range(matrix.cols):
            arr[r, c] = Float64(matrix[r, c])
    return arr

def matrix_from_numpy[dtype: DType = DType.float64](np_arr: PythonObject) raises -> Matrix[dtype]:
    """Converts a 2D NumPy ndarray to a Strata Matrix[dtype]."""
    var shape = np_arr.shape
    var rows = Int(py=shape[0])
    var cols = Int(py=shape[1])

    var res = Matrix[dtype](rows, cols, 0)
    for r in range(rows):
        for c in range(cols):
            res[r, c] = Scalar[dtype](Float64(py=np_arr[r, c]))
    return res^

def csr_to_scipy[dtype: DType](csr: CSRMatrix[dtype]) raises -> PythonObject:
    """Converts a Strata CSRMatrix[dtype] to a scipy.sparse.csr_matrix."""
    var sp = Python.import_module("scipy.sparse")
    var np = Python.import_module("numpy")

    var py_data = Python.list()
    for i in range(len(csr.data)):
        py_data.append(Float64(csr.data[i]))

    var py_indices = Python.list()
    for i in range(len(csr.indices)):
        py_indices.append(csr.indices[i])

    var py_indptr = Python.list()
    for i in range(len(csr.indptr)):
        py_indptr.append(csr.indptr[i])

    var data_arr = np.array(py_data, dtype="float64")
    var indices_arr = np.array(py_indices, dtype="int32")
    var indptr_arr = np.array(py_indptr, dtype="int32")

    var tuple_spec = Python.tuple(data_arr, indices_arr, indptr_arr)
    var shape_spec = Python.tuple(csr.rows, csr.cols)
    return sp.csr_matrix(tuple_spec, shape=shape_spec)

def csr_from_scipy[dtype: DType = DType.float64](sp_arr: PythonObject) raises -> CSRMatrix[dtype]:
    """Converts a scipy.sparse.csr_matrix to a Strata CSRMatrix[dtype]."""
    var data_py = sp_arr.data
    var indices_py = sp_arr.indices
    var indptr_py = sp_arr.indptr
    var shape = sp_arr.shape
    var rows = Int(py=shape[0])
    var cols = Int(py=shape[1])

    var nnz = Int(py=len(data_py))
    var data = List[Scalar[dtype]](capacity=nnz)
    var indices = List[Int](capacity=nnz)
    for i in range(nnz):
        data.append(Scalar[dtype](Float64(py=data_py[i])))
        indices.append(Int(py=indices_py[i]))

    var indptr_len = Int(py=len(indptr_py))
    var indptr = List[Int](capacity=indptr_len)
    for i in range(indptr_len):
        indptr.append(Int(py=indptr_py[i]))

    return CSRMatrix[dtype](rows, cols, data^, indices^, indptr^)
