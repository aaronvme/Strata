from std.python import PythonObject
from .types import ArrayLike
from .sparse import SparseMatrix
from .matrix import Matrix
from ..exceptions.errors import DimensionMismatchError

struct CSRMatrix[dtype: DType = DType.float64](ArrayLike, SparseMatrix, Copyable, Movable, Writable):
    var rows: Int
    var cols: Int
    var data: List[Scalar[Self.dtype]]
    var indices: List[Int]
    var indptr: List[Int]

    def __init__(
        out self,
        rows: Int,
        cols: Int,
        var data: List[Scalar[Self.dtype]],
        var indices: List[Int],
        var indptr: List[Int],
    ):
        self.rows = rows
        self.cols = cols
        self.data = data^
        self.indices = indices^
        self.indptr = indptr^

    @staticmethod
    def empty(rows: Int, cols: Int) -> Self:
        var indptr = List[Int](capacity=rows + 1)
        for _ in range(rows + 1):
            indptr.append(0)
        return Self(rows, cols, List[Scalar[Self.dtype]](), List[Int](), indptr^)

    @staticmethod
    def from_scipy(sp_arr: PythonObject) raises -> Self:
        from .interop import csr_from_scipy
        return csr_from_scipy[Self.dtype](sp_arr)

    def to_scipy(self) raises -> PythonObject:
        from .interop import csr_to_scipy
        return csr_to_scipy[Self.dtype](self)

    @staticmethod
    def from_dense(dense: Matrix[Self.dtype]) -> Self:
        var data = List[Scalar[Self.dtype]]()
        var indices = List[Int]()
        var indptr = List[Int](capacity=dense.rows + 1)
        indptr.append(0)

        for r in range(dense.rows):
            for c in range(dense.cols):
                var val = dense[r, c]
                if val != 0:
                    data.append(val)
                    indices.append(c)
            indptr.append(len(data))

        return Self(dense.rows, dense.cols, data^, indices^, indptr^)

    def to_dense(self) -> Matrix[Self.dtype]:
        var res = Matrix[Self.dtype](self.rows, self.cols, 0)
        for r in range(self.rows):
            var start = self.indptr[r]
            var end = self.indptr[r + 1]
            for idx in range(start, end):
                var c = self.indices[idx]
                var val = self.data[idx]
                res[r, c] = val
        return res^

    def num_rows(self) -> Int:
        return self.rows

    def num_cols(self) -> Int:
        return self.cols

    def num_elements(self) -> Int:
        return self.rows * self.cols

    def shape(self) -> Tuple[Int, Int]:
        return (self.rows, self.cols)

    def nnz(self) -> Int:
        return len(self.data)

    def dot_vec(self, vec: List[Scalar[Self.dtype]]) raises -> List[Scalar[Self.dtype]]:
        from .sparse_ops import spmv
        return spmv[Self.dtype](self, vec)

    def dot_dense(self, other: Matrix[Self.dtype]) raises -> Matrix[Self.dtype]:
        from .sparse_ops import spmm
        return spmm[Self.dtype](self, other)

    def dot_sparse(self, other: Self) raises -> Self:
        from .sparse_ops import spgemm
        return spgemm[Self.dtype](self, other)

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "CSRMatrix[",
            String(Self.dtype),
            "](",
            self.rows,
            "x",
            self.cols,
            ", nnz=",
            len(self.data),
            ")",
        )
