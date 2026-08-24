from .types import ArrayLike
from .sparse import SparseMatrix
from .matrix import Matrix
from ..exceptions.errors import DimensionMismatchError

struct CSCMatrix[dtype: DType = DType.float64](ArrayLike, SparseMatrix, Copyable, Movable, Writable):
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
    def from_dense(dense: Matrix[Self.dtype]) -> Self:
        var data = List[Scalar[Self.dtype]]()
        var indices = List[Int]()
        var indptr = List[Int](capacity=dense.cols + 1)
        indptr.append(0)

        for c in range(dense.cols):
            for r in range(dense.rows):
                var val = dense[r, c]
                if val != 0:
                    data.append(val)
                    indices.append(r)
            indptr.append(len(data))

        return Self(dense.rows, dense.cols, data^, indices^, indptr^)

    def to_dense(self) -> Matrix[Self.dtype]:
        var res = Matrix[Self.dtype](self.rows, self.cols, 0)
        for c in range(self.cols):
            var start = self.indptr[c]
            var end = self.indptr[c + 1]
            for idx in range(start, end):
                var r = self.indices[idx]
                var val = self.data[idx]
                res[r, c] = val
        return res^

    def num_rows(self) -> Int:
        return self.rows

    def num_cols(self) -> Int:
        return self.cols

    def num_elements(self) -> Int:
        return self.rows * self.cols

    def nnz(self) -> Int:
        return len(self.data)

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "CSCMatrix[",
            String(Self.dtype),
            "](",
            self.rows,
            "x",
            self.cols,
            ", nnz=",
            len(self.data),
            ")",
        )
