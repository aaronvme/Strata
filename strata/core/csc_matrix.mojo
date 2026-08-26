from .types import ArrayLike
from .sparse import SparseMatrix
from .matrix import Matrix
from .csr_matrix import CSRMatrix
from ..exceptions.errors import DimensionMismatchError


struct CSCMatrix[dtype: DType = DType.float64](
    ArrayLike, Copyable, Movable, SparseMatrix, Writable
):
    """Compressed Sparse Column (CSC) 2D matrix format."""

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
    ) raises:
        from ..utils.validation import check_sparse

        check_sparse(
            rows,
            cols,
            data,
            indices,
            indptr,
            is_csr=False,
            caller="CSCMatrix.__init__",
        )
        self.rows = rows
        self.cols = cols
        self.data = data^
        self.indices = indices^
        self.indptr = indptr^

    @staticmethod
    def from_dense(dense: Matrix[Self.dtype]) raises -> Self:
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

    def to_csr(self) raises -> CSRMatrix[Self.dtype]:
        var csr_indptr = List[Int](capacity=self.rows + 1)
        for _ in range(self.rows + 1):
            csr_indptr.append(0)

        for i in range(len(self.indices)):
            var r = self.indices[i]
            csr_indptr[r + 1] += 1

        for r in range(self.rows):
            csr_indptr[r + 1] += csr_indptr[r]

        var csr_data = List[Scalar[Self.dtype]](capacity=len(self.data))
        var csr_indices = List[Int](capacity=len(self.indices))
        for _ in range(len(self.data)):
            csr_data.append(0)
            csr_indices.append(0)

        var next_pos = csr_indptr.copy()
        for c in range(self.cols):
            var start = self.indptr[c]
            var end = self.indptr[c + 1]
            for idx in range(start, end):
                var r = self.indices[idx]
                var dest = next_pos[r]
                next_pos[r] += 1
                csr_data[dest] = self.data[idx]
                csr_indices[dest] = c

        return CSRMatrix[Self.dtype](
            self.rows, self.cols, csr_data^, csr_indices^, csr_indptr^
        )

    def dot_vec(
        self, vec: List[Scalar[Self.dtype]]
    ) raises -> List[Scalar[Self.dtype]]:
        if len(vec) != self.cols:
            raise DimensionMismatchError.error(
                "Vector length " + String(self.cols),
                String(len(vec)),
                "CSCMatrix.dot_vec",
            )
        var res = List[Scalar[Self.dtype]](capacity=self.rows)
        for _ in range(self.rows):
            res.append(0)
        for c in range(self.cols):
            var x_c = vec[c]
            if x_c == 0:
                continue
            var start = self.indptr[c]
            var end = self.indptr[c + 1]
            for idx in range(start, end):
                var r = self.indices[idx]
                res[r] += self.data[idx] * x_c
        return res^

    def dot_dense(self, dense: Matrix[Self.dtype]) raises -> Matrix[Self.dtype]:
        return self.to_csr().dot_dense(dense)

    def num_rows(self) -> Int:
        return self.rows

    def num_cols(self) -> Int:
        return self.cols

    def num_elements(self) -> Int:
        return self.rows * self.cols

    def shape(self) -> Tuple[Int, Int]:
        return (self.rows, self.cols)

    def cast[target_dtype: DType](self) raises -> CSCMatrix[target_dtype]:
        """Promotes or converts the CSCMatrix data elements to target_dtype."""
        var new_data = List[Scalar[target_dtype]](capacity=len(self.data))
        for i in range(len(self.data)):
            new_data.append(Scalar[target_dtype](self.data[i]))
        var indptr_copy = self.indptr.copy()
        var indices_copy = self.indices.copy()
        return CSCMatrix[target_dtype](
            self.rows, self.cols, new_data^, indices_copy^, indptr_copy^
        )

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
