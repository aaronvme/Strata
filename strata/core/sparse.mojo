from .matrix import Matrix
from .csr_matrix import CSRMatrix
from .csc_matrix import CSCMatrix


trait SparseMatrix:
    def num_rows(self) -> Int:
        ...

    def num_cols(self) -> Int:
        ...

    def nnz(self) -> Int:
        ...
