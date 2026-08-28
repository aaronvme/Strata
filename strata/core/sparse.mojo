from .matrix import Matrix
from .csr_matrix import CSRMatrix
from .csc_matrix import CSCMatrix


trait SparseMatrix:
    """Base interface trait for 2D sparse matrix representations.

    Provides common dimension queries (`num_rows`, `num_cols`) and structural
    sparsity counts (`nnz`) for Compressed Sparse Row (CSR) and Compressed Sparse Column (CSC) formats.
    """

    def num_rows(self) -> Int:
        ...

    def num_cols(self) -> Int:
        ...

    def nnz(self) -> Int:
        ...
