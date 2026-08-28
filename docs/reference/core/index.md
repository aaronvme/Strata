# `strata.core`

Fundamental 2D dense Matrix, MatrixView, CSRMatrix, CSCMatrix sparse representations, Dataset containers, and hardware-accelerated BLAS/LAPACK solvers.

---

## Structs & Classes

| Struct | Description |
| :--- | :--- |
| [`Matrix`](Matrix.md) | Dense 2D row-major matrix container with striding and view support. |
| [`MatrixView`](MatrixView.md) | Non-owning 2D view over a contiguous or strided matrix memory buffer. |
| [`DatasetSplit`](DatasetSplit.md) | Container holding train and test partitions of a Dataset. |
| [`Dataset`](Dataset.md) | Machine learning dataset container pairing a feature matrix with targets. |
| [`SVDResult`](SVDResult.md) | Result of Singular Value Decomposition ($A = U \Sigma V^T$). |
| [`QRResult`](QRResult.md) | Result of QR Decomposition ($A = Q R$). |
| [`EigResult`](EigResult.md) | Result of Symmetric Eigenvalue Decomposition ($A V = V \Lambda$). |

## Functions

| Function | Description |
| :--- | :--- |
| [`matrix_to_numpy`](matrix_to_numpy.md) | Converts a Strata Matrix[dtype] to a NumPy 2D array. |
| [`matrix_from_numpy`](matrix_from_numpy.md) | Converts a 2D NumPy ndarray to a Strata Matrix[dtype]. |
| [`csr_to_scipy`](csr_to_scipy.md) | Converts a Strata CSRMatrix[dtype] to a scipy.sparse.csr_matrix. |
| [`csr_from_scipy`](csr_from_scipy.md) | Converts a scipy.sparse.csr_matrix to a Strata CSRMatrix[dtype]. |
