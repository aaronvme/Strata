# Strata

Strata is a native machine learning and linear algebra library written in [Mojo](https://docs.modular.com/mojo/). It provides scikit-learn-compatible estimators, dense and sparse matrix containers, and composable pipelines with native SIMD and LAPACK acceleration.

---

## Installation

Strata uses [Pixi](https://pixi.sh/) for environment and dependency management.

```bash
git clone https://github.com/ethqnol/Strata.git
cd Strata
pixi install
```

---

## Quick Start

### Linear Algebra & Matrix Operations

```mojo
from strata import Matrix, gemm, svd, eigh, solve

def main() raises:
    # Dense matrices
    var A = Matrix[DType.float64](2, 2, 0)
    A[0, 0] = 4.0; A[0, 1] = 2.0
    A[1, 0] = 2.0; A[1, 1] = 3.0

    var B = Matrix[DType.float64].eye(2)

    # Matrix multiplication
    var C = gemm(A, B)

    # Factorizations (LAPACK)
    var svd_res = svd(A)
    var eig_res = eigh(A)

    # Linear system solve: A * x = b
    var b: List[Scalar[DType.float64]] = [8.0, 7.0]
    var x = solve(A, b)
```

### Pipelines & Estimators

```mojo
from strata import (
    Matrix,
    StandardScaler,
    PCA,
    LinearRegression,
    PipelineTransformer,
    PipelineRegressor,
)

def main() raises:
    var X = Matrix[DType.float64](5, 3, 0)
    for r in range(5):
        X[r, 0] = Float64(r + 1)
        X[r, 1] = Float64((r + 1) * 2)
        X[r, 2] = Float64((r + 1) * 3)

    var y: List[Scalar[DType.float64]] = [2.0, 4.0, 6.0, 8.0, 10.0]

    # Build a StandardScaler -> PCA -> LinearRegression pipeline
    var scaler = StandardScaler()
    var pca = PCA(n_components=2)
    var prep = PipelineTransformer(scaler^, pca^)
    var reg = LinearRegression(solver="cholesky")
    var pipe = PipelineRegressor(prep^, reg^)

    pipe.fit(X, y)
    var preds = pipe.predict(X)
```

---

## Implemented Modules

- **`strata.core`**:
  - `Matrix[dtype]`: Dense 2D row-major matrix.
  - `MatrixView[dtype, origin]`: Zero-copy strided 2D view.
  - `CSRMatrix[dtype]`, `CSCMatrix[dtype]`: Compressed sparse row/column matrices with `spmv`, `spmm`, `spgemm`, `sddmm`.
  - `linalg`: SIMD `gemm`, `dense_dot_vec`, and LAPACK bindings (`svd`, `eigh`, `qr`, `cholesky`, `lstsq`, `solve`, `inv`).
  - `dataset`: `Dataset` container for features, targets, and feature names.
  - `interop`: NumPy and SciPy sparse conversions.
- **`strata.linear_model`**: `LinearRegression`, `Ridge`, `LogisticRegression` (binary and multinomial).
- **`strata.decomposition`**: `PCA` (with whitening and sign-flip), `TruncatedSVD` (dense and sparse CSR via SpMM).
- **`strata.preprocessing`**: `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `OneHotEncoder`, `Binarizer`.
- **`strata.model_selection`**: `KFold`, `StratifiedKFold`, `cross_val_score`, `GridSearchRegressor`, `GridSearchClassifier`.
- **`strata.metrics`**:
  - Regression: `mean_squared_error`, `root_mean_squared_error`, `mean_absolute_error`, `r2_score`.
  - Classification: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`.
- **`strata.base`**: `Transformer`, `Regressor`, `Classifier`, `Clusterer`, and composable `Pipeline` structs.

---

## Development

```bash
# Run test suite
pixi run test-all

# Format code
pixi run format

# Typecheck / compile package
pixi run build
```

See [CONTRIBUTORS.md](./CONTRIBUTORS.md) for development setup and codebase conventions.  
See [ROADMAP.md](./ROADMAP.md) for planned features and progress.
