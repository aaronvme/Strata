# Strata

Strata is a native machine learning and high-performance linear algebra library written in [Mojo](https://docs.modular.com/mojo/). It delivers the expressive, user-friendly API design of `scikit-learn` combined with native performance, hardware SIMD vectorization, zero-copy strided views, and LAPACK matrix factorizations.

---

## Key Features

* **Multi-Precision Linear Algebra**: Homogeneous compute kernels across `Float64`, `Float32`, `BFloat16`, and `Float16`.
* **Hardware-Accelerated Kernels**:
  * **Native SIMD Operations**: Fast dense matrix multiplication (`gemm`) and matrix-vector products (`dense_dot_vec`) with zero external runtime dependencies.
  * **Production LAPACK Factorizations**: Singular Value Decomposition (`svd`), Symmetric Eigenvalue Decomposition (`eigh`), QR Decomposition (`qr`), Cholesky (`cholesky`), Least-Squares (`lstsq`), Linear Solve (`solve`), and Matrix Inversion (`inv`).
* **Compressed Sparse Formats**: Full `CSRMatrix` and `CSCMatrix` implementations with $O(\text{nnz})$ conversions, SpMV, SpVM, SpMM, Gustavson SpGEMM, and SDDMM.
* **Zero-Copy Views**: Strided 2D slicing with compile-time origin tracking via `MatrixView`.
* **Composable Pipelines**: Modular $N$-step composition chaining transformers, regressors, and classifiers with full type inference.
* **Numerically Stable Utilities**: Overflow-protected `softmax`, `sigmoid`, and `log_sum_exp`, alongside a 64-bit SplitMix64 PRNG with unbiased rejection sampling.

---

## Quick Start

### Installation

Strata uses [Pixi](https://pixi.sh/) for environment and dependency management:

```bash
git clone https://github.com/ethqnol/Strata.git
cd Strata
pixi install
```

### Basic Matrix Operations & LinAlg

```mojo
from strata import Matrix, gemm, svd, eigh, solve

def main() raises:
    # 1. Create matrices
    var A = Matrix[DType.float64](2, 2, 0)
    A[0, 0] = 4.0; A[0, 1] = 2.0
    A[1, 0] = 2.0; A[1, 1] = 3.0

    var B = Matrix[DType.float64].eye(2)

    # 2. Native SIMD matrix multiplication
    var C = gemm(A, B)

    # 3. Factorizations
    var svd_res = svd(A)
    var eig_res = eigh(A)

    # 4. Linear solve A * x = b
    var b: List[Scalar[DType.float64]] = [8.0, 7.0]
    var x = solve(A, b)
```

---

## Running Tests & Building

Strata maintains a comprehensive test suite covering mathematical invariants, factorizations, sparse algebra, and domain validation:

```bash
# Run the complete test suite (13 suites, 120 tests)
pixi run test-all

# Precompile package
pixi run build

# Format codebase
pixi run format
```

---

## Architecture & Roadmap

For contribution guidelines and codebase idioms, see [CONTRIBUTORS.md](./CONTRIBUTORS.md).  
For project milestones and upcoming algorithms, see [ROADMAP.md](./ROADMAP.md).
