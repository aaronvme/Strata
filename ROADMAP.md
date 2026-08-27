# Strata Roadmap

Strata is a native machine learning library for Mojo. The goal is to provide the familiar, expressive workflow of `scikit-learn` combined with native performance, zero-copy views, compile-time trait safety, and hardware acceleration out of the box—with zero C/C++ extensions or foreign runtimes.

---

## Current State

We have established a robust, thoroughly tested core engine:

- **Data structures & Zero-Copy Views**:
  - `Matrix[dtype]` (Dense 2D row-major matrix with strided views)
  - `MatrixView[dtype, origin]` (Non-owning strided 2D view with compile-time origin tracking)
  - `CSRMatrix[dtype]` & `CSCMatrix[dtype]` (Compressed sparse formats with $O(\text{nnz})$ conversions)
  - Upfront promotion via `.cast[target_dtype]()` across dense and sparse containers
- **Multi-Precision Homogeneous Linear Algebra**:
  - Generic support for `float32`, `float64`, `bfloat16`, and `float16` with strictly homogeneous kernels.
  - Dense: `gemm`, `dense_dot_vec`
  - Sparse: `spmv`, `spvm`, `spmm`, Gustavson `spgemm`, and `sddmm`
- **Composable Pipelines & Base Traits**:
  - Unified traits: `Transformer`, `Regressor`, `Classifier`, `Clusterer`
  - Composable pipelines: `PipelineTransformer`, `PipelineRegressor`, `PipelineClassifier`
  - Generic functional helpers for `Dataset` containers
- **Math Utilities & Numerical Stability**:
  - Overflow-protected `softmax`, `log_sum_exp`, and `sigmoid`
  - 64-bit SplitMix64 `PRNG` with unbiased Lemire rejection sampling, Fisher-Yates `permutation` & `shuffle`
- **Validation & Domain Errors**:
  - Domain exceptions: `DimensionMismatchError`, `NotFittedError`, `InvalidParameterError`, `DataConversionError`
  - Structural validators: `check_array`, `check_X_y`, `check_sparse`, `check_is_fitted`
- **Dimensionality Reduction (`strata.decomposition`)**:
  - `PCA` (Exact SVD, mean centering, whitening, deterministic sign flips, inverse reconstruction)
  - `TruncatedSVD` (Linear projection for dense and sparse `CSRMatrix` via hardware-vectorized `spmm`)
- **Test Coverage**:
  - 17 modular test suites (575 passing tests)

---

## Roadmap

### 1. Engine & Kernel Optimization
- [x] **Base traits & pipeline composition**: Unified 2-method pattern with composable $N$-step pipelines.
- [x] **Multi-precision homogeneous LinAlg & upfront promotion**: Support Float32, Float64, BFloat16, Float16 with strictly homogeneous execution and zero inner-loop casting.
- [x] **Math & PRNG foundation**: Stable softmax, log-sum-exp, sigmoid, and unbiased PRNG.
- [x] **SIMD vectorization**: Vectorized dense kernels (`gemm`, `dense_dot_vec`) with SIMD registers, FMA, and scalar tails.
- [x] **LAPACK FFI factorizations & solvers**: Production bindings (`dgesdd`, `dsyevd`, `dgeqrf`/`dorgqr`, `dpotrf`, `dgelss`, `dgesv`, `dgetrf`/`dgetri`) for exact SVD, Eigh, QR, Cholesky, Least-Squares, LU Solve, and Inversion.
- [ ] **Multi-threading**: Parallelize dense matrix products and row-wise operations using `parallel_for`.
- [ ] **Direct memory Python interop**: Fast NumPy buffer transfer via pointer/memcpy instead of element loops.

---

### 2. Core Estimators & Metrics
- [ ] **Evaluation Metrics (`strata.metrics`)**:
  - [x] Regression: `mean_squared_error`, `root_mean_squared_error`, `r2_score`, `mean_absolute_error`
  - [x] Classification: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`
  - [ ] Clustering: `silhouette_score`, `inertia`
- [x] **Linear Models (`strata.linear_model`)**:
  - [x] `LinearRegression` (Ordinary Least Squares with multi-solver support: `lstsq`, `qr`, `cholesky`, `solve`)
  - [x] `Ridge` ($L_2$-regularized closed-form solver: `auto`, `cholesky`, `svd`, `solve`)
  - [x] `LogisticRegression` (Binary and multinomial softmax classification with $L_2$ / unregularized optimization)
- [x] **Preprocessing (`strata.preprocessing`)**:
  - [x] `StandardScaler` (Zero mean, unit variance standardization)
  - [x] `MinMaxScaler` (Configurable feature range with optional clipping)
  - [x] `RobustScaler` (Median centering and quantile-range scaling)
  - [x] `OneHotEncoder` (Dense indicator expansion with `drop` and `handle_unknown` policies)
  - [x] `Binarizer` (Threshold-based 0/1 mapping)
- [x] **Model Selection (`strata.model_selection`)**:
  - [x] `KFold` and `StratifiedKFold`
  - [x] `cross_val_score`
  - [x] `GridSearchRegressor` and `GridSearchClassifier`
  - [x] `TimeSeriesSplit` (Rolling / expanding temporal cross-validation)
  - [ ] `cross_val_predict` (Out-of-fold prediction matrices)
  - [ ] `ShuffleSplit` & `StratifiedShuffleSplit` (Monte Carlo random splits)
  - [ ] `RandomizedSearchRegressor` & `RandomizedSearchClassifier`
  - [ ] `cross_validate` (Multi-metric evaluation)

---

### 3. Classical Machine Learning Algorithms
- [x] **Dimensionality Reduction (`strata.decomposition`)**:
  - [x] `PCA` (Exact SVD, empirical mean centering, whitening, and inverse reconstruction)
  - [x] `TruncatedSVD` (Linear dimensionality reduction for dense and sparse `CSRMatrix` via SpMM)
- [ ] **Clustering (`strata.cluster`)**:
  - `KMeans` (with KMeans++ initialization)
  - `KModes` (for categorical features)
  - `DBSCAN`
- [ ] **Trees & Ensembles (`strata.tree`, `strata.ensemble`)**:
  - `DecisionTreeClassifier` & `DecisionTreeRegressor`
  - `RandomForestClassifier` & `RandomForestRegressor`
  - `GradientBoosting`
- [ ] **Nearest Neighbors (`strata.neighbors`)**:
  - `KNeighborsClassifier` & `KNeighborsRegressor`
  - KD-Tree spatial index

---

### 4. Advanced Capabilities & Hardware Acceleration
- [ ] GPU acceleration (Mojo GPU backend for dense and sparse tensor operations).
- [ ] Out-of-core chunked streaming for large-scale datasets.
- [ ] Apache Arrow and Parquet zero-copy ingestion.

---

## Contributing

To contribute an algorithm or feature, check the tasks above, review [CONTRIBUTORS.md](./CONTRIBUTORS.md), and submit a pull request.
