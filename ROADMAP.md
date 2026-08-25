# Strata Roadmap

Strata is a native machine learning library for Mojo. The goal is straightforward is to give Mojo the same clean, familiar workflow as `scikit-learn`, but with native performance, zero-copy views, and hardware acceleration out of the box—no Cython or C extensions needed.

---

## Current State

We have the core data structures and linear algebra plumbing in place:

- **Data structures**:
  - `Matrix[dtype]` (dense 2D matrix)
  - `MatrixView` (zero-copy strided row/col/2D slicing)
  - `CSRMatrix` and `CSCMatrix` (sparse formats)
- **Sparse operations**:
  - SpMV, SpVM, SpMM, Gustavson SpGEMM, and SDDMM
- **Dataset & Preprocessing**:
  - Basic `Dataset` container with feature/target metadata
  - `train_test_split`
  - `StandardScaler`
- **Error handling**:
  - Custom error types (`DimensionMismatchError`, `NotFittedError`, `InvalidParameterError`, etc.)

---

## What needs to be completed:

### 1. Engine & Math Foundation
Before stacking up dozens of algorithms, we need our core engine to be rock-solid and fast.

- [x] **Fix base traits**: Update `Estimator`, `Transformer`, `Regressor`, and `Classifier` to have proper `Movable` contracts so pipelines compile cleanly.
- [ ] **Vectorize dense linalg**: Rewrite `gemm` and `dense_dot_vec` with SIMD (`sys.info.simdwidthof`) and cache-friendly tiling instead of naive loops.
- [ ] **Multi-threading**: Add parallel CPU execution (`parallel_for`) for dense matrix multiplications and row-wise operations.
- [ ] **Fast NumPy interop**: Replace element-by-element Python loops with direct pointer/buffer copies.
- [x] **Numerical stability**: Fix potential overflow in `softmax` (subtract max) and add standard math utils (log-sum-exp, better PRNG).

---

### 2. Core Estimators & Metrics
The goal here is a working end-to-end ML workflow: load data -> preprocess -> train a linear model -> evaluate.

- [ ] **Metrics (`strata.metrics`)**:
  - Regression: MSE, RMSE, MAE, R²
  - Classification: Accuracy, Precision, Recall, F1, Log Loss, ROC-AUC
  - Clustering: Silhouette score, Inertia
- [ ] **Linear Models (`strata.linear_model`)**:
  - `LinearRegression` (OLS via Normal Equations & SGD)
  - `Ridge` (L2 regularization)
  - `LogisticRegression` (Binary and multinomial classification)
- [ ] **More Preprocessing (`strata.preprocessing`)**:
  - `MinMaxScaler`
  - `RobustScaler`
  - `OneHotEncoder`
- [ ] **Model Selection (`strata.model_selection`)**:
  - `KFold` and `StratifiedKFold`
  - `cross_val_score`

---

### 3. Classical ML Algorithms

- [ ] **Clustering (`strata.cluster`)**:
  - `KMeans` (with KMeans++ initialization)
  - `KModes` (for categorical datasets)
  - `DBSCAN`
- [ ] **Trees & Ensembles (`strata.tree`, `strata.ensemble`)**:
  - `DecisionTreeClassifier` / `DecisionTreeRegressor`
  - `RandomForestClassifier` / `RandomForestRegressor`
  - `GradientBoosting`
- [ ] **Dimensionality Reduction (`strata.decomposition`)**:
  - `PCA` (via randomized SVD / power iteration)
  - `TruncatedSVD`
- [ ] **Nearest Neighbors (`strata.neighbors`)**:
  - `KNeighborsClassifier` / `KNeighborsRegressor`
  - KD-Tree / Ball-Tree spatial indices

---

### 4. Down the Line

- [ ] Full `Pipeline` and `ColumnTransformer` composition.
- [ ] GPU kernel acceleration (Mojo GPU backend for dense and sparse matmuls).
- [ ] Out-of-core chunked streaming for datasets that don't fit in RAM.
- [ ] Native Apache Arrow / Parquet data loading.

---

## Feedback & Ideas

If there's an algorithm or feature you want to see sooner, or if you want to help implement one, open an issue or grab a task from the list above and make a pr!
