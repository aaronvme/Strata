# Contributors & Development Guide

## Getting Started

Strata is built entirely in [Mojo](https://docs.modular.com/mojo/) and uses [Pixi](https://pixi.sh/) for reproducible environment and dependency management.

### 1. Environment Setup

#### Linux / macOS
Make sure you have [Pixi installed](https://pixi.sh/#installation), then clone and initialize the environment:

```bash
git clone https://github.com/ethqnol/Strata.git
cd Strata

# Install dependencies (Mojo, Python, NumPy, SciPy)
pixi install
```

#### Windows (via WSL 2)
Mojo runs natively on Linux (and macOS), so Windows contributors should use **WSL 2** (Windows Subsystem for Linux):

1. **Install WSL 2** (if not already installed):
   Open PowerShell as Administrator and run:
   ```powershell
   wsl --install
   ```
   Restart your PC if prompted, then launch the **Ubuntu** terminal from your Start menu.

2. **Install base packages in Ubuntu**:
   ```bash
   sudo apt update && sudo apt install -y curl git build-essential
   ```

3. **Install Pixi inside WSL**:
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   source ~/.bashrc
   ```

4. **Clone and install (inside the Linux filesystem)**:
   > **Note**: Always clone the repository into your WSL Linux home directory (`~/Code/` or `/home/<user>/`), **not** on the Windows mount (`/mnt/c/...`), for optimal disk I/O and compiler performance.

   ```bash
   mkdir -p ~/Code && cd ~/Code
   git clone https://github.com/ethqnol/Strata.git
   cd Strata
   pixi install
   ```

5. **VS Code / Cursor Integration**:
   - Install the **WSL** extension in VS Code / Cursor.
   - Run `code .` from the `Strata` directory inside your WSL terminal.
   - Install the official **Mojo** extension inside the remote WSL session.

---

### 2. Code Formatting

We use Mojo's built-in official formatter (`mojo format`). Since it is bundled with the compiler in our Pixi environment, no extra installations are required.

```bash
# Auto-format all source and test files
pixi run format

# Check formatting without modifying files
pixi run format-check
```

**Editor Configuration (VS Code / Cursor):**
To format automatically on save, add this to your `settings.json`:
```json
"[mojo]": {
    "editor.defaultFormatter": "modular.mojo",
    "editor.formatOnSave": true
}
```

---

### 3. Running Tests

We keep our test suites modular. You can run all tests or target specific subsystems:

```bash
# Run the full test suite
pixi run test-all

# Run specific test modules
pixi run test-matrix          # Dense GEMM, sparse SpMV/SpMM
pixi run test-view            # 2D zero-copy slicing
pixi run test-preprocessing   # Standard scaler, dataset transformations
pixi run test-split           # Train/test split, shuffling
pixi run test-interop         # NumPy / SciPy conversion roundtrips
pixi run test-large           # Large matrix benchmarks & stress tests
pixi run test-core            # Error types, validation routines, math utils
```

---

### 4. Precompiling the Package

To ensure all generic structs, traits, and module interfaces typecheck and compile cleanly:

```bash
pixi run build
```

---

## Codebase Architecture

Here is how the project is organized:

```
Strata/
├── strata/
│   ├── base/             # Core traits (Estimator, Transformer, Regressor, Classifier, Clusterer)
│   ├── core/             # Matrix, MatrixView, CSRMatrix, CSCMatrix, linalg, interop
│   ├── exceptions/       # Domain errors (DimensionMismatchError, NotFittedError, etc.)
│   ├── utils/            # Validation helpers (check_X_y), math (softmax), random (shuffle)
│   ├── preprocessing/    # Data transformers (StandardScaler)
│   ├── model_selection/  # Data splitting (train_test_split)
│   ├── metrics/          # Evaluation metrics (MSE, R2, Accuracy, F1)
│   ├── linear_model/     # Linear regression, Ridge, Logistic regression
│   ├── cluster/          # KMeans, KModes, DBSCAN
│   ├── tree/             # Decision trees
│   ├── ensemble/         # Random forests, gradient boosting
│   ├── decomposition/    # PCA, TruncatedSVD
│   └── neighbors/        # KNN, KD-Tree
└── tests/                # Unit and integration test suites
```

---

## Core Design Principles & Mojo Idioms

When contributing code to Strata, please keep these conventions in mind:

### 1. Ownership & Memory Efficiency
- **Avoid unnecessary copies**: For internal matrix operations and read-only slicing, use `MatrixView` rather than allocating new `Matrix` buffers.
- **Explicit transfers**: Use `^` (move operator) when transferring ownership of large arrays or structs into estimators or return values.
- **Explicit copying**: When an explicit clone is needed, call `.copy()`.

### 2. Trait Contracts & Ergonomic Call-Site Inference
To ensure compile-time trait enforcement, type safety, and effortless `model.predict(X)` call syntax matching scikit-learn:

- **Target DType Defaults & Developer Flexibility**:
  `target_dtype` defaults to `DType.float64` for Regressors and `DType.int32` for Classifiers. Because of this default, standard users get standard behavior out of the box, while advanced users have complete freedom to specify alternative precisions (e.g. `Float32` or `Int32`) if desired:
  ```mojo
  struct LinearRegression[
      target_dtype: DType = DType.float64,
      compute_dtype: DType = DType.float64,
  ](Regressor, Movable):
      ...
  ```

- **Transformers & Fitted DType Consistency**:
  Transformers (e.g. `StandardScaler`) accept arbitrary input precisions at `fit` time, enforce fitted `fit_dtype` consistency at `transform` time, and compute in `compute_dtype` (default `DType.float64`) for maximum numerical stability:
  ```mojo
  struct StandardScaler[
      compute_dtype: DType = DType.float64,
  ](Transformer, Movable, Copyable):
      ...
  ```

- **Clusterers & Discrete Cluster Assignments**:
  Clusterers (e.g. `KMeans`, `KModes`) implement generic `fit[dtype]` and `predict[dtype]` methods returning `List[Int]` cluster indices.

- **Clean, Zero-Type-Parameter Pipelines**:
  `PipelineRegressor` and `PipelineClassifier` wrap `[T: Transformer, R: Regressor]` and allow zero-type-parameter training and inference:
  ```mojo
  var scaler = StandardScaler()
  var model = LinearRegression()
  var pipe = PipelineRegressor(scaler^, model^)

  pipe.fit(X_train, y_train)
  var preds = pipe.predict(X_test)  # Fully inferred from X_test!
  ```

### 3. Validation & Clear Error Messages
- Use the shared validation functions in `strata.utils.validation`:
  - `check_array[dtype](X)` — checks for non-empty 2D matrices.
  - `check_X_y(X, y)` — verifies consistent sample counts between features and targets.
  - `check_is_fitted("EstimatorName", self.is_fitted)` — ensures models are trained before calling `predict` or `transform`.
- Raise domain-specific errors from `strata.exceptions.errors` (`DimensionMismatchError`, `InvalidParameterError`, `NotFittedError`).

---

## How to Add a New Estimator (Step-by-Step)

If you're implementing an estimator (e.g. from ROADMAP.md):

### Step 1: Define the Struct & Constructor
Implement the estimator conforming to `Movable` and the appropriate base trait (`Regressor`, `Classifier`, `Transformer`, or `Clusterer`), storing internal model parameters in `compute_dtype`:

```mojo
from ..base.estimator import Regressor
from ..core.matrix import Matrix
from ..core.dataset import Dataset
from ..utils.validation import check_is_fitted, check_X_y
from ..exceptions.errors import NotFittedError

struct MyRegressor[
    target_dtype: DType = DType.float64,
    compute_dtype: DType = DType.float64,
](Regressor, Movable):
    var is_fitted: Bool
    var coef_: List[Scalar[Self.compute_dtype]]      # Accumulates in compute precision!
    var intercept_: Scalar[Self.compute_dtype]

    def __init__(out self):
        self.is_fitted = False
        self.coef_ = List[Scalar[Self.compute_dtype]]()
        self.intercept_ = 0
```

### Step 2: Implement `fit` and `predict` (The 5-Method Pattern)
Implement the 5 methods to satisfy trait conformance while enabling zero-type-parameter call-site inference:

```mojo
    # 1. Universal fit accepting arbitrary input feature and target precisions
    def fit[
        feat_dtype: DType, in_target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[in_target_dtype]]) raises:
        check_X_y(X, y)
        # Compute coefficients using Self.compute_dtype arithmetic
        self.is_fitted = True

    # 2. Dataset fit overload forwarding to matrix fit
    def fit[
        feat_dtype: DType, in_target_dtype: DType
    ](mut self, dataset: Dataset[feat_dtype, in_target_dtype]) raises:
        self.fit(dataset.records, dataset.targets)

    # 3. Ergonomic zero-call-site-parameter predict (infers feat_dtype from X!)
    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Scalar[Self.target_dtype]]:
        return self.predict[feat_dtype, Self.target_dtype](X)

    # 4. Two-parameter predict satisfying trait Regressor
    def predict[
        feat_dtype: DType, out_target_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Scalar[out_target_dtype]]:
        check_is_fitted("MyRegressor", self.is_fitted)
        var preds = List[Scalar[out_target_dtype]](capacity=X.rows)
        # Compute predictions
        return preds^

    # 5. Dataset predict overload
    def predict[
        feat_dtype: DType, in_target_dtype: DType
    ](
        self,
        dataset: Dataset[feat_dtype, in_target_dtype],
    ) raises -> List[
        Scalar[in_target_dtype]
    ]:
        return self.predict[feat_dtype, in_target_dtype](dataset.records)
```

### Step 3: Export in Subpackage `__init__.mojo`
Export your struct in its folder's `__init__.mojo` (e.g., `strata/linear_model/__init__.mojo`) and add it to `strata/__init__.mojo`.

### Step 4: Write Unit Tests & Validate Parity
Create `tests/test_<feature>.mojo`. Where applicable, compare outputs against `scikit-learn` references to ensure numerical correctness.

---

## Pull Request Checklist

Before submitting a PR:

1. [ ] **Format code**: Run `pixi run format` to ensure code conforms to Mojo styling.
2. [ ] **Tests pass**: Run `pixi run test-all` and ensure all tests pass without errors.
3. [ ] **Package builds**: Run `pixi run build` to confirm package precompilation succeeds.
4. [ ] **Docstrings**: Ensure new public structs and methods have clear docstrings.
5. [ ] **Update Roadmap**: If completing an item from [ROADMAP.md](./ROADMAP.md), check off the task!
