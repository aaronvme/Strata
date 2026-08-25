# Contributors & Development Guide

Thanks for checking out Strata! Whether you're fixing a bug, adding a new algorithm, optimizing a kernel with SIMD, or writing tests, we welcome your contributions.

---

## Maintainers

- **Ethan Wu** ([@ethqnol](https://github.com/ethqnol)) — Creator & Lead Maintainer

---

## Getting Started

Strata is built entirely in [Mojo](https://docs.modular.com/mojo/) and uses [Pixi](https://pixi.sh/) for reproducible environment and dependency management.

### 1. Environment Setup

Make sure you have [Pixi installed](https://pixi.sh/#installation), then clone and initialize the environment:

```bash
git clone https://github.com/ethqnol/Strata.git
cd Strata

# Install dependencies (Mojo, Python, NumPy, SciPy)
pixi install
```

### 2. Running Tests

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

### 3. Precompiling the Package

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
│   ├── base/             # Core traits (Estimator, Transformer, Regressor, Classifier)
│   ├── core/             # Matrix, MatrixView, CSRMatrix, CSCMatrix, linalg, interop
│   ├── exceptions/       # Domain errors (DimensionMismatchError, NotFittedError, etc.)
│   ├── utils/            # Validation helpers (check_X_y), math (softmax), random (shuffle)
│   ├── preprocessing/    # Data transformers (StandardScaler, MinMaxScaler)
│   ├── model_selection/  # Data splitting (train_test_split, KFold)
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

### 2. Generics & Type Parameterization
- Core math routines and estimators should support standard floating point types via compile-time parameters:
  ```mojo
  struct MyEstimator[dtype: DType = DType.float64](Movable):
      var weights: List[Scalar[Self.dtype]]
  ```

### 3. Validation & Clear Error Messages
- Use the shared validation functions in `strata.utils.validation`:
  - `check_array(X)` — checks for non-empty 2D matrices.
  - `check_X_y(X, y)` — verifies consistent sample counts between features and targets.
  - `check_is_fitted("EstimatorName", self.is_fitted)` — ensures models are trained before calling `predict` or `transform`.
- Raise domain-specific errors from `strata.exceptions.errors` (`DimensionMismatchError`, `InvalidParameterError`, `NotFittedError`).

---

## How to Add a New Estimator (Step-by-Step)

If you're implementing an estimator (e.g. from the [ROADMAP.md](file:///home/ewu/Code/Strata/ROADMAP.md)):

### Step 1: Define the Struct & Constructor
Implement the estimator conforming to `Movable` and the appropriate base trait (`Regressor`, `Classifier`, or `Transformer`):

```mojo
from ..base.estimator import Regressor
from ..core.matrix import Matrix
from ..utils.validation import check_is_fitted, check_X_y
from ..exceptions.errors import NotFittedError

struct MyRegressor[dtype: DType = DType.float64](Regressor, Movable):
    var is_fitted: Bool
    var coef_: List[Scalar[Self.dtype]]
    var intercept_: Scalar[Self.dtype]

    def __init__(out self):
        self.is_fitted = False
        self.coef_ = List[Scalar[Self.dtype]]()
        self.intercept_ = 0
```

### Step 2: Implement `fit` and `predict` / `transform`
- Always call `check_X_y(X, y)` at the start of `fit`.
- Set `self.is_fitted = True` upon successful convergence.
- Call `check_is_fitted("MyRegressor", self.is_fitted)` at the beginning of `predict` / `transform`.

### Step 3: Export in Subpackage `__init__.mojo`
Export your struct in its folder's `__init__.mojo` (e.g., `strata/linear_model/__init__.mojo`) and add it to `strata/__init__.mojo`.

### Step 4: Write Unit Tests & Validate Parity
Create `tests/test_<feature>.mojo`. Where applicable, compare outputs against `scikit-learn` references to ensure numerical correctness.

---

## Pull Request Checklist

Before submitting a PR:

1. [ ] **Format and clean**: Check that variable and function names follow Mojo conventions (`snake_case` functions, `PascalCase` structs).
2. [ ] **Tests pass**: Run `pixi run test-all` and ensure all tests pass without errors or warnings.
3. [ ] **Package builds**: Run `pixi run build` to confirm package compilation succeeds.
4. [ ] **Docstrings**: Ensure new public structs and methods have clear docstrings.
5. [ ] **Add to Roadmap**: If completing an item from [ROADMAP.md](file:///home/ewu/Code/Strata/ROADMAP.md), check off the task!
