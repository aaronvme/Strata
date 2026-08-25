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

### 2. Generics & Target-DType Convention
To ensure complete consistency across the library as it scales:

- **Supervised Estimators (`Regressor`, `Classifier`)**:
  Supervised models map feature matrices to target vectors ($X \to y$). Features and targets may have different precision or types (e.g. `Float32` features with `Float64` continuous targets, or `Float32` features with `Int32` categorical labels).
  - **Always parameterize both**:
    ```mojo
    struct LinearRegression[
        feat_dtype: DType = DType.float64,
        target_dtype: DType = DType.float64,
    ](Regressor, Movable)
    ```
    ```mojo
    struct LogisticRegression[
        feat_dtype: DType = DType.float64,
        target_dtype: DType = DType.float64,
    ](Classifier, Movable)
    ```
  - Standard method signatures:
    - `fit(mut self, X: Matrix[Self.feat_dtype], y: List[Scalar[Self.target_dtype]]) raises`
    - `fit(mut self, dataset: Dataset[Self.feat_dtype, Self.target_dtype]) raises`
    - `predict(self, X: Matrix[Self.feat_dtype]) raises -> List[Scalar[Self.target_dtype]]`

- **Unsupervised Estimators & Transformers (`Transformer`, `Clusterer`)**:
  Transformers and clustering algorithms operate purely on feature representations:
  - **Struct parameter**: `[dtype: DType = DType.float64]` (operating on `Matrix[Self.dtype]`).
  - **Dataset passthrough overloads**: When accepting a `Dataset`, do *not* bind the struct to a fixed target type. Instead, genericize the method over `[target_dtype: DType]` so targets pass through untouched:
    ```mojo
    struct StandardScaler[dtype: DType = DType.float64](Transformer, Movable):
        def fit(mut self, X: Matrix[Self.dtype]) raises:
            ...
        def fit[target_dtype: DType](mut self, dataset: Dataset[Self.dtype, target_dtype]) raises:
            self.fit(dataset.records)
        def transform[target_dtype: DType](self, dataset: Dataset[Self.dtype, target_dtype]) raises -> Dataset[Self.dtype, target_dtype]:
            ...
    ```

- **Pipelines**:
  - `PipelineRegressor[feat_dtype: DType, target_dtype: DType, T: Transformer, R: Regressor]`
  - `PipelineClassifier[feat_dtype: DType, target_dtype: DType, T: Transformer, C: Classifier]`

### 3. Trait Conformance & Lifecycles
All components in Strata must adhere to standardized trait contracts in `strata.base`:
- **Lifecycles**: Every estimator and transformer struct **must conform to `Movable`** (and `Copyable` when feasible). This is required so estimators can be safely stored, moved into pipelines, or returned from helper functions.
- **Role traits**:
  - `Transformer`: Must implement `fit(mut self, X: Matrix[dtype]) raises` and `transform(self, X: Matrix[dtype]) raises -> Matrix[dtype]`.
  - `Regressor`: Must implement `fit(mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]]) raises` and `predict(self, X: Matrix[feat_dtype]) raises -> List[Scalar[target_dtype]]`.
  - `Classifier`: Must implement `fit(mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]]) raises` and `predict(self, X: Matrix[feat_dtype]) raises -> List[Scalar[target_dtype]]`.
  - `Clusterer`: Must implement `fit(mut self, X: Matrix[dtype]) raises` and `predict(self, X: Matrix[dtype]) raises -> List[Int]`.
- **Never create orphan estimators**: Do not define standalone model structs without conforming to their respective base traits in `strata.base.estimator`.

### 4. Validation & Clear Error Messages
- Use the shared validation functions in `strata.utils.validation`:
  - `check_array(X)` — checks for non-empty 2D matrices.
  - `check_X_y(X, y)` — verifies consistent sample counts between features and targets.
  - `check_is_fitted("EstimatorName", self.is_fitted)` — ensures models are trained before calling `predict` or `transform`.
- Raise domain-specific errors from `strata.exceptions.errors` (`DimensionMismatchError`, `InvalidParameterError`, `NotFittedError`).

---

## How to Add a New Estimator (Step-by-Step)

If you're implementing an estimator (e.g. from [ROADMAP.md](file:///home/ewu/Code/Strata/ROADMAP.md)):

### Step 1: Define the Struct & Constructor
Implement the estimator conforming to `Movable` and the appropriate base trait (`Regressor`, `Classifier`, or `Transformer`) with explicit `feat_dtype` and `target_dtype`:

```mojo
from ..base.estimator import Regressor
from ..core.matrix import Matrix
from ..core.dataset import Dataset
from ..utils.validation import check_is_fitted, check_X_y
from ..exceptions.errors import NotFittedError

struct MyRegressor[
    feat_dtype: DType = DType.float64,
    target_dtype: DType = DType.float64,
](Regressor, Movable):
    var is_fitted: Bool
    var coef_: List[Scalar[Self.feat_dtype]]
    var intercept_: Scalar[Self.target_dtype]

    def __init__(out self):
        self.is_fitted = False
        self.coef_ = List[Scalar[Self.feat_dtype]]()
        self.intercept_ = 0
```

### Step 2: Implement `fit` and `predict`
- Implement `fit(mut self, X: Matrix[Self.feat_dtype], y: List[Scalar[Self.target_dtype]]) raises`.
- Implement `fit(mut self, dataset: Dataset[Self.feat_dtype, Self.target_dtype]) raises`.
- Call `check_X_y(X, y)` at the start of `fit`.
- Set `self.is_fitted = True` upon successful convergence.
- Call `check_is_fitted("MyRegressor", self.is_fitted)` at the beginning of `predict`.

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
5. [ ] **Update Roadmap**: If completing an item from [ROADMAP.md](file:///home/ewu/Code/Strata/ROADMAP.md), check off the task!
