# Strata Docstring Style Guide

This specification defines the formatting, syntax, and structural requirements for docstrings in the Strata codebase. All public structs, traits, constructors, methods, and standalone functions must include docstrings adhering to this standard.

The automated documentation engine (`scripts/generate_docs.py`) parses these docstrings to generate the API reference documentation and search index.

---

## Table of Contents

1. [General Rules and Syntax](#1-general-rules-and-syntax)
2. [Mathematical Notation (LaTeX)](#2-mathematical-notation-latex)
3. [Docstring Structure by Symbol Type](#3-docstring-structure-by-symbol-type)
   - [A. Estimators and Structs](#a-estimators-and-structs)
   - [B. Constructors (`__init__`)](#b-constructors-__init__)
   - [C. Struct Methods](#c-struct-methods)
   - [D. Standalone Functions](#d-standalone-functions)
   - [E. Traits and Interfaces](#e-traits-and-interfaces)
4. [Section Header Keywords](#4-section-header-keywords)
5. [Reference Examples](#5-reference-examples)
6. [Anti-Patterns](#6-anti-patterns)

---

## 1. General Rules and Syntax

### 1.1 Placement and Quotes
- Use triple double quotes `"""` for all docstrings.
- Place the opening `"""` on the line immediately following the declaration (`struct`, `trait`, `def`, or `fn`).
- Place the closing `"""` on its own line matching the indentation of the declaration block.

### 1.2 Summary Line
- The first line must be a concise, single-line summary (maximum 80 characters).
- Use the imperative mood (for example, `"Fit the model..."`, `"Compute mean squared error..."`, `"Standardize features by removing the mean..."`).
- Always follow the summary line with a blank line before adding extended descriptions or sections.

```mojo
# Correct
struct StandardScaler(Transformer, Copyable, Movable):
    """Standardize features by removing the mean and scaling to unit variance.

    Centering and scaling happen independently on each feature by computing
    streaming SIMD statistics over the training samples.
    """

# Incorrect: Missing blank line after summary
struct StandardScaler(Transformer, Copyable, Movable):
    """Standardize features by removing the mean and scaling to unit variance.
    Centering and scaling happen independently on each feature.
    """
```

---

## 2. Mathematical Notation (LaTeX)

All mathematical expressions, objective functions, and loss formulations must be written in standard LaTeX notation.

Do not use ASCII bar representations (such as `||y - Xw||`) or pseudo-code math. Raw vertical bars interfere with Markdown table tokenization. Use standard LaTeX delimiters `$ ... $` for inline math and `$$ ... $$` for display equations.

### 2.1 Inline Math (`$ ... $`)
Use single dollar signs for mathematical variables, dimensions, norms, and complexity bounds:
- Dimensions: `$N \times D$`
- $L_2$ norm: `$\|w\|_2$`
- Regularization parameter: `$\alpha \ge 0$`
- Computational complexity: `$\mathcal{O}(N \cdot D \log N)$`

### 2.2 Display Math (`$$ ... $$`)
Use double dollar signs on dedicated lines for multi-line objectives, summations, and loss functions:

```text
Minimizes the penalized residual sum of squares:

$$
\min_{w} \frac{1}{2N} \|y - Xw\|_2^2 + \frac{\alpha}{2} \|w\|_2^2
$$
```

### 2.3 Common Machine Learning Formulations

| Concept | Standard LaTeX Format |
| :--- | :--- |
| **Ridge Objective** | `$$\min_{w} \|y - Xw\|_2^2 + \alpha \|w\|_2^2$$` |
| **Lasso Objective** | `$$\min_{w} \frac{1}{2N} \|y - Xw\|_2^2 + \alpha \|w\|_1$$` |
| **Gini Impurity** | `$$H(Q_m) = 1 - \sum_{k=1}^{K} p_{mk}^2$$` |
| **Entropy** | `$$H(Q_m) = -\sum_{k=1}^{K} p_{mk} \log_2(p_{mk})$$` |
| **Mean Squared Error** | `$$\text{MSE}(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$` |
| **Coefficient of Determination ($R^2$)** | `$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$` |
| **Logistic Sigmoid** | `$$\sigma(z) = \frac{1}{1 + e^{-z}}$$` |
| **Softmax Distribution** | `$$P(y = k \mid x) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$` |

---

## 3. Docstring Structure by Symbol Type

### A. Estimators and Structs

Struct docstrings provide architectural overviews, algorithm details, hyperparameter listings, and fitted attribute references.

```mojo
struct RandomForestClassifier[compute_dtype: DType = DType.float64](Classifier, Copyable, Movable):
    """Random Forest Classifier for ensemble classification.

    An ensemble of decision trees trained via bootstrap aggregation (bagging).
    Predictions are computed via soft voting (averaging predicted class probabilities
    across all trees and selecting the argmax class).

    Parameters:
        n_estimators: Number of trees in the forest. Default 100.
        criterion: Impurity split metric: 'gini', 'entropy', or 'log_loss'. Default 'gini'.
        max_depth: Maximum tree depth. -1 means unlimited. Default -1.
        min_samples_split: Minimum samples required to split an internal node. Default 2.
        min_samples_leaf: Minimum samples required to be a leaf node. Default 1.
        bootstrap: Whether to draw bootstrap samples when building trees. Default True.
        oob_score: Whether to compute out-of-bag accuracy score after fitting. Default False.
        random_state: PRNG seed for deterministic tree builds. Default 42.

    Attributes:
        classes_: Sorted list of unique class labels seen during fit.
        n_features_in_: Number of features seen during fit.
        feature_importances_: Gini feature importance vector normalized to sum to 1.0.
        oob_score_: Out-of-bag accuracy score (available when oob_score=True).

    Examples:
        ```mojo
        from strata.ensemble import RandomForestClassifier
        from strata.core import Matrix

        var rf = RandomForestClassifier[DType.float64](n_estimators=50, max_depth=5)
        rf.fit(X_train, y_train)
        var predictions = rf.predict(X_test)
        ```
    """
```

---

### B. Constructors (`__init__`)

Constructor docstrings document hyperparameter defaults and validation constraints.

```mojo
    def __init__(
        out self,
        alpha: Scalar[Self.compute_dtype] = 1.0,
        fit_intercept: Bool = True,
        solver: String = "auto",
    ) raises:
        """Initialize the Ridge regression estimator.

        Args:
            alpha: L2 regularization strength. Must be non-negative. Default 1.0.
            fit_intercept: Whether to calculate the intercept bias term. Default True.
            solver: Solver algorithm ('auto', 'svd', 'cholesky', 'solve'). Default 'auto'.

        Raises:
            InvalidParameterError: If alpha is negative or solver is unrecognized.
        """
```

---

### C. Struct Methods

Method docstrings specify arguments, explicit return structures, and runtime errors.

```mojo
    def predict_proba[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> Matrix[feat_dtype]:
        """Predict class probabilities for input feature matrix.

        Args:
            X: Input feature matrix of shape $(N, D)$.

        Returns:
            Matrix[feat_dtype]: Predicted class probabilities of shape $(N, K)$,
                where $K$ is the number of classes. Each row sums to $1.0$.

        Raises:
            NotFittedError: If the estimator has not been fitted.
            DimensionMismatchError: If the feature dimension of X does not match n_features_in_.
        """
```

---

### D. Standalone Functions

Standalone functions (preprocessing utilities, metric evaluators, dataset splitters) document all input tensors and return containers.

```mojo
def train_test_split[
    feat_dtype: DType, target_dtype: DType
](
    X: Matrix[feat_dtype],
    y: List[Scalar[target_dtype]],
    test_size: Float64 = 0.25,
    shuffle: Bool = True,
    seed: Int = 42,
) raises -> TrainTestSplit[feat_dtype, target_dtype]:
    """Split matrices and target vectors into random train and test subsets.

    Args:
        X: Input feature matrix of shape $(N, D)$.
        y: Target label vector of length $N$.
        test_size: Proportion of the dataset to include in the test split. Default 0.25.
        shuffle: Whether to shuffle data before splitting. Default True.
        seed: PRNG seed for deterministic random shuffling. Default 42.

    Returns:
        TrainTestSplit: Container holding X_train, X_test, y_train, and y_test.

    Raises:
        InvalidParameterError: If test_size is not in (0.0, 1.0) or if N < 2.
        DimensionMismatchError: If X.rows != len(y).
    """
```

---

### E. Traits and Interfaces

Trait docstrings define interface contracts and behavioral requirements for implementations.

```mojo
trait Transformer(Estimator):
    """Interface for feature transformers.

    Implementations must provide fit, transform, and fit_transform methods
    compatible with 2D Matrix buffers and Dataset containers.
    """
```

---

## 4. Section Header Keywords

Standard Google-style section headers are required:

| Header | Usage | Format |
| :--- | :--- | :--- |
| `Parameters:` / `Args:` | Input arguments and configuration options. | `name: Description.` |
| `Attributes:` | Model state populated after calling `fit()`. | `name_: Description.` |
| `Returns:` | Return value description and type. | `Type: Description.` |
| `Raises:` | Exceptions raised under specific error conditions. | `ErrorType: Condition.` |
| `Examples:` | Executable Mojo code block. | Fenced code block ` ```mojo ... ``` ` |
| `Notes:` | Algorithmic details, numerical stability, or references. | Text or LaTeX math blocks. |

---

## 5. Reference Examples

```mojo
struct LogisticRegression[
    compute_dtype: DType = DType.float64
](Classifier, Copyable, Movable):
    """Logistic Regression classifier with L2 regularization and L-BFGS optimization.

    Fits a linear decision boundary by minimizing binary cross-entropy:

    $$
    \mathcal{L}(w, b) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \ln(\sigma(w^T x_i + b)) + (1 - y_i) \ln(1 - \sigma(w^T x_i + b)) \right] + \frac{\lambda}{2} \|w\|_2^2
    $$

    Parameters:
        C: Inverse regularization strength ($C = \frac{1}{\lambda}$). Must be positive. Default 1.0.
        fit_intercept: Whether to calculate the intercept bias term. Default True.
        max_iter: Maximum number of L-BFGS optimization iterations. Default 100.
        tol: Tolerance for stopping criterion. Default 1e-4.

    Attributes:
        classes_: Sorted list of unique class labels.
        coef_: Learned feature weight coefficients of shape $(1, D)$.
        intercept_: Learned bias intercept scalar.
        n_iter_: Number of optimization iterations completed.

    Examples:
        ```mojo
        from strata.linear_model import LogisticRegression
        from strata.core import Matrix

        var clf = LogisticRegression[DType.float64](C=1.0, max_iter=200)
        clf.fit(X_train, y_train)
        var y_pred = clf.predict(X_test)
        ```
    """
```

---

## 6. Anti-Patterns

| Anti-Pattern | Issue | Correct Approach |
| :--- | :--- | :--- |
| `||y - Xw||^2_2` | ASCII bars break Markdown table parsing. | `$\|y - Xw\|_2^2$` (LaTeX) |
| Missing parameter descriptions | Type information missing from generated docs. | `X: Feature matrix of shape $(N, D)$.` |
| Single-line docstring on struct | Omitting the blank line breaks the summary parser. | Summary on line 1, blank line, then details. |
| Undocumented exceptions | Unhandled runtime errors for callers. | Document conditions in `Raises:` block. |
| Inconsistent attribute names | Deviates from Scikit-Learn conventions. | Trailing underscore: `classes_`, `coef_`, `intercept_` |
