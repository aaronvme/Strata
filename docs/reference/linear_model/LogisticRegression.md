# `LogisticRegression`

**Module**: [`strata.linear_model`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Classifier, Copyable, Movable`  
**Source**: [`strata/linear_model/logistic_regression.mojo`](file:////home/ewu/Code/Strata/strata/linear_model/logistic_regression.mojo)

```mojo
struct LogisticRegression[compute_dtype: DType = DType.float64](Classifier, Copyable, Movable)
```

```mojo
from strata.linear_model import LogisticRegression
```

**Logistic Regression classifier with L2 regularization.**

Supports binary and multiclass (multinomial) classification by minimizing
the regularized cross-entropy loss with gradient optimization:
$$
\min_{W, b} -\frac{1}{N} \sum_{i=1}^{N} \ln P(y_i \mid x_i; W, b) + \frac{1}{2C} \|W\|_F^2
$$

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`LogisticRegression.fit()`](#fit) | Fits the logistic regression model on training data (X, y). |
| [`LogisticRegression.predict_proba()`](#predict_proba) | Predict class probability distributions for samples in X. |
| [`LogisticRegression.predict()`](#predict) | Predict discrete class labels for samples in X. |

---

## Method Details

### `LogisticRegression.fit()`

```mojo
def fit[feat_dtype: DType, target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```

Fits the logistic regression model on training data (X, y).

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and target labels. |

---

### `LogisticRegression.predict_proba()`

```mojo
def predict_proba[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> Matrix[feat_dtype]
def predict_proba[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Matrix[feat_dtype]
```

Predict class probability distributions for samples in X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |

**Returns**: `Matrix[feat_dtype]` — Matrix[feat_dtype]: Probability matrix of shape $(N, K)$, where row $i$ contains the normalized probability distribution over $K$ classes.

---

### `LogisticRegression.predict()`

```mojo
def predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Int]
def predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Int]
```

Predict discrete class labels for samples in X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |

**Returns**: `List[Int]` — List[Int]: Predicted class labels vector of length $N$.
---

## Example

```mojo
from strata.linear_model import LogisticRegression
from strata.core import Matrix

var clf = LogisticRegression[DType.float64](C=1.0, max_iter=200)
clf.fit(X_train, y_train)
var probs = clf.predict_proba(X_test)
var preds = clf.predict(X_test)
```
