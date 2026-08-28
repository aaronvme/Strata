# `Ridge`

**Module**: [`strata.linear_model`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Regressor`  
**Source**: [`strata/linear_model/ridge.mojo`](file:////home/ewu/Code/Strata/strata/linear_model/ridge.mojo)

```mojo
struct Ridge[compute_dtype: DType = DType.float64](Copyable, Movable, Regressor)
```

```mojo
from strata.linear_model import Ridge
```

**Ridge regression with L2 regularization.**

Minimizes the penalized objective function:
$$
\min_{w} \|y - Xw\|_2^2 + \alpha \|w\|_2^2
$$

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`Ridge.fit()`](#fit) | Fit the Ridge regression model from training data. |
| [`Ridge.predict()`](#predict) | Predict continuous target values using the fitted linear model. |

---

## Method Details

### `Ridge.fit()`

```mojo
def fit[feat_dtype: DType, in_target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[in_target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```

Fit the Ridge regression model from training data.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[in_target_dtype]]` | Target vector / class labels. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. |

---

### `Ridge.predict()`

```mojo
def predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Scalar[feat_dtype]]
def predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Scalar[feat_dtype]]
```

Predict continuous target values using the fitted linear model.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |

**Returns**: `List[Scalar[feat_dtype]]` — List[Scalar[feat_dtype]]: Predicted target vector of length $N$.
---

## Example

```mojo
from strata.linear_model import Ridge
from strata.core import Matrix

var model = Ridge[DType.float64](alpha=0.5)
model.fit(X_train, y_train)
var preds = model.predict(X_test)
```
