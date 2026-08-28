# `DecisionTreeRegressor`

**Module**: [`strata.tree`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Regressor`  
**Source**: [`strata/tree/regressor.mojo`](file:////home/ewu/Code/Strata/strata/tree/regressor.mojo)

```mojo
struct DecisionTreeRegressor[compute_dtype: DType = DType.float64](Copyable, Movable, Regressor)
```

```mojo
from strata.tree import DecisionTreeRegressor
```

**Decision Tree Regressor for non-parametric continuous target regression.**

Builds a regression tree by minimizing sample variance (mean squared error)
or mean absolute deviation across recursive binary splits:
$$
H_{\text{MSE}}(Q) = \frac{1}{|Q|} \sum_{i \in Q} (y_i - \bar{y}_Q)^2
$$

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`DecisionTreeRegressor.fit()`](#fit) | Fits the decision tree regressor on (X, y). |
| [`DecisionTreeRegressor.predict()`](#predict) | Generates regression predictions for input matrix X. |
| [`DecisionTreeRegressor.get_depth()`](#get_depth) | Returns the maximum depth of the fitted tree. |
| [`DecisionTreeRegressor.get_n_leaves()`](#get_n_leaves) | Returns the total number of leaf nodes in the fitted tree. |

---

## Method Details

### `DecisionTreeRegressor.fit()`

```mojo
def fit[feat_dtype: DType, target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```

Fits the decision tree regressor on (X, y).

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. |

---

### `DecisionTreeRegressor.predict()`

```mojo
def predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Scalar[feat_dtype]]
def predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Scalar[feat_dtype]]
```

Generates regression predictions for input matrix X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |

**Returns**: `List[Scalar[feat_dtype]]`

---

### `DecisionTreeRegressor.get_depth()`

```mojo
def get_depth(self) -> Int
```

Returns the maximum depth of the fitted tree.

**Returns**: `Int`

---

### `DecisionTreeRegressor.get_n_leaves()`

```mojo
def get_n_leaves(self) -> Int
```

Returns the total number of leaf nodes in the fitted tree.

**Returns**: `Int`
---

## Example

```mojo
from strata.tree import DecisionTreeRegressor
from strata.core import Matrix

var reg = DecisionTreeRegressor[DType.float64](max_depth=4)
reg.fit(X_train, y_train)
var preds = reg.predict(X_test)
```
