# `GridSearchRegressor`

**Module**: [`strata.model_selection`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Regressor`  
**Source**: [`strata/model_selection/grid_search.mojo`](file:////home/ewu/Code/Strata/strata/model_selection/grid_search.mojo)

```mojo
struct GridSearchRegressor[ModelType: Regressor, feat_dtype: DType = DType.float64, target_dtype: DType = DType.float64](Copyable, Movable, Regressor)
```

```mojo
from strata.model_selection import GridSearchRegressor
```

**Exhaustive hyperparameter grid search for regression models.**

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`GridSearchRegressor.fit()`](#fit) | Runs cross-validation across all candidate models and fits the best one. |
| [`GridSearchRegressor.predict()`](#predict) | Predicts targets using the best discovered model configuration. |

---

## Method Details

### `GridSearchRegressor.fit()`

```mojo
def fit[in_feat_dtype: DType, in_target_dtype: DType](mut self, X: Matrix[in_feat_dtype], y: List[Scalar[in_target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```

Runs cross-validation across all candidate models and fits the best one.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[in_target_dtype]]` | Target vector / class labels. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. |

---

### `GridSearchRegressor.predict()`

```mojo
def predict[in_feat_dtype: DType](self, X: Matrix[in_feat_dtype]) -> List[Scalar[in_feat_dtype]]
def predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Scalar[feat_dtype]]
```

Predicts targets using the best discovered model configuration.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |

**Returns**: `List[Scalar[in_feat_dtype]]`

---
