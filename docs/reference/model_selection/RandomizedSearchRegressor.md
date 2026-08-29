# `RandomizedSearchRegressor`

**Module**: [`strata.model_selection`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Regressor`  
**Source**: [`strata/model_selection/randomized_search.mojo`](file:////home/ewu/Code/Strata/strata/model_selection/randomized_search.mojo)

```mojo
struct RandomizedSearchRegressor[ModelType: Regressor, feat_dtype: DType = DType.float64, target_dtype: DType = DType.float64](Copyable, Movable, Regressor)
```

```mojo
from strata.model_selection import RandomizedSearchRegressor
```

**Randomized hyperparameter search for regression models.**

Evaluates a random subset of the supplied candidate configurations rather
than the full grid, trading exhaustive coverage for a fixed search budget.

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`RandomizedSearchRegressor.fit()`](#fit) | Cross-validates a random subset of candidates and fits the best one. |
| [`RandomizedSearchRegressor.predict()`](#predict) | Predicts targets using the best discovered model configuration. |

---

## Method Details

### `RandomizedSearchRegressor.fit()`

```mojo
def fit[in_feat_dtype: DType, in_target_dtype: DType](mut self, X: Matrix[in_feat_dtype], y: List[Scalar[in_target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```

Cross-validates a random subset of candidates and fits the best one.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[in_target_dtype]]` | Target vector / class labels. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. |

---

### `RandomizedSearchRegressor.predict()`

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
