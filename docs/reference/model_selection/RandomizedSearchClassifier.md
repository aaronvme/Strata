# `RandomizedSearchClassifier`

**Module**: [`strata.model_selection`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Classifier, Copyable, Movable`  
**Source**: [`strata/model_selection/randomized_search.mojo`](file:////home/ewu/Code/Strata/strata/model_selection/randomized_search.mojo)

```mojo
struct RandomizedSearchClassifier[ModelType: Classifier, feat_dtype: DType = DType.float64, target_dtype: DType = DType.int32](Classifier, Copyable, Movable)
```

```mojo
from strata.model_selection import RandomizedSearchClassifier
```

**Randomized hyperparameter search for classification models.**

Evaluates a random subset of the supplied candidate configurations rather
than the full grid, trading exhaustive coverage for a fixed search budget.

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`RandomizedSearchClassifier.fit()`](#fit) | Cross-validates a random subset of candidates and fits the best one. |
| [`RandomizedSearchClassifier.predict()`](#predict) | Predicts class labels using the best discovered model configuration. |
| [`RandomizedSearchClassifier.predict_proba()`](#predict_proba) | Predicts class probabilities using the best model configuration. |

---

## Method Details

### `RandomizedSearchClassifier.fit()`

```mojo
def fit[in_feat_dtype: DType, in_target_dtype: DType](mut self, X: Matrix[in_feat_dtype], y: List[Scalar[in_target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```

Cross-validates a random subset of candidates and fits the best one.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[in_target_dtype]]` | Target vector / class labels. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and target labels. |

---

### `RandomizedSearchClassifier.predict()`

```mojo
def predict[in_feat_dtype: DType](self, X: Matrix[in_feat_dtype]) -> List[Int]
def predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Int]
```

Predicts class labels using the best discovered model configuration.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |

**Returns**: `List[Int]`

---

### `RandomizedSearchClassifier.predict_proba()`

```mojo
def predict_proba[in_feat_dtype: DType](self, X: Matrix[in_feat_dtype]) -> Matrix[in_feat_dtype]
def predict_proba[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Matrix[feat_dtype]
```

Predicts class probabilities using the best model configuration.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |

**Returns**: `Matrix[in_feat_dtype]`

---
