# `PipelineClassifier`

**Module**: [`strata.base`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Classifier, Copyable, Movable`  
**Source**: [`strata/base/pipeline.mojo`](file:////home/ewu/Code/Strata/strata/base/pipeline.mojo)

```mojo
struct PipelineClassifier[T: Transformer, C: Classifier, target_dtype: DType = DType.int32](Classifier, Copyable, Movable)
```

```mojo
from strata.base import PipelineClassifier
```

**Sequentially applies a transformer pipeline before fitting a classifier.**

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`PipelineClassifier.fit()`](#fit) | — |
| [`PipelineClassifier.predict()`](#predict) | — |
| [`PipelineClassifier.predict_proba()`](#predict_proba) | — |

---

## Method Details

### `PipelineClassifier.fit()`

```mojo
def fit[feat_dtype: DType, in_target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[in_target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```

Fits the classifier using a unified Dataset container.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[in_target_dtype]]` | Target vector / class labels. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and target labels. |

---

### `PipelineClassifier.predict()`

```mojo
def predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Int]
def predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Int]
```

Predicts class labels for a Dataset container.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |

**Returns**: `List[Int]`

---

### `PipelineClassifier.predict_proba()`

```mojo
def predict_proba[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> Matrix[feat_dtype]
def predict_proba[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Matrix[feat_dtype]
```

Predicts class probability distributions for a Dataset container.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |

**Returns**: `Matrix[feat_dtype]`

---
