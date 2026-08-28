# `MiniBatchKMeans`

**Module**: [`strata.cluster`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Clusterer, Copyable, Movable`  
**Source**: [`strata/cluster/minibatch_kmeans.mojo`](file:////home/ewu/Code/Strata/strata/cluster/minibatch_kmeans.mojo)

```mojo
struct MiniBatchKMeans[compute_dtype: DType = DType.float64](Clusterer, Copyable, Movable)
```

```mojo
from strata.cluster import MiniBatchKMeans
```

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`MiniBatchKMeans.partial_fit()`](#partial_fit) | — |
| [`MiniBatchKMeans.fit()`](#fit) | — |
| [`MiniBatchKMeans.predict()`](#predict) | — |
| [`MiniBatchKMeans.fit_predict()`](#fit_predict) | — |
| [`MiniBatchKMeans.transform()`](#transform) | — |
| [`MiniBatchKMeans.fit_transform()`](#fit_transform) | — |
| [`MiniBatchKMeans.score()`](#score) | — |

---

## Method Details

### `MiniBatchKMeans.partial_fit()`

```mojo
def partial_fit[in_dtype: DType](mut self, X: Matrix[in_dtype])
```

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |

---

### `MiniBatchKMeans.fit()`

```mojo
def fit[in_dtype: DType](mut self, X: Matrix[in_dtype])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```

Fits cluster centroids on Dataset feature records.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |

---

### `MiniBatchKMeans.predict()`

```mojo
def predict[in_dtype: DType](self, X: Matrix[in_dtype]) -> List[Int]
def predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Int]
```

Predicts closest cluster assignments for a Dataset container.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |

**Returns**: `List[Int]`

---

### `MiniBatchKMeans.fit_predict()`

```mojo
def fit_predict[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> List[Int]
```

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |

**Returns**: `List[Int]`

---

### `MiniBatchKMeans.transform()`

```mojo
def transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
```

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |

**Returns**: `Matrix[in_dtype]`

---

### `MiniBatchKMeans.fit_transform()`

```mojo
def fit_transform[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
```

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |

**Returns**: `Matrix[in_dtype]`

---

### `MiniBatchKMeans.score()`

```mojo
def score[in_dtype: DType](self, X: Matrix[in_dtype]) -> Scalar[Self.compute_dtype]
```

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |

**Returns**: `Scalar[Self.compute_dtype]`

---
