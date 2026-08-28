# `KMeans`

**Module**: [`strata.cluster`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Clusterer, Copyable, Movable`  
**Source**: [`strata/cluster/kmeans.mojo`](file:////home/ewu/Code/Strata/strata/cluster/kmeans.mojo)

```mojo
struct KMeans[compute_dtype: DType = DType.float64](Clusterer, Copyable, Movable)
```

```mojo
from strata.cluster import KMeans
```

**K-Means clustering using Lloyd's or Elkan's algorithm.**

Clusters $N$ observations into $K$ disjoint geometric partitions by
minimizing within-cluster inertia (sum-of-squared Euclidean distances):
$$
\arg\min_{C} \sum_{i=1}^{N} \min_{\mu_j \in C} \|x_i - \mu_j\|_2^2
$$

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`KMeans.fit()`](#fit) | — |
| [`KMeans.predict()`](#predict) | — |
| [`KMeans.fit_predict()`](#fit_predict) | — |
| [`KMeans.transform()`](#transform) | — |
| [`KMeans.fit_transform()`](#fit_transform) | — |
| [`KMeans.score()`](#score) | — |

---

## Method Details

### `KMeans.fit()`

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

### `KMeans.predict()`

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

### `KMeans.fit_predict()`

```mojo
def fit_predict[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> List[Int]
```

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |

**Returns**: `List[Int]`

---

### `KMeans.transform()`

```mojo
def transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
```

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |

**Returns**: `Matrix[in_dtype]`

---

### `KMeans.fit_transform()`

```mojo
def fit_transform[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
```

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |

**Returns**: `Matrix[in_dtype]`

---

### `KMeans.score()`

```mojo
def score[in_dtype: DType](self, X: Matrix[in_dtype]) -> Scalar[Self.compute_dtype]
```

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |

**Returns**: `Scalar[Self.compute_dtype]`
---

## Example

```mojo
from strata.cluster import KMeans
from strata.core import Matrix

var kmeans = KMeans[DType.float64](n_clusters=3, init="k-means++")
kmeans.fit(X_data)
var labels = kmeans.predict(X_data)
var distances = kmeans.transform(X_data)
```
