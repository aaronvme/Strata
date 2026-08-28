# `TruncatedSVD`

**Module**: [`strata.decomposition`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Transformer`  
**Source**: [`strata/decomposition/truncated_svd.mojo`](file:////home/ewu/Code/Strata/strata/decomposition/truncated_svd.mojo)

```mojo
struct TruncatedSVD[compute_dtype: DType = DType.float64](Copyable, Movable, Transformer)
```

```mojo
from strata.decomposition import TruncatedSVD
```

**Dimensionality reduction using truncated SVD.**

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`TruncatedSVD.fit()`](#fit) | Fits TruncatedSVD on dense matrix X. |
| [`TruncatedSVD.transform()`](#transform) | Projects dense matrix X onto the truncated components. |
| [`TruncatedSVD.fit_transform()`](#fit_transform) | Fits TruncatedSVD to X and returns the projected data. |
| [`TruncatedSVD.inverse_transform()`](#inverse_transform) | Transforms data back to its original space. |

---

## Method Details

### `TruncatedSVD.fit()`

```mojo
def fit[in_dtype: DType](mut self, X: Matrix[in_dtype])
def fit[in_dtype: DType](mut self, X: CSRMatrix[in_dtype])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```

Fits TruncatedSVD on dense matrix X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |

---

### `TruncatedSVD.transform()`

```mojo
def transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
def transform[in_dtype: DType](self, X: CSRMatrix[in_dtype]) -> Matrix[in_dtype]
def transform[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[ feat_dtype, target_dtype ]
```

Projects dense matrix X onto the truncated components.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |

**Returns**: `Matrix[in_dtype]`

---

### `TruncatedSVD.fit_transform()`

```mojo
def fit_transform[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
def fit_transform[in_dtype: DType](mut self, X: CSRMatrix[in_dtype]) -> Matrix[in_dtype]
def fit_transform[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[ feat_dtype, target_dtype ]
```

Fits TruncatedSVD to X and returns the projected data.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |

**Returns**: `Matrix[in_dtype]`

---

### `TruncatedSVD.inverse_transform()`

```mojo
def inverse_transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
```

Transforms data back to its original space.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |

**Returns**: `Matrix[in_dtype]`

---
