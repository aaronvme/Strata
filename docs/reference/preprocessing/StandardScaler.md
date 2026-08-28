# `StandardScaler`

**Module**: [`strata.preprocessing`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Transformer`  
**Source**: [`strata/preprocessing/scaler.mojo`](file:////home/ewu/Code/Strata/strata/preprocessing/scaler.mojo)

```mojo
struct StandardScaler[compute_dtype: DType = DType.float64](Copyable, Movable, Transformer)
```

```mojo
from strata.preprocessing import StandardScaler
```

**Standardize features by removing the mean and scaling to unit variance.**

The standard score of a sample $x$ is calculated as:
$$
z = \frac{x - \mu}{\sigma}
$$
where $\mu$ is the mean of the training samples and $\sigma$ is the standard deviation.

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`StandardScaler.fit()`](#fit) | — |
| [`StandardScaler.transform()`](#transform) | — |
| [`StandardScaler.fit_transform()`](#fit_transform) | — |

---

## Method Details

### `StandardScaler.fit()`

```mojo
def fit[in_dtype: DType](mut self, X: Matrix[in_dtype])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |

---

### `StandardScaler.transform()`

```mojo
def transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
def transform[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[ feat_dtype, target_dtype ]
```

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |

**Returns**: `Matrix[in_dtype]`

---

### `StandardScaler.fit_transform()`

```mojo
def fit_transform[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
def fit_transform[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[ feat_dtype, target_dtype ]
```

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |

**Returns**: `Matrix[in_dtype]`
---

## Example

```mojo
from strata.preprocessing import StandardScaler
from strata.core import Matrix

var scaler = StandardScaler[DType.float64]()
scaler.fit(X_train)
var X_scaled = scaler.transform(X_train)
```
