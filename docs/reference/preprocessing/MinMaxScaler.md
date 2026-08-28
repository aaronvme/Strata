# `MinMaxScaler`

**Module**: [`strata.preprocessing`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Transformer`  
**Source**: [`strata/preprocessing/scaler.mojo`](file:////home/ewu/Code/Strata/strata/preprocessing/scaler.mojo)

```mojo
struct MinMaxScaler[compute_dtype: DType = DType.float64](Copyable, Movable, Transformer)
```

```mojo
from strata.preprocessing import MinMaxScaler
```

**Transform features by scaling each feature to a specified range.**

Scales and translates each feature individually such that it is in the given
range on the training set, e.g. between zero and one:
$$
x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}} \cdot (\text{max} - \text{min}) + \text{min}
$$

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Computational precision data type. Default DType.float64. |

---

## Arguments (Runtime)

| Argument | Description |
| :--- | :--- |
| **`feature_range_min`** | Lower bound of the desired transformed range. Default 0.0. |
| **`feature_range_max`** | Upper bound of the desired transformed range. Default 1.0. |
| **`clip`** | Whether to clip transformed values to the feature range. Default False. |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`data_min_`** | Per-feature minimum seen in the training data. |
| **`data_max_`** | Per-feature maximum seen in the training data. |
| **`data_range_`** | Per-feature range ($x_{\max} - x_{\min}$) seen in the data. |
| **`scale_`** | Per-feature relative scaling factor. |
| **`min_`** | Per-feature minimum adjustment. |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`MinMaxScaler.fit()`](#fit) | — |
| [`MinMaxScaler.transform()`](#transform) | — |
| [`MinMaxScaler.fit_transform()`](#fit_transform) | — |
| [`MinMaxScaler.inverse_transform()`](#inverse_transform) | Undoes the scaling of X according to the fitted feature range. |

---

## Method Details

### `MinMaxScaler.fit()`

```mojo
def fit[in_dtype: DType](mut self, X: Matrix[in_dtype])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |

---

### `MinMaxScaler.transform()`

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

### `MinMaxScaler.fit_transform()`

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

### `MinMaxScaler.inverse_transform()`

```mojo
def inverse_transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
```

Undoes the scaling of X according to the fitted feature range.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |

**Returns**: `Matrix[in_dtype]`
---

## Example

```mojo
from strata.preprocessing import MinMaxScaler
from strata.core import Matrix

var scaler = MinMaxScaler[DType.float64](feature_range_min=0.0, feature_range_max=1.0)
scaler.fit(X_train)
var X_scaled = scaler.transform(X_train)
```
