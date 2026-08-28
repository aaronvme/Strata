# `RobustScaler`

**Module**: [`strata.preprocessing`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Transformer`  
**Source**: [`strata/preprocessing/scaler.mojo`](file:////home/ewu/Code/Strata/strata/preprocessing/scaler.mojo)

```mojo
struct RobustScaler[compute_dtype: DType = DType.float64](Copyable, Movable, Transformer)
```

```mojo
from strata.preprocessing import RobustScaler
```

**Scale features using statistics that are robust to outliers.**

Centers the data on the median and scales by the Interquartile Range (IQR):
$$
x_{\text{scaled}} = \frac{x - \text{median}}{\text{IQR}}
$$
where $\text{IQR} = Q_3 - Q_1$ (by default 75th percentile minus 25th percentile).

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Computational precision data type. Default DType.float64. |

---

## Arguments (Runtime)

| Argument | Description |
| :--- | :--- |
| **`with_centering`** | If True, center the data before scaling by subtracting the median. Default True. |
| **`with_scaling`** | If True, scale the data to interquartile range. Default True. |
| **`quantile_min`** | Lower quantile percentage of the scaling range ($0 <= q_{\min} < q_{\max} <= 100$). Default 25.0. |
| **`quantile_max`** | Upper quantile percentage of the scaling range. Default 75.0. |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`center_`** | Median value for each feature in the training set. |
| **`scale_`** | Interquartile range scaling factor for each feature. |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`RobustScaler.fit()`](#fit) | — |
| [`RobustScaler.transform()`](#transform) | — |
| [`RobustScaler.fit_transform()`](#fit_transform) | — |
| [`RobustScaler.inverse_transform()`](#inverse_transform) | Undoes the centering and scaling of X. |

---

## Method Details

### `RobustScaler.fit()`

```mojo
def fit[in_dtype: DType](mut self, X: Matrix[in_dtype])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |

---

### `RobustScaler.transform()`

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

### `RobustScaler.fit_transform()`

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

### `RobustScaler.inverse_transform()`

```mojo
def inverse_transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
```

Undoes the centering and scaling of X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |

**Returns**: `Matrix[in_dtype]`
---

## Example

```mojo
from strata.preprocessing import RobustScaler
from strata.core import Matrix

var scaler = RobustScaler[DType.float64]()
scaler.fit(X_train)
var X_scaled = scaler.transform(X_train)
```
