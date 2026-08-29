# `RandomForestRegressor`

**Module**: [`strata.ensemble`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Regressor`  
**Source**: [`strata/ensemble/forest.mojo`](file:////home/ewu/Code/Strata/strata/ensemble/forest.mojo)

```mojo
struct RandomForestRegressor[compute_dtype: DType = DType.float64](Copyable, Movable, Regressor)
```

```mojo
from strata.ensemble import RandomForestRegressor
```

**Random Forest Regressor ensemble estimator.**

An ensemble of decision trees trained via bootstrap aggregation (bagging).
Predictions are computed as the arithmetic mean of individual tree predictions.

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Computational precision data type. Default DType.float64. |

---

## Arguments (Runtime)

| Argument | Description |
| :--- | :--- |
| **`n_estimators`** | Number of trees in the forest. Default 100. |
| **`criterion`** | Impurity split criterion ('squared_error', 'friedman_mse', 'absolute_error'). Default 'squared_error'. |
| **`max_depth`** | Maximum tree depth. -1 means unlimited. Default -1. |
| **`min_samples_split`** | Minimum samples required to split an internal node. Default 2. |
| **`min_samples_leaf`** | Minimum samples required to be a leaf node. Default 1. |
| **`min_impurity_decrease`** | Split threshold if impurity decrease >= this value. Default 0.0. |
| **`max_features`** | Number of features to consider per split ('all', 'sqrt', 'log2'). Default 'sqrt'. |
| **`max_features_count`** | Exact number of features per split. Default -1 (disabled). |
| **`max_features_ratio`** | Proportion of features per split. Default 0.0 (disabled). |
| **`bootstrap`** | Whether to use bootstrap sampling. Default True. |
| **`max_samples_ratio`** | Proportion of samples drawn per tree when bootstrap=True. Default 1.0. |
| **`max_samples_count`** | Exact number of samples drawn per tree. Default -1 (disabled). |
| **`oob_score`** | Whether to compute out-of-bag $R^2$ score after fitting. Default False. |
| **`random_state`** | PRNG seed for deterministic tree builds. Default 42. |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`n_features_in_`** | Number of features seen during fit. |
| **`feature_importances_`** | Normalized impurity feature importance vector. |
| **`oob_score_`** | Out-of-bag $R^2$ score (available when oob_score=True). |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`RandomForestRegressor.fit()`](#fit) | Fits the random forest on (X, y). |
| [`RandomForestRegressor.predict()`](#predict) | Predicts regression targets as the arithmetic mean across all tree predictions. |
| [`RandomForestRegressor.get_n_estimators()`](#get_n_estimators) | Returns the number of fitted trees. |
| [`RandomForestRegressor.get_feature_importances()`](#get_feature_importances) | Returns normalized MDI feature importances (sums to 1.0). |
| [`RandomForestRegressor.get_oob_score()`](#get_oob_score) | Returns out-of-bag R² score. Requires oob_score=True and bootstrap=True. |

---

## Method Details

### `RandomForestRegressor.fit()`

```mojo
def fit[feat_dtype: DType, target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```

Fits the random forest on (X, y).

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. |

---

### `RandomForestRegressor.predict()`

```mojo
def predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Scalar[feat_dtype]]
def predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Scalar[feat_dtype]]
```

Predicts regression targets as the arithmetic mean across all tree predictions.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |

**Returns**: `List[Scalar[feat_dtype]]`

---

### `RandomForestRegressor.get_n_estimators()`

```mojo
def get_n_estimators(self) -> Int
```

Returns the number of fitted trees.

**Returns**: `Int`

---

### `RandomForestRegressor.get_feature_importances()`

```mojo
def get_feature_importances(self) -> List[Float64]
```

Returns normalized MDI feature importances (sums to 1.0).

**Returns**: `List[Float64]`

---

### `RandomForestRegressor.get_oob_score()`

```mojo
def get_oob_score(self) -> Float64
```

Returns out-of-bag R² score. Requires oob_score=True and bootstrap=True.

**Returns**: `Float64`
---

## Example

```mojo
from strata.ensemble import RandomForestRegressor
from strata.core import Matrix

var rf = RandomForestRegressor[DType.float64](n_estimators=50, max_depth=6)
rf.fit(X_train, y_train)
var preds = rf.predict(X_test)
```
