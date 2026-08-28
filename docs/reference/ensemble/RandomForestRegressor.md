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
