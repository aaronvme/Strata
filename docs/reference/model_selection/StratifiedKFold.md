# `StratifiedKFold`

**Module**: [`strata.model_selection`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Movable`  
**Source**: [`strata/model_selection/stratified_kfold.mojo`](file:////home/ewu/Code/Strata/strata/model_selection/stratified_kfold.mojo)

```mojo
struct StratifiedKFold(Movable)
```

```mojo
from strata.model_selection import StratifiedKFold
```

**Stratified K-Fold cross-validator for classification datasets.**

Splits dataset into k folds such that each fold preserves approximately
the same percentage of samples for each target class.

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`StratifiedKFold.get_n_splits()`](#get_n_splits) | Returns the number of splitting iterations. |
| [`StratifiedKFold.split()`](#split) | Generates stratified train and test indices from target labels. |

---

## Method Details

### `StratifiedKFold.get_n_splits()`

```mojo
def get_n_splits(self) -> Int
```

Returns the number of splitting iterations.

**Returns**: `Int`

---

### `StratifiedKFold.split()`

```mojo
def split[target_dtype: DType](self, y: List[Scalar[target_dtype]]) -> List[Split]
def split[feat_dtype: DType, target_dtype: DType](self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]]) -> List[Split]
```

Generates stratified train and test indices from target labels.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |

**Returns**: `List[Split]` — List of Split instances containing train and validation indices.

---
