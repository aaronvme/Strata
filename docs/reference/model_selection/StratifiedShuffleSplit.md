# `StratifiedShuffleSplit`

**Module**: [`strata.model_selection`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Movable`  
**Source**: [`strata/model_selection/stratified_shuffle_split.mojo`](file:////home/ewu/Code/Strata/strata/model_selection/stratified_shuffle_split.mojo)

```mojo
struct StratifiedShuffleSplit(Movable)
```

```mojo
from strata.model_selection import StratifiedShuffleSplit
```

**Stratified random permutation cross-validator.**

Yields indices to split data into training and test sets. Each split is an
independent random draw that preserves the percentage of samples for each
target class, so successive test sets may overlap.

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`StratifiedShuffleSplit.get_n_splits()`](#get_n_splits) | Returns the number of splitting iterations in the cross-validator. |
| [`StratifiedShuffleSplit.split()`](#split) | Generates class-balanced random train and test indices for each split. |

---

## Method Details

### `StratifiedShuffleSplit.get_n_splits()`

```mojo
def get_n_splits(self) -> Int
```

Returns the number of splitting iterations in the cross-validator.

**Returns**: `Int`

---

### `StratifiedShuffleSplit.split()`

```mojo
def split[target_dtype: DType](self, y: List[Scalar[target_dtype]]) -> List[Split]
def split[feat_dtype: DType, target_dtype: DType](self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]]) -> List[Split]
```

Generates class-balanced random train and test indices for each split.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |

**Returns**: `List[Split]` — List of Split objects containing train and validation indices.

---
