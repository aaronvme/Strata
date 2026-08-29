# `ShuffleSplit`

**Module**: [`strata.model_selection`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Movable`  
**Source**: [`strata/model_selection/shuffle_split.mojo`](file:////home/ewu/Code/Strata/strata/model_selection/shuffle_split.mojo)

```mojo
struct ShuffleSplit(Movable)
```

```mojo
from strata.model_selection import ShuffleSplit
```

**Random permutation cross-validator.**

Yields indices to split data into training and test sets. Each split is an
independent random permutation of the samples, so successive test sets may
overlap. Sizes are expressed as proportions of the total sample count.

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`ShuffleSplit.get_n_splits()`](#get_n_splits) | Returns the number of splitting iterations in the cross-validator. |
| [`ShuffleSplit.split()`](#split) | Generates randomly permuted train and test indices for each split. |

---

## Method Details

### `ShuffleSplit.get_n_splits()`

```mojo
def get_n_splits(self) -> Int
```

Returns the number of splitting iterations in the cross-validator.

**Returns**: `Int`

---

### `ShuffleSplit.split()`

```mojo
def split(self, n_samples: Int) -> List[Split]
def split[dtype: DType](self, X: Matrix[dtype]) -> List[Split]
```

Generates randomly permuted train and test indices for each split.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`n_samples`** | `Int` | — |
| **`X`** | `Matrix[dtype]` | Feature matrix. |

**Returns**: `List[Split]` — List of Split objects containing train and validation indices.

---
