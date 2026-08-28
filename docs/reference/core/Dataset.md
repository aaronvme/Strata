# `Dataset`

**Module**: [`strata.core`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable`  
**Source**: [`strata/core/dataset.mojo`](file:////home/ewu/Code/Strata/strata/core/dataset.mojo)

```mojo
struct Dataset[feat_dtype: DType = DType.float64, target_dtype: DType = DType.float64](Copyable, Movable)
```

```mojo
from strata.core import Dataset
```

**Machine learning dataset container pairing a feature matrix with targets.**

Encapsulates 2D feature observations, 1D target labels/values, and optional
feature/target name metadata.

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`Dataset.n_samples()`](#n_samples) | — |
| [`Dataset.n_features()`](#n_features) | — |
| [`Dataset.split_with_ratio()`](#split_with_ratio) | — |

---

## Method Details

### `Dataset.n_samples()`

```mojo
def n_samples(self) -> Int
```

**Returns**: `Int`

---

### `Dataset.n_features()`

```mojo
def n_features(self) -> Int
```

**Returns**: `Int`

---

### `Dataset.split_with_ratio()`

```mojo
def split_with_ratio(self, ratio: Float64 = 0.25, shuffle: Bool = True, seed: Int = 42) -> DatasetSplit[Self.feat_dtype, Self.target_dtype]
```

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`ratio`** | `Float64` | — |
| **`shuffle`** | `Bool` | — |
| **`seed`** | `Int` | — |

**Returns**: `DatasetSplit[Self.feat_dtype, Self.target_dtype]`
---

## Example

```mojo
from strata.core import Matrix, Dataset

var X = Matrix[DType.float64](100, 4, fill=1.0)
var y = List[Scalar[DType.float64]](capacity=100)
var ds = Dataset(X^, y^)
var split = ds.split_with_ratio(ratio=0.2)
```
