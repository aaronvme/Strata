# `DatasetSplit`

**Module**: [`strata.core`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Movable`  
**Source**: [`strata/core/dataset.mojo`](file:////home/ewu/Code/Strata/strata/core/dataset.mojo)

```mojo
struct DatasetSplit[feat_dtype: DType = DType.float64, target_dtype: DType = DType.float64](Movable)
```

```mojo
from strata.core import DatasetSplit
```

**Container holding train and test partitions of a Dataset.**
