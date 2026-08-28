# `SVDResult`

**Module**: [`strata.core`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable`  
**Source**: [`strata/core/linalg.mojo`](file:////home/ewu/Code/Strata/strata/core/linalg.mojo)

```mojo
struct SVDResult[dtype: DType = DType.float64](Copyable, Movable)
```

```mojo
from strata.core import SVDResult
```

**Result of Singular Value Decomposition ($A = U \Sigma V^T$).**
