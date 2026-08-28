# `DataConversionError`

**Module**: [`strata.utils`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Writable`  
**Source**: [`strata/exceptions/errors.mojo`](file:////home/ewu/Code/Strata/strata/exceptions/errors.mojo)

```mojo
struct DataConversionError(Copyable, Movable, Writable)
```

```mojo
from strata.utils import DataConversionError
```

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`DataConversionError.error()`](#error) | — |
| [`DataConversionError.write_to()`](#write_to) | — |

---

## Method Details

### `DataConversionError.error()`

```mojo
def error(msg: String) -> Error
```

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`msg`** | `String` | — |

**Returns**: `Error`

---

### `DataConversionError.write_to()`

```mojo
def write_to(self, mut writer: Some[Writer])
```

---
