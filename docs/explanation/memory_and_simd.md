# Memory Model, SIMD Execution & Value Semantics

This document explains the low-level design choices underpinning Strata's linear algebra engine, data layouts, and memory safety model in Mojo.


---

## 1. Contiguous Row-Major Allocation

Traditional Python libraries like NumPy and scikit-learn manage memory through the CPython heap with PyObject wrappers and reference counting.

In Strata, `Matrix[dtype]` manages a single contiguous heap buffer aligned to the hardware's SIMD vector width (e.g. 512-bit for AVX-512 / ARM Neon):

```mojo
struct Matrix[dtype: DType](Copyable, Movable):
    var rows: Int
    var cols: Int
    var data: UnsafePointer[Scalar[dtype]]
```

### Benefits:
- **Spatial Locality**: Row-major traversal matches CPU cache line loading patterns (64 bytes per line).
- **Direct SIMD Vectorization**: Inner loops load `simd_width = simdwidthof[dtype]()` elements per CPU cycle without memory shuffling.

---

## 2. Zero-Copy Views via `MatrixView`

When slicing a matrix or passing a subset of features to an estimator, creating a deep copy introduces $\mathcal{O}(M \times N)$ memory allocation churn.

Strata provides `MatrixView[dtype, origin]`:
- Tracks a pointer offset, row stride, and column count.
- Enforces compile-time origin lifetime tracking to guarantee that the view cannot outlive the parent matrix.

---

## 3. Mojo Value Semantics & Transfer Operators

Strata data structures strictly adhere to Mojo's value lifecycle:
- **`Copyable`**: Explicit `.copy()` when a new independent buffer is required.
- **`Movable`**: Transfer operator (`^`) moves pointers directly into target structs with zero-cost destruction of the source handle.

```mojo
# Moves the memory buffer into the estimator without heap reallocation
var model = PipelineRegressor(scaler^, regressor^)
```
