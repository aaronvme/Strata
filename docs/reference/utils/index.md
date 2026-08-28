# `strata.utils`

64-bit SplitMix64 PRNG with Lemire unbiased sampling, mathematical activations (softmax, sigmoid, log_sum_exp), and validation helpers.

---

## Structs & Classes

| Struct | Description |
| :--- | :--- |
| [`PRNG`](PRNG.md) | 64-bit SplitMix64 pseudo-random number generator with unbiased range generation. |
| [`NotFittedError`](NotFittedError.md) | Exception raised when an estimator is used before calling `fit`. |
| [`DimensionMismatchError`](DimensionMismatchError.md) | Exception raised when input matrix/vector dimensions do not match requirements. |
| [`ConvergenceError`](ConvergenceError.md) | Exception raised when iterative optimization fails to converge within max iterations. |
| [`InvalidParameterError`](InvalidParameterError.md) | Exception raised when an invalid hyperparameter value is supplied. |
| [`DataConversionError`](DataConversionError.md) | Exception raised when data type conversion or matrix array formatting fails. |

## Functions

| Function | Description |
| :--- | :--- |
| [`sigmoid`](sigmoid.md) | Compute the logistic sigmoid function $\sigma(x)$. |
| [`softmax`](softmax.md) | Compute numerically stable softmax probability distribution. |
| [`log_sum_exp`](log_sum_exp.md) | Compute numerically stable log-sum-exp: $\text{LSE}(x) = \ln \sum_i e^{x_i}$. |
| [`check_is_fitted`](check_is_fitted.md) | — |
| [`check_floating_dtype`](check_floating_dtype.md) | Asserts at compile time that the specified dtype is a floating-point type. |
| [`check_array`](check_array.md) | — |
| [`check_X_y`](check_X_y.md) | — |
| [`check_finite`](check_finite.md) | Rejects NaN and infinite entries in a target or prediction list. |
| [`check_consistent_length`](check_consistent_length.md) | — |
| [`check_consistent_length`](check_consistent_length.md) | — |
| [`check_sparse`](check_sparse.md) | Validates CSR/CSC sparse matrix format invariants. |
