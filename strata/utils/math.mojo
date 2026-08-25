from std.math import exp
from ..core.matrix import Matrix


def sigmoid[dtype: DType = DType.float64](x: Scalar[dtype]) -> Scalar[dtype]:
    """Numerically stable sigmoid function."""
    var z = Float64(x)
    if z >= 0.0:
        return Scalar[dtype](1.0 / (1.0 + exp(-z)))
    else:
        var ez = exp(z)
        return Scalar[dtype](ez / (1.0 + ez))


def softmax[
    dtype: DType = DType.float64
](x: List[Scalar[dtype]]) -> List[Scalar[dtype]]:
    """Numerically stable softmax function using max subtraction."""
    var n = len(x)
    var res = List[Scalar[dtype]](capacity=n)
    if n == 0:
        return res^

    var max_val: Float64 = Float64(x[0])
    for i in range(1, n):
        var v = Float64(x[i])
        if v > max_val:
            max_val = v

    var exp_vals = List[Float64](capacity=n)
    var sum_exp: Float64 = 0.0
    for i in range(n):
        var e = exp(Float64(x[i]) - max_val)
        exp_vals.append(e)
        sum_exp += e

    var inv_sum_exp = 1.0 / sum_exp
    for i in range(n):
        res.append(Scalar[dtype](exp_vals[i] * inv_sum_exp))
    return res^


def log_sum_exp[
    dtype: DType = DType.float64
](x: List[Scalar[dtype]]) -> Scalar[dtype]:
    """Numerically stable log-sum-exp: LSE(x) = max(x) + ln(sum(exp(x - max(x)))).
    """
    from std.math import log

    var n = len(x)
    if n == 0:
        return 0.0

    var max_val: Float64 = Float64(x[0])
    for i in range(1, n):
        var v = Float64(x[i])
        if v > max_val:
            max_val = v

    var sum_exp: Float64 = 0.0
    for i in range(n):
        sum_exp += exp(Float64(x[i]) - max_val)

    return Scalar[dtype](max_val + log(sum_exp))
