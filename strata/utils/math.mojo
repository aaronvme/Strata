from std.math import exp
from ..core.matrix import Matrix

def sigmoid[dtype: DType = DType.float64](x: Scalar[dtype]) -> Scalar[dtype]:
    var z = Float64(x)
    return Scalar[dtype](1.0 / (1.0 + exp(-z)))

def softmax[dtype: DType = DType.float64](x: List[Scalar[dtype]]) -> List[Scalar[dtype]]:
    var res = List[Scalar[dtype]](capacity=len(x))
    var sum_exp: Float64 = 0.0
    for i in range(len(x)):
        var e = exp(Float64(x[i]))
        sum_exp += e
    for i in range(len(x)):
        var p = exp(Float64(x[i])) / sum_exp
        res.append(Scalar[dtype](p))
    return res^
