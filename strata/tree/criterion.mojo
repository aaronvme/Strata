from std.math import log2, log, sqrt

from ..exceptions.errors import InvalidParameterError


def gini_impurity(counts: List[Int], total: Int) -> Float64:
    """Calculates Gini impurity from class counts: 1 - sum((count_c / total)^2).
    """
    if total <= 0:
        return 0.0
    var sum_sq_counts = 0
    for c in range(len(counts)):
        var cnt = counts[c]
        sum_sq_counts += cnt * cnt

    var total_f = Float64(total)
    return 1.0 - (Float64(sum_sq_counts) / (total_f * total_f))


def gini_from_sum_sq(sum_sq_counts: Int, total: Int) -> Float64:
    """Calculates Gini impurity from precomputed sum of squared counts."""
    if total <= 0:
        return 0.0
    var total_f = Float64(total)
    return 1.0 - (Float64(sum_sq_counts) / (total_f * total_f))


def entropy_impurity(
    counts: List[Int], total: Int, use_natural_log: Bool = False
) -> Float64:
    """Calculates Shannon entropy (base 2 or natural log) from class counts."""
    if total <= 0:
        return 0.0
    var total_f = Float64(total)
    var sum_n_log_n: Float64 = 0.0
    for c in range(len(counts)):
        var cnt = counts[c]
        if cnt > 0:
            var cnt_f = Float64(cnt)
            if use_natural_log:
                sum_n_log_n += cnt_f * log(cnt_f)
            else:
                sum_n_log_n += cnt_f * log2(cnt_f)

    if use_natural_log:
        return log(total_f) - (sum_n_log_n / total_f)
    return log2(total_f) - (sum_n_log_n / total_f)


def squared_error_impurity[
    dtype: DType
](y: List[Scalar[dtype]], indices: List[Int]) -> Float64:
    """Calculates sample variance for target subset."""
    var n = len(indices)
    if n <= 1:
        return 0.0
    var n_f = Float64(n)
    var sum_y: Float64 = 0.0
    var sum_sq_y: Float64 = 0.0

    for i in range(n):
        var val = Float64(y[indices[i]])
        sum_y += val
        sum_sq_y += val * val

    var mean_y = sum_y / n_f
    var var_y = (sum_sq_y / n_f) - (mean_y * mean_y)
    return var_y if var_y > 0.0 else 0.0


def variance_from_sums(sum_y: Float64, sum_sq_y: Float64, n: Int) -> Float64:
    """Calculates sample variance from precomputed sum and sum of squares."""
    if n <= 1:
        return 0.0
    var n_f = Float64(n)
    var mean = sum_y / n_f
    var v = (sum_sq_y / n_f) - (mean * mean)
    return v if v > 0.0 else 0.0


def absolute_error_impurity[
    dtype: DType
](y: List[Scalar[dtype]], indices: List[Int]) -> Float64:
    """Calculates Mean Absolute Error around the median for target subset."""
    var n = len(indices)
    if n <= 1:
        return 0.0

    var vals = List[Float64](capacity=n)
    for i in range(n):
        vals.append(Float64(y[indices[i]]))
    sort(vals)

    var median_y = vals[n // 2]
    var sum_abs: Float64 = 0.0
    for i in range(n):
        var diff = vals[i] - median_y
        sum_abs += -diff if diff < 0.0 else diff
    return sum_abs / Float64(n)


def friedman_mse_impurity[
    dtype: DType
](
    y: List[Scalar[dtype]],
    left_indices: List[Int],
    right_indices: List[Int],
) -> Float64:
    """Calculates Friedman's improvement score between split partitions."""
    var n_l = len(left_indices)
    var n_r = len(right_indices)
    if n_l == 0 or n_r == 0:
        return 0.0

    var sum_l: Float64 = 0.0
    for i in range(n_l):
        sum_l += Float64(y[left_indices[i]])
    var mean_l = sum_l / Float64(n_l)

    var sum_r: Float64 = 0.0
    for i in range(n_r):
        sum_r += Float64(y[right_indices[i]])
    var mean_r = sum_r / Float64(n_r)

    var diff = mean_l - mean_r
    var factor = (Float64(n_l) * Float64(n_r)) / Float64(n_l + n_r)
    return factor * diff * diff


def compute_impurity_decrease(
    parent_impurity: Float64,
    n_parent: Int,
    left_impurity: Float64,
    n_left: Int,
    right_impurity: Float64,
    n_right: Int,
) -> Float64:
    """Calculates weighted impurity decrease: parent - (n_l/n)*left - (n_r/n)*right.
    """
    if n_parent <= 0:
        return 0.0
    var n_p = Float64(n_parent)
    var decrease = (
        parent_impurity
        - (Float64(n_left) / n_p) * left_impurity
        - (Float64(n_right) / n_p) * right_impurity
    )
    return decrease if decrease > 0.0 else 0.0
