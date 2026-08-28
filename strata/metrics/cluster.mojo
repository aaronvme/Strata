from ..core.matrix import Matrix
from ..cluster._common import _euclidean_distance
from ..utils.validation import check_array, check_consistent_length
from ..exceptions.errors import InvalidParameterError
from .classification import _insert_label, _search_sorted


def silhouette_score[
    dtype: DType = DType.float64
](X: Matrix[dtype], labels: List[Int]) raises -> Float64:
    """Mean silhouette coefficient over all samples.

    For each sample the coefficient is (b - a) / max(a, b), where a is the mean
    distance to the other members of its own cluster and b is the mean distance
    to the members of the nearest other cluster. Samples alone in their cluster
    score 0.

    Args:
        X: Feature matrix with one row per sample.
        labels: Cluster label of each sample, as returned by a Clusterer.

    Returns:
        The mean silhouette coefficient, between -1.0 and 1.0.
    """
    check_array(X)
    check_consistent_length(X, labels)

    var n = X.rows
    var unique = List[Float64]()
    for i in range(n):
        _insert_label(unique, Float64(labels[i]))

    var k = len(unique)
    if k < 2 or k > n - 1:
        raise InvalidParameterError.error(
            "labels",
            "silhouette_score requires 2 to n_samples - 1 distinct labels, but "
            + String(k)
            + " were found for "
            + String(n)
            + " samples",
        )

    var encoded = List[Int](capacity=n)
    var sizes = List[Float64](length=k, fill=0.0)
    for i in range(n):
        var c = _search_sorted(unique, Float64(labels[i]))
        encoded.append(c)
        sizes[c] += 1.0

    var X_comp = X.cast[DType.float64]()
    var sums = Matrix[DType.float64](n, k, 0.0)
    for i in range(n):
        for j in range(i + 1, n):
            var d = Float64(_euclidean_distance(X_comp, i, X_comp, j))
            sums[i, encoded[j]] += d
            sums[j, encoded[i]] += d

    var total: Float64 = 0.0
    for i in range(n):
        var own = encoded[i]
        if sizes[own] == 1.0:
            continue

        var a = sums[i, own] / (sizes[own] - 1.0)
        var b = -1.0
        for c in range(k):
            if c == own:
                continue
            var mean_c = sums[i, c] / sizes[c]
            if b < 0.0 or mean_c < b:
                b = mean_c

        var denom = max(a, b)
        if denom == 0.0:
            continue
        total += (b - a) / denom

    return total / Float64(n)
