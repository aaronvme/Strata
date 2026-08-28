from std.math import sqrt, log2

from ..core.matrix import Matrix
from ..utils.random import PRNG, permutation
from .criterion import (
    gini_impurity,
    gini_from_sum_sq,
    entropy_impurity,
    squared_error_impurity,
    variance_from_sums,
    absolute_error_impurity,
    compute_impurity_decrease,
)


struct SplitResult(Movable):
    """Container for split evaluation results."""

    var found: Bool
    var feature_idx: Int
    var threshold: Float64
    var impurity_decrease: Float64
    var left_indices: List[Int]
    var right_indices: List[Int]
    var left_impurity: Float64
    var right_impurity: Float64

    def __init__(out self):
        self.found = False
        self.feature_idx = -1
        self.threshold = 0.0
        self.impurity_decrease = 0.0
        self.left_indices = List[Int]()
        self.right_indices = List[Int]()
        self.left_impurity = 0.0
        self.right_impurity = 0.0

    def __init__(
        out self,
        feature_idx: Int,
        threshold: Float64,
        impurity_decrease: Float64,
        var left_indices: List[Int],
        var right_indices: List[Int],
        left_impurity: Float64,
        right_impurity: Float64,
    ):
        self.found = True
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.impurity_decrease = impurity_decrease
        self.left_indices = left_indices^
        self.right_indices = right_indices^
        self.left_impurity = left_impurity
        self.right_impurity = right_impurity


struct FeatureValuePair(Comparable, Copyable, Movable):
    """Sample value pair for sorting along a single feature dimension."""

    var feat_val: Float64
    var sample_idx: Int
    var target_val: Float64
    var target_class: Int

    def __init__(
        out self,
        feat_val: Float64,
        sample_idx: Int,
        target_val: Float64 = 0.0,
        target_class: Int = 0,
    ):
        self.feat_val = feat_val
        self.sample_idx = sample_idx
        self.target_val = target_val
        self.target_class = target_class

    def __init__(out self, *, copy: Self):
        self.feat_val = copy.feat_val
        self.sample_idx = copy.sample_idx
        self.target_val = copy.target_val
        self.target_class = copy.target_class

    def __lt__(self, other: Self) -> Bool:
        return self.feat_val < other.feat_val

    def __le__(self, other: Self) -> Bool:
        return self.feat_val <= other.feat_val

    def __gt__(self, other: Self) -> Bool:
        return self.feat_val > other.feat_val

    def __ge__(self, other: Self) -> Bool:
        return self.feat_val >= other.feat_val

    def __eq__(self, other: Self) -> Bool:
        return self.feat_val == other.feat_val

    def __ne__(self, other: Self) -> Bool:
        return self.feat_val != other.feat_val


def select_features(
    n_total_features: Int,
    max_features: String,
    max_features_count: Int,
    max_features_ratio: Float64,
    mut rng: PRNG,
) -> List[Int]:
    """Selects candidate feature indices based on max_features strategy."""
    var k: Int
    if max_features_count > 0:
        k = (
            max_features_count if max_features_count
            < n_total_features else n_total_features
        )
    elif max_features_ratio > 0.0:
        var computed = Int(Float64(n_total_features) * max_features_ratio)
        k = computed if computed >= 1 else 1
        if k > n_total_features:
            k = n_total_features
    elif max_features == "sqrt":
        var computed = Int(sqrt(Float64(n_total_features)))
        k = computed if computed >= 1 else 1
    elif max_features == "log2":
        var computed = Int(log2(Float64(n_total_features)))
        k = computed if computed >= 1 else 1
    else:
        k = n_total_features

    if k >= n_total_features:
        var all_features = List[Int](capacity=n_total_features)
        for f in range(n_total_features):
            all_features.append(f)
        return all_features^

    var seed_val = Int(rng.next_u64() & 0x7FFFFFFFFFFFFFFF)
    var perm = permutation(n_total_features, seed=seed_val)
    var selected = List[Int](capacity=k)
    for i in range(k):
        selected.append(perm[i])
    return selected^


def find_best_split_classification[
    feat_dtype: DType
](
    X: Matrix[feat_dtype],
    y_encoded: List[Int],
    node_indices: List[Int],
    n_classes: Int,
    criterion: String,
    splitter: String,
    max_features: String,
    max_features_count: Int,
    max_features_ratio: Float64,
    min_samples_split: Int,
    min_samples_leaf: Int,
    min_impurity_decrease: Float64,
    mut rng: PRNG,
) -> SplitResult:
    """Finds best split for classification across candidate features."""
    var n_samples = len(node_indices)
    if n_samples < min_samples_split or n_samples < 2 * min_samples_leaf:
        return SplitResult()

    var parent_counts = List[Int](capacity=n_classes)
    for _ in range(n_classes):
        parent_counts.append(0)
    for i in range(n_samples):
        var c = y_encoded[node_indices[i]]
        if c >= 0 and c < n_classes:
            parent_counts[c] += 1

    var use_nat_log = criterion == "log_loss"
    var parent_impurity: Float64
    if criterion == "entropy" or criterion == "log_loss":
        parent_impurity = entropy_impurity(
            parent_counts, n_samples, use_natural_log=use_nat_log
        )
    else:
        parent_impurity = gini_impurity(parent_counts, n_samples)

    if parent_impurity <= 1e-12:
        return SplitResult()

    var selected_features = select_features(
        X.cols,
        max_features,
        max_features_count,
        max_features_ratio,
        rng,
    )

    var best_decrease: Float64 = -1.0
    var best_feature: Int = -1
    var best_threshold: Float64 = 0.0
    var best_left_imp: Float64 = 0.0
    var best_right_imp: Float64 = 0.0

    for f_i in range(len(selected_features)):
        var f = selected_features[f_i]

        var pairs = List[FeatureValuePair](capacity=n_samples)
        for i in range(n_samples):
            var s_idx = node_indices[i]
            var feat_val = X[s_idx, f].cast[DType.float64]()
            var cls_val = y_encoded[s_idx]
            pairs.append(FeatureValuePair(feat_val, s_idx, 0.0, cls_val))

        sort(pairs)

        if splitter == "random":
            var cand_positions = List[Int]()
            for i in range(min_samples_leaf - 1, n_samples - min_samples_leaf):
                if pairs[i].feat_val < pairs[i + 1].feat_val:
                    cand_positions.append(i)

            if len(cand_positions) == 0:
                continue

            var chosen_cand_idx = cand_positions[
                Int(rng.next_u64() % UInt64(len(cand_positions)))
            ]
            var cand_threshold = (
                pairs[chosen_cand_idx].feat_val
                + pairs[chosen_cand_idx + 1].feat_val
            ) * 0.5

            var l_counts = List[Int](capacity=n_classes)
            var r_counts = List[Int](capacity=n_classes)
            for _ in range(n_classes):
                l_counts.append(0)
                r_counts.append(0)

            var n_l = 0
            var n_r = 0
            for i in range(n_samples):
                var cls = pairs[i].target_class
                if pairs[i].feat_val <= cand_threshold:
                    if cls >= 0 and cls < n_classes:
                        l_counts[cls] += 1
                    n_l += 1
                else:
                    if cls >= 0 and cls < n_classes:
                        r_counts[cls] += 1
                    n_r += 1

            if n_l < min_samples_leaf or n_r < min_samples_leaf:
                continue

            var l_imp = gini_impurity(
                l_counts, n_l
            ) if criterion == "gini" else entropy_impurity(
                l_counts, n_l, use_natural_log=use_nat_log
            )
            var r_imp = gini_impurity(
                r_counts, n_r
            ) if criterion == "gini" else entropy_impurity(
                r_counts, n_r, use_natural_log=use_nat_log
            )
            var decrease = compute_impurity_decrease(
                parent_impurity, n_samples, l_imp, n_l, r_imp, n_r
            )

            if (
                decrease >= min_impurity_decrease
                and decrease > best_decrease
                and decrease > 0.0
            ):
                best_decrease = decrease
                best_feature = f
                best_threshold = cand_threshold
                best_left_imp = l_imp
                best_right_imp = r_imp
        else:
            var l_counts = List[Int](capacity=n_classes)
            var r_counts = List[Int](capacity=n_classes)
            for c in range(n_classes):
                l_counts.append(0)
                r_counts.append(parent_counts[c])

            var l_sum_sq_counts = 0
            var r_sum_sq_counts = 0
            for c in range(n_classes):
                r_sum_sq_counts += parent_counts[c] * parent_counts[c]

            var n_l = 0
            var n_r = n_samples

            for i in range(n_samples - 1):
                var cls = pairs[i].target_class
                if cls >= 0 and cls < n_classes:
                    var old_l_c = l_counts[cls]
                    var old_r_c = r_counts[cls]

                    l_counts[cls] += 1
                    r_counts[cls] -= 1
                    l_sum_sq_counts += (l_counts[cls] * l_counts[cls]) - (
                        old_l_c * old_l_c
                    )
                    r_sum_sq_counts += (r_counts[cls] * r_counts[cls]) - (
                        old_r_c * old_r_c
                    )

                n_l += 1
                n_r -= 1

                if pairs[i].feat_val == pairs[i + 1].feat_val:
                    continue

                if n_l < min_samples_leaf or n_r < min_samples_leaf:
                    continue

                var l_imp: Float64
                var r_imp: Float64
                if criterion == "entropy" or criterion == "log_loss":
                    l_imp = entropy_impurity(
                        l_counts, n_l, use_natural_log=use_nat_log
                    )
                    r_imp = entropy_impurity(
                        r_counts, n_r, use_natural_log=use_nat_log
                    )
                else:
                    l_imp = gini_from_sum_sq(l_sum_sq_counts, n_l)
                    r_imp = gini_from_sum_sq(r_sum_sq_counts, n_r)

                var decrease = compute_impurity_decrease(
                    parent_impurity, n_samples, l_imp, n_l, r_imp, n_r
                )

                if (
                    decrease >= min_impurity_decrease
                    and decrease > best_decrease
                    and decrease > 0.0
                ):
                    best_decrease = decrease
                    best_feature = f
                    best_threshold = (
                        pairs[i].feat_val + pairs[i + 1].feat_val
                    ) * 0.5
                    best_left_imp = l_imp
                    best_right_imp = r_imp

    if best_feature == -1:
        return SplitResult()

    var left_idx = List[Int](capacity=n_samples)
    var right_idx = List[Int](capacity=n_samples)
    for i in range(n_samples):
        var s = node_indices[i]
        var v = X[s, best_feature].cast[DType.float64]()
        if v <= best_threshold:
            left_idx.append(s)
        else:
            right_idx.append(s)

    if len(left_idx) < min_samples_leaf or len(right_idx) < min_samples_leaf:
        return SplitResult()

    return SplitResult(
        best_feature,
        best_threshold,
        best_decrease,
        left_idx^,
        right_idx^,
        best_left_imp,
        best_right_imp,
    )


def find_best_split_regression[
    feat_dtype: DType, target_dtype: DType
](
    X: Matrix[feat_dtype],
    y: List[Scalar[target_dtype]],
    node_indices: List[Int],
    criterion: String,
    splitter: String,
    max_features: String,
    max_features_count: Int,
    max_features_ratio: Float64,
    min_samples_split: Int,
    min_samples_leaf: Int,
    min_impurity_decrease: Float64,
    mut rng: PRNG,
) -> SplitResult:
    """Finds best split for regression across candidate features."""
    var n_samples = len(node_indices)
    if n_samples < min_samples_split or n_samples < 2 * min_samples_leaf:
        return SplitResult()

    var parent_impurity: Float64
    if criterion == "absolute_error":
        parent_impurity = absolute_error_impurity(y, node_indices)
    else:
        parent_impurity = squared_error_impurity(y, node_indices)

    if parent_impurity <= 1e-12:
        return SplitResult()

    var selected_features = select_features(
        X.cols,
        max_features,
        max_features_count,
        max_features_ratio,
        rng,
    )

    var best_decrease: Float64 = -1.0
    var best_feature: Int = -1
    var best_threshold: Float64 = 0.0
    var best_left_imp: Float64 = 0.0
    var best_right_imp: Float64 = 0.0

    var total_sum_y: Float64 = 0.0
    var total_sum_sq_y: Float64 = 0.0
    for i in range(n_samples):
        var v = y[node_indices[i]].cast[DType.float64]()
        total_sum_y += v
        total_sum_sq_y += v * v

    for f_i in range(len(selected_features)):
        var f = selected_features[f_i]

        var pairs = List[FeatureValuePair](capacity=n_samples)
        for i in range(n_samples):
            var s_idx = node_indices[i]
            var feat_val = X[s_idx, f].cast[DType.float64]()
            var tgt_val = y[s_idx].cast[DType.float64]()
            pairs.append(FeatureValuePair(feat_val, s_idx, tgt_val, 0))

        sort(pairs)

        if splitter == "random":
            var cand_positions = List[Int]()
            for i in range(min_samples_leaf - 1, n_samples - min_samples_leaf):
                if pairs[i].feat_val < pairs[i + 1].feat_val:
                    cand_positions.append(i)

            if len(cand_positions) == 0:
                continue

            var chosen_cand_idx = cand_positions[
                Int(rng.next_u64() % UInt64(len(cand_positions)))
            ]
            var cand_threshold = (
                pairs[chosen_cand_idx].feat_val
                + pairs[chosen_cand_idx + 1].feat_val
            ) * 0.5

            var l_idx = List[Int](capacity=n_samples)
            var r_idx = List[Int](capacity=n_samples)
            for i in range(n_samples):
                if pairs[i].feat_val <= cand_threshold:
                    l_idx.append(pairs[i].sample_idx)
                else:
                    r_idx.append(pairs[i].sample_idx)

            var n_l = len(l_idx)
            var n_r = len(r_idx)
            if n_l < min_samples_leaf or n_r < min_samples_leaf:
                continue

            var l_imp: Float64
            var r_imp: Float64
            if criterion == "absolute_error":
                l_imp = absolute_error_impurity(y, l_idx)
                r_imp = absolute_error_impurity(y, r_idx)
            else:
                l_imp = squared_error_impurity(y, l_idx)
                r_imp = squared_error_impurity(y, r_idx)

            var decrease = compute_impurity_decrease(
                parent_impurity, n_samples, l_imp, n_l, r_imp, n_r
            )
            if (
                decrease >= min_impurity_decrease
                and decrease > best_decrease
                and decrease > 0.0
            ):
                best_decrease = decrease
                best_feature = f
                best_threshold = cand_threshold
                best_left_imp = l_imp
                best_right_imp = r_imp
        else:
            if criterion == "squared_error" or criterion == "friedman_mse":
                var sum_l: Float64 = 0.0
                var sum_sq_l: Float64 = 0.0
                var sum_r: Float64 = total_sum_y
                var sum_sq_r: Float64 = total_sum_sq_y

                var n_l = 0
                var n_r = n_samples

                for i in range(n_samples - 1):
                    var val = pairs[i].target_val
                    sum_l += val
                    sum_sq_l += val * val
                    sum_r -= val
                    sum_sq_r -= val * val
                    n_l += 1
                    n_r -= 1

                    if pairs[i].feat_val == pairs[i + 1].feat_val:
                        continue

                    if n_l < min_samples_leaf or n_r < min_samples_leaf:
                        continue

                    var l_imp = variance_from_sums(sum_l, sum_sq_l, n_l)
                    var r_imp = variance_from_sums(sum_r, sum_sq_r, n_r)

                    var decrease: Float64
                    if criterion == "friedman_mse":
                        var mean_l = sum_l / Float64(n_l)
                        var mean_r = sum_r / Float64(n_r)
                        var diff = mean_l - mean_r
                        decrease = (
                            ((Float64(n_l) * Float64(n_r)) / Float64(n_l + n_r))
                            * diff
                            * diff
                        )
                    else:
                        decrease = compute_impurity_decrease(
                            parent_impurity, n_samples, l_imp, n_l, r_imp, n_r
                        )

                    if (
                        decrease >= min_impurity_decrease
                        and decrease > best_decrease
                        and decrease > 0.0
                    ):
                        best_decrease = decrease
                        best_feature = f
                        best_threshold = (
                            pairs[i].feat_val + pairs[i + 1].feat_val
                        ) * 0.5
                        best_left_imp = l_imp
                        best_right_imp = r_imp
            else:
                var l_idx = List[Int](capacity=n_samples)
                for i in range(n_samples - 1):
                    l_idx.append(pairs[i].sample_idx)
                    var n_l = i + 1
                    var n_r = n_samples - n_l

                    if pairs[i].feat_val == pairs[i + 1].feat_val:
                        continue

                    if n_l < min_samples_leaf or n_r < min_samples_leaf:
                        continue

                    var r_idx = List[Int](capacity=n_r)
                    for j in range(i + 1, n_samples):
                        r_idx.append(pairs[j].sample_idx)

                    var l_imp = absolute_error_impurity(y, l_idx)
                    var r_imp = absolute_error_impurity(y, r_idx)
                    var decrease = compute_impurity_decrease(
                        parent_impurity, n_samples, l_imp, n_l, r_imp, n_r
                    )

                    if (
                        decrease >= min_impurity_decrease
                        and decrease > best_decrease
                        and decrease > 0.0
                    ):
                        best_decrease = decrease
                        best_feature = f
                        best_threshold = (
                            pairs[i].feat_val + pairs[i + 1].feat_val
                        ) * 0.5
                        best_left_imp = l_imp
                        best_right_imp = r_imp

    if best_feature == -1:
        return SplitResult()

    var left_idx = List[Int](capacity=n_samples)
    var right_idx = List[Int](capacity=n_samples)
    for i in range(n_samples):
        var s = node_indices[i]
        var v = X[s, best_feature].cast[DType.float64]()
        if v <= best_threshold:
            left_idx.append(s)
        else:
            right_idx.append(s)

    if len(left_idx) < min_samples_leaf or len(right_idx) < min_samples_leaf:
        return SplitResult()

    return SplitResult(
        best_feature,
        best_threshold,
        best_decrease,
        left_idx^,
        right_idx^,
        best_left_imp,
        best_right_imp,
    )
