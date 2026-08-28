from ..core.matrix import Matrix
from ..tree.tree import Tree, Node
from ..utils.random import PRNG


def generate_bootstrap_indices(
    n_samples: Int,
    max_samples_count: Int,
    max_samples_ratio: Float64,
    mut rng: PRNG,
) -> List[Int]:
    """Generates bootstrap sample indices with replacement."""
    var k: Int
    if max_samples_count > 0:
        k = max_samples_count if max_samples_count < n_samples else n_samples
    elif max_samples_ratio > 0.0:
        var computed = Int(Float64(n_samples) * max_samples_ratio)
        k = computed if computed >= 1 else 1
        if k > n_samples:
            k = n_samples
    else:
        k = n_samples

    var indices = List[Int](capacity=k)
    var n_u64 = UInt64(n_samples)
    for _ in range(k):
        var idx = Int(rng.next_u64() % n_u64)
        indices.append(idx)
    return indices^


def get_oob_mask(n_samples: Int, sample_indices: List[Int]) -> List[Bool]:
    """Returns boolean mask indicating which samples were omitted from bootstrap sample."""
    var oob_mask = List[Bool](capacity=n_samples)
    for _ in range(n_samples):
        oob_mask.append(True)

    for i in range(len(sample_indices)):
        var idx = sample_indices[i]
        if idx >= 0 and idx < n_samples:
            oob_mask[idx] = False

    return oob_mask^


def compute_tree_feature_importances(
    tree: Tree, n_features: Int, total_samples: Int
) -> List[Float64]:
    """Calculates Mean Decrease in Impurity (MDI) feature importances for a single tree."""
    var importances = List[Float64](capacity=n_features)
    for _ in range(n_features):
        importances.append(0.0)

    if total_samples <= 0 or tree.node_count() <= 1:
        return importances^

    var total_f = Float64(total_samples)
    for i in range(len(tree.nodes)):
        if not tree.nodes[i].is_leaf:
            var f = tree.nodes[i].feature_idx
            if f >= 0 and f < n_features:
                var n_t = Float64(tree.nodes[i].n_node_samples)
                var l_idx = tree.nodes[i].left_child
                var r_idx = tree.nodes[i].right_child

                var n_l: Float64 = 0.0
                var imp_l: Float64 = 0.0
                if l_idx >= 0 and l_idx < len(tree.nodes):
                    n_l = Float64(tree.nodes[l_idx].n_node_samples)
                    imp_l = tree.nodes[l_idx].impurity

                var n_r: Float64 = 0.0
                var imp_r: Float64 = 0.0
                if r_idx >= 0 and r_idx < len(tree.nodes):
                    n_r = Float64(tree.nodes[r_idx].n_node_samples)
                    imp_r = tree.nodes[r_idx].impurity

                var decrease = (
                    n_t * tree.nodes[i].impurity - n_l * imp_l - n_r * imp_r
                ) / total_f
                if decrease > 0.0:
                    importances[f] += decrease

    return importances^


def normalize_feature_importances(
    importances: List[Float64], n_features: Int
) -> List[Float64]:
    """Normalizes raw impurity decreases so that the importance vector sums to 1.0."""
    var normalized = List[Float64](capacity=n_features)
    var sum_imp: Float64 = 0.0
    for j in range(n_features):
        sum_imp += importances[j]

    if sum_imp > 1e-12:
        for j in range(n_features):
            normalized.append(importances[j] / sum_imp)
    else:
        for _ in range(n_features):
            normalized.append(0.0)

    return normalized^
