from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_almost_equal,
    assert_raises,
)
from std.math import log
from strata import (
    sigmoid,
    softmax,
    log_sum_exp,
    PRNG,
    permutation,
    shuffle,
)


def test_sigmoid_numerical_stability() raises:
    # Standard values
    assert_equal(sigmoid[DType.float64](0.0), 0.5)

    # Positive and negative extremes (prevent overflow / NaN)
    assert_equal(sigmoid[DType.float64](1000.0), 1.0)
    assert_equal(sigmoid[DType.float64](-1000.0), 0.0)
    assert_equal(sigmoid[DType.float64](100000.0), 1.0)
    assert_equal(sigmoid[DType.float64](-100000.0), 0.0)

    # Symmetry invariant: sigmoid(-z) + sigmoid(z) == 1.0
    var z: Float64 = 2.5
    var s_pos = sigmoid[DType.float64](z)
    var s_neg = sigmoid[DType.float64](-z)
    assert_almost_equal(s_pos + s_neg, 1.0)

    # Float32 precision
    var s32 = sigmoid[DType.float32](0.0)
    assert_equal(s32, 0.5)


def test_softmax_properties() raises:
    # 1. Empty list
    var empty_list = List[Scalar[DType.float64]]()
    var empty_out = softmax(empty_list)
    assert_equal(len(empty_out), 0)

    # 2. Single element -> [1.0]
    var single: List[Scalar[DType.float64]] = [42.0]
    var single_out = softmax(single)
    assert_equal(len(single_out), 1)
    assert_equal(single_out[0], 1.0)

    # 3. Uniform distribution -> [1/3, 1/3, 1/3]
    var uniform: List[Scalar[DType.float64]] = [5.0, 5.0, 5.0]
    var uniform_out = softmax(uniform)
    assert_almost_equal(uniform_out[0], 1.0 / 3.0)
    assert_almost_equal(uniform_out[1], 1.0 / 3.0)
    assert_almost_equal(uniform_out[2], 1.0 / 3.0)

    # 4. Extreme logits (subtraction of max prevents overflow)
    var extreme: List[Scalar[DType.float64]] = [10000.0, 10000.0]
    var extreme_out = softmax(extreme)
    assert_equal(extreme_out[0], 0.5)
    assert_equal(extreme_out[1], 0.5)

    # 5. Extreme disparity -> [1.0, 0.0]
    var disparity: List[Scalar[DType.float64]] = [1000.0, -1000.0]
    var disparity_out = softmax(disparity)
    assert_equal(disparity_out[0], 1.0)
    assert_equal(disparity_out[1], 0.0)

    # 6. Sum of probabilities equals 1.0
    var logits: List[Scalar[DType.float64]] = [1.2, 3.4, -0.5, 2.1]
    var probs = softmax(logits)
    var total_prob: Float64 = 0.0
    for i in range(len(probs)):
        total_prob += Float64(probs[i])
    assert_almost_equal(total_prob, 1.0)


def test_log_sum_exp_properties() raises:
    # 1. Empty list -> 0.0
    var empty_list = List[Scalar[DType.float64]]()
    assert_equal(log_sum_exp(empty_list), 0.0)

    # 2. Single element -> x
    var single: List[Scalar[DType.float64]] = [7.5]
    assert_equal(log_sum_exp(single), 7.5)

    # 3. Two identical elements: LSE([x, x]) = x + ln(2)
    var two_same: List[Scalar[DType.float64]] = [10.0, 10.0]
    assert_almost_equal(log_sum_exp(two_same), 10.0 + log(2.0))

    # 4. Extreme values without overflow
    var extreme: List[Scalar[DType.float64]] = [5000.0, 5000.0]
    assert_almost_equal(log_sum_exp(extreme), 5000.0 + log(2.0))

    # 5. Shift invariance: LSE(x + c) == LSE(x) + c
    var vals: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0]
    var c: Float64 = 100.0
    var shifted = List[Scalar[DType.float64]]()
    for i in range(len(vals)):
        shifted.append(vals[i] + c)
    assert_almost_equal(log_sum_exp(shifted), log_sum_exp(vals) + c)


def test_prng_determinism_and_rejection_sampling() raises:
    # 1. Determinism: Same seed produces identical sequence
    var rng1 = PRNG(12345)
    var rng2 = PRNG(12345)
    for _ in range(20):
        assert_equal(rng1.next_u64(), rng2.next_u64())

    # 2. Distinct seeds produce distinct sequences
    var rng3 = PRNG(54321)
    assert_true(rng1.next_u64() != rng3.next_u64())

    # 3. Negative seed handling
    var rng_neg = PRNG(-99)
    var val = rng_neg.next_u64()
    assert_true(val > 0)

    # 4. next_int bounds check [0, upper_bound)
    var rng = PRNG(42)
    for _ in range(100):
        var r = rng.next_int(7)
        assert_true(r >= 0 and r < 7)

    # 5. upper_bound <= 1 returns 0
    assert_equal(rng.next_int(1), 0)
    assert_equal(rng.next_int(0), 0)


def test_prng_uniformity_distribution() raises:
    var rng = PRNG(2026)
    var num_buckets = 10
    var samples_per_bucket = 1000
    var total_samples = num_buckets * samples_per_bucket

    var buckets = List[Int](capacity=num_buckets)
    for _ in range(num_buckets):
        buckets.append(0)

    for _ in range(total_samples):
        var bucket = rng.next_int(num_buckets)
        buckets[bucket] += 1

    # Check that all buckets are populated evenly (within statistical 3-sigma bound: ~[850, 1150])
    for b in range(num_buckets):
        assert_true(buckets[b] >= 800 and buckets[b] <= 1200)


def test_permutation_and_shuffle() raises:
    # Permutation of 0, 1, 10
    var p0 = permutation(0, seed=42)
    assert_equal(len(p0), 0)

    var p1 = permutation(1, seed=42)
    assert_equal(len(p1), 1)
    assert_equal(p1[0], 0)

    var p10 = permutation(10, seed=42)
    assert_equal(len(p10), 10)

    # Verify all integers 0..9 are present exactly once
    var counts = List[Int](capacity=10)
    for _ in range(10):
        counts.append(0)
    for i in range(10):
        counts[p10[i]] += 1
    for i in range(10):
        assert_equal(counts[i], 1)

    # In-place shuffle
    var original: List[Int] = [100, 200, 300, 400, 500]
    var to_shuffle = original.copy()
    shuffle(to_shuffle, seed=99)
    assert_equal(len(to_shuffle), 5)


def test_sigmoid_strictly_monotonic() raises:
    var s_prev = sigmoid[DType.float64](-10.0)
    for i in range(1, 20):
        var z = -10.0 + Float64(i)
        var s = sigmoid[DType.float64](z)
        assert_true(s > s_prev)
        s_prev = s


def test_sigmoid_derivative_relation() raises:
    # d/dz sigmoid(z) = sigmoid(z) * (1 - sigmoid(z))
    var z: Float64 = 1.0
    var eps: Float64 = 1e-6
    var num_deriv = (
        sigmoid[DType.float64](z + eps) - sigmoid[DType.float64](z - eps)
    ) / (2.0 * eps)
    var s = sigmoid[DType.float64](z)
    var exact_deriv = s * (1.0 - s)
    assert_almost_equal(num_deriv, exact_deriv, rtol=1e-4)


def test_softmax_order_preservation() raises:
    var logits: List[Scalar[DType.float64]] = [1.0, 3.0, 2.0, 5.0, 4.0]
    var probs = softmax(logits)
    assert_true(probs[3] > probs[4])
    assert_true(probs[4] > probs[1])
    assert_true(probs[1] > probs[2])
    assert_true(probs[2] > probs[0])


def test_softmax_float32_precision() raises:
    var logits: List[Scalar[DType.float32]] = [1.0, 2.0, 3.0]
    var probs = softmax(logits)
    assert_equal(len(probs), 3)
    var total: Float32 = probs[0] + probs[1] + probs[2]
    assert_almost_equal(Float64(total), 1.0, rtol=1e-4)


def test_softmax_all_negative_logits() raises:
    var logits: List[Scalar[DType.float64]] = [-100.0, -200.0, -300.0]
    var probs = softmax(logits)
    assert_equal(len(probs), 3)
    assert_true(probs[0] > 0.99)
    assert_true(probs[1] < 0.01)
    assert_true(probs[2] < 0.01)


def test_log_sum_exp_float32_precision() raises:
    var vals: List[Scalar[DType.float32]] = [1.0, 2.0, 3.0]
    var res = log_sum_exp(vals)
    assert_true(Float64(res) > 3.0)


def test_log_sum_exp_mathematical_bounds() raises:
    # max(x) <= LSE(x) <= max(x) + ln(N)
    var vals: List[Scalar[DType.float64]] = [2.0, 5.0, 1.0, 4.0]
    var lse = log_sum_exp(vals)
    var max_x: Float64 = 5.0
    var ln_N = log(Float64(4))
    assert_true(lse >= max_x)
    assert_true(lse <= max_x + ln_N + 1e-6)


def test_prng_seed_reproducibility() raises:
    var rng1 = PRNG(777)
    var seq1 = List[Int]()
    for _ in range(10):
        seq1.append(rng1.next_int(100))

    var rng2 = PRNG(777)
    for i in range(10):
        assert_equal(rng2.next_int(100), seq1[i])


def test_shuffle_preserves_multiset_counts() raises:
    var original: List[Int] = [1, 2, 2, 3, 3, 3, 4]
    var shuffled = original.copy()
    shuffle(shuffled, seed=42)

    var count_2: Int = 0
    var count_3: Int = 0
    for i in range(len(shuffled)):
        if shuffled[i] == 2:
            count_2 += 1
        elif shuffled[i] == 3:
            count_3 += 1

    assert_equal(count_2, 2)
    assert_equal(count_3, 3)
    assert_equal(len(shuffled), 7)


def test_permutation_large_n_elements() raises:
    var n = 100
    var p = permutation(n, seed=123)
    assert_equal(len(p), n)

    var seen = List[Bool](capacity=n)
    for _ in range(n):
        seen.append(False)

    for i in range(n):
        var val = p[i]
        assert_true(val >= 0 and val < n)
        assert_true(not seen[val])
        seen[val] = True


def test_shuffle_empty_and_single_element() raises:
    var empty_list = List[Int]()
    shuffle(empty_list, seed=42)
    assert_equal(len(empty_list), 0)

    var single: List[Int] = [42]
    shuffle(single, seed=42)
    assert_equal(len(single), 1)
    assert_equal(single[0], 42)


def test_prng_prime_modulus_bounds() raises:
    var rng = PRNG(999)
    for _ in range(50):
        var r = rng.next_int(17)
        assert_true(r >= 0 and r < 17)


def test_prng_power_of_two_bounds() raises:
    var rng = PRNG(888)
    for _ in range(50):
        var r = rng.next_int(64)
        assert_true(r >= 0 and r < 64)


def test_sigmoid_bfloat16_precision() raises:
    var s_bf = sigmoid[DType.bfloat16](0.0)
    assert_almost_equal(Float64(s_bf), 0.5, rtol=1e-2)


def test_softmax_bfloat16_precision() raises:
    var logits: List[Scalar[DType.bfloat16]] = [1.0, 1.0]
    var probs = softmax(logits)
    assert_almost_equal(Float64(probs[0]), 0.5, rtol=1e-2)
    assert_almost_equal(Float64(probs[1]), 0.5, rtol=1e-2)


def test_log_sum_exp_bfloat16_precision() raises:
    var vals: List[Scalar[DType.bfloat16]] = [0.0, 0.0]
    var res = log_sum_exp(vals)
    assert_almost_equal(Float64(res), log(2.0), rtol=1e-2)


def test_permutation_seed_difference() raises:
    var p1 = permutation(20, seed=1)
    var p2 = permutation(20, seed=2)
    var has_diff = False
    for i in range(20):
        if p1[i] != p2[i]:
            has_diff = True
            break
    assert_true(has_diff)


def test_shuffle_floating_point_list() raises:
    var flist: List[Float64] = [1.1, 2.2, 3.3, 4.4, 5.5]
    shuffle(flist, seed=7)
    assert_equal(len(flist), 5)


def test_sigmoid_domain_limits() raises:
    var s_min = sigmoid[DType.float64](-1e8)
    assert_equal(s_min, 0.0)
    var s_max = sigmoid[DType.float64](1e8)
    assert_equal(s_max, 1.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
