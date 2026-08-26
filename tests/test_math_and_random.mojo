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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
