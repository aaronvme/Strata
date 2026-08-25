from std.testing import assert_equal, assert_true, assert_raises, TestSuite
from strata import (
    Matrix,
    NotFittedError,
    check_is_fitted,
    check_array,
    check_X_y,
    sigmoid,
    softmax,
)


def test_domain_errors() raises:
    var err = NotFittedError.error("Estimator", "call fit first")
    assert_true(String(err).byte_length() > 0)

    with assert_raises():
        check_is_fitted("TestModel", False)


def test_validation() raises:
    var empty_m = Matrix[DType.float64](0, 0, 0)
    with assert_raises():
        check_array(empty_m, allow_empty=False)

    var valid_m = Matrix[DType.float64](3, 2, 1.0)
    var wrong_y: List[Scalar[DType.float64]] = [1.0, 2.0]
    with assert_raises():
        check_X_y(valid_m, wrong_y)


def test_math_utils() raises:
    var s0 = sigmoid[DType.float64](0.0)
    assert_equal(s0, 0.5)

    var logits: List[Scalar[DType.float64]] = [1.0, 1.0]
    var probs = softmax[DType.float64](logits)
    assert_equal(probs[0], 0.5)
    assert_equal(probs[1], 0.5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
