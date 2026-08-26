from std.testing import (
    assert_equal,
    assert_true,
    assert_raises,
    assert_almost_equal,
    TestSuite,
)
from strata import (
    Matrix,
    Dataset,
    NotFittedError,
    Regressor,
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

    var valid_y: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0]
    check_X_y(valid_m, valid_y)


def test_math_utils() raises:
    var s0 = sigmoid[DType.float64](0.0)
    assert_equal(s0, 0.5)
    var s_neg = sigmoid[DType.float64](-1000.0)
    assert_equal(s_neg, 0.0)
    var s_pos = sigmoid[DType.float64](1000.0)
    assert_equal(s_pos, 1.0)

    # Numerically stable softmax on extreme logits
    var logits_extreme: List[Scalar[DType.float64]] = [1000.0, 1000.0]
    var probs_extreme = softmax[DType.float64](logits_extreme)
    assert_equal(probs_extreme[0], 0.5)
    assert_equal(probs_extreme[1], 0.5)

    # Numerically stable log_sum_exp
    from strata import log_sum_exp, PRNG
    from std.math import log

    var lse = log_sum_exp[DType.float64](logits_extreme)
    # LSE([1000, 1000]) = 1000 + ln(2)
    assert_almost_equal(lse, 1000.0 + log(2.0))

    # Test PRNG
    var rng = PRNG(123)
    var r_val = rng.next_int(10)
    assert_true(r_val >= 0 and r_val < 10)

    # Verify negative seeds do not alias positive seeds
    var rng_pos = PRNG(5)
    var rng_neg = PRNG(-5)
    assert_true(rng_pos.next_u64() != rng_neg.next_u64())


@fieldwise_init
struct MockRegressor(Movable, Regressor):
    var slope: Float64

    def fit[
        feat_dtype: DType, in_target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[in_target_dtype]]) raises:
        pass

    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Scalar[feat_dtype]]:
        var res = List[Scalar[feat_dtype]](capacity=X.rows)
        for r in range(X.rows):
            var val = Float64(X[r, 0]) * self.slope
            res.append(Scalar[feat_dtype](val))
        return res^


def test_pipeline_and_traits() raises:
    from strata import (
        StandardScaler,
        PipelineRegressor,
        PipelineTransformer,
        Dataset,
    )
    from strata.base.estimator import fit as fit_ds, predict as predict_ds

    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = 10.0
    X[1, 0] = 20.0
    X[2, 0] = 30.0
    X[3, 0] = 40.0

    var y: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0, 4.0]

    # Standard Float64 usage with 2-method MockRegressor
    var scaler = StandardScaler()
    var reg = MockRegressor(2.0)
    var pipe = PipelineRegressor(scaler^, reg^)

    var ds = Dataset(X^, y^)
    # Fits using generic Dataset free function or matrix methods
    fit_ds(pipe, ds)
    var preds = pipe.predict(ds.records)
    assert_equal(len(preds), 4)

    # Test N-step composable PipelineTransformer
    var scaler1 = StandardScaler(with_mean=True, with_std=False)
    var scaler2 = StandardScaler(with_mean=False, with_std=True)
    var chained_prep = PipelineTransformer(scaler1^, scaler2^)
    var reg2 = MockRegressor(3.0)
    var deep_pipe = PipelineRegressor(chained_prep^, reg2^)

    var X_deep = Matrix[DType.float64](3, 1, 10.0)
    var y_deep: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0]
    deep_pipe.fit(X_deep, y_deep)
    var deep_preds = deep_pipe.predict(X_deep)
    assert_equal(len(deep_preds), 3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

