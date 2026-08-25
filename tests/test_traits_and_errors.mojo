from std.testing import assert_equal, assert_true, assert_raises, TestSuite
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


@fieldwise_init
struct MockRegressor[
    target_dtype: DType = DType.float64,
](Movable, Regressor):
    var slope: Float64

    def fit[
        feat_dtype: DType, in_target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[in_target_dtype]]) raises:
        pass

    def fit[
        feat_dtype: DType, in_target_dtype: DType
    ](mut self, dataset: Dataset[feat_dtype, in_target_dtype]) raises:
        self.fit(dataset.records, dataset.targets)

    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Scalar[Self.target_dtype]]:
        var res = List[Scalar[Self.target_dtype]](capacity=X.rows)
        for r in range(X.rows):
            var val = Float64(X[r, 0]) * self.slope
            res.append(Scalar[Self.target_dtype](val))
        return res^

    def predict[
        feat_dtype: DType, out_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Scalar[out_dtype]]:
        var res = List[Scalar[out_dtype]](capacity=X.rows)
        for r in range(X.rows):
            var val = Float64(X[r, 0]) * self.slope
            res.append(Scalar[out_dtype](val))
        return res^

    def predict[
        feat_dtype: DType, in_target_dtype: DType
    ](
        self,
        dataset: Dataset[feat_dtype, in_target_dtype],
    ) raises -> List[
        Scalar[in_target_dtype]
    ]:
        return self.predict[feat_dtype, in_target_dtype](dataset.records)


def test_pipeline_and_traits() raises:
    from strata import StandardScaler, PipelineRegressor, Dataset

    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = 10.0
    X[1, 0] = 20.0
    X[2, 0] = 30.0
    X[3, 0] = 40.0

    var y: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0, 4.0]

    # Standard Float64 usage
    var scaler = StandardScaler()
    var reg = MockRegressor(2.0)
    var pipe = PipelineRegressor(scaler^, reg^)

    var ds = Dataset(X^, y^)
    pipe.fit[DType.float64, DType.float64](ds)
    var preds = pipe.predict[DType.float64, DType.float64](ds.records)
    assert_equal(len(preds), 4)

    # User freedom: Int32 regressor allowed if explicitly requested
    var int_reg = MockRegressor[DType.int32](2.0)
    var X_int = Matrix[DType.float64](2, 1, 5.0)
    var int_preds = int_reg.predict(X_int)
    assert_equal(int_preds[0], 10)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
