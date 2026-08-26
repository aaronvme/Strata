from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_raises,
)
from strata import (
    Matrix,
    Dataset,
    StandardScaler,
    PipelineTransformer,
    PipelineRegressor,
    PipelineClassifier,
    Regressor,
    Classifier,
)
from strata.base.estimator import (
    fit as fit_ds,
    predict as predict_ds,
    predict_proba as predict_proba_ds,
    transform as transform_ds,
    fit_transform as fit_transform_ds,
)


@fieldwise_init
struct MockLinearRegressor(Movable, Regressor):
    var weight: Float64
    var bias: Float64

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]]) raises:
        pass

    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Scalar[feat_dtype]]:
        var res = List[Scalar[feat_dtype]](capacity=X.rows)
        for r in range(X.rows):
            var val = Float64(X[r, 0]) * self.weight + self.bias
            res.append(Scalar[feat_dtype](val))
        return res^


@fieldwise_init
struct MockLogisticClassifier(Classifier, Movable):
    var threshold: Float64

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]]) raises:
        pass

    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Int]:
        var res = List[Int](capacity=X.rows)
        for r in range(X.rows):
            if Float64(X[r, 0]) >= self.threshold:
                res.append(1)
            else:
                res.append(0)
        return res^

    def predict_proba[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> Matrix[feat_dtype]:
        var res = Matrix[feat_dtype](X.rows, 2, 0)
        for r in range(X.rows):
            var prob1: Float64 = (
                0.8 if Float64(X[r, 0]) >= self.threshold else 0.2
            )
            var prob0: Float64 = 1.0 - prob1
            res[r, 0] = Scalar[feat_dtype](prob0)
            res[r, 1] = Scalar[feat_dtype](prob1)
        return res^


def test_pipeline_transformer_chaining() raises:
    # Scale with mean then scale with std in 2 chained steps
    var s1 = StandardScaler(with_mean=True, with_std=False)
    var s2 = StandardScaler(with_mean=False, with_std=True)
    var prep = PipelineTransformer(s1^, s2^)

    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = 10.0
    X[1, 0] = 20.0
    X[2, 0] = 30.0
    X[3, 0] = 40.0

    var X_trans = prep.fit_transform(X)
    assert_equal(X_trans.rows, 4)
    assert_equal(X_trans.cols, 1)


def test_pipeline_regressor_end_to_end() raises:
    var scaler = StandardScaler()
    var reg = MockLinearRegressor(2.0, 1.0)
    var pipe = PipelineRegressor(scaler^, reg^)

    var X = Matrix[DType.float64](3, 1, 0)
    X[0, 0] = 2.0
    X[1, 0] = 4.0
    X[2, 0] = 6.0
    var y: List[Scalar[DType.float64]] = [5.0, 9.0, 13.0]

    pipe.fit(X, y)
    var preds = pipe.predict(X)
    assert_equal(len(preds), 3)


def test_pipeline_classifier_and_probabilities() raises:
    var scaler = StandardScaler(with_mean=False, with_std=False)
    var clf = MockLogisticClassifier(0.0)
    var pipe = PipelineClassifier(scaler^, clf^)

    var X = Matrix[DType.float32](2, 1, 0)
    X[0, 0] = 5.0
    X[1, 0] = -5.0
    var y: List[Scalar[DType.int32]] = [1, 0]

    pipe.fit(X, y)

    # 1. Discrete Int predictions
    var preds = pipe.predict(X)
    assert_equal(len(preds), 2)
    assert_equal(preds[0], 1)
    assert_equal(preds[1], 0)

    # 2. Probability matrix matching Float32 precision
    var probs = pipe.predict_proba(X)
    assert_equal(probs.rows, 2)
    assert_equal(probs.cols, 2)
    assert_equal(probs[0, 1], 0.8)
    assert_equal(probs[1, 1], 0.2)


def test_generic_dataset_helpers() raises:
    var scaler = StandardScaler()
    var reg = MockLinearRegressor(1.0, 0.0)
    var pipe = PipelineRegressor(scaler^, reg^)

    var X = Matrix[DType.float64](3, 1, 10.0)
    var y: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0]
    var ds = Dataset(X^, y^)

    # Test top-level generic functional helpers
    fit_ds(pipe, ds)
    var preds = predict_ds(pipe, ds)
    assert_equal(len(preds), 3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
