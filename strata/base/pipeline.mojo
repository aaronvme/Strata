from ..core.matrix import Matrix
from ..core.dataset import Dataset
from .estimator import Estimator, Transformer, Regressor, Classifier


@fieldwise_init
struct PipelineTransformer[
    T1: Transformer,
    T2: Transformer,
](Copyable, Movable, Transformer):
    """Chains two data transformers into a single composite transformer."""

    var step1: Self.T1
    var step2: Self.T2

    def fit[in_dtype: DType](mut self, X: Matrix[in_dtype]) raises:
        self.step1.fit[in_dtype](X)
        var X_trans = self.step1.transform[in_dtype](X)
        self.step2.fit[in_dtype](X_trans)

    def transform[
        in_dtype: DType
    ](self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        var X_1 = self.step1.transform[in_dtype](X)
        return self.step2.transform[in_dtype](X_1)

    def fit_transform[
        in_dtype: DType
    ](mut self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        var X_1 = self.step1.fit_transform[in_dtype](X)
        return self.step2.fit_transform[in_dtype](X_1)


@fieldwise_init
struct PipelineRegressor[
    T: Transformer,
    R: Regressor,
    target_dtype: DType = DType.float64,
](Copyable, Movable, Regressor):
    """Sequentially applies a transformer pipeline before fitting a regressor.
    """

    var transformer: Self.T
    var regressor: Self.R

    def fit[
        feat_dtype: DType, in_target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[in_target_dtype]]) raises:
        var X_trans = self.transformer.fit_transform[feat_dtype](X)
        self.regressor.fit[feat_dtype, in_target_dtype](X_trans, y)

    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Scalar[feat_dtype]]:
        var X_trans = self.transformer.transform[feat_dtype](X)
        return self.regressor.predict[feat_dtype](X_trans)


@fieldwise_init
struct PipelineClassifier[
    T: Transformer,
    C: Classifier,
    target_dtype: DType = DType.int32,
](Classifier, Copyable, Movable):
    """Sequentially applies a transformer pipeline before fitting a classifier.
    """

    var transformer: Self.T
    var classifier: Self.C

    def fit[
        feat_dtype: DType, in_target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[in_target_dtype]]) raises:
        var X_trans = self.transformer.fit_transform[feat_dtype](X)
        self.classifier.fit[feat_dtype, in_target_dtype](X_trans, y)

    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Int]:
        var X_trans = self.transformer.transform[feat_dtype](X)
        return self.classifier.predict[feat_dtype](X_trans)

    def predict_proba[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> Matrix[feat_dtype]:
        var X_trans = self.transformer.transform[feat_dtype](X)
        return self.classifier.predict_proba[feat_dtype](X_trans)
