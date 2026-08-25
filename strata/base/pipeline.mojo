from ..core.matrix import Matrix
from ..core.dataset import Dataset
from .estimator import Estimator, Transformer, Regressor, Classifier


struct PipelineRegressor[
    T: Transformer,
    R: Regressor,
](Movable, Regressor):
    var transformer: Self.T
    var regressor: Self.R

    def __init__(out self, var transformer: Self.T, var regressor: Self.R):
        self.transformer = transformer^
        self.regressor = regressor^

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]],) raises:
        self.transformer.fit[feat_dtype](X)
        var X_trans = self.transformer.transform[feat_dtype](X)
        self.regressor.fit[feat_dtype, target_dtype](X_trans, y)

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, dataset: Dataset[feat_dtype, target_dtype],) raises:
        self.fit[feat_dtype, target_dtype](dataset.records, dataset.targets)

    def predict[
        feat_dtype: DType, target_dtype: DType
    ](self, X: Matrix[feat_dtype],) raises -> List[Scalar[target_dtype]]:
        var X_trans = self.transformer.transform[feat_dtype](X)
        return self.regressor.predict[feat_dtype, target_dtype](X_trans)

    def predict[
        feat_dtype: DType, target_dtype: DType
    ](
        self,
        dataset: Dataset[feat_dtype, target_dtype],
    ) raises -> List[
        Scalar[target_dtype]
    ]:
        return self.predict[feat_dtype, target_dtype](dataset.records)


struct PipelineClassifier[
    T: Transformer,
    C: Classifier,
](Classifier, Movable):
    var transformer: Self.T
    var classifier: Self.C

    def __init__(out self, var transformer: Self.T, var classifier: Self.C):
        self.transformer = transformer^
        self.classifier = classifier^

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]],) raises:
        self.transformer.fit[feat_dtype](X)
        var X_trans = self.transformer.transform[feat_dtype](X)
        self.classifier.fit[feat_dtype, target_dtype](X_trans, y)

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, dataset: Dataset[feat_dtype, target_dtype],) raises:
        self.fit[feat_dtype, target_dtype](dataset.records, dataset.targets)

    def predict[
        feat_dtype: DType, target_dtype: DType
    ](self, X: Matrix[feat_dtype],) raises -> List[Scalar[target_dtype]]:
        var X_trans = self.transformer.transform[feat_dtype](X)
        return self.classifier.predict[feat_dtype, target_dtype](X_trans)

    def predict[
        feat_dtype: DType, target_dtype: DType
    ](
        self,
        dataset: Dataset[feat_dtype, target_dtype],
    ) raises -> List[
        Scalar[target_dtype]
    ]:
        return self.predict[feat_dtype, target_dtype](dataset.records)

    def predict_proba[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype],) raises -> Matrix[DType.float64]:
        var X_trans = self.transformer.transform[feat_dtype](X)
        return self.classifier.predict_proba[feat_dtype](X_trans)

    def predict_proba[
        feat_dtype: DType, target_dtype: DType
    ](
        self,
        dataset: Dataset[feat_dtype, target_dtype],
    ) raises -> Matrix[
        DType.float64
    ]:
        return self.predict_proba[feat_dtype](dataset.records)
