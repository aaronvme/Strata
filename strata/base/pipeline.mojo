from ..core.matrix import Matrix
from .estimator import Estimator, Transformer, Regressor, Classifier


@fieldwise_init
struct PipelineRegressor[
    feat_dtype: DType,
    target_dtype: DType,
    T: Transformer,
    R: Regressor,
](Movable, Regressor):
    var transformer: Self.T
    var regressor: Self.R


@fieldwise_init
struct PipelineClassifier[
    feat_dtype: DType,
    target_dtype: DType,
    T: Transformer,
    C: Classifier,
](Classifier, Movable):
    var transformer: Self.T
    var classifier: Self.C
