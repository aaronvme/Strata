from ..core.matrix import Matrix
from .estimator import Estimator, Transformer, Regressor, Classifier

@fieldwise_init
struct PipelineRegressor[
    dtype: DType,
    T: Transformer,
    R: Regressor,
](Regressor, Movable):
    var transformer: Self.T
    var regressor: Self.R

@fieldwise_init
struct PipelineClassifier[
    dtype: DType,
    label_dtype: DType,
    T: Transformer,
    C: Classifier,
](Classifier, Movable):
    var transformer: Self.T
    var classifier: Self.C
