from ..core.matrix import Matrix
from ..core.dataset import Dataset


trait Estimator(Deinitable, Movable):
    pass


trait Transformer(Estimator):
    def fit[dtype: DType](mut self, X: Matrix[dtype]) raises:
        ...

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, dataset: Dataset[feat_dtype, target_dtype]) raises:
        ...

    def transform[dtype: DType](self, X: Matrix[dtype]) raises -> Matrix[dtype]:
        ...

    def transform[
        feat_dtype: DType, target_dtype: DType
    ](self, dataset: Dataset[feat_dtype, target_dtype]) raises -> Dataset[
        feat_dtype, target_dtype
    ]:
        ...


trait Regressor(Estimator):
    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]]) raises:
        ...

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, dataset: Dataset[feat_dtype, target_dtype]) raises:
        ...

    def predict[
        feat_dtype: DType, target_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Scalar[target_dtype]]:
        ...

    def predict[
        feat_dtype: DType, target_dtype: DType
    ](self, dataset: Dataset[feat_dtype, target_dtype]) raises -> List[
        Scalar[target_dtype]
    ]:
        ...


trait Classifier(Estimator):
    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]]) raises:
        ...

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, dataset: Dataset[feat_dtype, target_dtype]) raises:
        ...

    def predict[
        feat_dtype: DType, target_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Scalar[target_dtype]]:
        ...

    def predict[
        feat_dtype: DType, target_dtype: DType
    ](self, dataset: Dataset[feat_dtype, target_dtype]) raises -> List[
        Scalar[target_dtype]
    ]:
        ...

    def predict_proba[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> Matrix[DType.float64]:
        ...

    def predict_proba[
        feat_dtype: DType, target_dtype: DType
    ](self, dataset: Dataset[feat_dtype, target_dtype]) raises -> Matrix[
        DType.float64
    ]:
        ...


trait Clusterer(Estimator):
    def fit[dtype: DType](mut self, X: Matrix[dtype]) raises:
        ...

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, dataset: Dataset[feat_dtype, target_dtype]) raises:
        ...

    def predict[dtype: DType](self, X: Matrix[dtype]) raises -> List[Int]:
        ...

    def predict[
        feat_dtype: DType, target_dtype: DType
    ](self, dataset: Dataset[feat_dtype, target_dtype]) raises -> List[Int]:
        ...
