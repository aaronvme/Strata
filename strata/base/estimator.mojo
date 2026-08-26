from ..core.matrix import Matrix
from ..core.dataset import Dataset


trait Estimator(Deinitable, Movable):
    pass


trait Transformer(Estimator):
    def fit[in_dtype: DType](mut self, X: Matrix[in_dtype]) raises:
        ...

    def transform[
        in_dtype: DType
    ](self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        ...

    def fit_transform[
        in_dtype: DType
    ](mut self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        ...


trait Regressor(Estimator):
    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]]) raises:
        ...

    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Scalar[feat_dtype]]:
        ...


trait Classifier(Estimator):
    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]]) raises:
        ...

    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Int]:
        ...

    def predict_proba[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> Matrix[feat_dtype]:
        ...


trait Clusterer(Estimator):
    def fit[in_dtype: DType](mut self, X: Matrix[in_dtype]) raises:
        ...

    def predict[in_dtype: DType](self, X: Matrix[in_dtype]) raises -> List[Int]:
        ...

    def fit_predict[
        in_dtype: DType
    ](mut self, X: Matrix[in_dtype]) raises -> List[Int]:
        ...


def fit[
    R: Regressor, feat_dtype: DType, target_dtype: DType
](mut model: R, dataset: Dataset[feat_dtype, target_dtype]) raises:
    """Fits a Regressor using a Dataset container."""
    model.fit[feat_dtype, target_dtype](dataset.records, dataset.targets)


def fit[
    C: Classifier, feat_dtype: DType, target_dtype: DType
](mut model: C, dataset: Dataset[feat_dtype, target_dtype]) raises:
    """Fits a Classifier using a Dataset container."""
    model.fit[feat_dtype, target_dtype](dataset.records, dataset.targets)


def fit[
    T: Transformer, feat_dtype: DType, target_dtype: DType
](mut model: T, dataset: Dataset[feat_dtype, target_dtype]) raises:
    """Fits a Transformer on dataset records."""
    model.fit[feat_dtype](dataset.records)


def fit[
    K: Clusterer, feat_dtype: DType, target_dtype: DType
](mut model: K, dataset: Dataset[feat_dtype, target_dtype]) raises:
    """Fits a Clusterer on dataset records."""
    model.fit[feat_dtype](dataset.records)


def predict[
    R: Regressor, feat_dtype: DType, target_dtype: DType
](model: R, dataset: Dataset[feat_dtype, target_dtype]) raises -> List[
    Scalar[feat_dtype]
]:
    """Predicts regression targets for a Dataset container."""
    return model.predict[feat_dtype](dataset.records)


def predict[
    C: Classifier, feat_dtype: DType, target_dtype: DType
](model: C, dataset: Dataset[feat_dtype, target_dtype]) raises -> List[Int]:
    """Predicts classification labels for a Dataset container."""
    return model.predict[feat_dtype](dataset.records)


def predict_proba[
    C: Classifier, feat_dtype: DType, target_dtype: DType
](model: C, dataset: Dataset[feat_dtype, target_dtype]) raises -> Matrix[
    feat_dtype
]:
    """Predicts class probabilities for a Dataset container."""
    return model.predict_proba[feat_dtype](dataset.records)


def predict[
    K: Clusterer, feat_dtype: DType, target_dtype: DType
](model: K, dataset: Dataset[feat_dtype, target_dtype]) raises -> List[Int]:
    """Predicts cluster assignments for a Dataset container."""
    return model.predict[feat_dtype](dataset.records)


def transform[
    T: Transformer, feat_dtype: DType, target_dtype: DType
](model: T, dataset: Dataset[feat_dtype, target_dtype]) raises -> Dataset[
    feat_dtype, target_dtype
]:
    """Transforms dataset records and returns a new Dataset preserving labels and names.
    """
    var transformed_records = model.transform[feat_dtype](dataset.records)
    return Dataset[feat_dtype, target_dtype](
        transformed_records^,
        dataset.targets.copy(),
        dataset.feature_names.copy(),
        dataset.target_names.copy(),
    )


def fit_transform[
    T: Transformer, feat_dtype: DType, target_dtype: DType
](mut model: T, dataset: Dataset[feat_dtype, target_dtype]) raises -> Dataset[
    feat_dtype, target_dtype
]:
    """Fits transformer and transforms dataset records in place."""
    var transformed_records = model.fit_transform[feat_dtype](dataset.records)
    return Dataset[feat_dtype, target_dtype](
        transformed_records^,
        dataset.targets.copy(),
        dataset.feature_names.copy(),
        dataset.target_names.copy(),
    )
