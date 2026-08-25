from ..core.matrix import Matrix


trait Estimator(Movable):
    pass


trait Transformer(Estimator):
    pass


trait Regressor(Estimator):
    pass


trait Classifier(Estimator):
    pass


trait Clusterer(Estimator):
    pass
