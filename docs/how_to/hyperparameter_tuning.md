# Hyperparameter Tuning with Grid & Randomized Search

This guide demonstrates how to tune model hyperparameters in Strata using `GridSearchRegressor` and `RandomizedSearchClassifier` across $K$-Fold cross-validation splits.


---

## 1. Tuning a Decision Tree with Grid Search

`GridSearchRegressor` exhaustively evaluates candidate hyperparameter parameter grids:

```mojo
from strata.core.matrix import Matrix
from strata.tree.regressor import DecisionTreeRegressor
from strata.model_selection.grid_search import GridSearchRegressor

def main() raises:
    var X = Matrix[DType.float64](20, 2)
    var y = List[Scalar[DType.float64]](capacity=20)
    for i in range(20):
        X[i, 0] = Float64(i)
        X[i, 1] = Float64(i * 2)
        y.append(Float64(i * 3 + 1))

    # Define hyperparameter grid
    var max_depth_candidates = List[Int](1, 2, 3, 5)
    var min_samples_split_candidates = List[Int](2, 4)

    # Perform 5-fold cross-validated grid search
    var grid_search = GridSearchRegressor[DType.float64](
        estimator=DecisionTreeRegressor[DType.float64](),
        cv=5,
        max_depth_grid=max_depth_candidates,
        min_samples_split_grid=min_samples_split_candidates
    )

    grid_search.fit(X, y)

    print("Best Score (MSE):", grid_search.best_score_)
    print("Best max_depth:", grid_search.best_params_["max_depth"])

    # Predict using the refitted best model
    var best_preds = grid_search.predict(X)
```

---

## 2. Randomized Search for High-Dimensional Parameter Spaces

When evaluating wide continuous ranges or large forests, `RandomizedSearchClassifier` randomly samples $N$ configurations from the parameter distributions:

```mojo
from strata.ensemble.forest import RandomForestClassifier
from strata.model_selection.randomized_search import RandomizedSearchClassifier

def run_random_search() raises:
    # Set n_iter to evaluate a fixed budget of configurations
    var random_search = RandomizedSearchClassifier[DType.float64](
        estimator=RandomForestClassifier[DType.float64](),
        n_iter=10,
        cv=3,
        random_state=42
    )
```

---

## Related References
- [strata.model_selection Reference](../reference/model_selection.md)
- [strata.tree Reference](../reference/tree.md)
