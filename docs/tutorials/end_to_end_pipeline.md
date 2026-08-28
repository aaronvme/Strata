# Composing End-to-End ML Pipelines

Strata provides pipeline wrappers like `PipelineRegressor` and `PipelineClassifier` to chain data transformers (such as `StandardScaler`) and estimators into a single object.

---

## How Pipelines Work in Mojo

In Python libraries like scikit-learn, pipelines use dynamic runtime dispatch to pass data between steps. In Mojo, Strata pipelines use compile-time generic types:

```mojo
PipelineRegressor[
    TransformerT: Transformer,
    RegressorT: Regressor,
    target_dtype: DType
]
```

Because concrete types are known at compile time, calls to `transform()` and `predict()` are inlined directly without virtual table lookup overhead.

---

## Step-by-Step Example

Create `pipeline_demo.mojo`:

```mojo
from strata.core.matrix import Matrix
from strata.preprocessing.scaler import StandardScaler
from strata.linear_model.ridge import Ridge
from strata.base.pipeline import PipelineRegressor
from strata.metrics.regression import mean_squared_error, r2_score

def main() raises:
    # 10 samples with 3 features on different scales
    var X = Matrix[DType.float64](10, 3)
    var y = List[Scalar[DType.float64]]()

    for i in range(10):
        X[i, 0] = Float64(i * 1000)      # High-scale feature
        X[i, 1] = Float64(i) * 0.01      # Low-scale feature
        X[i, 2] = Float64(i % 2)         # Binary feature
        y.append(Float64(i * 5 + 2))     # Linear target

    # Instantiate the scaler and regressor
    var scaler = StandardScaler[DType.float64]()
    var ridge = Ridge[DType.float64](alpha=1.0)

    # Compose into a pipeline. Note the transfer operator (^) passing ownership
    var pipe = PipelineRegressor[
        StandardScaler[DType.float64],
        Ridge[DType.float64],
        DType.float64
    ](scaler^, ridge^)

    # Fit the pipeline: standardizes X, then fits Ridge on scaled features
    pipe.fit(X, y)

    # Predict: automatically scales input features before running inference
    var preds = pipe.predict(X)

    # Calculate metrics
    var mse = mean_squared_error(y, preds)
    var r2 = r2_score(y, preds)

    print("MSE:", mse)
    print("R²:", r2)
```

Run the script:

```bash
pixi run mojo run -I . -Xlinker -L$CONDA_PREFIX/lib -Xlinker -llapack pipeline_demo.mojo
```

---

## Ownership and Transfer Operator (`^`)

In Mojo, passing an object into a struct constructor requires explicit ownership handling:
- `scaler^` uses Mojo's transfer operator `^` to move ownership of `scaler` into `PipelineRegressor`.
- Once moved, the original `scaler` variable can no longer be accessed, preventing unintended mutations or double-free errors.
