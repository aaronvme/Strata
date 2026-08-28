# Composing End-to-End ML Pipelines

In this guide, you will build an end-to-end Machine Learning pipeline that chains data standardization (`StandardScaler`) with a regularized linear model (`Ridge`) using Strata's `PipelineRegressor`.


---

## 1. Why Pipelines in Mojo?

In Python, pipelines rely on dynamic dispatch (`getattr`, `**kwargs`) and boxed memory objects. In Mojo, Strata pipelines use compile-time generic specialization:

```mojo
PipelineRegressor[
    TransformerT: Transformer,
    RegressorT: Regressor,
    target_dtype: DType
]
```

This guarantees:
- **Zero dynamic dispatch**: Method calls inline directly into hardware SIMD loops.
- **Strict memory safety**: Intermediate feature matrices are passed efficiently without dangling references.

---

## 2. Step-by-Step Implementation

Create `pipeline_demo.mojo`:

```mojo
from strata.core.matrix import Matrix
from strata.preprocessing.scaler import StandardScaler
from strata.linear_model.ridge import Ridge
from strata.base.pipeline import PipelineRegressor
from strata.metrics.regression import mean_squared_error, r2_score

def main() raises:
    # Generate 10 samples with 3 unscaled features
    var X = Matrix[DType.float64](10, 3)
    var y = List[Scalar[DType.float64]](capacity=10)

    for i in range(10):
        X[i, 0] = Float64(i * 1000)      # High-scale feature
        X[i, 1] = Float64(i) * 0.01      # Micro-scale feature
        X[i, 2] = Float64(i % 2)         # Binary indicator
        y.append(Float64(i * 5 + 2))     # Target: linear relation

    # 1. Instantiate the individual components
    var scaler = StandardScaler[DType.float64]()
    var ridge = Ridge[DType.float64](alpha=1.0)

    # 2. Compose into a unified pipeline
    var pipe = PipelineRegressor[
        StandardScaler[DType.float64],
        Ridge[DType.float64],
        DType.float64
    ](scaler^, ridge^)

    # 3. Fit the pipeline (Standardizes X, then trains Ridge)
    pipe.fit(X, y)
    print("Pipeline successfully fitted!")

    # 4. Predict on new data (Automatically scales input before inference)
    var preds = pipe.predict(X)

    # 5. Evaluate regression performance
    var mse = mean_squared_error(y, preds)
    var r2 = r2_score(y, preds)
    print("Mean Squared Error:", mse)
    print("R² Score:", r2)
```

Run with:

```bash
pixi run mojo run -I . -Xlinker -L$CONDA_PREFIX/lib -Xlinker -llapack pipeline_demo.mojo
```

---

## 3. Key Takeaways

- `PipelineRegressor` takes ownership of its components using Mojo's transfer operator (`^`).
- When calling `pipe.predict(X)`, features are automatically transformed through all preprocessing steps before reaching the final estimator.
