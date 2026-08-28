# Getting Started with Strata

Welcome to Strata! This tutorial guides you through installing Strata, creating your first dataset, training a `RandomForestClassifier`, and evaluating predictions in native Mojo.


---

## 1. Prerequisites & Installation

Strata uses [`pixi`](https://pixi.sh) to manage the Mojo toolchain and hardware-accelerated linear algebra libraries (LAPACK).

Clone the repository and install dependencies:

```bash
git clone https://github.com/ethqnol/Strata.git
cd Strata
pixi install
```

Verify your environment by running the test suite:

```bash
pixi run test-ensemble
```

---

## 2. Your First ML Pipeline in Mojo

Create a file named `main.mojo` in your workspace:

```mojo
from strata.core.matrix import Matrix
from strata.ensemble.forest import RandomForestClassifier
from strata.metrics.classification import accuracy_score

def main() raises:
    # 1. Create a 2D feature matrix (8 samples, 2 features)
    var X = Matrix[DType.float64](8, 2)
    
    # Class 0 points (cluster near -3.0)
    X[0, 0] = -3.0; X[0, 1] = -2.0
    X[1, 0] = -2.0; X[1, 1] = -3.0
    X[2, 0] = -4.0; X[2, 1] = -2.5
    X[3, 0] = -2.5; X[3, 1] = -4.0

    # Class 1 points (cluster near +3.0)
    X[4, 0] = 3.0; X[4, 1] = 2.0
    X[5, 0] = 2.0; X[5, 1] = 3.0
    X[6, 0] = 4.0; X[6, 1] = 2.5
    X[7, 0] = 2.5; X[7, 1] = 4.0

    # 2. Target labels
    var y = List[Scalar[DType.int32]](capacity=8)
    for _ in range(4): y.append(0)
    for _ in range(4): y.append(1)

    # 3. Instantiate and train a Random Forest
    var rf = RandomForestClassifier[DType.float64](
        n_estimators=20,
        max_depth=4,
        random_state=42
    )
    rf.fit(X, y)
    print("Random Forest successfully trained!")

    # 4. Predict probabilities and discrete classes
    var preds = rf.predict(X)
    var proba = rf.predict_proba(X)
    
    # 5. Measure accuracy
    var acc = accuracy_score(y, preds)
    print("Training Accuracy:", acc)
```

Run the script with pixi:

```bash
pixi run mojo run -I . main.mojo
```

---

## 3. What You Learned

- How to allocate a contiguous, SIMD-aligned 2D `Matrix[dtype]`.
- How to instantiate and fit a `RandomForestClassifier`.
- How to inspect probability simplex outputs (`predict_proba`) and compute metrics (`accuracy_score`).

Next, continue to the [End-to-End Pipeline Tutorial](end_to_end_pipeline.md) to learn how to compose Scalers, PCA, and Regressors into type-safe pipelines!
