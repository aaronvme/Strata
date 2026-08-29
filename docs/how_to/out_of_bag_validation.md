# Out-of-Bag Validation & Feature Importances

In Random Forests, every tree is trained on a bootstrap sample drawn with replacement ($\approx 63.2\%$ unique samples). The remaining $\approx 36.8\%$ of samples are **Out-of-Bag (OOB)** and can be used as an integrated test set without requiring a separate train/test split.


---

## 1. Enabling OOB Estimation

Set `bootstrap=True` and `oob_score=True` in `RandomForestClassifier` or `RandomForestRegressor`:

```mojo
from strata.core.matrix import Matrix
from strata.ensemble.forest import RandomForestClassifier

def main() raises:
    var X = Matrix[DType.float64](50, 4)
    var y = List[Scalar[DType.int32]](capacity=50)
    # Populate X and y...

    # Enable OOB evaluation
    var rf = RandomForestClassifier[DType.float64](
        n_estimators=50,
        bootstrap=True,
        oob_score=True,
        random_state=42
    )

    rf.fit(X, y)

    # Retrieve out-of-bag classification accuracy
    var oob_acc = rf.get_oob_score()
    print("Out-of-Bag Accuracy:", oob_acc)
```

---

## 2. Inspecting MDI Feature Importances

Mean Decrease in Impurity (MDI) importances measure the total normalized reduction in Gini/Entropy/MSE brought by each feature:

```mojo
    var importances = rf.get_feature_importances()
    for j in range(X.cols):
        print("Feature", j, "Importance:", importances[j])
```

---

## Related References
- [strata.ensemble Reference](../reference/ensemble.md)
- [Explanation: Tree Algorithms & MDI Mechanics](../explanation/tree_algorithms.md)
