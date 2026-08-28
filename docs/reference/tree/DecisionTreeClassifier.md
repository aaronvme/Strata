# `DecisionTreeClassifier`

**Module**: [`strata.tree`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Classifier, Copyable, Movable`  
**Source**: [`strata/tree/classifier.mojo`](file:////home/ewu/Code/Strata/strata/tree/classifier.mojo)

```mojo
struct DecisionTreeClassifier[compute_dtype: DType = DType.float64](Classifier, Copyable, Movable)
```

```mojo
from strata.tree import DecisionTreeClassifier
```

**Decision Tree Classifier for non-parametric supervised classification.**

Splits internal nodes to maximize impurity reduction based on Gini impurity
or Shannon entropy:
$$
H_{\text{gini}}(Q) = 1 - \sum_{k=1}^{K} p_k^2, \quad H_{\text{entropy}}(Q) = -\sum_{k=1}^{K} p_k \log_2(p_k)
$$

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Computational precision data type. Default DType.float64. |

---

## Arguments (Runtime)

| Argument | Description |
| :--- | :--- |
| **`criterion`** | The function to measure the quality of a split ('gini', 'entropy', 'log_loss'). Default 'gini'. |
| **`splitter`** | Strategy used to choose the split at each node ('best', 'random'). Default 'best'. |
| **`max_depth`** | Maximum tree depth. -1 indicates unlimited depth. Default -1. |
| **`min_samples_split`** | Minimum samples required to split an internal node. Default 2. |
| **`min_samples_leaf`** | Minimum samples required to be at a leaf node. Default 1. |
| **`min_impurity_decrease`** | Split threshold if impurity decrease >= this value. Default 0.0. |
| **`max_features`** | Number of features to consider when looking for best split ('all', 'sqrt', 'log2', 'custom'). Default 'all'. |
| **`max_features_count`** | Explicit number of features to evaluate when max_features='custom'. Default -1. |
| **`max_features_ratio`** | Proportion of features to evaluate when max_features='custom'. Default 0.0. |
| **`random_state`** | PRNG seed for deterministic feature and split selection. Default 42. |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`classes_`** | Sorted list of unique class labels observed during fit. |
| **`n_classes_`** | Number of unique classes observed. |
| **`n_features_in_`** | Number of features seen during fit. |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`DecisionTreeClassifier.fit()`](#fit) | Fits the decision tree classifier on (X, y). |
| [`DecisionTreeClassifier.predict()`](#predict) | Generates discrete class predictions for input matrix X. |
| [`DecisionTreeClassifier.predict_proba()`](#predict_proba) | Generates class probability estimates for input matrix X. |
| [`DecisionTreeClassifier.get_depth()`](#get_depth) | Returns the maximum depth of the fitted tree. |
| [`DecisionTreeClassifier.get_n_leaves()`](#get_n_leaves) | Returns the total number of leaf nodes in the fitted tree. |

---

## Method Details

### `DecisionTreeClassifier.fit()`

```mojo
def fit[feat_dtype: DType, target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```

Fits the decision tree classifier on (X, y).

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and target labels. |

---

### `DecisionTreeClassifier.predict()`

```mojo
def predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Int]
def predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Int]
```

Generates discrete class predictions for input matrix X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |

**Returns**: `List[Int]`

---

### `DecisionTreeClassifier.predict_proba()`

```mojo
def predict_proba[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> Matrix[feat_dtype]
def predict_proba[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Matrix[feat_dtype]
```

Generates class probability estimates for input matrix X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |

**Returns**: `Matrix[feat_dtype]`

---

### `DecisionTreeClassifier.get_depth()`

```mojo
def get_depth(self) -> Int
```

Returns the maximum depth of the fitted tree.

**Returns**: `Int`

---

### `DecisionTreeClassifier.get_n_leaves()`

```mojo
def get_n_leaves(self) -> Int
```

Returns the total number of leaf nodes in the fitted tree.

**Returns**: `Int`
---

## Example

```mojo
from strata.tree import DecisionTreeClassifier
from strata.core import Matrix

var tree = DecisionTreeClassifier[DType.float64](max_depth=5, criterion="gini")
tree.fit(X_train, y_train)
var preds = tree.predict(X_test)
```
