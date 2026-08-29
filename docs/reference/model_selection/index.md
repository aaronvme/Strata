# `strata.model_selection`

Cross-validation splitters (K-Fold, Stratified, TimeSeries, Shuffle), cross_val_score, cross_validate, and Grid/Randomized hyperparameter search.

---

## Structs & Classes

| Struct | Description |
| :--- | :--- |
| [`Split`](Split.md) | Pair of train and validation sample indices for a cross-validation fold. |
| [`KFold`](KFold.md) | K-Fold cross-validator. |
| [`StratifiedKFold`](StratifiedKFold.md) | Stratified K-Fold cross-validator for classification datasets. |
| [`TimeSeriesSplit`](TimeSeriesSplit.md) | Time Series cross-validator. |
| [`ShuffleSplit`](ShuffleSplit.md) | Random permutation cross-validator. |
| [`StratifiedShuffleSplit`](StratifiedShuffleSplit.md) | Stratified random permutation cross-validator. |
| [`CrossValidateResult`](CrossValidateResult.md) | Per-fold scores for one or more metrics from a cross-validation run. |
| [`GridSearchRegressor`](GridSearchRegressor.md) | Exhaustive hyperparameter grid search for regression models. |
| [`GridSearchClassifier`](GridSearchClassifier.md) | Exhaustive hyperparameter grid search for classification models. |
| [`RandomizedSearchRegressor`](RandomizedSearchRegressor.md) | Randomized hyperparameter search for regression models. |
| [`RandomizedSearchClassifier`](RandomizedSearchClassifier.md) | Randomized hyperparameter search for classification models. |

## Functions

| Function | Description |
| :--- | :--- |
| [`train_test_split`](train_test_split.md) | Split a Dataset container into random train and test partitions. |
| [`train_test_split`](train_test_split.md) | Split feature matrix and target list into train and test partitions. |
| [`cross_val_score`](cross_val_score.md) | Evaluate regression scores by cross-validation across K folds. |
| [`cross_val_score`](cross_val_score.md) | Evaluate regression scores by cross-validation on predefined splits. |
| [`cross_val_score`](cross_val_score.md) | Evaluates classification scores by stratified cross-validation across K folds. |
| [`cross_val_score`](cross_val_score.md) | Evaluates classification scores by cross-validation on pre-defined splits. |
| [`cross_val_predict`](cross_val_predict.md) | Generates out-of-fold regression predictions across K folds. |
| [`cross_val_predict`](cross_val_predict.md) | Generates out-of-fold regression predictions on pre-defined splits. |
| [`cross_val_predict`](cross_val_predict.md) | Generates out-of-fold class label predictions across K stratified folds. |
| [`cross_val_predict`](cross_val_predict.md) | Generates out-of-fold class label predictions on pre-defined splits. |
