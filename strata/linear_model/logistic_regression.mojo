from ..core.matrix import Matrix
from ..base.estimator import Classifier
from ..utils.validation import (
    check_X_y,
    check_array,
    check_floating_dtype,
    check_is_fitted,
)
from ..utils.math import softmax
from ..exceptions.errors import InvalidParameterError, DimensionMismatchError


struct LogisticRegression[
    compute_dtype: DType = DType.float64,
](Classifier, Movable, Copyable):
    """Logistic Regression classifier with L2 regularization.

    Supports binary and multiclass (multinomial) classification with
    gradient optimization.

    Parameters:
        compute_dtype: Floating point precision for internal computations (default: Float64).
    """

    var is_fitted: Bool
    var penalty: String
    var C: Scalar[Self.compute_dtype]
    var fit_intercept: Bool
    var max_iter: Int
    var tol: Scalar[Self.compute_dtype]
    var learning_rate: Scalar[Self.compute_dtype]
    var classes_: List[Int]
    var coef_: Matrix[Self.compute_dtype]
    var intercept_: List[Scalar[Self.compute_dtype]]

    def __init__(
        out self,
        penalty: String = "l2",
        C: Scalar[Self.compute_dtype] = 1.0,
        fit_intercept: Bool = True,
        max_iter: Int = 100,
        tol: Scalar[Self.compute_dtype] = 1e-4,
        learning_rate: Scalar[Self.compute_dtype] = 0.1,
    ) raises:
        """Initializes the LogisticRegression estimator.

        Args:
            penalty: Regularization norm ('l2' or 'none').
            C: Inverse regularization strength (must be strictly positive).
            fit_intercept: Whether to calculate the intercept / bias term.
            max_iter: Maximum number of optimization iterations.
            tol: Tolerance for stopping criterion.
            learning_rate: Initial step size for gradient updates.
        """
        check_floating_dtype[Self.compute_dtype, "LogisticRegression"]()
        if penalty != "l2" and penalty != "none" and penalty != "None":
            raise InvalidParameterError.error(
                "penalty",
                "Unsupported penalty '"
                + penalty
                + "'. Expected 'l2' or 'none'.",
            )
        if C <= 0:
            raise InvalidParameterError.error(
                "C",
                "C must be strictly positive, got " + String(C),
            )
        if max_iter <= 0:
            raise InvalidParameterError.error(
                "max_iter",
                "max_iter must be strictly positive, got " + String(max_iter),
            )
        if tol < 0:
            raise InvalidParameterError.error(
                "tol",
                "tol must be non-negative, got " + String(tol),
            )
        self.is_fitted = False
        self.penalty = penalty
        self.C = C
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.classes_ = List[Int]()
        self.coef_ = Matrix[Self.compute_dtype](0, 0, 0)
        self.intercept_ = List[Scalar[Self.compute_dtype]]()

    def __init__(out self, *, copy: Self):
        """Copies an existing LogisticRegression instance."""
        self.is_fitted = copy.is_fitted
        self.penalty = copy.penalty
        self.C = copy.C
        self.fit_intercept = copy.fit_intercept
        self.max_iter = copy.max_iter
        self.tol = copy.tol
        self.learning_rate = copy.learning_rate
        self.classes_ = copy.classes_.copy()
        self.coef_ = copy.coef_.copy()
        self.intercept_ = copy.intercept_.copy()

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]]) raises:
        """Fits the logistic regression model on training data (X, y).

        Args:
            X: Training feature matrix (N x D).
            y: Discrete target class labels (length N).
        """
        check_X_y(X, y)

        var N = X.rows
        var D = X.cols

        var raw_classes = List[Int]()
        for i in range(len(y)):
            var label = Int(y[i])
            var found = False
            for c_idx in range(len(raw_classes)):
                if raw_classes[c_idx] == label:
                    found = True
                    break
            if not found:
                raw_classes.append(label)

        for i in range(len(raw_classes)):
            for j in range(i + 1, len(raw_classes)):
                if raw_classes[j] < raw_classes[i]:
                    var temp = raw_classes[i]
                    raw_classes[i] = raw_classes[j]
                    raw_classes[j] = temp

        var K = len(raw_classes)
        if K < 2:
            raise InvalidParameterError.error(
                "y",
                "LogisticRegression requires at least 2 distinct classes, got "
                + String(K),
            )
        self.classes_ = raw_classes^

        var y_idx = List[Int](capacity=N)
        for i in range(N):
            var target_val = Int(y[i])
            for k in range(K):
                if self.classes_[k] == target_val:
                    y_idx.append(k)
                    break

        var X_comp = X.cast[Self.compute_dtype]()

        var W = Matrix[Self.compute_dtype](K, D, 0)
        var b = List[Scalar[Self.compute_dtype]](capacity=K)
        for _ in range(K):
            b.append(0)

        var use_l2 = self.penalty == "l2"
        var n_samples = Scalar[Self.compute_dtype](N)
        var lambda_reg: Scalar[Self.compute_dtype] = (
            1.0 / (self.C * n_samples) if use_l2 else 0.0
        )

        for _ in range(self.max_iter):
            var grad_W = Matrix[Self.compute_dtype](K, D, 0)
            var grad_b = List[Scalar[Self.compute_dtype]](capacity=K)
            for _ in range(K):
                grad_b.append(0)

            for i in range(N):
                var logits = List[Scalar[Self.compute_dtype]](capacity=K)
                for k in range(K):
                    var z: Scalar[Self.compute_dtype] = b[
                        k
                    ] if self.fit_intercept else 0
                    for j in range(D):
                        z += W[k, j] * X_comp[i, j]
                    logits.append(z)

                var probs = softmax[Self.compute_dtype](logits)

                for k in range(K):
                    var target_k: Scalar[Self.compute_dtype] = (
                        1.0 if y_idx[i] == k else 0.0
                    )
                    var err = probs[k] - target_k
                    for j in range(D):
                        grad_W[k, j] += err * X_comp[i, j]
                    if self.fit_intercept:
                        grad_b[k] += err

            var grad_norm_sq: Scalar[Self.compute_dtype] = 0
            for k in range(K):
                for j in range(D):
                    var g_w = grad_W[k, j] / n_samples + lambda_reg * W[k, j]
                    grad_norm_sq += g_w * g_w
                if self.fit_intercept:
                    var g_b = grad_b[k] / n_samples
                    grad_norm_sq += g_b * g_b

            if grad_norm_sq < self.tol * self.tol:
                break

            for k in range(K):
                for j in range(D):
                    var g_w = grad_W[k, j] / n_samples + lambda_reg * W[k, j]
                    W[k, j] -= self.learning_rate * g_w
                if self.fit_intercept:
                    b[k] -= self.learning_rate * (grad_b[k] / n_samples)

        self.coef_ = W^
        self.intercept_ = b^
        self.is_fitted = True

    def predict_proba[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> Matrix[feat_dtype]:
        """Predicts class probability distributions for samples in X.

        Args:
            X: Feature matrix of shape (N x D).

        Returns:
            Matrix of shape (N x K) where row i contains normalized class probabilities.
        """
        check_is_fitted("LogisticRegression", self.is_fitted)
        check_array[feat_dtype](X)
        var D = self.coef_.cols
        if X.cols != D:
            raise DimensionMismatchError.error(
                "X.cols == " + String(D),
                "X.cols == " + String(X.cols),
                "LogisticRegression.predict_proba",
            )

        var N = X.rows
        var K = len(self.classes_)
        var X_comp = X.cast[Self.compute_dtype]()
        var probs_data = List[Scalar[feat_dtype]](capacity=N * K)

        for i in range(N):
            var logits = List[Scalar[Self.compute_dtype]](capacity=K)
            for k in range(K):
                var z: Scalar[Self.compute_dtype] = self.intercept_[
                    k
                ] if self.fit_intercept else 0
                for j in range(D):
                    z += self.coef_[k, j] * X_comp[i, j]
                logits.append(z)

            var probs = softmax[Self.compute_dtype](logits)
            for k in range(K):
                probs_data.append(Scalar[feat_dtype](probs[k]))

        return Matrix[feat_dtype](N, K, probs_data^)

    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Int]:
        """Predicts discrete class labels for samples in X.

        Args:
            X: Feature matrix of shape (N x D).

        Returns:
            List of predicted class labels (length N).
        """
        check_is_fitted("LogisticRegression", self.is_fitted)
        check_array[feat_dtype](X)
        var D = self.coef_.cols
        if X.cols != D:
            raise DimensionMismatchError.error(
                "X.cols == " + String(D),
                "X.cols == " + String(X.cols),
                "LogisticRegression.predict",
            )

        var N = X.rows
        var K = len(self.classes_)
        var X_comp = X.cast[Self.compute_dtype]()

        var preds = List[Int](capacity=N)
        for i in range(N):
            var best_k = 0
            var max_logit: Scalar[Self.compute_dtype] = 0
            for k in range(K):
                var z: Scalar[Self.compute_dtype] = self.intercept_[
                    k
                ] if self.fit_intercept else 0
                for j in range(D):
                    z += self.coef_[k, j] * X_comp[i, j]
                if k == 0 or z > max_logit:
                    max_logit = z
                    best_k = k
            preds.append(self.classes_[best_k])

        return preds^
