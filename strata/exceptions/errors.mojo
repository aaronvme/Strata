@fieldwise_init
struct NotFittedError(Copyable, Movable, Writable):
    """Exception raised when an estimator is used before calling `fit`."""

    var estimator_name: String
    var message: String

    @staticmethod
    def error(estimator_name: String, msg: String = "") -> Error:
        """Create a formatted NotFittedError message."""
        if msg != "":
            return Error("NotFittedError: " + estimator_name + ": " + msg)
        return Error(
            "NotFittedError: This "
            + estimator_name
            + " instance is not fitted yet. Call 'fit' before using this"
            " estimator."
        )

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "NotFittedError(", self.estimator_name, "): ", self.message
        )


@fieldwise_init
struct DimensionMismatchError(Copyable, Movable, Writable):
    """Exception raised when input matrix/vector dimensions do not match requirements."""

    var expected: String
    var actual: String
    var message: String

    @staticmethod
    def error(expected: String, actual: String, context: String = "") -> Error:
        """Create a formatted DimensionMismatchError message."""
        var msg = (
            "DimensionMismatchError: Expected "
            + expected
            + ", but got "
            + actual
        )
        if context != "":
            msg += " (Context: " + context + ")"
        return Error(msg)

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "DimensionMismatchError: expected ",
            self.expected,
            ", got ",
            self.actual,
        )


@fieldwise_init
struct ConvergenceError(Copyable, Movable, Writable):
    """Exception raised when iterative optimization fails to converge within max iterations."""

    var estimator_name: String
    var max_iter: Int
    var final_loss: Float64

    @staticmethod
    def error(
        estimator_name: String, max_iter: Int, loss: Float64 = 0.0
    ) -> Error:
        """Create a formatted ConvergenceError message."""
        return Error(
            "ConvergenceError: "
            + estimator_name
            + " failed to converge after "
            + String(max_iter)
            + " iterations (final loss: "
            + String(loss)
            + ")"
        )

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "ConvergenceError(",
            self.estimator_name,
            ", max_iter=",
            self.max_iter,
            ")",
        )


@fieldwise_init
struct InvalidParameterError(Copyable, Movable, Writable):
    """Exception raised when an invalid hyperparameter value is supplied."""

    var param_name: String
    var reason: String

    @staticmethod
    def error(param_name: String, reason: String) -> Error:
        """Create a formatted InvalidParameterError message."""
        return Error(
            "InvalidParameterError: Parameter '"
            + param_name
            + "' is invalid: "
            + reason
        )


    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "InvalidParameterError: ", self.param_name, ": ", self.reason
        )


@fieldwise_init
struct DataConversionError(Copyable, Movable, Writable):
    var message: String

    @staticmethod
    def error(msg: String) -> Error:
        return Error("DataConversionError: " + msg)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("DataConversionError: ", self.message)
