from std.math import sqrt
from .types import ArrayLike
from ..exceptions.errors import DimensionMismatchError

struct Matrix[dtype: DType = DType.float64](ArrayLike, Copyable, Movable, Writable):
    var rows: Int
    var cols: Int
    var data: List[Scalar[Self.dtype]]

    def __init__(out self, rows: Int, cols: Int, fill: Scalar[Self.dtype] = 0):
        self.rows = rows
        self.cols = cols
        var total = rows * cols
        var d = List[Scalar[Self.dtype]](capacity=total)
        for _ in range(total):
            d.append(fill)
        self.data = d^

    def __init__(out self, rows: Int, cols: Int, var data: List[Scalar[Self.dtype]]):
        self.rows = rows
        self.cols = cols
        self.data = data^

    @staticmethod
    def zeros(rows: Int, cols: Int) -> Self:
        return Self(rows, cols, 0)

    @staticmethod
    def ones(rows: Int, cols: Int) -> Self:
        return Self(rows, cols, 1)

    def num_rows(self) -> Int:
        return self.rows

    def num_cols(self) -> Int:
        return self.cols

    def num_elements(self) -> Int:
        return self.rows * self.cols

    def shape(self) -> (Int, Int):
        return (self.rows, self.cols)

    def __getitem__(self, r: Int, c: Int) -> Scalar[Self.dtype]:
        return self.data[r * self.cols + c]

    def __setitem__(mut self, r: Int, c: Int, val: Scalar[Self.dtype]):
        self.data[r * self.cols + c] = val

    def row(self, r: Int) -> List[Scalar[Self.dtype]]:
        var res = List[Scalar[Self.dtype]](capacity=self.cols)
        var start_idx = r * self.cols
        for c in range(self.cols):
            res.append(self.data[start_idx + c])
        return res^

    def col(self, c: Int) -> List[Scalar[Self.dtype]]:
        var res = List[Scalar[Self.dtype]](capacity=self.rows)
        for r in range(self.rows):
            res.append(self.data[r * self.cols + c])
        return res^

    def transpose(self) -> Self:
        var res = Self(self.cols, self.rows)
        for r in range(self.rows):
            for c in range(self.cols):
                res[c, r] = self[r, c]
        return res^

    def dot(self, other: Self) raises -> Self:
        from .linalg import gemm
        return gemm[Self.dtype](self, other)

    def dot_vec(self, vec: List[Scalar[Self.dtype]]) raises -> List[Scalar[Self.dtype]]:
        from .linalg import dense_dot_vec
        return dense_dot_vec[Self.dtype](self, vec)

    def mean_along_axis_0(self) -> List[Scalar[Self.dtype]]:
        var means = List[Scalar[Self.dtype]](capacity=self.cols)
        var n_rows = Float64(self.rows)
        for c in range(self.cols):
            var total: Float64 = 0.0
            for r in range(self.rows):
                total += Float64(self[r, c])
            means.append(Scalar[Self.dtype](total / n_rows))
        return means^

    def std_along_axis_0(self, means: List[Scalar[Self.dtype]]) -> List[Scalar[Self.dtype]]:
        var stds = List[Scalar[Self.dtype]](capacity=self.cols)
        var n_rows = Float64(self.rows)
        for c in range(self.cols):
            var var_sum: Float64 = 0.0
            var col_mean = Float64(means[c])
            for r in range(self.rows):
                var diff = Float64(self[r, c]) - col_mean
                var_sum += diff * diff
            var s = sqrt(var_sum / n_rows)
            if s == 0.0:
                s = 1.0
            stds.append(Scalar[Self.dtype](s))
        return stds^

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Matrix[", String(Self.dtype), "](", self.rows, "x", self.cols, ")\n[")
        for r in range(self.rows):
            if r > 0:
                writer.write(" ")
            writer.write("[")
            for c in range(self.cols):
                writer.write(self[r, c])
                if c + 1 < self.cols:
                    writer.write(", ")
            writer.write("]")
            if r + 1 < self.rows:
                writer.write("\n")
        writer.write("]")
