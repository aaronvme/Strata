comptime Float32Type = DType.float32
comptime Float64Type = DType.float64
comptime Int32Type = DType.int32
comptime Int64Type = DType.int64


trait ArrayLike:
    """Interface for 2D dimensional containers providing shape queries."""

    def num_rows(self) -> Int:
        ...

    def num_cols(self) -> Int:
        ...

    def num_elements(self) -> Int:
        ...

    def shape(self) -> Tuple[Int, Int]:
        ...
