from .types import ArrayLike
from .matrix import Matrix
from ..exceptions.errors import DimensionMismatchError

struct MatrixView[
    dtype: DType,
    origin: Origin,
](ArrayLike, Copyable, Movable):
    """Non-owning 2D view over a contiguous or strided matrix memory buffer."""

    var ptr: Pointer[Scalar[Self.dtype], Self.origin]
    var rows: Int
    var cols: Int
    var row_stride: Int
    var col_stride: Int

    def __init__(
        out self,
        ptr: Pointer[Scalar[Self.dtype], Self.origin],
        rows: Int,
        cols: Int,
        row_stride: Int,
        col_stride: Int = 1,
    ):
        self.ptr = ptr
        self.rows = rows
        self.cols = cols
        self.row_stride = row_stride
        self.col_stride = col_stride

    def num_rows(self) -> Int:
        return self.rows

    def num_cols(self) -> Int:
        return self.cols

    def num_elements(self) -> Int:
        return self.rows * self.cols

    def shape(self) -> Tuple[Int, Int]:
        return (self.rows, self.cols)

    def __getitem__(self, r: Int, c: Int) -> Scalar[Self.dtype]:
        return self.ptr[unsafe_offset=r * self.row_stride + c * self.col_stride]

    def slice_rows(
        self, start_row: Int, end_row: Int
    ) raises -> MatrixView[Self.dtype, Self.origin]:
        if start_row < 0 or end_row > self.rows or start_row > end_row:
            raise DimensionMismatchError.error(
                "Valid row slice [0, " + String(self.rows) + "]",
                "[" + String(start_row) + ", " + String(end_row) + "]",
                "MatrixView.slice_rows",
            )
        var new_ptr = self.ptr.unsafe_offset(start_row * self.row_stride)
        return MatrixView[Self.dtype, Self.origin](
            new_ptr,
            end_row - start_row,
            self.cols,
            self.row_stride,
            self.col_stride,
        )

    def slice_cols(
        self, start_col: Int, end_col: Int
    ) raises -> MatrixView[Self.dtype, Self.origin]:
        if start_col < 0 or end_col > self.cols or start_col > end_col:
            raise DimensionMismatchError.error(
                "Valid col slice [0, " + String(self.cols) + "]",
                "[" + String(start_col) + ", " + String(end_col) + "]",
                "MatrixView.slice_cols",
            )
        var new_ptr = self.ptr.unsafe_offset(start_col * self.col_stride)
        return MatrixView[Self.dtype, Self.origin](
            new_ptr,
            self.rows,
            end_col - start_col,
            self.row_stride,
            self.col_stride,
        )

    def slice_2d(
        self,
        start_row: Int,
        end_row: Int,
        start_col: Int,
        end_col: Int,
    ) raises -> MatrixView[Self.dtype, Self.origin]:
        var row_view = self.slice_rows(start_row, end_row)
        return row_view.slice_cols(start_col, end_col)

    def to_matrix(self) -> Matrix[Self.dtype]:
        var res = Matrix[Self.dtype](self.rows, self.cols, 0)
        for r in range(self.rows):
            for c in range(self.cols):
                res[r, c] = self[r, c]
        return res^
