from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_raises,
    assert_almost_equal,
)
from strata import Matrix, MatrixView, DimensionMismatchError


def test_matrix_view_basic() raises:
    var m = Matrix[DType.float64](4, 4, 0)
    for r in range(4):
        for c in range(4):
            m[r, c] = Float64(r * 10 + c)

    var v = m.view()
    assert_equal(v.rows, 4)
    assert_equal(v.cols, 4)
    assert_equal(v.num_rows(), 4)
    assert_equal(v.num_cols(), 4)
    assert_equal(v.num_elements(), 16)
    assert_equal(v[2, 3], 23.0)


def test_matrix_view_slicing() raises:
    var m = Matrix[DType.float64](4, 5, 0)
    for r in range(4):
        for c in range(5):
            m[r, c] = Float64(r * 10 + c)

    var row_sub = m.slice_rows(1, 3)
    assert_equal(row_sub.rows, 2)
    assert_equal(row_sub.cols, 5)
    assert_equal(row_sub[0, 0], 10.0)
    assert_equal(row_sub[1, 4], 24.0)

    var col_sub = m.slice_cols(2, 4)
    assert_equal(col_sub.rows, 4)
    assert_equal(col_sub.cols, 2)
    assert_equal(col_sub[0, 0], 2.0)
    assert_equal(col_sub[3, 1], 33.0)

    var sub2d = m.slice_2d(1, 3, 2, 4)
    assert_equal(sub2d.rows, 2)
    assert_equal(sub2d.cols, 2)
    assert_equal(sub2d[0, 0], 12.0)
    assert_equal(sub2d[0, 1], 13.0)
    assert_equal(sub2d[1, 0], 22.0)
    assert_equal(sub2d[1, 1], 23.0)

    var owned_m = sub2d.to_matrix()
    assert_equal(owned_m.rows, 2)
    assert_equal(owned_m.cols, 2)
    assert_equal(owned_m[1, 1], 23.0)


def test_matrix_view_bounds_validation() raises:
    var m = Matrix[DType.float64](4, 4, 0)

    with assert_raises():
        _ = m.slice_rows(-1, 2)

    with assert_raises():
        _ = m.slice_rows(0, 5)

    with assert_raises():
        _ = m.slice_rows(3, 2)

    with assert_raises():
        _ = m.slice_cols(-1, 2)

    with assert_raises():
        _ = m.slice_cols(0, 5)

    with assert_raises():
        _ = m.slice_cols(3, 2)


def test_view_nested_subview_slicing() raises:
    var m = Matrix[DType.float64](6, 6, 0)
    for r in range(6):
        for c in range(6):
            m[r, c] = Float64(r * 10 + c)

    var v1 = m.view()
    var v2 = v1.slice_2d(1, 5, 1, 5)  # 4x4
    assert_equal(v2.rows, 4)
    assert_equal(v2.cols, 4)
    assert_equal(v2[0, 0], 11.0)
    assert_equal(v2[3, 3], 44.0)

    var v3 = v2.slice_2d(1, 3, 1, 3)  # 2x2
    assert_equal(v3.rows, 2)
    assert_equal(v3.cols, 2)
    assert_equal(v3[0, 0], 22.0)
    assert_equal(v3[1, 1], 33.0)


def test_view_nested_slice_rows() raises:
    var m = Matrix[DType.float64](5, 3, 0)
    for r in range(5):
        for c in range(3):
            m[r, c] = Float64(r * 10 + c)

    var v = m.view()
    var v_rows = v.slice_rows(1, 4)  # 3x3
    var v_sub_rows = v_rows.slice_rows(1, 3)  # 2x3
    assert_equal(v_sub_rows.rows, 2)
    assert_equal(v_sub_rows.cols, 3)
    assert_equal(v_sub_rows[0, 0], 20.0)
    assert_equal(v_sub_rows[1, 2], 32.0)


def test_view_nested_slice_cols() raises:
    var m = Matrix[DType.float64](3, 6, 0)
    for r in range(3):
        for c in range(6):
            m[r, c] = Float64(r * 10 + c)

    var v = m.view()
    var v_cols = v.slice_cols(1, 5)  # 3x4
    var v_sub_cols = v_cols.slice_cols(1, 3)  # 3x2
    assert_equal(v_sub_cols.rows, 3)
    assert_equal(v_sub_cols.cols, 2)
    assert_equal(v_sub_cols[0, 0], 2.0)
    assert_equal(v_sub_cols[2, 1], 23.0)


def test_view_single_element_slice() raises:
    var m = Matrix[DType.float64](4, 4, 0)
    m[2, 3] = 42.0
    var sub1x1 = m.slice_2d(2, 3, 3, 4)
    assert_equal(sub1x1.rows, 1)
    assert_equal(sub1x1.cols, 1)
    assert_equal(sub1x1[0, 0], 42.0)


def test_view_single_row_slice() raises:
    var m = Matrix[DType.float64](3, 4, 0)
    for c in range(4):
        m[1, c] = Float64(c + 1)
    var row_view = m.slice_rows(1, 2)
    assert_equal(row_view.rows, 1)
    assert_equal(row_view.cols, 4)
    for c in range(4):
        assert_equal(row_view[0, c], Float64(c + 1))


def test_view_single_col_slice() raises:
    var m = Matrix[DType.float64](4, 3, 0)
    for r in range(4):
        m[r, 2] = Float64(r + 1)
    var col_view = m.slice_cols(2, 3)
    assert_equal(col_view.rows, 4)
    assert_equal(col_view.cols, 1)
    for r in range(4):
        assert_equal(col_view[r, 0], Float64(r + 1))


def test_view_full_matrix_slice() raises:
    var m = Matrix[DType.float64](3, 3, 7.0)
    var full_view = m.slice_2d(0, 3, 0, 3)
    assert_equal(full_view.rows, 3)
    assert_equal(full_view.cols, 3)
    for r in range(3):
        for c in range(3):
            assert_equal(full_view[r, c], 7.0)


def test_view_to_matrix_deep_copy_isolation() raises:
    var m = Matrix[DType.float64](3, 3, 1.0)
    var sub = m.slice_2d(0, 2, 0, 2)
    var owned = sub.to_matrix()
    owned[0, 0] = 999.0
    assert_equal(m[0, 0], 1.0)
    assert_equal(sub[0, 0], 1.0)
    assert_equal(owned[0, 0], 999.0)


def test_view_float32_slicing() raises:
    var m = Matrix[DType.float32](3, 3, 0)
    m[1, 1] = 3.14
    var v = m.slice_2d(1, 2, 1, 2)
    assert_almost_equal(Float64(v[0, 0]), 3.14, rtol=1e-4)


def test_view_int32_slicing() raises:
    var m = Matrix[DType.int32](3, 3, 0)
    m[2, 2] = -42
    var v = m.slice_2d(2, 3, 2, 3)
    assert_equal(v[0, 0], -42)


def test_view_shape_and_num_elements_queries() raises:
    var m = Matrix[DType.float64](5, 8, 0)
    var v = m.slice_2d(1, 4, 2, 7)  # 3x5
    assert_equal(v.shape()[0], 3)
    assert_equal(v.shape()[1], 5)
    assert_equal(v.num_elements(), 15)


def test_view_empty_slice_zero_rows() raises:
    var m = Matrix[DType.float64](4, 4, 1.0)
    var empty_v = m.slice_rows(2, 2)
    assert_equal(empty_v.rows, 0)
    assert_equal(empty_v.cols, 4)
    assert_equal(empty_v.num_elements(), 0)


def test_view_empty_slice_zero_cols() raises:
    var m = Matrix[DType.float64](4, 4, 1.0)
    var empty_v = m.slice_cols(2, 2)
    assert_equal(empty_v.rows, 4)
    assert_equal(empty_v.cols, 0)
    assert_equal(empty_v.num_elements(), 0)


def test_view_nested_subview_bounds_errors() raises:
    var m = Matrix[DType.float64](6, 6, 0)
    var v = m.slice_2d(1, 4, 1, 4)  # 3x3

    with assert_raises():
        _ = v.slice_rows(-1, 2)

    with assert_raises():
        _ = v.slice_rows(0, 4)  # Exceeds 3

    with assert_raises():
        _ = v.slice_cols(0, 4)  # Exceeds 3


def test_view_bfloat16_slicing() raises:
    var m = Matrix[DType.bfloat16](3, 3, 2.5)
    var v = m.slice_2d(0, 2, 0, 2)
    assert_almost_equal(Float64(v[0, 0]), 2.5, rtol=1e-2)


def test_view_strided_access_non_first_column() raises:
    var m = Matrix[DType.float64](3, 4, 0)
    m[0, 3] = 100.0
    m[1, 3] = 200.0
    m[2, 3] = 300.0

    var last_col = m.slice_cols(3, 4)
    assert_equal(last_col[0, 0], 100.0)
    assert_equal(last_col[1, 0], 200.0)
    assert_equal(last_col[2, 0], 300.0)


def test_view_non_square_to_matrix() raises:
    var m = Matrix[DType.float64](2, 5, 0)
    for c in range(5):
        m[0, c] = Float64(c)
        m[1, c] = Float64(c + 10)

    var v = m.slice_2d(0, 2, 1, 4)  # 2x3
    var owned = v.to_matrix()
    assert_equal(owned.rows, 2)
    assert_equal(owned.cols, 3)
    assert_equal(owned[0, 0], 1.0)
    assert_equal(owned[1, 2], 13.0)


def test_view_identity_matrix_view() raises:
    var Eye = Matrix[DType.float64].eye(4)
    var v = Eye.view()
    for r in range(4):
        for c in range(4):
            assert_equal(v[r, c], 1.0 if r == c else 0.0)


def test_view_copy_constructible() raises:
    var m = Matrix[DType.float64](3, 3, 5.0)
    var v1 = m.view()
    var v2 = v1.copy()
    assert_equal(v2.rows, 3)
    assert_equal(v2.cols, 3)
    assert_equal(v2[1, 1], 5.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
