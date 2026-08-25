from std.testing import TestSuite, assert_equal, assert_true
from strata import Matrix, MatrixView

def test_matrix_view_basic() raises:
    var m = Matrix[DType.float64](4, 4, 0)
    for r in range(4):
        for c in range(4):
            m[r, c] = Float64(r * 10 + c)

    var v = m.view()
    assert_equal(v.rows, 4)
    assert_equal(v.cols, 4)
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

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
