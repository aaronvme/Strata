from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_almost_equal,
    assert_raises,
)
from std.math import sqrt
from strata import (
    Matrix,
    CSRMatrix,
    Dataset,
    PCA,
    TruncatedSVD,
    StandardScaler,
    LinearRegression,
    Ridge,
    LogisticRegression,
    PipelineRegressor,
    PipelineClassifier,
    NotFittedError,
    DimensionMismatchError,
    InvalidParameterError,
    gemm,
)


def test_pca_2d_projection_and_orthogonality() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = -2.0
    X[0, 1] = -4.0
    X[1, 0] = -1.0
    X[1, 1] = -2.0
    X[2, 0] = 1.0
    X[2, 1] = 2.0
    X[3, 0] = 2.0
    X[3, 1] = 4.0

    var pca = PCA(n_components=2)
    pca.fit(X)

    assert_true(pca.is_fitted)
    assert_equal(pca.n_components_, 2)
    assert_equal(pca.n_features_in_, 2)

    assert_true(pca.explained_variance_ratio_[0] > 0.999)
    assert_true(pca.explained_variance_ratio_[1] < 0.001)

    var comp = pca.components_.copy()
    var comp_t = comp.transpose()
    var gram = gemm(comp, comp_t)

    assert_almost_equal(gram[0, 0], 1.0, atol=1e-5)
    assert_almost_equal(gram[1, 1], 1.0, atol=1e-5)
    assert_almost_equal(gram[0, 1], 0.0, atol=1e-5)


def test_pca_explained_variance_and_ratio_sum() raises:
    var X = Matrix[DType.float64](6, 3, 0)
    for r in range(6):
        var fr = Float64(r)
        X[r, 0] = fr
        X[r, 1] = fr * 2.0 + 1.0
        X[r, 2] = fr * fr

    var pca = PCA(n_components=3)
    pca.fit(X)

    var ratio_sum: Float64 = 0
    for i in range(3):
        ratio_sum += pca.explained_variance_ratio_[i]

    assert_almost_equal(ratio_sum, 1.0, atol=1e-5)


def test_pca_reconstruction_perfect_full_rank() raises:
    var X = Matrix[DType.float64](5, 3, 0)
    for r in range(5):
        for c in range(3):
            X[r, c] = Float64((r + 1) * (c + 2))

    var pca = PCA(n_components=3)
    var X_proj = pca.fit_transform(X)
    var X_rec = pca.inverse_transform(X_proj)

    for r in range(5):
        for c in range(3):
            assert_almost_equal(X_rec[r, c], X[r, c], atol=1e-4)


def test_pca_whitening_unit_variance() raises:
    var X = Matrix[DType.float64](10, 2, 0)
    for r in range(10):
        var fr = Float64(r)
        X[r, 0] = fr * 10.0
        X[r, 1] = fr * 2.0

    var pca = PCA(n_components=2, whiten=True)
    var X_white = pca.fit_transform(X)

    var var0: Float64 = 0
    var var1: Float64 = 0
    for r in range(10):
        var0 += X_white[r, 0] * X_white[r, 0]
        var1 += X_white[r, 1] * X_white[r, 1]
    var0 /= 9.0

    assert_almost_equal(var0, 1.0, atol=1e-4)


def test_pca_deterministic_sign_flip() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 2.0
    X[1, 0] = 2.0
    X[1, 1] = 4.0
    X[2, 0] = 3.0
    X[2, 1] = 6.0
    X[3, 0] = 4.0
    X[3, 1] = 8.0

    var pca1 = PCA(n_components=1)
    pca1.fit(X)

    var pca2 = PCA(n_components=1)
    pca2.fit(X)

    assert_equal(pca1.components_[0, 0], pca2.components_[0, 0])
    assert_equal(pca1.components_[0, 1], pca2.components_[0, 1])
    assert_true(pca1.components_[0, 1] > 0)


def test_pca_single_component_reduction() raises:
    var X = Matrix[DType.float64](6, 3, 1.0)
    for r in range(6):
        X[r, 0] = Float64(r * 5)
        X[r, 1] = Float64(r * 10)
        X[r, 2] = Float64(r * 15)

    var pca = PCA(n_components=1)
    var X_1d = pca.fit_transform(X)

    assert_equal(X_1d.rows, 6)
    assert_equal(X_1d.cols, 1)


def test_pca_dataset_container_pipeline() raises:
    var X = Matrix[DType.float64](4, 3, 0)
    for r in range(4):
        X[r, 0] = Float64(r)
        X[r, 1] = Float64(r * 2)
        X[r, 2] = Float64(r * 3)

    var y: List[Scalar[DType.float64]] = [0.0, 1.0, 0.0, 1.0]
    var ds = Dataset(X^, y^)

    var pca = PCA(n_components=2)
    var ds_pca = pca.fit_transform(ds)

    assert_equal(ds_pca.n_samples(), 4)
    assert_equal(ds_pca.n_features(), 2)
    assert_equal(ds_pca.targets[1], 1.0)


def test_pca_copy_constructor_isolation() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    for r in range(4):
        X[r, 0] = Float64(r)
        X[r, 1] = Float64(r * 3)

    var pca1 = PCA(n_components=1)
    pca1.fit(X)

    var pca2 = pca1.copy()
    assert_true(pca2.is_fitted)
    assert_equal(pca2.n_components_, 1)

    var X_new = Matrix[DType.float64](4, 2, 0)
    for r in range(4):
        X_new[r, 0] = Float64(r * 100)
        X_new[r, 1] = Float64(r * 500)
    pca2.fit(X_new)

    assert_true(pca1.mean_[0] != pca2.mean_[0])


def test_pca_not_fitted_error() raises:
    var pca = PCA(n_components=2)
    var X = Matrix[DType.float64](4, 2, 1.0)

    with assert_raises():
        _ = pca.transform(X)

    with assert_raises():
        _ = pca.inverse_transform(X)


def test_pca_dimension_mismatch_transform() raises:
    var X_train = Matrix[DType.float64](4, 3, 1.0)
    var pca = PCA(n_components=2)
    pca.fit(X_train)

    var X_bad = Matrix[DType.float64](4, 5, 1.0)
    with assert_raises():
        _ = pca.transform(X_bad)


def test_pca_dimension_mismatch_inverse() raises:
    var X_train = Matrix[DType.float64](4, 3, 1.0)
    var pca = PCA(n_components=2)
    pca.fit(X_train)

    var X_bad_proj = Matrix[DType.float64](4, 4, 1.0)
    with assert_raises():
        _ = pca.inverse_transform(X_bad_proj)


def test_pca_invalid_n_components() raises:
    var X = Matrix[DType.float64](4, 2, 1.0)

    var pca_neg = PCA(n_components=-1)
    with assert_raises():
        pca_neg.fit(X)

    var pca_too_large = PCA(n_components=5)
    with assert_raises():
        pca_too_large.fit(X)


def test_pca_float32_casting() raises:
    var X = Matrix[DType.float32](5, 3, 0)
    for r in range(5):
        for c in range(3):
            X[r, c] = Float32((r + 1) * (c + 1))

    var pca = PCA(n_components=2)
    var X_proj = pca.fit_transform(X)

    assert_equal(X_proj.rows, 5)
    assert_equal(X_proj.cols, 2)


def test_pca_bfloat16_casting() raises:
    var X = Matrix[DType.bfloat16](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 2.0
    X[1, 0] = 2.0
    X[1, 1] = 4.0
    X[2, 0] = 3.0
    X[2, 1] = 6.0
    X[3, 0] = 4.0
    X[3, 1] = 8.0

    var pca = PCA(n_components=1)
    var X_proj = pca.fit_transform(X)
    assert_equal(X_proj.rows, 4)
    assert_equal(X_proj.cols, 1)


def test_pca_all_components_default() raises:
    var X = Matrix[DType.float64](6, 4, 1.0)
    for r in range(6):
        for c in range(4):
            X[r, c] = Float64(r * c + r)

    var pca = PCA()
    pca.fit(X)
    assert_equal(pca.n_components_, 4)


def test_pca_centered_data_mean_zero() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = -2.0
    X[0, 1] = -4.0
    X[1, 0] = -1.0
    X[1, 1] = -2.0
    X[2, 0] = 1.0
    X[2, 1] = 2.0
    X[3, 0] = 2.0
    X[3, 1] = 4.0

    var pca = PCA(n_components=2)
    pca.fit(X)

    assert_almost_equal(pca.mean_[0], 0.0, atol=1e-5)
    assert_almost_equal(pca.mean_[1], 0.0, atol=1e-5)


def test_pca_pipeline_with_linear_regression() raises:
    var X = Matrix[DType.float64](6, 3, 0)
    var y: List[Scalar[DType.float64]] = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    for r in range(6):
        X[r, 0] = Float64(r + 1)
        X[r, 1] = Float64((r + 1) * 2)
        X[r, 2] = Float64((r + 1) * 3)

    var pca = PCA(n_components=1)
    var reg = LinearRegression()
    var pipe = PipelineRegressor(pca^, reg^)

    pipe.fit(X, y)
    var preds = pipe.predict(X)

    assert_equal(len(preds), 6)
    assert_almost_equal(preds[0], 2.0, atol=1e-3)
    assert_almost_equal(preds[5], 12.0, atol=1e-3)


def test_pca_pipeline_with_logistic_regression() raises:
    var X = Matrix[DType.float64](6, 4, 0)
    var y: List[Scalar[DType.int32]] = [0, 0, 0, 1, 1, 1]
    for r in range(3):
        X[r, 0] = -10.0 + Float64(r)
        X[r, 1] = -5.0
    for r in range(3, 6):
        X[r, 0] = 10.0 + Float64(r)
        X[r, 1] = 5.0

    var pca = PCA(n_components=2)
    var clf = LogisticRegression()
    var pipe = PipelineClassifier(pca^, clf^)

    pipe.fit(X, y)
    var preds = pipe.predict(X)

    assert_equal(preds[0], 0)
    assert_equal(preds[5], 1)


def test_pca_constant_feature_handling() raises:
    var X = Matrix[DType.float64](5, 3, 0)
    for r in range(5):
        X[r, 0] = Float64(r)
        X[r, 1] = 100.0
        X[r, 2] = Float64(r * 2)

    var pca = PCA(n_components=2)
    pca.fit(X)

    assert_equal(pca.mean_[1], 100.0)
    assert_equal(pca.n_components_, 2)


def test_pca_high_dimensional_fat_matrix() raises:
    var N = 8
    var D = 20
    var X = Matrix[DType.float64](N, D, 0)
    for r in range(N):
        for c in range(D):
            X[r, c] = Float64((r + 1) * (c + 1))

    var pca = PCA(n_components=4)
    var X_proj = pca.fit_transform(X)

    assert_equal(X_proj.rows, N)
    assert_equal(X_proj.cols, 4)


def test_pca_large_scale_sample() raises:
    var N = 100
    var D = 10
    var X = Matrix[DType.float64](N, D, 0)
    for r in range(N):
        for c in range(D):
            X[r, c] = Float64(r + c)

    var pca = PCA(n_components=3)
    var X_proj = pca.fit_transform(X)

    assert_equal(X_proj.rows, N)
    assert_equal(X_proj.cols, 3)


def test_pca_whiten_inverse_reconstruction() raises:
    var X = Matrix[DType.float64](6, 3, 0)
    for r in range(6):
        X[r, 0] = Float64(r + 1)
        X[r, 1] = Float64((r + 1) * 3)
        X[r, 2] = Float64((r + 1) * 5)

    var pca = PCA(n_components=3, whiten=True)
    var X_proj = pca.fit_transform(X)
    var X_rec = pca.inverse_transform(X_proj)

    for r in range(6):
        for c in range(3):
            assert_almost_equal(X_rec[r, c], X[r, c], atol=1e-4)


def test_pca_whiten_zero_variance_component_handling() raises:
    var X = Matrix[DType.float64](5, 3, 0)
    for r in range(5):
        X[r, 0] = Float64(r + 1)
        X[r, 1] = 42.0
        X[r, 2] = Float64((r + 1) * 2)

    var pca = PCA(n_components=3, whiten=True)
    var X_proj = pca.fit_transform(X)
    var X_rec = pca.inverse_transform(X_proj)

    assert_equal(X_rec.rows, 5)
    assert_equal(X_rec.cols, 3)
    for r in range(5):
        assert_almost_equal(X_rec[r, 1], 42.0, atol=1e-4)


def test_truncated_svd_dense_projection() raises:
    var X = Matrix[DType.float64](4, 3, 0)
    for r in range(4):
        for c in range(3):
            X[r, c] = Float64((r + 1) * (c + 1))

    var svd_model = TruncatedSVD(n_components=2)
    var X_proj = svd_model.fit_transform(X)

    assert_true(svd_model.is_fitted)
    assert_equal(X_proj.rows, 4)
    assert_equal(X_proj.cols, 2)
    assert_equal(svd_model.n_components_, 2)
    assert_equal(svd_model.n_features_in_, 3)


def test_truncated_svd_sparse_csr_projection() raises:
    var dense = Matrix[DType.float64](4, 4, 0)
    dense[0, 0] = 5.0
    dense[1, 1] = 10.0
    dense[2, 2] = 15.0
    dense[3, 3] = 20.0

    var csr = CSRMatrix[DType.float64].from_dense(dense)
    var svd_model = TruncatedSVD(n_components=2)
    var X_proj = svd_model.fit_transform(csr)

    assert_equal(X_proj.rows, 4)
    assert_equal(X_proj.cols, 2)


def test_truncated_svd_dense_sparse_parity() raises:
    var dense = Matrix[DType.float64](5, 3, 0)
    dense[0, 0] = 1.0
    dense[1, 2] = 2.0
    dense[2, 1] = 3.0
    dense[3, 0] = 4.0
    dense[4, 2] = 5.0

    var csr = CSRMatrix[DType.float64].from_dense(dense)

    var svd_dense = TruncatedSVD(n_components=2)
    var proj_dense = svd_dense.fit_transform(dense)

    var svd_sparse = TruncatedSVD(n_components=2)
    var proj_sparse = svd_sparse.fit_transform(csr)

    for r in range(5):
        for c in range(2):
            assert_almost_equal(proj_dense[r, c], proj_sparse[r, c], atol=1e-5)


def test_truncated_svd_components_orthogonality() raises:
    var X = Matrix[DType.float64](5, 4, 0)
    for r in range(5):
        for c in range(4):
            X[r, c] = Float64((r + 2) * (c + 1))

    var svd_model = TruncatedSVD(n_components=3)
    svd_model.fit(X)

    var comp = svd_model.components_.copy()
    var comp_t = comp.transpose()
    var gram = gemm(comp, comp_t)

    assert_almost_equal(gram[0, 0], 1.0, atol=1e-5)
    assert_almost_equal(gram[1, 1], 1.0, atol=1e-5)
    assert_almost_equal(gram[2, 2], 1.0, atol=1e-5)
    assert_almost_equal(gram[0, 1], 0.0, atol=1e-5)


def test_truncated_svd_reconstruction_full_rank() raises:
    var X = Matrix[DType.float64](4, 3, 0)
    for r in range(4):
        for c in range(3):
            X[r, c] = Float64((r + 1) * (c + 2))

    var svd_model = TruncatedSVD(n_components=3)
    var X_proj = svd_model.fit_transform(X)
    var X_rec = svd_model.inverse_transform(X_proj)

    for r in range(4):
        for c in range(3):
            assert_almost_equal(X_rec[r, c], X[r, c], atol=1e-4)


def test_truncated_svd_copy_constructor_isolation() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    for r in range(4):
        X[r, 0] = Float64(r + 1)
        X[r, 1] = Float64((r + 1) * 3)

    var s1 = TruncatedSVD(n_components=1)
    s1.fit(X)

    var s2 = s1.copy()
    assert_true(s2.is_fitted)
    assert_equal(s2.n_components_, 1)

    var X_new = Matrix[DType.float64](4, 2, 10.0)
    s2.fit(X_new)

    assert_true(s1.singular_values_[0] != s2.singular_values_[0])


def test_truncated_svd_not_fitted_error() raises:
    var svd_model = TruncatedSVD(n_components=2)
    var X = Matrix[DType.float64](4, 2, 1.0)

    with assert_raises():
        _ = svd_model.transform(X)

    with assert_raises():
        _ = svd_model.inverse_transform(X)


def test_truncated_svd_dimension_mismatch_transform() raises:
    var X_train = Matrix[DType.float64](4, 3, 1.0)
    var svd_model = TruncatedSVD(n_components=2)
    svd_model.fit(X_train)

    var X_bad = Matrix[DType.float64](4, 5, 1.0)
    with assert_raises():
        _ = svd_model.transform(X_bad)


def test_truncated_svd_dimension_mismatch_inverse() raises:
    var X_train = Matrix[DType.float64](4, 3, 1.0)
    var svd_model = TruncatedSVD(n_components=2)
    svd_model.fit(X_train)

    var X_bad_proj = Matrix[DType.float64](4, 5, 1.0)
    with assert_raises():
        _ = svd_model.inverse_transform(X_bad_proj)


def test_truncated_svd_invalid_n_components() raises:
    var X = Matrix[DType.float64](4, 2, 1.0)

    var svd_zero = TruncatedSVD(n_components=0)
    with assert_raises():
        svd_zero.fit(X)

    var svd_too_large = TruncatedSVD(n_components=5)
    with assert_raises():
        svd_too_large.fit(X)


def test_truncated_svd_dataset_container() raises:
    var X = Matrix[DType.float64](4, 3, 0)
    for r in range(4):
        X[r, 0] = Float64(r)
        X[r, 1] = Float64(r * 2)
        X[r, 2] = Float64(r * 3)

    var y: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0, 4.0]
    var ds = Dataset(X^, y^)

    var svd_model = TruncatedSVD(n_components=2)
    var ds_proj = svd_model.fit_transform(ds)

    assert_equal(ds_proj.n_samples(), 4)
    assert_equal(ds_proj.n_features(), 2)
    assert_equal(ds_proj.targets[0], 1.0)


def test_truncated_svd_pipeline_with_ridge() raises:
    var X = Matrix[DType.float64](6, 3, 0)
    var y: List[Scalar[DType.float64]] = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    for r in range(6):
        X[r, 0] = Float64(r + 1)
        X[r, 1] = Float64((r + 1) * 2)
        X[r, 2] = Float64((r + 1) * 3)

    var svd_model = TruncatedSVD(n_components=1)
    var reg = Ridge(alpha=0.1)
    var pipe = PipelineRegressor(svd_model^, reg^)

    pipe.fit(X, y)
    var preds = pipe.predict(X)

    assert_equal(len(preds), 6)
    assert_almost_equal(preds[0], 2.0, atol=0.2)


def test_truncated_svd_float32_native() raises:
    var X = Matrix[DType.float32](4, 3, 1.0)
    var svd_model = TruncatedSVD(n_components=2)
    var X_proj = svd_model.fit_transform(X)

    assert_equal(X_proj.rows, 4)
    assert_equal(X_proj.cols, 2)


def test_truncated_svd_bfloat16_native() raises:
    var X = Matrix[DType.bfloat16](4, 3, 2.0)
    var svd_model = TruncatedSVD(n_components=2)
    var X_proj = svd_model.fit_transform(X)

    assert_equal(X_proj.rows, 4)
    assert_equal(X_proj.cols, 2)


def test_truncated_svd_large_sparse_matrix() raises:
    var dim = 100
    var dense = Matrix[DType.float64].eye(dim)
    var csr = CSRMatrix[DType.float64].from_dense(dense)

    var svd_model = TruncatedSVD(n_components=5)
    var X_proj = svd_model.fit_transform(csr)

    assert_equal(X_proj.rows, dim)
    assert_equal(X_proj.cols, 5)


def test_truncated_svd_high_dimensional_fat_matrix() raises:
    var N = 8
    var D = 25
    var X = Matrix[DType.float64](N, D, 0)
    for r in range(N):
        for c in range(D):
            X[r, c] = Float64((r + 1) * (c + 1))

    var svd_model = TruncatedSVD(n_components=4)
    var X_proj = svd_model.fit_transform(X)

    assert_equal(X_proj.rows, N)
    assert_equal(X_proj.cols, 4)


def test_truncated_svd_single_component_reduction() raises:
    var X = Matrix[DType.float64](6, 3, 0)
    for r in range(6):
        X[r, 0] = Float64(r * 5)
        X[r, 1] = Float64(r * 10)
        X[r, 2] = Float64(r * 15)

    var svd_model = TruncatedSVD(n_components=1)
    var X_1d = svd_model.fit_transform(X)

    assert_equal(X_1d.rows, 6)
    assert_equal(X_1d.cols, 1)


def test_truncated_svd_uncentered_variance_ratio_bounds() raises:
    # Uncentered data with non-zero column means
    var X = Matrix[DType.float64](6, 3, 0)
    for r in range(6):
        var fr = Float64(r)
        X[r, 0] = fr + 100.0
        X[r, 1] = fr * 2.0 + 50.0
        X[r, 2] = fr * 3.0 + 25.0

    var svd_model = TruncatedSVD(n_components=2)
    svd_model.fit(X)

    # Explained variance ratios must be non-negative and sum to <= 1.0
    var sum_ratios: Float64 = 0
    for j in range(2):
        assert_true(svd_model.explained_variance_ratio_[j] >= 0.0)
        assert_true(svd_model.explained_variance_ratio_[j] <= 1.0)
        sum_ratios += svd_model.explained_variance_ratio_[j]
    assert_true(sum_ratios <= 1.0001)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
