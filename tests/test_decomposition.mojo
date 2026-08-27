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
    Dataset,
    PCA,
    StandardScaler,
    LinearRegression,
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
