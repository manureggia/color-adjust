import numpy as np

from color_adjust.grading import apply_color_grade
from color_adjust.lut import LUT3D, fit_lut
from color_adjust.metrics import psnr, ssim


def test_identity_lut_leaves_image_unchanged():
    rng = np.random.default_rng(1)
    image = rng.random((12, 10, 3), dtype=np.float32)
    output = LUT3D.identity(17).apply(image)
    np.testing.assert_allclose(output, image, atol=1e-6)


def test_fit_lut_approximates_smooth_color_transform():
    rng = np.random.default_rng(2)
    source = rng.random((80, 80, 3), dtype=np.float32)
    target = np.clip(source * np.array([0.75, 0.9, 1.1], dtype=np.float32) + 0.06, 0, 1)

    lut = fit_lut(
        source,
        target,
        size=17,
        max_samples=None,
        seed=0,
        smooth_iterations=8,
        identity_weight=0.001,
    )
    output = lut.apply(source)
    assert psnr(target, output) > 31.0
    assert ssim(target, output) > 0.98


def test_cube_roundtrip(tmp_path):
    lut = LUT3D.identity(5)
    path = tmp_path / "identity.cube"
    lut.save_cube(path)
    loaded = LUT3D.load_cube(path)
    np.testing.assert_allclose(loaded.table, lut.table, atol=1e-7)


def test_grading_creates_valid_recoloring():
    image = np.full((8, 8, 3), [0.45, 0.50, 0.55], dtype=np.float32)
    output = apply_color_grade(image, "warm-sunset", strength=1.0)
    assert output.shape == image.shape
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0
    assert output[..., 0].mean() > image[..., 0].mean()


def test_luma_tolerance_filters_out_incoherent_pairs():
    source = np.zeros((10, 10, 3), dtype=np.float32)
    source[:5] = 0.2
    source[5:] = 0.8
    target = source.copy()
    target[:5] = 1.0
    target[5:] = 0.7

    lut = fit_lut(
        source,
        target,
        size=5,
        max_samples=None,
        luma_tolerance=0.2,
        smooth_iterations=0,
    )
    mapped = lut.apply(np.full((1, 1, 3), 0.8, dtype=np.float32))
    np.testing.assert_allclose(mapped, 0.7, atol=0.08)
