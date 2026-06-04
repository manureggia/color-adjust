import json

import numpy as np

from color_adjust.cli import main
from color_adjust.image import save_rgb
from color_adjust.metrics import psnr
from color_adjust.residual import (
    compute_residual_mask,
    mask_to_rgb,
    refine_with_masked_residual,
)


def test_masked_residual_improves_local_unexplained_change():
    baseline = np.full((20, 20, 3), 0.35, dtype=np.float32)
    target = baseline.copy()
    target[6:14, 6:14] = [0.9, 0.2, 0.2]

    refinement = refine_with_masked_residual(
        target,
        baseline,
        alpha=1.0,
        threshold=0.05,
        softness=0.02,
        blur_radius=0,
        max_residual=None,
    )

    assert refinement.mask[8, 8] > 0.95
    assert refinement.mask[0, 0] == 0.0
    assert psnr(target, refinement.output) > psnr(target, baseline)


def test_positive_luma_only_suppresses_dark_residuals():
    baseline = np.full((12, 12, 3), 0.6, dtype=np.float32)
    target = baseline.copy()
    target[3:9, 3:9] = 0.2

    mask = compute_residual_mask(
        target,
        baseline,
        threshold=0.05,
        softness=0.02,
        blur_radius=0,
        positive_luma_only=True,
    )

    assert float(mask.max()) == 0.0


def test_mask_to_rgb_creates_savable_rgb_image():
    mask = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
    rgb = mask_to_rgb(mask)

    assert rgb.shape == (2, 2, 3)
    np.testing.assert_allclose(rgb[..., 0], mask)
    np.testing.assert_allclose(rgb[..., 1], mask)
    np.testing.assert_allclose(rgb[..., 2], mask)


def test_refine_command_saves_default_output_bundle(tmp_path, monkeypatch):
    input_path = tmp_path / "inputs" / "sample.jpg"
    target_path = tmp_path / "targets" / "target local.png"
    source = np.full((18, 18, 3), 0.4, dtype=np.float32)
    target = source.copy()
    target[5:13, 5:13] = [0.85, 0.2, 0.2]
    save_rgb(input_path, source)
    save_rgb(target_path, target)

    monkeypatch.chdir(tmp_path)
    assert (
        main(
            [
                "refine",
                "--input",
                str(input_path),
                "--target",
                str(target_path),
                "--size",
                "5",
                "--samples",
                "0",
                "--smooth-iterations",
                "0",
                "--mask-blur-radius",
                "0",
            ]
        )
        == 0
    )

    output = tmp_path / "sample" / "sample_from_target-local_refined.png"
    lut_output = tmp_path / "sample" / "sample_from_target-local_refined_lut.png"
    lut = tmp_path / "sample" / "sample_from_target-local_refined_lut.cube"
    mask = tmp_path / "sample" / "sample_from_target-local_refined_mask.png"
    report = tmp_path / "sample" / "sample_from_target-local_refined_metrics.json"
    comparison = tmp_path / "sample" / "sample_from_target-local_refined_comparison.png"
    assert output.exists()
    assert lut_output.exists()
    assert lut.exists()
    assert mask.exists()
    assert report.exists()
    assert comparison.exists()

    metrics = json.loads(report.read_text(encoding="utf-8"))["metrics"]
    assert set(metrics) == {"lut", "refined"}
    assert metrics["refined"]["psnr"] >= metrics["lut"]["psnr"]


def test_pipeline_refine_saves_additional_outputs(tmp_path, monkeypatch):
    input_path = tmp_path / "inputs" / "sample.jpg"
    target_path = tmp_path / "targets" / "target local.png"
    source = np.full((18, 18, 3), 0.4, dtype=np.float32)
    target = source.copy()
    target[5:13, 5:13] = [0.85, 0.2, 0.2]
    save_rgb(input_path, source)
    save_rgb(target_path, target)

    monkeypatch.chdir(tmp_path)
    assert (
        main(
            [
                "pipeline",
                "--input",
                str(input_path),
                "--target",
                str(target_path),
                "--refine",
                "--size",
                "5",
                "--samples",
                "0",
                "--smooth-iterations",
                "0",
                "--mask-blur-radius",
                "0",
            ]
        )
        == 0
    )

    base = tmp_path / "sample" / "sample_from_target-local_lut"
    assert base.with_suffix(".png").exists()
    assert base.with_suffix(".cube").exists()
    assert (tmp_path / "sample" / "sample_from_target-local_lut_refined.png").exists()
    assert (tmp_path / "sample" / "sample_from_target-local_lut_refined_mask.png").exists()
    assert (tmp_path / "sample" / "sample_from_target-local_lut_refined_metrics.json").exists()
    assert (tmp_path / "sample" / "sample_from_target-local_lut_refined_comparison.png").exists()
