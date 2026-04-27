import json

import numpy as np

from color_adjust.cli import main
from color_adjust.image import save_rgb


def test_grade_uses_default_output_folder(tmp_path, monkeypatch):
    input_path = tmp_path / "inputs" / "Street One.jpg"
    image = np.full((12, 10, 3), [0.45, 0.50, 0.55], dtype=np.float32)
    save_rgb(input_path, image)

    monkeypatch.chdir(tmp_path)
    assert main(["grade", "--input", str(input_path), "--style", "warm-sunset"]) == 0

    expected = tmp_path / "Street-One" / "Street-One_grade_warm-sunset_target.png"
    assert expected.exists()


def test_fit_uses_default_output_bundle(tmp_path, monkeypatch):
    input_path = tmp_path / "inputs" / "sample.jpg"
    target_path = tmp_path / "targets" / "target warm.png"
    source = np.full((12, 10, 3), [0.45, 0.50, 0.55], dtype=np.float32)
    target = np.clip(source * np.array([1.1, 1.0, 0.9], dtype=np.float32), 0, 1)
    save_rgb(input_path, source)
    save_rgb(target_path, target)

    monkeypatch.chdir(tmp_path)
    assert (
        main(
            [
                "fit",
                "--input",
                str(input_path),
                "--target",
                str(target_path),
                "--samples",
                "0",
                "--smooth-iterations",
                "0",
            ]
        )
        == 0
    )

    output = tmp_path / "sample" / "sample_from_target-warm_lut.png"
    lut = tmp_path / "sample" / "sample_from_target-warm_lut.cube"
    report = tmp_path / "sample" / "sample_from_target-warm_lut_metrics.json"
    comparison = tmp_path / "sample" / "sample_from_target-warm_lut_comparison.png"
    assert output.exists()
    assert lut.exists()
    assert report.exists()
    assert comparison.exists()
    assert json.loads(report.read_text(encoding="utf-8"))["metrics"]["ssim"] > 0.9
