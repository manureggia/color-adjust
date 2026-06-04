import numpy as np
from PIL import Image

from color_adjust.image import save_comparison


def test_save_comparison_includes_title_and_footer_space(tmp_path):
    image = np.full((10, 10, 3), 0.5, dtype=np.float32)
    output = tmp_path / "comparison.png"

    save_comparison(
        output,
        [("original", image), ("target", image), ("lut result", image)],
        panel_width=20,
        title="Prompt: Make the whole image warmer while preserving geometry.",
        footer="PSNR: 31.500 dB | SSIM: 0.9876",
    )

    comparison = Image.open(output)
    assert comparison.width == 3 * 20 + 4 * 14
    assert comparison.height > 20 + 28 + 2 * 14
