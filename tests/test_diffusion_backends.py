from color_adjust.cli import _pipeline_output_paths, build_parser
from color_adjust.diffusion import resolve_backend_config


def test_default_diffusion_backend_is_instruct_pix2pix():
    config = resolve_backend_config()
    assert config.backend == "instruct-pix2pix"
    assert config.model_id == "timbrooks/instruct-pix2pix"
    assert config.pipeline == "instruct-pix2pix"
    assert config.strength is None
    assert config.image_guidance_scale == 1.2


def test_model_id_can_still_infer_img2img_backend():
    config = resolve_backend_config(model_id="stabilityai/stable-diffusion-xl-base-1.0")
    assert config.backend == "sdxl"
    assert config.pipeline == "img2img"
    assert config.strength == 0.55


def test_sdxl_turbo_backend_has_turbo_defaults():
    config = resolve_backend_config(backend="sdxl-turbo")
    assert config.model_id == "stabilityai/sdxl-turbo"
    assert config.steps == 4
    assert config.guidance_scale == 0.0
    assert config.strength == 0.75


def test_pipeline_default_names_include_backend(tmp_path, monkeypatch):
    parser = build_parser()
    args = parser.parse_args(
        [
            "pipeline",
            "--input",
            str(tmp_path / "inputs" / "street-1.jpg"),
            "--prompt",
            "make it warm",
            "--backend",
            "instruct-pix2pix",
        ]
    )
    monkeypatch.chdir(tmp_path)
    paths = _pipeline_output_paths(args)

    assert paths["generated_target"] == (
        tmp_path
        / "street-1"
        / "street-1_diffusion_instruct-pix2pix_igs120_seed0_target.png"
    )
    assert paths["output"] == (
        tmp_path
        / "street-1"
        / "street-1_diffusion_instruct-pix2pix_igs120_seed0_lut.png"
    )

