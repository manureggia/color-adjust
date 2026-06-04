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


def test_explicit_img2img_backend_uses_low_strength_sd15_defaults():
    config = resolve_backend_config(backend="img2img")
    assert config.backend == "img2img"
    assert config.model_id == "runwayml/stable-diffusion-v1-5"
    assert config.pipeline == "img2img"
    assert config.strength == 0.25
    assert config.guidance_scale == 7.5
    assert config.steps == 30


def test_instruct_pix2pix_underscore_alias_is_supported():
    config = resolve_backend_config(backend="instruct_pix2pix")
    assert config.backend == "instruct-pix2pix"
    assert config.pipeline == "instruct-pix2pix"


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


def test_pipeline_accepts_requested_diffusion_backend_and_steps_aliases(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "pipeline",
            "--input",
            str(tmp_path / "inputs" / "street-1.jpg"),
            "--prompt",
            "cinematic warm sunset color grading, same scene, same objects",
            "--diffusion_backend",
            "img2img",
            "--strength",
            "0.25",
            "--guidance_scale",
            "7.5",
            "--num_inference_steps",
            "30",
            "--seed",
            "42",
        ]
    )

    config = resolve_backend_config(
        backend=args.backend,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        steps=args.steps,
    )
    assert config.backend == "img2img"
    assert config.strength == 0.25
    assert config.guidance_scale == 7.5
    assert config.steps == 30
    assert args.seed == 42
