from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image


DiffusionBackend = Literal[
    "instruct-pix2pix",
    "instruct_pix2pix",
    "img2img",
    "sd15",
    "sdxl",
    "sdxl-turbo",
]
CanonicalDiffusionBackend = Literal[
    "instruct-pix2pix",
    "img2img",
    "sd15",
    "sdxl",
    "sdxl-turbo",
]


@dataclass(frozen=True)
class BackendPreset:
    model_id: str
    pipeline: Literal["img2img", "instruct-pix2pix"]
    strength: float | None
    guidance_scale: float
    image_guidance_scale: float | None
    steps: int
    max_side: int


@dataclass(frozen=True)
class BackendConfig:
    backend: CanonicalDiffusionBackend
    model_id: str
    pipeline: Literal["img2img", "instruct-pix2pix"]
    strength: float | None
    guidance_scale: float
    image_guidance_scale: float | None
    steps: int
    max_side: int


BACKEND_PRESETS: dict[CanonicalDiffusionBackend, BackendPreset] = {
    "instruct-pix2pix": BackendPreset(
        model_id="timbrooks/instruct-pix2pix",
        pipeline="instruct-pix2pix",
        strength=None,
        guidance_scale=7.5,
        image_guidance_scale=1.2,
        steps=30,
        max_side=768,
    ),
    "img2img": BackendPreset(
        model_id="runwayml/stable-diffusion-v1-5",
        pipeline="img2img",
        strength=0.25,
        guidance_scale=7.5,
        image_guidance_scale=None,
        steps=30,
        max_side=768,
    ),
    "sd15": BackendPreset(
        model_id="runwayml/stable-diffusion-v1-5",
        pipeline="img2img",
        strength=0.55,
        guidance_scale=8.0,
        image_guidance_scale=None,
        steps=50,
        max_side=768,
    ),
    "sdxl": BackendPreset(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        pipeline="img2img",
        strength=0.55,
        guidance_scale=9.0,
        image_guidance_scale=None,
        steps=50,
        max_side=768,
    ),
    "sdxl-turbo": BackendPreset(
        model_id="stabilityai/sdxl-turbo",
        pipeline="img2img",
        strength=0.75,
        guidance_scale=0.0,
        image_guidance_scale=None,
        steps=4,
        max_side=768,
    ),
}
BACKEND_ALIASES: dict[str, CanonicalDiffusionBackend] = {
    "instruct_pix2pix": "instruct-pix2pix",
    "instruct-pix2pix": "instruct-pix2pix",
    "img2img": "img2img",
    "sd15": "sd15",
    "sdxl": "sdxl",
    "sdxl-turbo": "sdxl-turbo",
}
BACKEND_CHOICES = tuple(BACKEND_ALIASES)
DEFAULT_BACKEND: DiffusionBackend = "instruct-pix2pix"


def generate_recoloring(
    input_path: str | Path,
    output_path: str | Path,
    prompt: str,
    negative_prompt: str | None = None,
    backend: DiffusionBackend | None = None,
    model_id: str | None = None,
    strength: float | None = None,
    guidance_scale: float | None = None,
    image_guidance_scale: float | None = None,
    steps: int | None = None,
    seed: int | None = 0,
    device: str | None = None,
    max_side: int | None = None,
) -> Path:
    """Generate a recolored target image.

    Dependencies are optional. Install with: pip install -e ".[diffusion]".
    """
    config = resolve_backend_config(
        backend=backend,
        model_id=model_id,
        strength=strength,
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale,
        steps=steps,
        max_side=max_side,
    )

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Diffusers support is not installed. Run: pip install -e \".[diffusion]\""
        ) from exc

    if config.strength is not None and not (0.0 <= config.strength <= 1.0):
        raise ValueError("strength must be in [0, 1].")
    if config.steps < 1:
        raise ValueError("steps must be >= 1.")
    if config.max_side < 8:
        raise ValueError("max_side must be >= 8.")
    if config.image_guidance_scale is not None and config.image_guidance_scale < 1.0:
        raise ValueError("image_guidance_scale must be >= 1 for InstructPix2Pix.")

    input_path = Path(input_path)
    output_path = Path(output_path)
    source = Image.open(input_path).convert("RGB")
    original_size = source.size
    work_image = _resize_for_diffusion(source, max_side=config.max_side)

    device = _resolve_device(torch, device)
    dtype = torch.float16 if device == "cuda" else torch.float32

    generator = None
    if seed is not None:
        generator_device = "cuda" if device == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(seed)

    if config.pipeline == "instruct-pix2pix":
        result = _generate_instruct_pix2pix(
            torch=torch,
            model_id=config.model_id,
            dtype=dtype,
            device=device,
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=work_image,
            guidance_scale=config.guidance_scale,
            image_guidance_scale=config.image_guidance_scale or 1.2,
            steps=config.steps,
            generator=generator,
        )
    else:
        result = _generate_img2img(
            torch=torch,
            backend=config.backend,
            model_id=config.model_id,
            dtype=dtype,
            device=device,
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=work_image,
            strength=config.strength if config.strength is not None else 0.25,
            guidance_scale=config.guidance_scale,
            steps=config.steps,
            generator=generator,
        )

    if result.size != original_size:
        result = result.resize(original_size, Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    return output_path


def resolve_backend_config(
    backend: DiffusionBackend | None = None,
    model_id: str | None = None,
    strength: float | None = None,
    guidance_scale: float | None = None,
    image_guidance_scale: float | None = None,
    steps: int | None = None,
    max_side: int | None = None,
) -> BackendConfig:
    selected_backend = _normalize_backend(backend or _infer_backend(model_id))
    preset = BACKEND_PRESETS[selected_backend]
    return BackendConfig(
        backend=selected_backend,
        model_id=model_id or preset.model_id,
        pipeline=preset.pipeline,
        strength=preset.strength if strength is None else strength,
        guidance_scale=preset.guidance_scale if guidance_scale is None else guidance_scale,
        image_guidance_scale=(
            preset.image_guidance_scale
            if image_guidance_scale is None
            else image_guidance_scale
        ),
        steps=preset.steps if steps is None else steps,
        max_side=preset.max_side if max_side is None else max_side,
    )


def _normalize_backend(backend: DiffusionBackend | str) -> CanonicalDiffusionBackend:
    try:
        return BACKEND_ALIASES[backend]
    except KeyError as exc:
        choices = ", ".join(BACKEND_CHOICES)
        raise ValueError(f"Unknown diffusion backend: {backend}. Choices: {choices}") from exc


def _infer_backend(model_id: str | None) -> CanonicalDiffusionBackend:
    if model_id is None:
        return _normalize_backend(DEFAULT_BACKEND)
    normalized = model_id.lower()
    if "instruct-pix2pix" in normalized:
        return "instruct-pix2pix"
    if "turbo" in normalized:
        return "sdxl-turbo"
    if "xl" in normalized or "sdxl" in normalized:
        return "sdxl"
    return "img2img"


def _resolve_device(torch, requested: str | None) -> str:
    if requested is not None:
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested device 'cuda', but CUDA is not available.")
        if requested == "mps" and not _mps_available(torch):
            raise RuntimeError("Requested device 'mps', but MPS is not available.")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if _mps_available(torch):
        return "mps"
    return "cpu"


def _mps_available(torch) -> bool:
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


def _generate_img2img(
    torch,
    backend: CanonicalDiffusionBackend,
    model_id: str,
    dtype,
    device: str,
    prompt: str,
    negative_prompt: str | None,
    image: Image.Image,
    strength: float,
    guidance_scale: float,
    steps: int,
    generator,
) -> Image.Image:
    if backend in {"img2img", "sd15"}:
        from diffusers import StableDiffusionImg2ImgPipeline

        pipeline_cls = StableDiffusionImg2ImgPipeline
    else:
        from diffusers import AutoPipelineForImage2Image

        pipeline_cls = AutoPipelineForImage2Image

    pipeline = pipeline_cls.from_pretrained(model_id, torch_dtype=dtype)
    pipeline = pipeline.to(device)
    with torch.inference_mode():
        return pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=generator,
        ).images[0].convert("RGB")


def _generate_instruct_pix2pix(
    torch,
    model_id: str,
    dtype,
    device: str,
    prompt: str,
    negative_prompt: str | None,
    image: Image.Image,
    guidance_scale: float,
    image_guidance_scale: float,
    steps: int,
    generator,
) -> Image.Image:
    from diffusers import (
        EulerAncestralDiscreteScheduler,
        StableDiffusionInstructPix2PixPipeline,
    )

    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config
    )
    pipeline = pipeline.to(device)
    with torch.inference_mode():
        return pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            generator=generator,
        ).images[0].convert("RGB")


def _resize_for_diffusion(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    new_width = max(8, int(round(width * scale / 8)) * 8)
    new_height = max(8, int(round(height * scale / 8)) * 8)
    if (new_width, new_height) == image.size:
        return image
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
