from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from .diffusion import BACKEND_CHOICES, generate_recoloring, resolve_backend_config
from .grading import STYLES, apply_color_grade
from .image import ensure_same_size, load_rgb, save_comparison, save_rgb
from .lut import LUT3D, fit_lut
from .metrics import psnr, ssim


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="color-adjust",
        description="Fit and apply 3D LUTs for content-preserving recoloring.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="Fit a LUT from input/target images.")
    _add_fit_io_args(fit_parser)
    _add_fit_params(fit_parser)
    fit_parser.set_defaults(func=cmd_fit)

    apply_parser = subparsers.add_parser("apply", help="Apply a saved .cube LUT.")
    apply_parser.add_argument("--input", required=True, type=Path)
    apply_parser.add_argument("--lut", required=True, type=Path)
    apply_parser.add_argument("--output", type=Path, help="Optional image output path.")
    apply_parser.set_defaults(func=cmd_apply)

    grade_parser = subparsers.add_parser(
        "grade",
        help="Create a deterministic recolored target without diffusion.",
    )
    grade_parser.add_argument("--input", required=True, type=Path)
    grade_parser.add_argument("--output", type=Path, help="Optional target image output path.")
    grade_parser.add_argument("--style", choices=STYLES, default="warm-sunset")
    grade_parser.add_argument("--strength", type=float, default=1.0)
    grade_parser.set_defaults(func=cmd_grade)

    diffusion_parser = subparsers.add_parser(
        "diffuse",
        help="Generate a target recoloring using HuggingFace Diffusers.",
    )
    _add_diffusion_args(diffusion_parser)
    diffusion_parser.add_argument("--output", type=Path, help="Optional diffusion target output path.")
    diffusion_parser.set_defaults(func=cmd_diffuse)

    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Generate or load a target, fit a LUT, and save the LUT recoloring.",
    )
    pipeline_parser.add_argument("--input", required=True, type=Path)
    target_group = pipeline_parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--target", type=Path, help="Existing recolored target image.")
    target_group.add_argument("--prompt", type=str, help="Prompt for Diffusers image-to-image.")
    pipeline_parser.add_argument("--generated-target", type=Path, help="Optional generated target path.")
    pipeline_parser.add_argument("--output", type=Path, help="Optional LUT recoloring output path.")
    pipeline_parser.add_argument("--lut", type=Path, help="Optional .cube LUT output path.")
    pipeline_parser.add_argument("--report-json", type=Path, help="Optional metrics JSON output path.")
    pipeline_parser.add_argument("--comparison", type=Path, help="Optional comparison image output path.")
    _add_fit_params(pipeline_parser)
    _add_diffusion_params(pipeline_parser)
    pipeline_parser.set_defaults(func=cmd_pipeline)

    return parser


def _add_fit_io_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Optional LUT recoloring output path.")
    parser.add_argument("--lut", type=Path, help="Optional .cube LUT output path.")
    parser.add_argument("--report-json", type=Path, help="Optional metrics JSON output path.")
    parser.add_argument("--comparison", type=Path, help="Optional comparison image output path.")


def _add_fit_params(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--size", type=int, default=17, help="LUT grid size.")
    parser.add_argument("--samples", type=int, default=200_000, help="Max pixel pairs to sample.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--method",
        choices=["weighted", "nearest-mean", "nearest-median"],
        default="weighted",
    )
    parser.add_argument("--smooth-iterations", type=int, default=24)
    parser.add_argument("--identity-weight", type=float, default=0.02)
    parser.add_argument(
        "--luma-tolerance",
        type=float,
        help="Optional filter: discard pixel pairs whose luma differs more than this value.",
    )


def _add_diffusion_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=str)
    _add_diffusion_params(parser, include_seed=True)


def _add_diffusion_params(parser: argparse.ArgumentParser, include_seed: bool = False) -> None:
    parser.add_argument(
        "--backend",
        choices=BACKEND_CHOICES,
        help="Diffusion backend preset. Default: instruct-pix2pix unless --model-id is provided.",
    )
    parser.add_argument("--negative-prompt", type=str)
    parser.add_argument("--model-id", help="Advanced override for the model id.")
    parser.add_argument(
        "--strength",
        type=float,
        help="Img2img denoising strength. Ignored by instruct-pix2pix.",
    )
    parser.add_argument("--guidance-scale", type=float, help="Text guidance scale.")
    parser.add_argument(
        "--image-guidance-scale",
        type=float,
        help="InstructPix2Pix source-image guidance. Higher preserves the input more.",
    )
    parser.add_argument("--steps", type=int, help="Number of diffusion inference steps.")
    if include_seed:
        parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str)
    parser.add_argument("--max-side", type=int, help="Resize longest side before diffusion.")


def cmd_fit(args: argparse.Namespace) -> int:
    paths = _fit_output_paths(args)
    source = load_rgb(args.input)
    target = ensure_same_size(source, load_rgb(args.target))
    lut = _fit_from_args(source, target, args)
    output = lut.apply(source)
    save_rgb(paths["output"], output)
    lut.save_cube(paths["lut"])
    save_comparison(
        paths["comparison"],
        [("original", source), ("target", target), ("lut result", output)],
    )
    metrics = _metrics_dict(target, output)
    _write_report(paths["report_json"], metrics, args)
    _print_metrics(metrics)
    _print_saved_paths(paths)
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    output_path = _apply_output_path(args)
    source = load_rgb(args.input)
    lut = LUT3D.load_cube(args.lut)
    save_rgb(output_path, lut.apply(source))
    print(f"Saved recolored image: {output_path}")
    return 0


def cmd_grade(args: argparse.Namespace) -> int:
    output_path = _grade_output_path(args)
    source = load_rgb(args.input)
    output = apply_color_grade(source, args.style, strength=args.strength)
    save_rgb(output_path, output)
    print(f"Saved graded target: {output_path}")
    return 0


def cmd_diffuse(args: argparse.Namespace) -> int:
    output_path = _diffusion_target_path(args)
    output = generate_recoloring(
        input_path=args.input,
        output_path=output_path,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        backend=args.backend,
        model_id=args.model_id,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        steps=args.steps,
        seed=args.seed,
        device=args.device,
        max_side=args.max_side,
    )
    print(f"Saved diffusion target: {output}")
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    paths = _pipeline_output_paths(args)
    source = load_rgb(args.input)
    target_path = args.target
    target_label = "target"
    if target_path is None:
        target_label = "diffusion"
        target_path = generate_recoloring(
            input_path=args.input,
            output_path=paths["generated_target"],
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            backend=args.backend,
            model_id=args.model_id,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            image_guidance_scale=args.image_guidance_scale,
            steps=args.steps,
            seed=args.seed,
            device=args.device,
            max_side=args.max_side,
        )
    target = ensure_same_size(source, load_rgb(target_path))
    lut = _fit_from_args(source, target, args)
    output = lut.apply(source)
    save_rgb(paths["output"], output)
    lut.save_cube(paths["lut"])
    save_comparison(
        paths["comparison"],
        [("original", source), (target_label, target), ("lut result", output)],
    )
    metrics = _metrics_dict(target, output)
    _write_report(paths["report_json"], metrics, args)
    _print_metrics(metrics)
    _print_saved_paths(paths)
    return 0


def _fit_from_args(source, target, args: argparse.Namespace) -> LUT3D:
    return fit_lut(
        source=source,
        target=target,
        size=args.size,
        max_samples=args.samples,
        seed=args.seed,
        method=args.method,
        smooth_iterations=args.smooth_iterations,
        identity_weight=args.identity_weight,
        luma_tolerance=args.luma_tolerance,
    )


def _metrics_dict(target, output) -> dict[str, float | None]:
    psnr_value = psnr(target, output)
    return {
        "psnr": None if math.isinf(psnr_value) else psnr_value,
        "psnr_is_infinite": math.isinf(psnr_value),
        "ssim": ssim(target, output),
    }


def _write_report(
    path: Path | None,
    metrics: dict[str, float | None],
    args: argparse.Namespace,
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": metrics,
        "parameters": {
            "size": getattr(args, "size", None),
            "samples": getattr(args, "samples", None),
            "method": getattr(args, "method", None),
            "smooth_iterations": getattr(args, "smooth_iterations", None),
            "identity_weight": getattr(args, "identity_weight", None),
            "luma_tolerance": getattr(args, "luma_tolerance", None),
            "backend": getattr(args, "backend", None),
            "model_id": getattr(args, "model_id", None),
            "strength": getattr(args, "strength", None),
            "guidance_scale": getattr(args, "guidance_scale", None),
            "image_guidance_scale": getattr(args, "image_guidance_scale", None),
            "steps": getattr(args, "steps", None),
            "max_side": getattr(args, "max_side", None),
        },
    }
    if getattr(args, "prompt", None) is not None:
        config = resolve_backend_config(
            backend=getattr(args, "backend", None),
            model_id=getattr(args, "model_id", None),
            strength=getattr(args, "strength", None),
            guidance_scale=getattr(args, "guidance_scale", None),
            image_guidance_scale=getattr(args, "image_guidance_scale", None),
            steps=getattr(args, "steps", None),
            max_side=getattr(args, "max_side", None),
        )
        payload["diffusion"] = {
            "backend": config.backend,
            "model_id": config.model_id,
            "pipeline": config.pipeline,
            "strength": config.strength,
            "guidance_scale": config.guidance_scale,
            "image_guidance_scale": config.image_guidance_scale,
            "steps": config.steps,
            "max_side": config.max_side,
        }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _print_metrics(metrics: dict[str, float | None]) -> None:
    psnr_value = "inf" if metrics["psnr_is_infinite"] else f"{metrics['psnr']:.3f} dB"
    print(f"PSNR: {psnr_value}")
    print(f"SSIM: {metrics['ssim']:.4f}")


def _fit_output_paths(args: argparse.Namespace) -> dict[str, Path]:
    base = f"{_input_stem(args.input)}_from_{_slug(args.target.stem)}_lut"
    output = args.output or _default_output_dir(args.input) / f"{base}.png"
    return {
        "output": output,
        "lut": args.lut or output.with_suffix(".cube"),
        "report_json": args.report_json or _sidecar_path(output, "_metrics.json"),
        "comparison": args.comparison or _sidecar_path(output, "_comparison.png"),
    }


def _apply_output_path(args: argparse.Namespace) -> Path:
    base = f"{_input_stem(args.input)}_apply_{_slug(args.lut.stem)}"
    return args.output or _default_output_dir(args.input) / f"{base}.png"


def _grade_output_path(args: argparse.Namespace) -> Path:
    base = f"{_input_stem(args.input)}_grade_{_slug(args.style)}_target"
    return args.output or _default_output_dir(args.input) / f"{base}.png"


def _diffusion_target_path(args: argparse.Namespace) -> Path:
    base = f"{_input_stem(args.input)}_{_diffusion_variant(args)}_target"
    return args.output or _default_output_dir(args.input) / f"{base}.png"


def _pipeline_output_paths(args: argparse.Namespace) -> dict[str, Path | None]:
    if args.target is None:
        base = f"{_input_stem(args.input)}_{_diffusion_variant(args)}"
        generated_target = (
            args.generated_target
            or _default_output_dir(args.input) / f"{base}_target.png"
        )
    else:
        base = f"{_input_stem(args.input)}_from_{_slug(args.target.stem)}"
        generated_target = None

    output = args.output or _default_output_dir(args.input) / f"{base}_lut.png"
    return {
        "generated_target": generated_target,
        "output": output,
        "lut": args.lut or output.with_suffix(".cube"),
        "report_json": args.report_json or _sidecar_path(output, "_metrics.json"),
        "comparison": args.comparison or _sidecar_path(output, "_comparison.png"),
    }


def _default_output_dir(input_path: Path) -> Path:
    return Path.cwd() / _input_stem(input_path)


def _input_stem(path: Path) -> str:
    return _slug(path.stem)


def _diffusion_variant(args: argparse.Namespace) -> str:
    config = resolve_backend_config(
        backend=args.backend,
        model_id=args.model_id,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        steps=args.steps,
        max_side=args.max_side,
    )
    model = _slug(config.model_id.split("/")[-1])
    seed = _slug(str(getattr(args, "seed", "none")))
    parts = ["diffusion", _slug(config.backend)]
    if model != _slug(config.backend):
        parts.append(model)
    if config.strength is not None:
        parts.append(f"s{_float_slug(config.strength)}")
    if config.image_guidance_scale is not None:
        parts.append(f"igs{_float_slug(config.image_guidance_scale)}")
    parts.append(f"seed{seed}")
    return "_".join(parts)


def _float_slug(value: float) -> str:
    return f"{value:.2f}".replace("-", "m").replace(".", "")


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    slug = slug.strip("-._")
    return slug or "output"


def _sidecar_path(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}{suffix}")


def _print_saved_paths(paths: dict[str, Path | None]) -> None:
    for label, path in paths.items():
        if path is not None:
            print(f"Saved {label}: {path}")
