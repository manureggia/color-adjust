from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

from .diffusion import BACKEND_CHOICES, generate_recoloring, resolve_backend_config
from .grading import STYLES, apply_color_grade
from .image import ensure_same_size, load_rgb, save_comparison, save_rgb
from .lut import LUT3D, fit_lut
from .metrics import psnr, ssim
from .residual import mask_to_rgb, refine_with_masked_residual


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

    refine_parser = subparsers.add_parser(
        "refine",
        help="Fit a LUT, then refine it with a masked local residual.",
    )
    _add_refine_io_args(refine_parser)
    _add_fit_params(refine_parser)
    _add_refine_params(refine_parser)
    refine_parser.set_defaults(func=cmd_refine)

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
    pipeline_parser.add_argument(
        "--refine",
        action="store_true",
        help="Also save a masked residual refinement after the LUT result.",
    )
    pipeline_parser.add_argument("--refined-output", type=Path, help="Optional refined output path.")
    pipeline_parser.add_argument("--refined-mask", type=Path, help="Optional residual mask output path.")
    pipeline_parser.add_argument(
        "--refined-report-json",
        type=Path,
        help="Optional refined metrics JSON output path.",
    )
    pipeline_parser.add_argument(
        "--refined-comparison",
        type=Path,
        help="Optional four-panel refined comparison image output path.",
    )
    _add_fit_params(pipeline_parser)
    _add_diffusion_params(pipeline_parser)
    _add_refine_params(pipeline_parser)
    pipeline_parser.set_defaults(func=cmd_pipeline)

    return parser


def _add_fit_io_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Optional LUT recoloring output path.")
    parser.add_argument("--lut", type=Path, help="Optional .cube LUT output path.")
    parser.add_argument("--report-json", type=Path, help="Optional metrics JSON output path.")
    parser.add_argument("--comparison", type=Path, help="Optional comparison image output path.")


def _add_refine_io_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Optional refined output path.")
    parser.add_argument("--lut-output", type=Path, help="Optional baseline LUT image output path.")
    parser.add_argument("--lut", type=Path, help="Optional .cube LUT output path.")
    parser.add_argument("--mask", type=Path, help="Optional residual mask output path.")
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


def _add_refine_params(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--residual-alpha",
        "--alpha",
        dest="residual_alpha",
        type=float,
        default=0.5,
        help="Strength of the masked residual correction.",
    )
    parser.add_argument(
        "--mask-threshold",
        dest="mask_threshold",
        type=float,
        default=0.08,
        help="Minimum color difference before residual transfer starts.",
    )
    parser.add_argument(
        "--mask-softness",
        dest="mask_softness",
        type=float,
        default=0.08,
        help="Soft ramp width above --mask-threshold.",
    )
    parser.add_argument(
        "--mask-blur-radius",
        dest="mask_blur_radius",
        type=int,
        default=4,
        help="Box blur radius for the residual mask, in pixels.",
    )
    parser.add_argument(
        "--max-residual",
        dest="max_residual",
        type=float,
        default=0.35,
        help="Maximum absolute residual per RGB channel before blending.",
    )
    parser.add_argument(
        "--positive-luma-only",
        action="store_true",
        help="Gate residual transfer to areas that get brighter in the target.",
    )


def _add_diffusion_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=str)
    _add_diffusion_params(parser, include_seed=True)


def _add_diffusion_params(parser: argparse.ArgumentParser, include_seed: bool = False) -> None:
    parser.add_argument(
        "--backend",
        "--diffusion-backend",
        "--diffusion_backend",
        dest="backend",
        choices=BACKEND_CHOICES,
        help="Diffusion backend preset. Default: instruct-pix2pix unless --model-id is provided.",
    )
    parser.add_argument(
        "--negative-prompt",
        "--negative_prompt",
        dest="negative_prompt",
        type=str,
    )
    parser.add_argument(
        "--model-id",
        "--model_id",
        dest="model_id",
        help="Advanced override for the model id.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        help="Img2img denoising strength. Ignored by instruct-pix2pix.",
    )
    parser.add_argument(
        "--guidance-scale",
        "--guidance_scale",
        dest="guidance_scale",
        type=float,
        help="Text guidance scale.",
    )
    parser.add_argument(
        "--image-guidance-scale",
        "--image_guidance_scale",
        dest="image_guidance_scale",
        type=float,
        help="InstructPix2Pix source-image guidance. Higher preserves the input more.",
    )
    parser.add_argument(
        "--steps",
        "--num-inference-steps",
        "--num_inference_steps",
        dest="steps",
        type=int,
        help="Number of diffusion inference steps.",
    )
    if include_seed:
        parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str)
    parser.add_argument(
        "--max-side",
        "--max_side",
        dest="max_side",
        type=int,
        help="Resize longest side before diffusion.",
    )


def cmd_fit(args: argparse.Namespace) -> int:
    paths = _fit_output_paths(args)
    source = load_rgb(args.input)
    target = ensure_same_size(source, load_rgb(args.target))
    lut = _fit_from_args(source, target, args)
    output = lut.apply(source)
    metrics = _metrics_dict(target, output)
    save_rgb(paths["output"], output)
    lut.save_cube(paths["lut"])
    save_comparison(
        paths["comparison"],
        [("original", source), ("target", target), ("lut result", output)],
        footer=_format_metrics(metrics),
    )
    _write_report(paths["report_json"], metrics, args)
    _print_metrics(metrics)
    _print_saved_paths(paths)
    return 0


def cmd_refine(args: argparse.Namespace) -> int:
    paths = _refine_output_paths(args)
    source = load_rgb(args.input)
    target = ensure_same_size(source, load_rgb(args.target))
    lut = _fit_from_args(source, target, args)
    lut_output = lut.apply(source)
    refinement = _refine_from_args(target, lut_output, args)
    metrics = {
        "lut": _metrics_dict(target, lut_output),
        "refined": _metrics_dict(target, refinement.output),
    }

    save_rgb(paths["output"], refinement.output)
    save_rgb(paths["lut_output"], lut_output)
    lut.save_cube(paths["lut"])
    save_rgb(paths["mask"], mask_to_rgb(refinement.mask))
    save_comparison(
        paths["comparison"],
        [
            ("original", source),
            ("target", target),
            ("lut result", lut_output),
            ("refined", refinement.output),
        ],
        footer=_format_refine_metrics(metrics),
    )
    _write_refine_report(paths["report_json"], metrics, args)
    _print_refine_metrics(metrics)
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
    try:
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
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Saved diffusion target: {output}")
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    paths = _pipeline_output_paths(args)
    source = load_rgb(args.input)
    target_path = args.target
    target_label = "target"
    if target_path is None:
        target_label = "diffusion"
        print(
            "Warning: pixel-wise LUT fitting assumes the diffusion target preserves "
            "the original geometry and composition.",
            flush=True,
        )
        try:
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
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    target = ensure_same_size(source, load_rgb(target_path))
    lut = _fit_from_args(source, target, args)
    output = lut.apply(source)
    metrics = _metrics_dict(target, output)
    save_rgb(paths["output"], output)
    lut.save_cube(paths["lut"])
    save_comparison(
        paths["comparison"],
        [("original", source), (target_label, target), ("lut result", output)],
        title=_comparison_title(args),
        footer=_format_metrics(metrics),
    )
    _write_report(paths["report_json"], metrics, args)
    if args.refine:
        refinement = _refine_from_args(target, output, args)
        refined_metrics = {
            "lut": metrics,
            "refined": _metrics_dict(target, refinement.output),
        }
        save_rgb(paths["refined_output"], refinement.output)
        save_rgb(paths["refined_mask"], mask_to_rgb(refinement.mask))
        save_comparison(
            paths["refined_comparison"],
            [
                ("original", source),
                (target_label, target),
                ("lut result", output),
                ("refined", refinement.output),
            ],
            title=_comparison_title(args),
            footer=_format_refine_metrics(refined_metrics),
        )
        _write_refine_report(paths["refined_report_json"], refined_metrics, args)
    _print_metrics(metrics)
    if args.refine:
        _print_refine_metrics(refined_metrics)
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


def _refine_from_args(target, baseline, args: argparse.Namespace):
    return refine_with_masked_residual(
        target=target,
        baseline=baseline,
        alpha=args.residual_alpha,
        threshold=args.mask_threshold,
        softness=args.mask_softness,
        blur_radius=args.mask_blur_radius,
        max_residual=args.max_residual,
        positive_luma_only=args.positive_luma_only,
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
            "seed": getattr(args, "seed", None),
            "negative_prompt": getattr(args, "negative_prompt", None),
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
            "seed": getattr(args, "seed", None),
            "negative_prompt": getattr(args, "negative_prompt", None),
            "max_side": config.max_side,
        }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_refine_report(
    path: Path | None,
    metrics: dict[str, dict[str, float | None]],
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
            "residual_alpha": getattr(args, "residual_alpha", None),
            "mask_threshold": getattr(args, "mask_threshold", None),
            "mask_softness": getattr(args, "mask_softness", None),
            "mask_blur_radius": getattr(args, "mask_blur_radius", None),
            "max_residual": getattr(args, "max_residual", None),
            "positive_luma_only": getattr(args, "positive_luma_only", None),
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _print_metrics(metrics: dict[str, float | None]) -> None:
    psnr_value = _format_psnr(metrics)
    print(f"PSNR: {psnr_value}")
    print(f"SSIM: {metrics['ssim']:.4f}")


def _print_refine_metrics(metrics: dict[str, dict[str, float | None]]) -> None:
    print(f"LUT PSNR: {_format_psnr(metrics['lut'])}")
    print(f"LUT SSIM: {metrics['lut']['ssim']:.4f}")
    print(f"Refined PSNR: {_format_psnr(metrics['refined'])}")
    print(f"Refined SSIM: {metrics['refined']['ssim']:.4f}")


def _comparison_title(args: argparse.Namespace) -> str | None:
    prompt = getattr(args, "prompt", None)
    if prompt is None:
        return None
    return f"Prompt: {prompt}"


def _format_metrics(metrics: dict[str, float | None]) -> str:
    return f"PSNR: {_format_psnr(metrics)} | SSIM: {metrics['ssim']:.4f}"


def _format_refine_metrics(metrics: dict[str, dict[str, float | None]]) -> str:
    return (
        f"LUT PSNR: {_format_psnr(metrics['lut'])} | "
        f"LUT SSIM: {metrics['lut']['ssim']:.4f} | "
        f"Refined PSNR: {_format_psnr(metrics['refined'])} | "
        f"Refined SSIM: {metrics['refined']['ssim']:.4f}"
    )


def _format_psnr(metrics: dict[str, float | None]) -> str:
    return "inf" if metrics["psnr_is_infinite"] else f"{metrics['psnr']:.3f} dB"


def _fit_output_paths(args: argparse.Namespace) -> dict[str, Path]:
    base = f"{_input_stem(args.input)}_from_{_slug(args.target.stem)}_lut"
    output = args.output or _default_output_dir(args.input) / f"{base}.png"
    return {
        "output": output,
        "lut": args.lut or output.with_suffix(".cube"),
        "report_json": args.report_json or _sidecar_path(output, "_metrics.json"),
        "comparison": args.comparison or _sidecar_path(output, "_comparison.png"),
    }


def _refine_output_paths(args: argparse.Namespace) -> dict[str, Path]:
    base = f"{_input_stem(args.input)}_from_{_slug(args.target.stem)}_refined"
    output = args.output or _default_output_dir(args.input) / f"{base}.png"
    return {
        "output": output,
        "lut_output": args.lut_output or _sidecar_path(output, "_lut.png"),
        "lut": args.lut or _sidecar_path(output, "_lut.cube"),
        "mask": args.mask or _sidecar_path(output, "_mask.png"),
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
    refined_output = args.refined_output or _sidecar_path(output, "_refined.png")
    return {
        "generated_target": generated_target,
        "output": output,
        "lut": args.lut or output.with_suffix(".cube"),
        "report_json": args.report_json or _sidecar_path(output, "_metrics.json"),
        "comparison": args.comparison or _sidecar_path(output, "_comparison.png"),
        "refined_output": refined_output if args.refine else None,
        "refined_mask": (
            args.refined_mask or _sidecar_path(refined_output, "_mask.png")
            if args.refine
            else None
        ),
        "refined_report_json": (
            args.refined_report_json or _sidecar_path(refined_output, "_metrics.json")
            if args.refine
            else None
        ),
        "refined_comparison": (
            args.refined_comparison or _sidecar_path(refined_output, "_comparison.png")
            if args.refine
            else None
        ),
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
