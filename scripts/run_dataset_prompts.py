#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


@dataclass(frozen=True)
class PromptSet:
    prompts: list[str]
    negative_prompt: str | None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    dataset_dir = args.dataset_dir.resolve()
    output_root = args.output_dir.resolve()
    prompt_set = read_prompts(args.prompts)
    negative_prompt = args.negative_prompt or prompt_set.negative_prompt
    images = find_images(dataset_dir, recursive=args.recursive)

    if not images:
        print(f"No images found in {dataset_dir}", file=sys.stderr)
        return 1
    if not prompt_set.prompts:
        print(f"No prompts found in {args.prompts}", file=sys.stderr)
        return 1

    print(f"Images: {len(images)}")
    print(f"Prompts: {len(prompt_set.prompts)}")
    print(f"Results: {output_root}")

    failures: list[tuple[Path, int, int]] = []
    for image_index, image_path in enumerate(images, start=1):
        image_name = image_output_name(dataset_dir, image_path)
        image_output_dir = output_root / image_name
        image_output_dir.mkdir(parents=True, exist_ok=True)

        for prompt_index, prompt in enumerate(prompt_set.prompts, start=1):
            prefix = f"{image_name}_prompt{prompt_index}"
            command = build_color_adjust_command(
                args=args,
                image_path=image_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                image_output_dir=image_output_dir,
                prefix=prefix,
            )
            write_run_manifest(
                image_output_dir / f"{prefix}_run.json",
                image=image_path,
                prompt=prompt,
                prompt_index=prompt_index,
                negative_prompt=negative_prompt,
                command=command,
            )

            print(
                f"[{image_index}/{len(images)} {prompt_index}/{len(prompt_set.prompts)}] "
                f"{image_path.name} -> {image_output_dir / prefix}"
            )
            if args.dry_run:
                print(shlex.join(command))
                continue

            env = subprocess_env(repo_root)
            result = subprocess.run(
                command,
                cwd=repo_root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if result.stdout:
                print(result.stdout, end="")
            if result.returncode != 0:
                failures.append((image_path, prompt_index, result.returncode))
                if not args.keep_going:
                    return result.returncode

    if failures:
        print("Failures:", file=sys.stderr)
        for image_path, prompt_index, returncode in failures:
            print(
                f"- {image_path} prompt {prompt_index}: exit code {returncode}",
                file=sys.stderr,
            )
        return 1

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run color-adjust pipeline for every image in a dataset and every prompt "
            "listed in a prompts file."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/dataset"))
    parser.add_argument("--prompts", type=Path, default=Path("data/prompts.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/results"))
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Only scan images directly inside --dataset-dir.",
    )
    parser.set_defaults(recursive=True)
    parser.add_argument(
        "--negative-prompt",
        help="Override the negative prompt. By default a 'negative:' line is read from the prompts file.",
    )
    parser.add_argument("--backend", "--diffusion-backend", dest="backend")
    parser.add_argument("--model-id", "--model_id", dest="model_id")
    parser.add_argument("--strength", type=float)
    parser.add_argument("--guidance-scale", "--guidance_scale", dest="guidance_scale", type=float)
    parser.add_argument(
        "--image-guidance-scale",
        "--image_guidance_scale",
        dest="image_guidance_scale",
        type=float,
    )
    parser.add_argument("--steps", "--num-inference-steps", dest="steps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--max-side", "--max_side", dest="max_side", type=int)
    parser.add_argument("--size", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--method", choices=["weighted", "nearest-mean", "nearest-median"])
    parser.add_argument("--smooth-iterations", dest="smooth_iterations", type=int)
    parser.add_argument("--identity-weight", dest="identity_weight", type=float)
    parser.add_argument("--luma-tolerance", dest="luma_tolerance", type=float)
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue with remaining image/prompt pairs after a failed run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and write manifests without running color-adjust.",
    )
    return parser


def read_prompts(path: Path) -> PromptSet:
    prompts: list[str] = []
    negative_prompt: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("negative:"):
            negative_prompt = line.split(":", 1)[1].strip() or None
            continue
        prompts.append(line)

    return PromptSet(prompts=prompts, negative_prompt=negative_prompt)


def find_images(dataset_dir: Path, recursive: bool) -> list[Path]:
    if not dataset_dir.exists():
        return []
    paths = dataset_dir.rglob("*") if recursive else dataset_dir.iterdir()
    return sorted(
        path.resolve()
        for path in paths
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_output_name(dataset_dir: Path, image_path: Path) -> str:
    relative = image_path.relative_to(dataset_dir)
    parts = list(relative.with_suffix("").parts)
    return slug("_".join(parts))


def build_color_adjust_command(
    *,
    args: argparse.Namespace,
    image_path: Path,
    prompt: str,
    negative_prompt: str | None,
    image_output_dir: Path,
    prefix: str,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "color_adjust",
        "pipeline",
        "--input",
        str(image_path),
        "--prompt",
        prompt,
        "--generated-target",
        str(image_output_dir / f"{prefix}_target.png"),
        "--output",
        str(image_output_dir / f"{prefix}_lut.png"),
        "--lut",
        str(image_output_dir / f"{prefix}_lut.cube"),
        "--report-json",
        str(image_output_dir / f"{prefix}_metrics.json"),
        "--comparison",
        str(image_output_dir / f"{prefix}_comparison.png"),
        "--seed",
        str(args.seed),
    ]
    if negative_prompt:
        command.extend(["--negative-prompt", negative_prompt])

    add_optional(command, "--backend", args.backend)
    add_optional(command, "--model-id", args.model_id)
    add_optional(command, "--strength", args.strength)
    add_optional(command, "--guidance-scale", args.guidance_scale)
    add_optional(command, "--image-guidance-scale", args.image_guidance_scale)
    add_optional(command, "--steps", args.steps)
    add_optional(command, "--device", args.device)
    add_optional(command, "--max-side", args.max_side)
    add_optional(command, "--size", args.size)
    add_optional(command, "--samples", args.samples)
    add_optional(command, "--method", args.method)
    add_optional(command, "--smooth-iterations", args.smooth_iterations)
    add_optional(command, "--identity-weight", args.identity_weight)
    add_optional(command, "--luma-tolerance", args.luma_tolerance)
    return command


def add_optional(command: list[str], option: str, value: object | None) -> None:
    if value is not None:
        command.extend([option, str(value)])


def subprocess_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{src_path}{os.pathsep}{current_pythonpath}" if current_pythonpath else src_path
    )
    return env


def write_run_manifest(
    path: Path,
    *,
    image: Path,
    prompt: str,
    prompt_index: int,
    negative_prompt: str | None,
    command: list[str],
) -> None:
    payload = {
        "image": str(image),
        "prompt_index": prompt_index,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "command": command,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    normalized = normalized.strip("-._")
    return normalized or "image"


if __name__ == "__main__":
    raise SystemExit(main())
