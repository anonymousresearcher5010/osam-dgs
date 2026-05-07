#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import yaml


def build_test_case_item(folder: Path, args) -> dict:
    video_path = folder / args.video_name
    image_path = folder / args.image_name

    if not video_path.exists() or not image_path.exists():
        return None

    return {
        "ref_video_path": str(video_path),
        "ref_image_path": str(image_path),
        "num_frames": args.num_frames,
        "resolution": args.resolution,
        "frames_overlap": args.frames_overlap,
        "num_inference_steps": args.num_inference_steps,
        "noise_aug_strength": args.noise_aug_strength,
        "guidance_scale": args.guidance_scale,
        "sample_stride": args.sample_stride,
        "fps": args.fps,
        "seed": args.seed,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate MimicMotion inference YAML from input folders.")
    parser.add_argument("--input-root", default="data/experiments/video_generation/input")
    parser.add_argument("--output-yaml", default="data/experiments/video_generation/OSAM-DGS-video-generation.yaml")
    parser.add_argument("--base-model-path", default="stabilityai/stable-video-diffusion-img2vid-xt-1-1")
    parser.add_argument("--ckpt-path", default="/storage/model_storage/MimicMotion/models/MimicMotion_1-1.pth")
    parser.add_argument("--output-dir", default="/storage/experiments/video_generation/output")
    parser.add_argument("--video-name", default="video.mp4")
    parser.add_argument("--image-name", default="reference_image.png")

    # Inference params
    parser.add_argument("--num-frames", type=int, default=72)
    parser.add_argument("--resolution", type=int, default=576)
    parser.add_argument("--frames-overlap", type=int, default=6)
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--noise-aug-strength", type=float, default=0.0)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=100)

    args = parser.parse_args()

    root = Path(args.input_root)
    if not root.exists():
        raise FileNotFoundError(f"Input root does not exist: {root}")

    test_cases = []
    for folder in sorted([p for p in root.iterdir() if p.is_dir()]):
        item = build_test_case_item(folder, args)
        if item is not None:
            test_cases.append(item)

    if not test_cases:
        raise RuntimeError(f"No valid (video+reference image) pairs found under: {root}")

    config = {
        "base_model_path": args.base_model_path,
        "ckpt_path": args.ckpt_path,
        "output_dir": args.output_dir,
        "test_case": test_cases,
    }

    out_path = Path(args.output_yaml)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"Generated {out_path} with {len(test_cases)} test_case entries.")


if __name__ == "__main__":
    main()