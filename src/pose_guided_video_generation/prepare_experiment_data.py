#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import pandas as pd


DEFAULT_CSV = Path("data/CSV_views/OSAM-DGS_sentences_dataset.csv")
DEFAULT_OUTPUT_DIR = Path("data/experiments/video_generation/input")
DEFAULT_REFERENCE_IMAGES_DIR = Path("data/reference_images")
EXPECTED_FPS = 50
DEFAULT_MIN_FRAMES = 72
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def word_count(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return len(text.strip().split())


def pick_video_column(speaker_value: str) -> str:
    speaker = str(speaker_value).strip().upper()
    if speaker == "A":
        return "video_reference_speaker_A"
    if speaker == "B":
        return "video_reference_speaker_B"
    raise ValueError(f"Unsupported speaker value: {speaker_value!r}")


def get_video_fps(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    fps_raw = result.stdout.strip()
    if not fps_raw:
        raise ValueError(f"Could not read FPS for video: {video_path}")
    return float(Fraction(fps_raw))


def get_video_frame_count(video_path: Path) -> int:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    value = result.stdout.strip()
    if not value or value == "N/A":
        raise ValueError(f"Could not read frame count for video: {video_path}")
    return int(value)


def cut_video_by_frames(
    video_src: Path,
    video_dst: Path,
    start_frame: int,
    end_frame: int,
) -> None:
    if start_frame < 0 or end_frame < 0:
        raise ValueError(f"start_frame/end_frame must be non-negative, got {start_frame}, {end_frame}")
    if end_frame < start_frame:
        raise ValueError(f"end_frame must be >= start_frame, got {start_frame}, {end_frame}")

    # ffmpeg trim uses end_frame as exclusive -> +1 makes it inclusive
    end_frame_exclusive = end_frame + 1

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_src),
        "-vf",
        f"trim=start_frame={start_frame}:end_frame={end_frame_exclusive},setpts=PTS-STARTPTS",
        "-an",
        str(video_dst),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Randomly sample sentence IDs and prepare video_generation/input folders."
    )
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="Number of sentence_ids to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Path to sentences CSV (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--reference-images-dir",
        type=Path,
        default=DEFAULT_REFERENCE_IMAGES_DIR,
        help=f"Directory with reference images (default: {DEFAULT_REFERENCE_IMAGES_DIR})",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=DEFAULT_MIN_FRAMES,
        help=f"Minimum sentence length in frames using (end_frame - start_frame + 1). Default: {DEFAULT_MIN_FRAMES}",
    )
    args = parser.parse_args()

    if args.min_frames < 1:
        raise ValueError(f"--min-frames must be >= 1, got {args.min_frames}")

    csv_path = args.csv
    output_dir = args.output_dir
    reference_images_dir = args.reference_images_dir

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if not reference_images_dir.exists() or not reference_images_dir.is_dir():
        raise FileNotFoundError(f"Reference images directory not found: {reference_images_dir}")

    reference_images = [
        p for p in reference_images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not reference_images:
        raise ValueError(f"No reference images found in {reference_images_dir}")

    rng = random.Random(args.seed)

    df = pd.read_csv(csv_path)

    required_columns = {
        "sentence_id",
        "speaker",
        "german_translation",
        "start_frame",
        "end_frame",
        "video_reference_speaker_A",
        "video_reference_speaker_B",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    # Numeric frame columns for robust filtering
    df["start_frame_num"] = pd.to_numeric(df["start_frame"], errors="coerce")
    df["end_frame_num"] = pd.to_numeric(df["end_frame"], errors="coerce")
    df["sentence_frame_count"] = df["end_frame_num"] - df["start_frame_num"] + 1

    # Filter by translation length >= 10 words and minimum frame count
    eligible = df[
        (df["german_translation"].apply(word_count) >= 10)
        & (df["sentence_frame_count"] >= args.min_frames)
    ].copy()

    if eligible.empty:
        raise ValueError(
            f"No rows found where german_translation has at least 10 words "
            f"and sentence length is at least {args.min_frames} frames."
        )

    # Ensure one row per sentence_id for sampling
    eligible_unique = eligible.drop_duplicates(subset=["sentence_id"]).copy()

    if args.count > len(eligible_unique):
        raise ValueError(
            f"Requested --count={args.count}, but only {len(eligible_unique)} eligible unique sentence_ids available."
        )

    sampled = eligible_unique.sample(n=args.count, random_state=args.seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    for _, row in sampled.iterrows():
        sentence_id = str(row["sentence_id"])
        speaker = str(row["speaker"]).strip().upper()

        video_col = pick_video_column(speaker)
        video_rel = row[video_col]
        if not isinstance(video_rel, str) or not video_rel.strip():
            print(f"[WARN] Skipping {sentence_id}: empty video reference in {video_col}")
            continue

        video_src = Path(video_rel)
        if not video_src.is_absolute():
            # Interpret CSV paths relative to project root (current working directory)
            video_src = Path.cwd() / video_src

        if not video_src.exists():
            print(f"[WARN] Skipping {sentence_id}: video not found at {video_src}")
            continue

        try:
            fps = get_video_fps(video_src)
        except (subprocess.CalledProcessError, ValueError) as exc:
            print(f"[WARN] Skipping {sentence_id}: cannot read FPS ({exc})")
            continue

        if abs(fps - EXPECTED_FPS) > 1e-6:
            print(f"[WARN] Skipping {sentence_id}: unsupported FPS {fps} (expected {EXPECTED_FPS})")
            continue

        try:
            start_frame = int(row["start_frame"])
            end_frame = int(row["end_frame"])
        except (TypeError, ValueError):
            print(f"[WARN] Skipping {sentence_id}: invalid start_frame/end_frame values")
            continue

        frame_count = end_frame - start_frame + 1
        if frame_count < args.min_frames:
            print(
                f"[WARN] Skipping {sentence_id}: too short ({frame_count} frames, min required {args.min_frames})"
            )
            continue

        target_dir = output_dir / sentence_id
        target_dir.mkdir(parents=True, exist_ok=True)

        video_dst = target_dir / "video.mp4"

        try:
            cut_video_by_frames(video_src, video_dst, start_frame, end_frame)
        except (subprocess.CalledProcessError, ValueError) as exc:
            print(f"[WARN] Skipping {sentence_id}: ffmpeg trim failed ({exc})")
            shutil.rmtree(target_dir, ignore_errors=True)
            continue

        # Validate output clip length as final guard
        try:
            produced_frames = get_video_frame_count(video_dst)
        except (subprocess.CalledProcessError, ValueError) as exc:
            print(f"[WARN] Skipping {sentence_id}: cannot read output frame count ({exc})")
            shutil.rmtree(target_dir, ignore_errors=True)
            continue

        if produced_frames < args.min_frames:
            print(
                f"[WARN] Skipping {sentence_id}: output too short "
                f"({produced_frames} frames, min required {args.min_frames})"
            )
            shutil.rmtree(target_dir, ignore_errors=True)
            continue

        # Store used row as tiny CSV
        row_df = pd.DataFrame([row])
        row_df.to_csv(target_dir / "metadata.csv", index=False)

        # Copy random reference image
        selected_image = rng.choice(reference_images)
        image_dst = target_dir / f"reference_image{selected_image.suffix.lower()}"
        shutil.copy2(selected_image, image_dst)

        created += 1
        print(
            f"[OK] {sentence_id} -> {target_dir} "
            f"(frames={produced_frames}, image={selected_image.name})"
        )

    print(f"\nDone. Prepared {created}/{args.count} samples in: {output_dir}")


if __name__ == "__main__":
    main()