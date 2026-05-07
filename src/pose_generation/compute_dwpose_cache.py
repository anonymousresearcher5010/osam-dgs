import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Set

import torch
import decord
from tqdm import tqdm

from mimicmotion.dwpose.dwpose_detector import dwpose_detector as dwprocessor
from mimicmotion.pose_generation.training_utils import get_dataset_video_paths, filter_paths_by_processed, video_cache_key


@torch.inference_mode()
def precompute_for_video(
    video_path: str,
    cache_dir: str,
    sample_stride: int = 1,
    force: bool = False,
    every_frame: bool = False,
    batch_size: int = 256,
) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    out_name = video_cache_key(video_path)
    out_path = os.path.join(cache_dir, out_name)

    if os.path.exists(out_path) and not force:
        return out_path

    # Load video
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    fps = float(vr.get_avg_fps())

    # When every_frame is True, disable FPS-based stride scaling and use stride=1
    if every_frame:
        adjusted_stride = 1
    else:
        adjusted_stride = sample_stride * max(1, int(fps / 24))

    # Select frames to process
    idxs = list(range(0, len(vr), adjusted_stride))

    # Process in chunks to avoid loading all frames into RAM at once
    short_path = '/'.join(video_path.split('/')[-2:])
    dets: List[Dict[str, Any]] = []
    batches = range(0, len(idxs), batch_size)
    for i, start in enumerate(tqdm(
            batches,
            desc=f"DWPose precompute: {short_path}"
    )):
        chunk_ids = idxs[start:start + batch_size]
        frames = vr.get_batch(chunk_ids).asnumpy()  # [B,H,W,3] uint8
        for frm in frames:
            det = dwprocessor(frm)
            dets.append(det)

        print(f"Processed batch {i + 1}/{len(batches)} of {short_path}")

    dwprocessor.release_memory()

    H, W = int(vr[0].shape[0]), int(vr[0].shape[1])
    payload = {
        "meta": {
            "video_path": str(Path(video_path).resolve()),
            "orig_size": (H, W),
            "fps": fps,
            "indices": idxs,
            "sample_stride": sample_stride,
            "adjusted_stride": adjusted_stride,
            "num_frames": len(vr),
            "every_frame": bool(every_frame),
        },
        "dets": dets,  # list of detector dicts
    }
    torch.save(payload, out_path)

    return out_path


def main(args):
    print('Pre-processing videos with DWPose')

    # Get list of videos to process
    all_video_paths: Set[str] = get_dataset_video_paths(args.data_path)

    # Resume work
    video_path_to_process: Set[str] = filter_paths_by_processed(all_video_paths, args.cache_dir)

    results = []
    for video_path in tqdm(video_path_to_process, desc="Precomputing videos", unit="video"):
        path = precompute_for_video(
            video_path,
            args.cache_dir,
            args.sample_stride,
            args.force,
            every_frame=args.every_frame,
            batch_size=args.batch_size,
        )
        results.append(path)
        print(f"Cached: {video_path} -> {path}")

    print(f"Done. {len(results)} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute DWPose outputs for videos")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--sample_stride", type=int, default=2, help="Base sampling stride; adjusted by FPS.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing cache.")
    parser.add_argument("--every_frame", action="store_true", help="Precompute detections for every frame.")
    parser.add_argument("--batch_size", type=int, default=4096, help="Frame batch size for decoding/inference.")
    arguments = parser.parse_args()

    main(arguments)

