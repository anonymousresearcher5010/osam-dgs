import hashlib
import os
from pathlib import Path

from typing import Set


def get_dataset_video_paths(data_path: str) -> Set[str]:

    video_paths: Set[str] = set()

    # Iterate through all directories starting with 'entry_'
    for directory in os.listdir(data_path):
        if directory.startswith('entry_'):
            dir_path = os.path.join(data_path, directory)

            # Get all mp4 files
            all_files = os.listdir(dir_path)
            video_files: Set[str] = {
                os.path.join(dir_path, f)
                for f in all_files
                if f.endswith('.mp4')
            }

            if len(video_files) < 1:
                print(f'WARNING: No video files found in {directory}! Files: {all_files}')
            else:
                video_paths |= video_files

    return video_paths


def filter_paths_by_processed(all_video_paths: Set[str], cache_path: str) -> Set[str]:

    # read already processed videos -> Set of cache keys
    # all cach keys
    # get delta, only unprocessed
    # iterate over all_video_paths and check if in delta or filter by delta

    existing_cache_keys = {f for f in os.listdir(cache_path) if f.endswith('.pt')}
    all_cache_keys = {video_cache_key(video_path) for video_path in all_video_paths}
    unprocessed_cache_keys = all_cache_keys - existing_cache_keys

    filtered_video_paths: Set[str] = {
        path
        for path in all_video_paths
        if video_cache_key(path) in unprocessed_cache_keys
    }

    assert len(unprocessed_cache_keys) == len(filtered_video_paths)
    print(f'Filtered out {len(all_video_paths) - len(filtered_video_paths)} already cached videos out of {len(all_video_paths)}')

    return filtered_video_paths


def video_cache_key(video_path: str) -> str:
    # Stable name based on path and mtime to avoid collisions
    p = Path(video_path)
    stat = p.stat()
    h = hashlib.sha256(f'{str(p.resolve())}|{stat.st_size}|{int(stat.st_mtime)}'.encode()).hexdigest()[:16]
    return f'{p.stem}_{h}.pt'
