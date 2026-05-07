import re
from typing import Optional
from datetime import timedelta
import logging

import ffmpeg

from src.data_processing.types import SRTCue, FPS, TimeSegment, ConversationPaths


logger = logging.getLogger(__name__)


def parse_time(t: str) -> timedelta:
    """Parse a time string into a timedelta object."""
    h, m, rest = t.split(':')
    s, ms = rest.split(',')
    return timedelta(
        hours=int(h),
        minutes=int(m),
        seconds=int(s),
        milliseconds=int(ms)
    )


def split_timestamp(cue: SRTCue) -> tuple[timedelta, timedelta]:
    """Split a timestamp string into start and end timedeltas."""
    timestamps: list[str] = cue.timestamp.split(' --> ')

    if len(timestamps) > 2: raise ValueError(f'More than 2 timestamps found in cue: {cue}')

    start: timedelta = parse_time(timestamps[0])
    end: timedelta = parse_time(timestamps[1])
    return start, end


def frame_from_second(second: float, frames_per_sec: FPS) -> int:
    return int(second * frames_per_sec)


def get_time_segment_from_cues_list(transcript_cues: list[SRTCue], _fps: FPS) -> TimeSegment:

    start_time_first, _ = split_timestamp(transcript_cues[0])
    _, end_time_last = split_timestamp(transcript_cues[-1])
    start_time_first_s: float = start_time_first.total_seconds()
    end_time_last_s: float = end_time_last.total_seconds()

    start_frame: int = frame_from_second(start_time_first_s, _fps)
    end_frame: int = frame_from_second(end_time_last_s, _fps)

    return TimeSegment(
        start_time_s=start_time_first_s,
        end_time_s=end_time_last_s,
        start_frame=start_frame,
        end_frame=end_frame
    )


def get_time_segment_from_single_cue(transcript_cue: SRTCue, _fps: FPS) -> TimeSegment:

    start_time, end_time = split_timestamp(transcript_cue)
    start_time_s: float = start_time.total_seconds()
    end_time_s: float = end_time.total_seconds()

    start_frame: int = frame_from_second(start_time_s, _fps)
    end_frame: int = frame_from_second(end_time_s, _fps)

    return TimeSegment(
        start_time_s=start_time_s,
        end_time_s=end_time_s,
        start_frame=start_frame,
        end_frame=end_frame
    )


def get_fps(conversation_paths: ConversationPaths) -> Optional[FPS]:

    video_a_exists: bool = conversation_paths.video_a.exists()
    video_b_exists: bool = conversation_paths.video_b.exists()

    if not video_a_exists and not video_b_exists:
        logger.warning(f'No videos found in {conversation_paths.conversation_dir}.')
        return None

    fps_video_a: Optional[FPS] = (
        int(ffmpeg.probe(str(conversation_paths.video_a))['streams'][0]['r_frame_rate'].split('/')[0])
        if video_a_exists else None
    )

    fps_video_b: Optional[FPS] = (
        int(ffmpeg.probe(str(conversation_paths.video_b))['streams'][0]['r_frame_rate'].split('/')[0])
        if video_b_exists else None
    )

    only_single_video: bool = (video_a_exists and not video_b_exists) or (not video_a_exists and video_b_exists)
    has_same_fps: bool = only_single_video or (fps_video_a == fps_video_b)

    if not has_same_fps:
        error_message: str = f'FPS mismatch in {conversation_paths.conversation_dir}: video-a.mp4={fps_video_a}, video-b.mp4={fps_video_b}'
        logger.error(error_message)
        return None

    fps: Optional[FPS] = fps_video_a or fps_video_b

    # only support 50 fps for now
    if fps != 50:
        logger.warning(f'Unsupported FPS {fps} in {conversation_paths.conversation_dir} (expected 50).')
        return None

    return fps


def time_segments_match(segment1: TimeSegment, segment2: TimeSegment) -> bool:
    """
    Check if two time segments are exactly the same.

    Returns True if the segments are identical (same start and end frames).
    """
    return segment1.start_frame == segment2.start_frame and segment1.end_frame == segment2.end_frame


def time_segments_overlap(segment1: TimeSegment, segment2: TimeSegment) -> bool:
    """
    Check if two time segments overlap.

    Returns True if the segments overlap (any frame range intersection).
    Segments that only touch at boundaries (end_frame == start_frame) are not considered overlapping.
    """
    return (segment1.start_frame < segment2.end_frame
            and segment2.start_frame < segment1.end_frame)
