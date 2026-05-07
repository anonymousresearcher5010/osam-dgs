import csv
import json
from pathlib import Path
from typing import Optional, Union
import logging

from src.data_processing.types import ConversationPaths

logger = logging.getLogger(__name__)


def load_gloss_types(csv_path: Path) -> set[str]:
    """Load gloss types from a CSV file that were retrieved from the DGS-Korpus Release 3 web page."""
    gloss_types: set[str] = set()

    try:
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # Skip the header row

            for row in reader:
                gloss_types.add(row[0].strip())

        logger.info(f'Loaded {len(gloss_types)} gloss types from {csv_path}.')

    except Exception as e:
        logger.exception(f'Could not load gloss types from {csv_path}: {e}')

    return gloss_types


def read_transcript(transcript_path: Path) -> Optional[str]:
    conversation_has_transcript: bool = transcript_path.exists()

    if conversation_has_transcript:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            content: str = f.read()

        return content
    else:
        return None


def get_conversation_id(conversation_dir: Path) -> str:
    return conversation_dir.name.split('/')[-1].split('_')[1]


def read_lines(file_path: Path) -> Union[list[str], str]:
    with open(file_path, 'r') as f:
        data = f.read().splitlines()

    return data


def get_path_reference(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    else:
        return str(path)


def persist_dict_as_json(dict_to_save: dict, file_path: Path) -> None:
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dict_to_save, f, ensure_ascii=False, indent=4)


def make_paths(conversation_dir: Path) -> ConversationPaths:

    return ConversationPaths(
        conversation_dir=conversation_dir,
        srt=conversation_dir / 'transcript.srt',
        video_a=conversation_dir / 'video-a.mp4',
        video_b=conversation_dir / 'video-b.mp4',
        openpose_gz=conversation_dir / 'openpose.json.gz',
        openpose_json=conversation_dir / 'openpose.json',
        video_ab=conversation_dir / 'video-ab.mp4',
        video_long_shot=conversation_dir / 'video-totale.mp4',
        ilex=conversation_dir / 'transcript.ilex',
        conversation_topics=conversation_dir / 'topics.txt',
        conversation_format=conversation_dir / 'format.txt',
        conversation_age_group=conversation_dir / 'age-group.txt'
    )
