import base64
import hashlib
from functools import reduce
import logging
from typing import cast

from src.data_processing.types import SRTCue, Speaker

logger = logging.getLogger(__name__)


def contains_full_sentence(_text: str) -> bool:
    """
    Checks if a cue text in the SRT files contains a full sentence.
    """
    
    text_without_speaker_prefix: str = remove_speaker_prefix(_text)

    if not text_without_speaker_prefix:
        raise ValueError(f'Empty text: {_text}')

    # Anomaly handling of full sentence - return match.
    positive_full_sentence_anomalies = [
        'Dort hat man sich auf Sport, Bildung und dass alle die gleichen Persönlichkeitsrechte haben konzentriert',
        '500/ Packung/ 4',
        'zum Beispiel',
        'Und sonst-',
        'Ausflüge -',
        'Ahrensburg',
        'ich gebärde Skype so wobei die anderen Gehörlosen Skype so gebärden. ooVoo wird nämlich so gebärdet. Aber ich gebärde Skype so.',
        'ooVoo/ Ja. ooVoo da, ich weiß nicht, ob es das bei Skype gibt, aber bei ooVoo kann man eine Video-Konferenz machen.',
        'zum Beispiel',
        'ooVoo.',
        'das TBW/',
        'Ahrensburg',
        '?'
    ]
    if text_without_speaker_prefix in positive_full_sentence_anomalies:
        return True

    # Allowed chars for full sentence ending.
    if text_without_speaker_prefix[-1] not in ':.!?/“])";-”\'':
        return False

    core = text_without_speaker_prefix[:-1].strip()
    if not core:
        return False

    # Optional leading ellipsis.
    if core.startswith('…'):
        core = core[1:].lstrip()

    # First visible character rule:
    # - normal case: uppercase / digit / opening quote
    # - after ellipsis: uppercase or lowercase / digit / opening quote
    if not core:
        return False

    first_char = core[0]
    normal_start = first_char.isupper() or first_char.isdigit() or first_char in '„"(‘’‚#\'-'
    ellipsis_start = first_char.isalpha() or first_char.isdigit() or first_char in '„"(‘’‚#'

    if text_without_speaker_prefix.startswith('…'):
        if not ellipsis_start:
            return False
    else:
        if not normal_start:
            return False

    # Allow common German sentence characters inside.
    allowed = set(
        'abcdefghijklmnopqrstuvwxyz'
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        'äöüÄÖÜß'
        '0123456789'
        ' –-_.,;:!?()[]{}\'"/…„“”‘’´`‚#€%&=§'
    )

    is_match = all(ch.isalnum() or ch.isspace() or ch in allowed for ch in core)

    return is_match


def text_is_full_sentence(text: str) -> bool:
    return '_FULL_SENTENCE' in text


def mark_full_sentence(text: str) -> str:
    """Marks full sentences with a special suffix."""
    return f'{text}_FULL_SENTENCE'


def short_hash(value: str, digest_size: int = 8) -> str:
    digest: bytes = hashlib.blake2b(value.encode('utf-8'), digest_size=digest_size).digest()
    hash_str: str = base64.b32encode(digest).decode('ascii').rstrip('=')
    return hash_str


def generate_id(srt_cues: list[SRTCue], prefix: str) -> str:

    cues_text: str = reduce(
        lambda concatenated_text, cue: concatenated_text + cue.text,
        srt_cues,
        ''
    )

    text_hash: str = short_hash(cues_text)
    _id: str = f'{prefix}_{text_hash}'

    return _id


def extract_german_sentence(raw_text: str):

    speaker_separator: str = ': '
    full_sentence_marker: str = '_FULL_SENTENCE'

    # Remove speaker prefix (and keep other ': ' appearances intact).
    speaker_separation: str = speaker_separator.join(raw_text.split(speaker_separator)[1:])

    cleaned_text: str = speaker_separation.split(full_sentence_marker)[0].strip()

    return cleaned_text


def remove_speaker_prefix(text: str, speaker_separator: str=': ') -> str:
    return text.split(speaker_separator, maxsplit=1)[1].strip()


def get_cue_speaker(cue: SRTCue) -> Speaker:
    return cast(Speaker, cue.text.split(':')[0])
