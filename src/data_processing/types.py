from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Literal


TypeCategory = Literal['type', 'subtype', 'productive sign']
RetrievedTypes = tuple[str, TypeCategory, Union[str, None]]

HandUsage = Literal['both', 'left', 'right']
SignCategory = Literal['type', 'subtype']
Speaker = Literal['A', 'B', 'C']
# currently only 50 fps is expected/supported
FPS = Literal[50]

GlossAndMouthingClassificationResult = dict[str, Union[str, bool]]


@dataclass
class RetrievalConfig:
    """Technical and structural configuration parameters for the retrieval."""
    url: str
    data_dir: Path
    SRT_required: bool
    max_conversations: int
    retrieve_video_ab: bool
    retrieve_video_long_shot: bool
    retrieve_ilex: bool
    base_url: str
    request_timeout: float
    delay_seconds: float
    user_agent: str
    overwrite_logs: bool


@dataclass
class RetrieveTypesConfig:
    """Technical and structural configuration parameters for the types retrieval."""
    url: str
    data_dir: Path
    output_file: str
    request_timeout: float
    user_agent: str


@dataclass
class ConversationMeta:
    conversation_id: str
    srt_url: str
    video_a_url: Optional[str]
    video_b_url: Optional[str]
    openpose_url: Optional[str]
    video_ab_url: Optional[str]
    video_long_shot_url: Optional[str]
    ilex_url: Optional[str]
    topics: list[str]
    format: str
    age_group: str


@dataclass
class ConversationPaths:
    conversation_dir: Path
    srt: Path
    video_a: Path
    video_b: Path
    openpose_gz: Path
    openpose_json: Path
    video_ab: Path
    video_long_shot: Path
    ilex: Path
    conversation_topics: Path
    conversation_format: Path
    conversation_age_group: Path


@dataclass
class SRTCue:
    conversation: str
    index: str
    timestamp: str
    text: str


@dataclass
class TimeSegment:
    start_time_s: float
    end_time_s: float
    start_frame: int
    end_frame: int


@dataclass
class MouthingOrMouthGesture:
    mouthing_token: str
    is_regular_mouthing: bool
    is_incomplete_mouthing: bool
    is_unclear_mouthing: bool
    is_mouth_gesture: bool
    is_oral_articulation: bool
    is_undocumented_mouthing: bool
    is_sound_imitation: bool


@dataclass
class Gloss:
    gloss_id: str
    time_segment: TimeSegment
    lexical_sign: str
    is_type: bool
    is_subtype: bool
    has_modified_from_citation_form: bool
    is_double_signing: bool
    is_second_hand_only: bool
    is_name_sign: bool
    is_organisation_name_sign: bool
    is_city_name_sign: bool
    is_unknown_regional_sign: bool
    is_bound_german_morpheme: bool
    is_foreign_sign: bool
    is_productive_sign: bool
    is_pointing_sign: bool
    is_finger_spelling: bool
    is_initialisation_sign: bool
    is_number_sign: bool
    is_list_buoy_sign: bool
    is_gesture_sign: bool
    is_mouthing_without_manual_activity: bool
    is_cued_speech: bool
    is_unclear_linguistic_manual_activity: bool
    is_extra_linguistic_manual_activity: bool
    is_articulation_sign: bool
    is_date_sign: bool
    mouthing_or_mouth_gesture: Optional[MouthingOrMouthGesture]


@dataclass
class Sentence:
    sentence_id: str
    time_segment: TimeSegment
    german_translation: str
    back_translations: list[str]
    glosses: list[Gloss]


@dataclass
class Turn:
    turn_id: str
    time_segment: TimeSegment
    speaker: Speaker
    moderation_sentences_german_translation: Optional[list[str]]
    sentences: list[Sentence]


@dataclass
class Conversation:
    conversation_id: str
    time_segment: Optional[TimeSegment]
    video_reference_speaker_A: Optional[str]
    video_reference_speaker_B: Optional[str]
    fps: Optional[FPS]
    openpose_reference: Optional[str]
    video_reference_AB: Optional[str]
    video_reference_long_shot: Optional[str]
    conversation_topics: list[str]
    conversation_format: str
    conversation_age_group: str
    turns: Optional[list[Turn]]


@dataclass
class CanonicalData:
    dataset_version: str
    creation_date: str
    conversations: list[Conversation]