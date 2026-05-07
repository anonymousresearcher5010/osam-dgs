import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data_processing.processing_io_utils import get_conversation_id, read_lines, get_path_reference, make_paths
from src.data_processing.text_processors import generate_id, text_is_full_sentence, extract_german_sentence, \
    get_cue_speaker
from src.data_processing.time_processors import get_time_segment_from_cues_list, get_fps, \
    get_time_segment_from_single_cue
from src.data_processing.transcript_processors import process_transcript, split_cues_by_turns, get_moderation_sentence, split_turn_cues_by_sentences, split_cues_into_glosses_and_mouthings, match_cues
from src.data_processing.types import CanonicalData, SRTCue, ConversationPaths, Conversation, FPS, Turn, Sentence, \
    TimeSegment, Speaker, Gloss, GlossAndMouthingClassificationResult, MouthingOrMouthGesture

logger = logging.getLogger(__name__)


def produce_canonical_data(dataset_path: Path, _now_rfc3339: str) -> CanonicalData:

    def create_conversation_data(_conversation_paths: ConversationPaths, _gloss_types: pd.DataFrame) -> Optional[Conversation]:

        def get_sentences(
                _cues_by_sentences: list[list[SRTCue]],
                _turn_id: str,
                _fps: FPS,
                gl_table: pd.DataFrame
        ) -> list[Sentence]:

            _sentences: list[Sentence] = []
            for sentence_cues in _cues_by_sentences:

                # Sentence id
                sentence_id: str = generate_id(sentence_cues, _turn_id)

                # Time segment
                sentence_time_segment: TimeSegment = get_time_segment_from_cues_list(sentence_cues, _fps)

                # Full German Sentence
                raw_sentence: str = sentence_cues[0].text
                assert text_is_full_sentence(raw_sentence)
                german_sentence: str = extract_german_sentence(raw_sentence)

                # Back-translation
                # (created separately in another process and injected into dataset after the canonical dataset JSON is created)
                back_translations: list[str] = []

                # Get mouthings and glosses
                annotated_split_cues = split_cues_into_glosses_and_mouthings(sentence_cues, gl_table)
                annotated_gloss_cues: list[tuple[SRTCue, GlossAndMouthingClassificationResult]] = annotated_split_cues[0]
                annotated_mouthing_cues: list[tuple[SRTCue, GlossAndMouthingClassificationResult]] = annotated_split_cues[1]
                mouthing_cues: list[SRTCue] = [cue for cue, _ in annotated_mouthing_cues]

                glosses: list[Gloss] = []
                for gloss_cue, gloss_result in annotated_gloss_cues:

                    # Get gloss id
                    gloss_id: str = generate_id([gloss_cue], sentence_id)

                    # Get time segment
                    gloss_time_segment: TimeSegment = get_time_segment_from_single_cue(gloss_cue, _fps)


                    mouthing_cue_match_idx: Optional[int] = match_cues(
                        gloss_cue,
                        mouthing_cues,
                        _fps
                    )

                    if mouthing_cue_match_idx is not None:

                        matched_mouthing: tuple[SRTCue, GlossAndMouthingClassificationResult] = annotated_mouthing_cues[mouthing_cue_match_idx]
                        matched_mouthing_cue: SRTCue = matched_mouthing[0]
                        matched_mouthing_metadata: GlossAndMouthingClassificationResult = matched_mouthing[1]

                        mouthing = MouthingOrMouthGesture(
                            mouthing_token=matched_mouthing_metadata['mouthing_token'],
                            is_regular_mouthing=matched_mouthing_metadata['is_regular_mouthing'],
                            is_incomplete_mouthing=matched_mouthing_metadata['is_incomplete_mouthing'],
                            is_unclear_mouthing=matched_mouthing_metadata['is_unclear_mouthing'],
                            is_mouth_gesture=matched_mouthing_metadata['is_mouth_gesture'],
                            is_oral_articulation=matched_mouthing_metadata['is_oral_articulation'],
                            is_sound_imitation=matched_mouthing_metadata['is_sound_imitation'],
                            is_undocumented_mouthing=matched_mouthing_metadata['is_undocumented_mouthing']
                        )
                    else:
                        mouthing = None

                    gloss = Gloss(
                        gloss_id=gloss_id,
                        time_segment=gloss_time_segment,
                        lexical_sign=gloss_result['lexical_sign'],
                        is_type=gloss_result['is_type'],
                        is_subtype=gloss_result['is_subtype'],
                        has_modified_from_citation_form=gloss_result['has_modified_from_citation_form'],
                        is_double_signing=gloss_result['is_double_signing'],
                        is_second_hand_only=gloss_result['is_second_hand_only'],
                        is_name_sign=gloss_result['is_name_sign'],
                        is_organisation_name_sign=gloss_result['is_organisation_name_sign'],
                        is_city_name_sign=gloss_result['is_city_name_sign'],
                        is_unknown_regional_sign=gloss_result['is_unknown_regional_sign'],
                        is_bound_german_morpheme=gloss_result['is_bound_german_morpheme'],
                        is_foreign_sign=gloss_result['is_foreign_sign'],
                        is_productive_sign=gloss_result['is_productive_sign'],
                        is_pointing_sign=gloss_result['is_pointing_sign'],
                        is_finger_spelling=gloss_result['is_finger_spelling'],
                        is_initialisation_sign=gloss_result['is_initialisation_sign'],
                        is_number_sign=gloss_result['is_number_sign'],
                        is_list_buoy_sign=gloss_result['is_list_buoy_sign'],
                        is_gesture_sign=gloss_result['is_gesture_sign'],
                        is_mouthing_without_manual_activity=gloss_result['is_mouthing_without_manual_activity'],
                        is_cued_speech=gloss_result['is_cued_speech'],
                        is_unclear_linguistic_manual_activity=gloss_result['is_unclear_linguistic_manual_activity'],
                        is_extra_linguistic_manual_activity=gloss_result['is_extra_linguistic_manual_activity'],
                        is_articulation_sign=gloss_result['is_articulation_sign'],
                        is_date_sign=gloss_result['is_date_sign'],
                        mouthing_or_mouth_gesture=mouthing
                    )

                    glosses.append(gloss)

                sentence: Sentence = Sentence(
                    sentence_id=sentence_id,
                    time_segment=sentence_time_segment,
                    german_translation=german_sentence,
                    back_translations=back_translations,
                    glosses=glosses
                )

                _sentences.append(sentence)

            return _sentences

        def get_turns(
                _transcript_cues_by_turns: Optional[list[list[SRTCue]]],
                _conversation_id: str,
                _moderation_cues: Optional[list[SRTCue]],
                _fps: Optional[FPS],
                _gloss_types_table: pd.DataFrame
        ) -> Optional[list[Turn]]:

            if not _transcript_cues_by_turns:
                return None

            assert _fps is not None

            _turns: list[Turn] = []
            for turn_cues in _transcript_cues_by_turns:
                # Turn id
                turn_id: str = generate_id(turn_cues, _conversation_id)

                # Time segment
                turn_time_segment: TimeSegment = get_time_segment_from_cues_list(turn_cues, _fps)

                # Speaker
                speaker: Speaker = get_cue_speaker(turn_cues[0])

                # Moderation sentences
                moderation_sentences: Optional[list[str]] = get_moderation_sentence(
                    turn_time_segment,
                    _moderation_cues,
                    _fps
                )

                # Split turn cues into sentences
                cues_by_sentences: list[list[SRTCue]] = split_turn_cues_by_sentences(turn_cues)

                # Sentences
                sentences: list[Sentence] = get_sentences(cues_by_sentences, turn_id, _fps, _gloss_types_table)

                turn: Turn = Turn(
                    turn_id=turn_id,
                    time_segment=turn_time_segment,
                    speaker=speaker,
                    moderation_sentences_german_translation=moderation_sentences,
                    sentences=sentences
                )

                _turns.append(turn)

            return _turns


        # ID
        conversation_id: str = get_conversation_id(_conversation_paths.conversation_dir)

        # File reference data
        video_reference_speaker_a: Optional[str] = get_path_reference(_conversation_paths.video_a)
        video_reference_speaker_b: str = get_path_reference(_conversation_paths.video_b)
        openpose_reference: Optional[str] = get_path_reference(_conversation_paths.openpose_json)
        video_reference_ab: Optional[str] = get_path_reference(_conversation_paths.video_ab)
        video_reference_long_shot: Optional[str] = get_path_reference(_conversation_paths.video_long_shot)

        # FPS
        fps: Optional[FPS] = get_fps(_conversation_paths)
        if fps is None:
            logger.warning(f'Skipping conversation {conversation_id} due to missing/invalid FPS')
            return None

        # Metadata
        conversation_topics: list[str] = read_lines(_conversation_paths.conversation_topics)
        conversation_format_list: list[str] = read_lines(_conversation_paths.conversation_format)
        conversation_age_group_list: list[str] = read_lines(_conversation_paths.conversation_age_group)

        assert len(conversation_format_list) == 1, f'Unexpected data format. {conversation_format_list}'
        assert len(conversation_age_group_list) == 1, f'Unexpected data format. {conversation_age_group_list}'

        conversation_format: str = conversation_format_list[0]
        conversation_age_group: str = conversation_age_group_list[0]

        # Transcript data
        transcript_cues, moderation_cues, conversation_time_segment = process_transcript(
            _conversation_paths.srt,
            conversation_id,
            fps
        )

        # Split transcript cues into turns
        transcript_cues_by_turns: Optional[list[list[SRTCue]]] = split_cues_by_turns(transcript_cues)

        # Extract turns
        turns: Optional[list[Turn]] = get_turns(
            transcript_cues_by_turns,
            conversation_id,
            moderation_cues,
            fps,
            _gloss_types
        )

        # Create conversation object
        conversation: Conversation = Conversation(
            conversation_id=conversation_id,
            time_segment=conversation_time_segment,
            video_reference_speaker_A=video_reference_speaker_a,
            video_reference_speaker_B=video_reference_speaker_b,
            fps=fps,
            openpose_reference=openpose_reference,
            video_reference_AB=video_reference_ab,
            video_reference_long_shot=video_reference_long_shot,
            conversation_topics=conversation_topics,
            conversation_format=conversation_format,
            conversation_age_group=conversation_age_group,
            turns=turns
        )

        # Done
        return conversation


    # Load gloss types from CSV file
    # gloss_types: pd.DataFrame = pd.read_csv('/Volumes/LaCie2of4TB/datasets/dgs_korpus_release_3_raw/dgs-types.csv')
    gloss_types: pd.DataFrame = pd.read_csv('data/dgs_korpus_release_3_raw/dgs-types.csv')

    # Sanity check
    unique_gloss_types: set[str] = set(gloss_types['lexical_sign'])
    assert len(unique_gloss_types) == len(gloss_types)

    # Get list of conversations folders
    all_conversation_dirs: list[Path] = [folder for folder in dataset_path.iterdir() if folder.is_dir()]

    # Create paths for each conversation
    all_conversation_paths: list[ConversationPaths] = [
        make_paths(conversation_dir)
        for conversation_dir in all_conversation_dirs
    ]

    # Count total conversations
    total_conversations: int = len(all_conversation_paths)

    # Create data for each conversation
    conversations: list[Conversation] = []
    logger.info(f'Starting processing of {total_conversations} conversations...')
    for i, conversation_paths in enumerate(all_conversation_paths):
        # Progress logging
        progress_percent: float = (i / total_conversations) * 100
        logger.info(f'Processing {conversation_paths.conversation_dir} ({i}/{total_conversations}) - {progress_percent:.1f}% complete')

        conversation_data: Optional[Conversation] = create_conversation_data(conversation_paths, gloss_types)

        if conversation_data is None:
            logger.warning(f'Skipping conversation {conversation_paths.conversation_dir}')
            continue
        else:
            conversations.append(conversation_data)

    canonical_data = CanonicalData(
        dataset_version='cfg.dataset_version',
        creation_date=_now_rfc3339,
        conversations=conversations
    )

    return canonical_data
