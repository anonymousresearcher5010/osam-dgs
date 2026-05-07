from datetime import timedelta
from functools import reduce
from pathlib import Path
from typing import Callable, Optional
import logging

import pandas as pd

from src.data_processing.gloss_and_mouthing_parsing import get_mouthing_metadata, get_gloss_metadata
from src.data_processing.processing_io_utils import read_transcript
from src.data_processing.text_processors import contains_full_sentence, \
    text_is_full_sentence, mark_full_sentence, get_cue_speaker
from src.data_processing.time_processors import split_timestamp, get_time_segment_from_cues_list, \
    get_time_segment_from_single_cue, time_segments_match, time_segments_overlap
from src.data_processing.types import SRTCue, Speaker, TimeSegment, FPS, GlossAndMouthingClassificationResult

logger = logging.getLogger(__name__)


def parse_transcript(transcript: str, conversation_id: str) -> list[SRTCue]:
    """Parse the transcript text into a list of SRTCue objects."""

    def parse_cue(cue: str, _conversation: str) -> SRTCue:
        """
        Parse a single SRT cue block into a SRTCue object,
        where each line is a separate attribute of the object.
        """
        lines = cue.splitlines()

        # Ensure the block has exactly 3 lines: index, timestamp, text
        assert len(lines) == 3

        return SRTCue(
            conversation=_conversation,
            index=lines[0],
            timestamp=lines[1],
            text=lines[2]  # Single text line (e.g., 'A: GLOSS')
        )

    # Split content in SRT file into cues (everytime someone says something)
    transcript_cues = transcript.strip().split('\n\n')

    parsed_transcript_cues: list[SRTCue] = [parse_cue(cue, conversation_id) for cue in transcript_cues]

    return parsed_transcript_cues


def mark_full_sentences_factory() -> Callable[[SRTCue], SRTCue]:
    """Returns a function that marks full sentences in a cue with a special suffix."""

    def mark_single_cue(cue: SRTCue) -> SRTCue:
        """Marks full sentences in a single cue with a special suffix."""
        text: str = cue.text

        has_full_sentence: bool = contains_full_sentence(text)

        if has_full_sentence:
            updated_text: str = mark_full_sentence(text)
            updated_cue: SRTCue = SRTCue(
                cue.conversation,
                cue.index,
                cue.timestamp,
                updated_text
            )
            return updated_cue
        else:
            return cue

    return mark_single_cue


def process_transcript_cues(
        srt_cues: list[SRTCue],
        processor_fnc: Callable[[SRTCue], SRTCue]
) -> list[SRTCue]:
    """Applies a processor function to each cue in the list."""

    processed_cues: list[SRTCue] = [processor_fnc(cue) for cue in srt_cues]

    return processed_cues


def process_transcript(
        srt_path: Path,
        conversation_id: str,
        fps: FPS
) -> tuple[
    Optional[list[SRTCue]],
    Optional[list[SRTCue]],
    Optional[TimeSegment]
]:
    """
    A rather high-level function that:
    - reads the transcript and converts it into a list of SRTCue objects
    - separates moderation cues
    - gets the time segment of the conversation
    - fixes the order of full sentences and glosses
    - disentangles speaker concurrency

    Returns a tuple containing:
    - the processed transcript cues
    - the moderation cues
    - the time segment of the conversation
    """
    # Read transcript file
    transcript: Optional[str] = read_transcript(srt_path)

    if transcript is None:
        return (None, None, None)

    # Create SRT objects
    transcript_cues: list[SRTCue] = parse_transcript(transcript, conversation_id)

    # Split SRT object lists into speaker and moderation
    processed_transcript_cues, moderation_cues = extract_moderation_speaker_cues(transcript_cues)

    # Get timestamp
    time_segment: TimeSegment = get_time_segment_from_cues_list(processed_transcript_cues, fps)

    # Mark full German sentences for easier processing
    processed_transcript_cues = process_transcript_cues(
        processed_transcript_cues,
        mark_full_sentences_factory()
    )

    # Fix order in the sense, that full sentence always come first in the transcripts
    processed_transcript_cues = align_gloss_sentence_order(processed_transcript_cues)

    # Disentangle speaker concurrency to consecutive order
    processed_transcript_cues = disentangle_speaker_concurrency(processed_transcript_cues)

    return processed_transcript_cues, moderation_cues, time_segment


def sum_speakers(srt_cues: list[SRTCue]) -> dict[str, int]:
    speakers_sum: dict[str, int] = {'A': 0, 'B': 0, 'C': 0}
    speakers: list[str] = [get_cue_speaker(cue) for cue in srt_cues]
    for speaker in speakers: speakers_sum[speaker] += 1
    return speakers_sum


def reindex_cues(srt_cues: list[SRTCue]) -> list[SRTCue]:
    """
    Reindexes the cues in the list starting from 1
    based on their position in the list.
    """
    for i, cue in enumerate(srt_cues):
        cue.index = str(i+1)

    return srt_cues


def extract_moderation_speaker_cues(
        srt_cues: Optional[list[SRTCue]]
) -> tuple[Optional[list[SRTCue]], Optional[list[SRTCue]]]:

    if srt_cues is None:
        return (None, None)

    pruned_srt_cues: list[SRTCue] = []
    moderation_speaker_cues: Optional[list[SRTCue]] = []

    for cue in srt_cues:
        cue_speaker: Speaker = get_cue_speaker(cue)
        if cue_speaker == 'A' or cue_speaker == 'B':
            pruned_srt_cues.append(cue)
        elif cue_speaker == 'C':
            moderation_speaker_cues.append(cue)
        else:
            raise ValueError(f'Speaker {cue_speaker} not recognized.')

    if len(moderation_speaker_cues) == 0:
        moderation_speaker_cues = None


    return pruned_srt_cues, moderation_speaker_cues


def align_gloss_sentence_order(srt_cues: Optional[list[SRTCue]], debug=False) -> list[SRTCue]:
    """
    Aligns the order of gloss and full sentence cues based on their timestamps.
    Some sentence starting-glosses appear before the full sentence cue they are associated with.
    For consistency, these should be after the full sentence cue.

    # Original (Problematic):
    # 24
    # 00:00:07,160 --> 00:00:09,600
    # B: $GEST-NM-KOPFNICKEN1

    # 25
    # 00:00:07,160 --> 00:00:09,600
    # B: Ja!_FULL_SENTENCE

    # Becomes (Fixed):
    # 24
    # 00:00:07,159 --> 00:00:09,600
    # B: Ja!_FULL_SENTENCE

    # 25
    # 00:00:07,160 --> 00:00:09,600
    # B: $GEST-NM-KOPFNICKEN1
    """

    def parse_cue_timestamp(_cue: Optional[SRTCue]) -> tuple[timedelta, timedelta] | tuple[None, None]:
        """Parse a cue timestamp into start and end time deltas. Returns None if cue is None."""
        if _cue is None:
            return (None, None)
        else:
            return split_timestamp(_cue)


    if srt_cues is None:
        return None

    speakers_sum_before = sum_speakers(srt_cues)

    full_sentence_cues: list[SRTCue] = list(filter(lambda _cue: text_is_full_sentence(_cue.text), srt_cues))
    non_sentence_cues: list[SRTCue] = list(filter(lambda _cue: not text_is_full_sentence(_cue.text), srt_cues))

    assert len(full_sentence_cues) + len(non_sentence_cues) == len(srt_cues)

    srt_cues_sorted: list[SRTCue] = []
    non_sentence_cues_to_check: list[SRTCue] = non_sentence_cues.copy()
    remaining_non_sentence_cues: list[SRTCue] = []

    for fs_cue_idx, fs_cue in enumerate(full_sentence_cues):

        # Append full sentence cue to sorted list
        srt_cues_sorted.append(fs_cue)

        # Get start and end timestamps of full sentence in timedelta format
        start_fs, end_fs = parse_cue_timestamp(fs_cue)

        # Get start and end timestamps of next full sentence in timedelta format
        next_fs_cue: SRTCue = full_sentence_cues[fs_cue_idx + 1] if fs_cue_idx < len(full_sentence_cues) - 1 else None
        start_next_fs, end_next_fs = parse_cue_timestamp(next_fs_cue)

        # Search for non-sentence cues that overlap with full sentence
        for ns_cue_idx, ns_cue in enumerate(non_sentence_cues_to_check):
            # Get start and end timestamps of non-sentence cue in timedelata format
            start_ns, end_ns = parse_cue_timestamp(ns_cue)

            # Check if non-sentence cue overlaps with full sentence
            ns_cue_is_after_fs: bool = start_ns >= start_fs
            ns_cue_is_before_next_fs: bool = (start_next_fs is None) or (start_ns < start_next_fs)
            ns_cue_overlaps_fs_cue: bool = ns_cue_is_after_fs and ns_cue_is_before_next_fs

            # If overlap, append non-sentence cue to sorted list
            if ns_cue_overlaps_fs_cue:
                # add match to sorted cues list
                srt_cues_sorted.append(ns_cue)
            else:
                # Add non-match to remaining list
                remaining_non_sentence_cues.append(ns_cue)

        # After iterating over all non-sentence cues, update with of remaining non-sentence cues for next iteration
        non_sentence_cues_to_check = remaining_non_sentence_cues.copy()
        remaining_non_sentence_cues = []

    assert len(srt_cues_sorted) == len(srt_cues), f'Expected {len(srt_cues)} cues, but got {len(srt_cues_sorted)}'

    # Log/debug alignments
    if debug:
        for i, cue in enumerate(srt_cues):
            changed: bool = srt_cues[i].text != srt_cues_sorted[i].text
            if changed:
                logger.debug(f'Re-ordering of cues in conversation {srt_cues[i].conversation} at index: {srt_cues[i].index}: Original {srt_cues[i].text} Sorted: {srt_cues_sorted[i].text}.')

    # Update index of cues based on sorted order
    srt_cues_sorted = reindex_cues(srt_cues_sorted)

    return srt_cues_sorted


def disentangle_speaker_concurrency(srt_cues: list[SRTCue]) -> list[SRTCue]:

    # Build the final sorted list
    disentangled_cues: list[SRTCue] = []

    for i, fs_cue_candidate in enumerate(srt_cues):

        is_fs_cue: bool = text_is_full_sentence(fs_cue_candidate.text)
        if is_fs_cue:
            fs_cue: SRTCue = fs_cue_candidate
            fs_cue_speaker: Speaker = get_cue_speaker(fs_cue)

            # Append the full sentence cue
            disentangled_cues.append(fs_cue)

            # Find all following cues that are transcriptions of the full sentence
            # criteria: all non full sentence cues that come before the next full sentence cue of same speaker
            for j in range(i + 1, len(srt_cues)):
                next_cue: SRTCue = srt_cues[j]

                # Check if next cue is a full sentence cue from same speaker = breaking criteria
                is_fs_cue: bool = text_is_full_sentence(next_cue.text)

                # Check if next cue is the same speaker as the current full sentence cue
                next_cue_speaker: Speaker = get_cue_speaker(next_cue)
                is_same_speaker: bool = fs_cue_speaker == next_cue_speaker

                # Stopping criteria
                if is_fs_cue and is_same_speaker:
                    break

                # Appending criteria
                if not is_fs_cue and is_same_speaker:
                    disentangled_cues.append(next_cue)

                # Note: this simple criteria is sufficient, even it appears incomplete on first sight,
                # because it does not stop when encountering a consecutive full sentence from the other speaker.
                # Once the stopping criteria applies, a consecutive full sentence from the other speaker
                # is placed next anyway in the top level loop.

    # Reindex the cues
    disentangled_cues = reindex_cues(disentangled_cues)

    return disentangled_cues


def split_cues_by_turns(srt_cues: Optional[list[SRTCue]]) -> Optional[list[list[SRTCue]]]:

    if srt_cues is None:
        return None

    cues_by_turns: list[list[SRTCue]] = []
    # Initialize first turn
    turn_cues: list[SRTCue] = [srt_cues[0]]
    turn_speaker: Speaker = get_cue_speaker(srt_cues[0])
    for i in range(1, len(srt_cues)):
        cue: SRTCue = srt_cues[i]
        current_speaker: Speaker = get_cue_speaker(cue)
        is_same_speaker: bool = turn_speaker == current_speaker

        if is_same_speaker:
            turn_cues.append(cue)
        else:
            # Current turn done, append to list and initialize new turn
            cues_by_turns.append(turn_cues)
            turn_cues = [cue]
            turn_speaker = current_speaker

    # Append also last turn
    cues_by_turns.append(turn_cues)

    # Sanity check
    assert len(srt_cues) == reduce(
        lambda total_length, turn: total_length + len(turn),
        cues_by_turns,
        0
    )

    return cues_by_turns


def get_moderation_sentence(
        turn_time_segment: TimeSegment,
        moderation_cues: list[SRTCue],
        fps: FPS
) -> Optional[list[str]]:

    if moderation_cues is None:
        return None

    turn_start_frame: int = turn_time_segment.start_frame
    turn_end_frame: int = turn_time_segment.end_frame

    found_sentences: list[str] = []
    for cue in moderation_cues:
        cue_time_segment: TimeSegment = get_time_segment_from_single_cue(cue, fps)
        cue_start_frame: int = cue_time_segment.start_frame

        if turn_start_frame <= cue_start_frame <= turn_end_frame:
            found_sentences.append(cue.text)

    if len(found_sentences) == 0:
        return None
    else:
        return found_sentences


def split_turn_cues_by_sentences(srt_cues: list[SRTCue]) -> list[list[SRTCue]]:

    cues_by_sentences: list[list[SRTCue]] = []
    sentence_cues: list[SRTCue] = [srt_cues[0]]

    for i in range(1, len(srt_cues)):
        cue: SRTCue = srt_cues[i]
        if text_is_full_sentence(cue.text):
            cues_by_sentences.append(sentence_cues)
            sentence_cues = [cue]
        else:
            sentence_cues.append(cue)

    # Append also last sentence
    cues_by_sentences.append(sentence_cues)

    # Sanity check
    assert len(srt_cues) == reduce(
        lambda total_length, _sentence_cues: total_length + len(_sentence_cues),
        cues_by_sentences,
        0
    )

    return cues_by_sentences


def split_cues_into_glosses_and_mouthings(
        srt_cues: list[SRTCue],
        gloss_types: pd.DataFrame
) -> tuple[
    list[tuple[SRTCue, GlossAndMouthingClassificationResult]],
    list[tuple[SRTCue, GlossAndMouthingClassificationResult]]
]:
    mouthing_cues: list[tuple[SRTCue, GlossAndMouthingClassificationResult]] = []
    gloss_cues: list[tuple[SRTCue, GlossAndMouthingClassificationResult]] = []

    for cue in srt_cues:
        if not text_is_full_sentence(cue.text):
            mouthing_metadata: Optional[GlossAndMouthingClassificationResult] = get_mouthing_metadata(cue.text)
            gloss_metadata: Optional[GlossAndMouthingClassificationResult] = get_gloss_metadata(cue.text, gloss_types)

            is_mouthing: bool = mouthing_metadata is not None and gloss_metadata is None
            is_gloss: bool = mouthing_metadata is None and gloss_metadata is not None
            is_not_both: bool = not (is_mouthing and is_gloss)
            is_not_none_of_both: bool = mouthing_metadata is not None and gloss_metadata is not None

            assert is_not_both or is_not_none_of_both

            if is_mouthing:
                mouthing_cues.append((cue, mouthing_metadata))
            elif is_gloss:
                gloss_cues.append((cue, gloss_metadata))
            else:
                raise ValueError(f'Cue text "{cue.text}" is neither mouthing nor gloss.')

    return gloss_cues, mouthing_cues


def match_cues(
        cue_to_match: SRTCue,
        cue_candidates: list[SRTCue],
        fps: FPS
) -> Optional[int]:
    cue_to_match_time_segment: TimeSegment = get_time_segment_from_single_cue(cue_to_match, fps)

    matching_indices: list[int] = []

    for idx, cue_cand in enumerate(cue_candidates):
        candidate_time_segment: TimeSegment = get_time_segment_from_single_cue(cue_cand, fps)
        exact_match: bool = time_segments_match(cue_to_match_time_segment, candidate_time_segment)
        overlap: bool = time_segments_overlap(cue_to_match_time_segment, candidate_time_segment)

        if exact_match or overlap:
            matching_indices.append(idx)

    if len(matching_indices) > 1:
        raise ValueError(f'Multiple mouthing found for gloss "{cue_to_match.text}".')

    if len(matching_indices) == 0:
        return None
    else:
        return matching_indices[0]
