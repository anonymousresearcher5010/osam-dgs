from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional

CSV_HEADER = [
    'sentence_id',
    'german_translation',
    'transcript',
    'start_time_s',
    'start_frame',
    'end_time_s',
    'end_frame',
    'backtranslations',
    'turn_id',
    'speaker',
    'moderation_sentence_german_translation',
    'conversation_id',
    'conversation_topics',
    'conversation_format',
    'conversation_age_group',
    'video_reference_speaker_A',
    'video_reference_speaker_B',
    'pose_reference',
    'video_reference_AB',
    'video_reference_long_shot',
]


def join_list(values: Optional[list[str]]) -> str:
    if not values:
        return ''
    return ' || '.join(values)


def true_flags(obj: dict[str, Any], *, exclude: set[str]) -> list[str]:
    """
    Return all boolean keys that are True, excluding selected keys.
    """
    flags: list[str] = []
    for key, value in obj.items():
        if key in exclude:
            continue
        if isinstance(value, bool) and value:
            flags.append(f'<{key}>')
    return flags


def build_transcript(glosses: Optional[list[dict[str, Any]]], inline_structured_output: bool) -> str:
    """
    Build one sentence-level string by joining all glosses.

    For each gloss:
      lexical_sign (+ all true gloss flags as <flag> if structured_output is True)
      then mouthing part (if structured_output is True):
        [mouthing_token] + all true mouthing flags as <flag>
    """
    if not glosses:
        return ''

    gloss_chunks: list[str] = []

    for gloss in glosses:
        lexical_sign = str(gloss.get('lexical_sign', '')).strip()

        if not inline_structured_output:
            gloss_chunks.append(lexical_sign)
            continue

        # Add annotations to glosses and mouthings
        gloss_flag_exclude = {'gloss_id', 'time_segment', 'lexical_sign', 'mouthing_or_mouth_gesture'}
        gloss_flags = true_flags(gloss, exclude=gloss_flag_exclude)

        chunk_parts: list[str] = []
        base = lexical_sign + ''.join(gloss_flags)
        if base:
            chunk_parts.append(base)

        mouthing = gloss.get('mouthing_or_mouth_gesture')
        if isinstance(mouthing, dict):
            mouthing_token = str(mouthing.get('mouthing_token', '')).strip()
            mouthing_flag_exclude = {'mouthing_token'}
            mouthing_flags = true_flags(mouthing, exclude=mouthing_flag_exclude)

            mouthing_part = f'[{mouthing_token}]' + ''.join(mouthing_flags)
            chunk_parts.append(mouthing_part)

        gloss_chunks.append(' '.join(part for part in chunk_parts if part))

    return ' '.join(gloss_chunks)


def export_sentence_csv(input_json: Path, output_csv: Path, inline_structured_output: bool) -> None:
    with input_json.open('r', encoding='utf-8') as f:
        data = json.load(f)

    conversations = data.get('conversations', [])

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()

        for conversation in conversations:
            conversation_id = conversation.get('conversation_id', '')
            conversation_topics = join_list(conversation.get('conversation_topics'))
            conversation_format = conversation.get('conversation_format', '')
            conversation_age_group = conversation.get('conversation_age_group', '')
            video_reference_speaker_a = conversation.get('video_reference_speaker_A', '')
            video_reference_speaker_b = conversation.get('video_reference_speaker_B', '')
            pose_reference = conversation.get('openpose_reference', '')
            video_reference_ab = conversation.get('video_reference_AB', '')
            video_reference_long_shot = conversation.get('video_reference_long_shot', '')

            turns = conversation.get('turns') or []
            for turn in turns:
                turn_id = turn.get('turn_id', '')
                speaker = turn.get('speaker', '')
                moderation_sentence = join_list(turn.get('moderation_sentences_german_translation'))

                sentences = turn.get('sentences') or []
                for sentence in sentences:
                    time_segment = sentence.get('time_segment', {}) or {}
                    writer.writerow(
                        {
                            'sentence_id': sentence.get('sentence_id', ''),
                            'german_translation': sentence.get('german_translation', ''),
                            'transcript': build_transcript(sentence.get('glosses'), inline_structured_output),
                            'start_time_s': time_segment.get('start_time_s', ''),
                            'start_frame': time_segment.get('start_frame', ''),
                            'end_time_s': time_segment.get('end_time_s', ''),
                            'end_frame': time_segment.get('end_frame', ''),
                            'backtranslations': join_list(sentence.get('back_translations')),
                            'turn_id': turn_id,
                            'speaker': speaker,
                            'moderation_sentence_german_translation': moderation_sentence,
                            'conversation_id': conversation_id,
                            'conversation_topics': conversation_topics,
                            'conversation_format': conversation_format,
                            'conversation_age_group': conversation_age_group,
                            'video_reference_speaker_A': video_reference_speaker_a,
                            'video_reference_speaker_B': video_reference_speaker_b,
                            'pose_reference': pose_reference,
                            'video_reference_AB': video_reference_ab,
                            'video_reference_long_shot': video_reference_long_shot,
                        }
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Export sentence-level CSV from canonical_dataset.json')
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('data/canonical_dataset/OSAM-DGS_canonical_dataset.json'),
        help='Path to canonical_dataset.json',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/CSV_views/OSAM-DGS_sentences_dataset.csv'),
        help='Path to output CSV file',
    )
    parser.add_argument(
        '--structured',
        type=bool,
        default=False,
        help='Whether to export the structured data or the raw data. '
             'Structured includes the gloss annotations and mouthings and their annotations.',

    )
    args = parser.parse_args()

    export_sentence_csv(args.input, args.output, args.structured)
    print(f'CSV exported to: {args.output}')


if __name__ == '__main__':
    main()