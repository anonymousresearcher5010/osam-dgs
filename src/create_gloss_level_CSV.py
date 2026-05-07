from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional

GLOSS_BOOL_FIELDS = [
    "is_type",
    "is_subtype",
    "has_modified_from_citation_form",
    "is_double_signing",
    "is_second_hand_only",
    "is_name_sign",
    "is_city_name_sign",
    "is_organisation_name_sign",
    "is_unknown_regional_sign",
    "is_bound_german_morpheme",
    "is_foreign_sign",
    "is_productive_sign",
    "is_pointing_sign",
    "is_finger_spelling",
    "is_initialisation_sign",
    "is_number_sign",
    "is_list_buoy_sign",
    "is_gesture_sign",
    "is_mouthing_without_manual_activity",
    "is_cued_speech",
    "is_unclear_linguistic_manual_activity",
    "is_extra_linguistic_manual_activity",
    "is_date_sign",
]

MOUTHING_BOOL_FIELDS = [
    "is_regular_mouthing",
    "is_incomplete_mouthing",
    "is_unclear_mouthing",
    "is_mouth_gesture",
    "is_oral_articulation",
    "is_sound_imitation",
    "is_undocumented_mouthing",
]

CSV_HEADER = [
    "gloss_id",
    "lexical_sign",
    *GLOSS_BOOL_FIELDS,
    "mouthing_or_mouth_gesture",
    *MOUTHING_BOOL_FIELDS,
    "start_time_s",
    "start_frame",
    "end_time_s",
    "end_frame",
    "sentence_id",
    "german_translation",
    "backtranslations",
    "turn_id",
    "speaker",
    "moderation_sentence_german_translation",
    "conversation_id",
    "conversation_topics",
    "conversation_format",
    "conversation_age_group",
    "video_reference_speaker_A",
    "video_reference_speaker_B",
    "pose_reference",
    "video_reference_AB",
    "video_reference_long_shot",
]


def join_list(values: Optional[list[str]]) -> str:
    if not values:
        return ""
    return " || ".join(values)


def export_gloss_csv(input_json: Path, output_csv: Path) -> None:
    with input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    conversations = data.get("conversations", [])
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()

        for conversation in conversations:
            conversation_id = conversation.get("conversation_id", "")
            conversation_topics = join_list(conversation.get("conversation_topics"))
            conversation_format = conversation.get("conversation_format", "")
            conversation_age_group = conversation.get("conversation_age_group", "")
            video_reference_speaker_a = conversation.get("video_reference_speaker_A", "")
            video_reference_speaker_b = conversation.get("video_reference_speaker_B", "")
            pose_reference = conversation.get("openpose_reference", "")
            video_reference_ab = conversation.get("video_reference_AB", "")
            video_reference_long_shot = conversation.get("video_reference_long_shot", "")

            turns = conversation.get("turns") or []
            for turn in turns:
                turn_id = turn.get("turn_id", "")
                speaker = turn.get("speaker", "")
                moderation_sentence = join_list(turn.get("moderation_sentences_german_translation"))

                sentences = turn.get("sentences") or []
                for sentence in sentences:
                    sentence_id = sentence.get("sentence_id", "")
                    german_translation = sentence.get("german_translation", "")
                    backtranslations = join_list(sentence.get("back_translations"))

                    glosses = sentence.get("glosses") or []
                    for gloss in glosses:
                        gloss_time_segment = gloss.get("time_segment", {}) or {}
                        mouthing = gloss.get("mouthing_or_mouth_gesture")

                        row: dict[str, Any] = {
                            "gloss_id": gloss.get("gloss_id", ""),
                            "lexical_sign": gloss.get("lexical_sign", ""),
                            "mouthing_or_mouth_gesture": (
                                mouthing.get("mouthing_token", "")
                                if isinstance(mouthing, dict)
                                else ""
                            ),
                            "start_time_s": gloss_time_segment.get("start_time_s", ""),
                            "start_frame": gloss_time_segment.get("start_frame", ""),
                            "end_time_s": gloss_time_segment.get("end_time_s", ""),
                            "end_frame": gloss_time_segment.get("end_frame", ""),
                            "sentence_id": sentence_id,
                            "german_translation": german_translation,
                            "backtranslations": backtranslations,
                            "turn_id": turn_id,
                            "speaker": speaker,
                            "moderation_sentence_german_translation": moderation_sentence,
                            "conversation_id": conversation_id,
                            "conversation_topics": conversation_topics,
                            "conversation_format": conversation_format,
                            "conversation_age_group": conversation_age_group,
                            "video_reference_speaker_A": video_reference_speaker_a,
                            "video_reference_speaker_B": video_reference_speaker_b,
                            "pose_reference": pose_reference,
                            "video_reference_AB": video_reference_ab,
                            "video_reference_long_shot": video_reference_long_shot,
                        }

                        for field in GLOSS_BOOL_FIELDS:
                            row[field] = bool(gloss.get(field, False))

                        for field in MOUTHING_BOOL_FIELDS:
                            row[field] = bool(mouthing.get(field, False)) if isinstance(mouthing, dict) else False

                        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export gloss-level CSV from canonical_dataset.json")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/canonical_dataset/OSAM-DGS_canonical_dataset.json"),
        help="Path to canonical_dataset.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/CSV_views/OSAM-DGS_glosses_dataset.csv"),
        help="Path to output CSV file",
    )
    args = parser.parse_args()

    export_gloss_csv(args.input, args.output)
    print(f"CSV exported to: {args.output}")


if __name__ == "__main__":
    main()