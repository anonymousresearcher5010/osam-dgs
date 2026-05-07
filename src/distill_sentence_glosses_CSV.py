from __future__ import annotations

import argparse
import csv
import unicodedata
from pathlib import Path


OUTPUT_HEADER = ['sentence_id', 'full_sentence', 'glosses']
DOUBLE_GLOSS_TOKEN = '||'
BACKTRANSLATION_SEPARATOR = ' || '


def normalize_text(value: str) -> str:
    """
    Normalize full-sentence text.
    - Unicode NFKC normalization
    - Strip leading/trailing whitespace
    - Collapse internal whitespace to single spaces
    """
    if value is None:
        return ''
    value = unicodedata.normalize('NFKC', str(value))
    return ' '.join(value.strip().split())


def split_backtranslations(value: str) -> list[str]:
    """
    Split backtranslations by the literal separator ' || '.
    """
    if not value:
        return []
    return [part for part in value.split(BACKTRANSLATION_SEPARATOR) if part]


def clean_double_glossing_marker(transcript: str) -> str:
    """
    Remove/normalize double-gloss marker '||' in transcript:
    - leading/trailing marker is removed
    - in-between marker becomes a single space
    """
    if not transcript or DOUBLE_GLOSS_TOKEN not in transcript:
        return transcript

    parts = [part.strip() for part in transcript.split(DOUBLE_GLOSS_TOKEN) if part.strip()]
    return ' '.join(parts)


def clean_double_glossing_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    """
    Apply double-gloss marker cleanup to all rows.
    Returns cleaned rows and number of rows that changed.
    """
    changed = 0
    cleaned_rows: list[dict[str, str]] = []

    for row in rows:
        old_glosses = row.get('glosses', '')
        new_glosses = clean_double_glossing_marker(old_glosses)

        if new_glosses != old_glosses:
            changed += 1

        cleaned_rows.append(
            {
                'sentence_id': row['sentence_id'],
                'full_sentence': row['full_sentence'],
                'glosses': new_glosses,
            }
        )

    return cleaned_rows, changed


def deduplicate_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    """
    Deduplicate exact rows by (sentence_id, full_sentence, glosses),
    preserving first occurrence order.
    """
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, str]] = []

    for row in rows:
        key = (row['sentence_id'], row['full_sentence'], row['glosses'])
        if key not in seen:
            seen.add(key)
            deduped.append(row)

    removed = len(rows) - len(deduped)
    return deduped, removed


def count_duplicate_pairs_wo_id(rows: list[dict[str, str]]) -> int:
    """
    Count duplicate (full_sentence, glosses) pairs ignoring sentence_id.
    """
    seen_pairs: set[tuple[str, str]] = set()
    duplicates = 0
    for row in rows:
        key = (row['full_sentence'], row['glosses'])
        if key in seen_pairs:
            duplicates += 1
        else:
            seen_pairs.add(key)
    return duplicates


def distill_csv(input_csv: Path, output_csv: Path, clean_double_glossing: bool) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    source_rows = 0
    all_pairs: list[dict[str, str]] = []

    with input_csv.open('r', encoding='utf-8', newline='') as fin:
        reader = csv.DictReader(fin)

        required_cols = {'sentence_id', 'german_translation', 'transcript'}
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f'Input CSV is missing required columns: {sorted(missing)}. '
                f'Found: {reader.fieldnames}'
            )

        for row in reader:
            source_rows += 1

            # Keep sentence_id and transcript unnormalized by request.
            sentence_id = row.get('sentence_id', '')
            transcript = row.get('transcript', '')

            german_translation = normalize_text(row.get('german_translation', ''))

            # Unnormalized backtranslations
            raw_backtranslations = split_backtranslations(row.get('backtranslations', ''))

            candidates: list[str] = []
            if german_translation:
                candidates.append(german_translation)

            # Normalization is applied only after splitting.
            for bt in raw_backtranslations:
                bt_norm = normalize_text(bt)
                if bt_norm:
                    candidates.append(bt_norm)

            for full_sentence in candidates:
                all_pairs.append(
                    {
                        'sentence_id': sentence_id,
                        'full_sentence': full_sentence,
                        'glosses': transcript,
                    }
                )

    expanded_rows_before_cleanup = len(all_pairs)

    rows_for_dedup = all_pairs
    double_gloss_rows_changed = 0
    if clean_double_glossing:
        rows_for_dedup, double_gloss_rows_changed = clean_double_glossing_rows(all_pairs)

    rows_before_dedup = len(rows_for_dedup)
    duplicate_pairs_wo_id_count = count_duplicate_pairs_wo_id(rows_for_dedup)

    deduped_rows, dedup_removed = deduplicate_rows(rows_for_dedup)
    dedup_pct = (dedup_removed / rows_before_dedup * 100.0) if rows_before_dedup else 0.0

    with output_csv.open('w', encoding='utf-8', newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=OUTPUT_HEADER)
        writer.writeheader()
        writer.writerows(deduped_rows)

    print('Distillation complete:')
    print(f'  Input file: {input_csv}')
    print(f'  Output file: {output_csv}')
    print(f'  Source rows read: {source_rows}')
    print(f'  Expanded rows (before cleanup): {expanded_rows_before_cleanup}')
    print(f'  Double-gloss cleanup enabled: {clean_double_glossing}')
    print(f'  Rows changed by double-gloss cleanup: {double_gloss_rows_changed}')
    print(f'  Rows before dedup: {rows_before_dedup}')
    print(f'  Output rows (after dedup): {len(deduped_rows)}')
    print(f'  Rows deduplicated: {dedup_removed} ({dedup_pct:.2f}%)')
    print(f'  Duplicate (full_sentence, glosses) pairs ignoring sentence_id: {duplicate_pairs_wo_id_count}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Distill sentence-level CSV into sentence_id/full_sentence/glosses with '
            'backtranslation expansion and post-build deduplication.'
        )
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('data/CSV_views/OSAM-DGS_sentences_dataset_bt.csv'),
        help='Path to source sentence-level CSV.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/CSV_views/OSAM-DGS_distilled_sentence_gloss_data_bt.csv'),
        help='Path to distilled output CSV.',
    )
    parser.add_argument(
        '--clean-double-glossing',
        dest='clean_double_glossing',
        action='store_true',
        help='Enable cleanup of double-gloss marker "||" in transcripts before deduplication.',
    )
    args = parser.parse_args()

    distill_csv(args.input, args.output, args.clean_double_glossing)


if __name__ == '__main__':
    main()