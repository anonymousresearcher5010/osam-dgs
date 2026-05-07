from __future__ import annotations

import argparse
import csv
from pathlib import Path

OUTPUT_HEADER = ['sentence_id', 'full_sentence', 'glosses']
DEFAULT_INPUT_FILES = [
    'PHOENIX-2014-T.train.corpus.csv',
    'PHOENIX-2014-T.test.corpus.csv',
    'PHOENIX-2014-T.dev.corpus.csv',
]


def distilled_output_name(input_name: str) -> str:
    if input_name.endswith('.csv'):
        return input_name[:-4] + '_distilled.csv'
    return input_name + '_distilled.csv'


def distill_phoenix_file(input_csv: Path, output_csv: Path) -> None:
    with input_csv.open('r', encoding='utf-8', newline='') as fin:
        reader = csv.DictReader(fin, delimiter='|')
        fieldnames = set(reader.fieldnames or [])

        required = {'name', 'translation'}
        missing = required - fieldnames
        if missing:
            raise ValueError(
                f'Input CSV "{input_csv}" is missing required columns: {sorted(missing)}. '
                f'Found: {reader.fieldnames}'
            )

        gloss_col = 'ortho' if 'ortho' in fieldnames else ('orth' if 'orth' in fieldnames else None)
        if gloss_col is None:
            raise ValueError(
                f'Input CSV "{input_csv}" must contain "ortho" or "orth" column. '
                f'Found: {reader.fieldnames}'
            )

        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append(
                {
                    'sentence_id': row.get('name', ''),
                    'full_sentence': row.get('translation', ''),
                    'glosses': row.get(gloss_col, ''),
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open('w', encoding='utf-8', newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=OUTPUT_HEADER)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Created: {output_csv} ({len(rows)} rows)')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Create distilled PHOENIX CSVs with sentence_id/full_sentence/glosses.'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('data/CSV_views'),
        help='Directory containing PHOENIX-2014-T.*.corpus.csv files.',
    )
    args = parser.parse_args()

    for file_name in DEFAULT_INPUT_FILES:
        input_csv = args.input_dir / file_name
        output_csv = args.input_dir / distilled_output_name(file_name)

        if not input_csv.exists():
            print(f'Skipping missing file: {input_csv}')
            continue

        distill_phoenix_file(input_csv, output_csv)


if __name__ == '__main__':
    main()