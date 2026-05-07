import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(
        description='Deterministic split by sentence_id (no sentence_id overlap across splits).'
    )
    parser.add_argument('--input_csv', required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--id_column', type=str, default='sentence_id')
    parser.add_argument(
        '--output_prefix',
        type=str,
        default='',
        help='Optional prefix for output split file names, e.g. my_dataset -> my_dataset_train.csv',
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    if args.id_column not in df.columns:
        raise ValueError(
            f'Required id column "{args.id_column}" not found in CSV columns: {list(df.columns)}'
        )

    unique_ids = (
        df[args.id_column]
        .dropna()
        .drop_duplicates()
        .to_numpy(dtype=object)
    )

    train_ids, temp_ids = train_test_split(
        unique_ids,
        test_size=0.2,
        random_state=args.seed,
    )
    dev_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.5,
        random_state=args.seed + 1,
    )

    train_df = df[df[args.id_column].isin(train_ids)].copy()
    dev_df = df[df[args.id_column].isin(dev_ids)].copy()
    test_df = df[df[args.id_column].isin(test_ids)].copy()

    train_set = set(train_df[args.id_column].dropna().unique())
    dev_set = set(dev_df[args.id_column].dropna().unique())
    test_set = set(test_df[args.id_column].dropna().unique())

    assert train_set.isdisjoint(dev_set), 'Leakage detected: train/dev sentence_id overlap'
    assert train_set.isdisjoint(test_set), 'Leakage detected: train/test sentence_id overlap'
    assert dev_set.isdisjoint(test_set), 'Leakage detected: dev/test sentence_id overlap'

    prefix = f'{args.output_prefix}_' if args.output_prefix else ''
    train_path = output_dir / f'{prefix}train.csv'
    dev_path = output_dir / f'{prefix}dev.csv'
    test_path = output_dir / f'{prefix}test.csv'

    train_df.to_csv(train_path, index=False)
    dev_df.to_csv(dev_path, index=False)
    test_df.to_csv(test_path, index=False)

    print('Split complete (by sentence_id):')
    print(f'  train rows: {len(train_df)} | unique sentence_id: {len(train_set)}')
    print(f'  dev rows:   {len(dev_df)} | unique sentence_id: {len(dev_set)}')
    print(f'  test rows:  {len(test_df)} | unique sentence_id: {len(test_set)}')
    print(f'  train path: {train_path}')
    print(f'  dev path:   {dev_path}')
    print(f'  test path:  {test_path}')


if __name__ == "__main__":
    main()