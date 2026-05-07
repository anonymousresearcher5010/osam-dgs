from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import pipeline

from src.validation.validate_json_schema import validate_dataset
from src.utils.logging_utils import setup_logger

DEFAULT_NUM_BACKTRANSLATIONS = 2
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_CHECKPOINT_EVERY = 128
DEFAULT_BATCH_SIZE = 64
MAX_LENGTH = 512


def is_degenerate(
    sentence: str,
    min_words: int = 5,
    max_ngram: int = 10,
    repetition_threshold: float = 0.5,
) -> bool:
    words = sentence.split()
    if not words or len(words) < min_words:
        return False

    total_words = len(words)

    for n in range(1, min(max_ngram, total_words) + 1):
        ngrams = [' '.join(words[i : i + n]) for i in range(total_words - n + 1)]
        if not ngrams:
            continue

        counts = Counter(ngrams)
        _, freq = counts.most_common(1)[0]
        if freq > 1:
            proportion = (freq * n) / total_words
            if proportion > repetition_threshold:
                return True

    return False


def chunks(seq: list[int], size: int) -> list[list[int]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def back_translate_batch_once(
    source_texts: list[str],
    translator_de_en: Any,
    translator_en_de: Any,
    batch_size: int,
) -> list[str]:
    en_out = translator_de_en(
        source_texts,
        batch_size=batch_size,
        max_length=MAX_LENGTH,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        no_repeat_ngram_size=3,
        repetition_penalty=2.0,
        early_stopping=True,
    )
    en_texts = [item['translation_text'].strip() for item in en_out]

    de_out = translator_en_de(
        en_texts,
        batch_size=batch_size,
        max_length=MAX_LENGTH,
        no_repeat_ngram_size=3,
        repetition_penalty=2.0,
        early_stopping=True,
    )
    return [item['translation_text'].strip() for item in de_out]


def generate_backtranslations_batched(
    source_texts: list[str],
    existing_lists: list[list[str]],
    translator_de_en: Any,
    translator_en_de: Any,
    num_backtranslations: int,
    max_attempts: int,
    batch_size: int,
) -> list[list[str]]:
    n = len(source_texts)
    results: list[list[str]] = [[] for _ in range(n)]
    excludes: list[set[str]] = []

    for i in range(n):
        src = source_texts[i].strip()
        exclude = {src}
        for bt in existing_lists[i]:
            bt_clean = bt.strip()
            if bt_clean:
                exclude.add(bt_clean)
        excludes.append(exclude)

    attempts = [0] * n

    while True:
        active_indices = [
            i
            for i in range(n)
            if attempts[i] < max_attempts and len(results[i]) < num_backtranslations
        ]
        if not active_indices:
            break

        for idx_chunk in chunks(active_indices, batch_size):
            batch_inputs = [source_texts[i] for i in idx_chunk]
            batch_candidates = back_translate_batch_once(
                batch_inputs,
                translator_de_en=translator_de_en,
                translator_en_de=translator_en_de,
                batch_size=batch_size,
            )

            for i, candidate in zip(idx_chunk, batch_candidates, strict=True):
                attempts[i] += 1
                cand = candidate.strip()

                if not cand:
                    continue
                if cand in excludes[i]:
                    continue
                if is_degenerate(cand):
                    continue

                results[i].append(cand)
                excludes[i].add(cand)

    return results


def iter_sentences(data: dict[str, Any]) -> list[dict[str, Any]]:
    sentences_out: list[dict[str, Any]] = []

    conversations = data.get('conversations') or []
    for conversation in conversations:
        turns = conversation.get('turns') or []
        for turn in turns:
            sentences = turn.get('sentences') or []
            for sentence in sentences:
                if isinstance(sentence, dict):
                    sentences_out.append(sentence)

    return sentences_out


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')

    with tmp_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    tmp_path.replace(path)


def enrich_dataset_with_backtranslations(
    input_json: Path,
    output_json: Path,
    num_backtranslations: int,
    max_attempts: int,
    overwrite_existing: bool,
    checkpoint_every: int,
    batch_size: int,
    resume_from_output: bool,
    logger: logging.Logger,
) -> None:

    source_json = output_json if resume_from_output and output_json.exists() else input_json

    with source_json.open('r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info('Loaded dataset from: %s', source_json)

    device = 0 if torch.cuda.is_available() else -1
    logger.info('Loading translation pipelines (device=%s)...', device)

    translator_de_en = pipeline('translation_de_to_en', model='Helsinki-NLP/opus-mt-de-en', device=device)
    translator_en_de = pipeline('translation_en_to_de', model='Helsinki-NLP/opus-mt-en-de', device=device)

    sentences = iter_sentences(data)
    total = len(sentences)
    logger.info('Found %d sentence objects.', total)

    updated_count = 0
    skipped_count = 0
    processed_count = 0

    pending_sentence_indices: list[int] = []
    pending_sources: list[str] = []
    pending_existing: list[list[str]] = []

    def flush_pending() -> None:
        nonlocal updated_count, processed_count, pending_sentence_indices, pending_sources, pending_existing
        if not pending_sources:
            return

        generated_lists = generate_backtranslations_batched(
            source_texts=pending_sources,
            existing_lists=pending_existing,
            translator_de_en=translator_de_en,
            translator_en_de=translator_en_de,
            num_backtranslations=num_backtranslations,
            max_attempts=max_attempts,
            batch_size=batch_size,
        )

        for sentence_idx, generated in zip(pending_sentence_indices, generated_lists, strict=True):
            sentence = sentences[sentence_idx]
            sentence['back_translations'] = generated
            updated_count += 1
            processed_count += 1

            logger.info(
                'Processed %d/%d (generated=%d).',
                sentence_idx + 1,
                total,
                len(generated),
            )

            if checkpoint_every > 0 and processed_count % checkpoint_every == 0:
                write_json_atomic(output_json, data)
                logger.info('Checkpoint written at processed=%d/%d: %s', processed_count, total, output_json)

        pending_sentence_indices = []
        pending_sources = []
        pending_existing = []

    for index, sentence in enumerate(sentences):
        source_text = str(sentence.get('german_translation', '')).strip()
        existing_backtranslations = sentence.get('back_translations')

        if not source_text:
            skipped_count += 1
            logger.info('Skipped %d/%d (missing german_translation).', index + 1, total)
            continue

        if not overwrite_existing and isinstance(existing_backtranslations, list) and existing_backtranslations:
            skipped_count += 1
            logger.info('Skipped %d/%d (already has backtranslations).', index + 1, total)
            continue

        existing_list = existing_backtranslations if isinstance(existing_backtranslations, list) else []

        pending_sentence_indices.append(index)
        pending_sources.append(source_text)
        pending_existing.append(existing_list)

        if len(pending_sources) >= batch_size:
            flush_pending()

    flush_pending()

    logger.info('Done. Updated sentences: %d', updated_count)
    logger.info('Skipped sentences: %d', skipped_count)
    logger.info('Output written to: %s', output_json)

    logger.info('Saving final output...')
    write_json_atomic(output_json, data)

    logger.info('Validating output...')
    dataset_is_valid: bool = validate_dataset(data)
    logger.info(f'Validation finished. Dataset is valid: {str(dataset_is_valid).upper()}.')


def main() -> None:
    now_dt: datetime = datetime.now(timezone.utc)
    now_compact: str = now_dt.strftime('%Y%m%d%H%M%S')

    parser = argparse.ArgumentParser(description='Generate sentence-level backtranslations in canonical dataset JSON.')
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('data/canonical_dataset/OSAM-DGS_canonical_dataset.json'),
        help='Path to canonical dataset JSON.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/canonical_dataset/OSAM-DGS_canonical_dataset_with_backtranslations.json'),
        help='Path to output JSON.',
    )
    parser.add_argument(
        '--num-backtranslations',
        type=int,
        default=DEFAULT_NUM_BACKTRANSLATIONS,
        help='Number of backtranslations to generate per sentence.',
    )
    parser.add_argument(
        '--max-attempts',
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help='Maximum attempts per sentence to obtain unique/non-degenerate outputs.',
    )
    parser.add_argument(
        '--overwrite-existing',
        action='store_false',
        help='If set, existing sentence["backtranslations"] values are overwritten.',
    )
    parser.add_argument(
        '--checkpoint-every',
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help='Write intermediate output every N processed sentence objects (0 disables checkpoints).',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help='Batch size for GPU inference through HF pipelines.',
    )
    parser.add_argument(
        '--resume-from-output',
        action='store_true',
        help='If set and output exists, resume from output file instead of input file.',
    )
    parser.add_argument(
        '--logs-path',
        type=Path,
        default=Path('data/canonical_dataset/backtranslation_generation.log'),
        help='Path to log file.',
    )

    args = parser.parse_args()

    logger_path: Path = args.logs_path.parent / f'{args.logs_path.stem}_{now_compact}.log'
    logger_path.parent.mkdir(parents=True, exist_ok=True)
    logger: logging.Logger = setup_logger(logger_path, debug=True)

    enrich_dataset_with_backtranslations(
        input_json=args.input,
        output_json=args.output,
        num_backtranslations=args.num_backtranslations,
        max_attempts=args.max_attempts,
        overwrite_existing=args.overwrite_existing,
        checkpoint_every=args.checkpoint_every,
        batch_size=args.batch_size,
        resume_from_output=args.resume_from_output,
        logger=logger,
    )


if __name__ == '__main__':
    main()