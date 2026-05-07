from dataclasses import asdict
from pathlib import Path
from datetime import datetime, timezone
import logging

from src.data_processing.canonical_dataset_creation import produce_canonical_data
from src.validation.validate_json_schema import validate_dataset
from src.data_processing.processing_io_utils import persist_dict_as_json
from src.data_processing.types import CanonicalData
from src.utils.logging_utils import setup_logger


def main():
    now_dt: datetime = datetime.now(timezone.utc)
    now_compact: str = now_dt.strftime('%Y%m%d%H%M%S')  # for file names
    now_rfc3339: str = now_dt.isoformat(timespec='seconds').replace('+00:00', 'Z')  # for JSON schema date-time

    dataset_dir: Path = Path('data/dgs_korpus_release_3_raw')
    canonical_dataset_path: Path = Path('data/canonical_dataset')
    canonical_file_name: str = 'OSAM-DGS_canonical_dataset.json'

    canonical_dataset_path.mkdir(parents=True, exist_ok=True)

    logger_path: Path = canonical_dataset_path / f'OSAM-DGS_canonical_dataset_creation_{now_compact}.log'

    # logger_path.parent.mkdir(parents=True, exist_ok=True)
    logger: logging.Logger = setup_logger(logger_path, debug=True)

    logger.info(f'Creating canonical dataset from {dataset_dir}...')

    # Creation of the canonical dataset
    canonical_data: CanonicalData = produce_canonical_data(dataset_dir, now_rfc3339)
    canonical_data_as_dict = asdict(canonical_data)
    logger.info(f'Creation finished.')

    # Validation of the created data
    logger.info('Validating canonical dataset...')
    dataset_is_valid: bool = validate_dataset(canonical_data_as_dict)
    logger.info(f'Validation finished. Dataset is valid: {str(dataset_is_valid).upper()}.')

    logger.info('Persisting canonical dataset...')
    canonical_dataset_path.mkdir(parents=True, exist_ok=True)
    persist_dict_as_json(canonical_data_as_dict, canonical_dataset_path / canonical_file_name)
    logger.info('Persisting canonical dataset...')

    logger.info('Process finished.')


if __name__ == '__main__':
    main()
