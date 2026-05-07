import json

from jsonschema import Draft202012Validator, ValidationError
from jsonschema.protocols import Validator

from src.data_processing.canonical_dataset_creation import logger


def validate_dataset(data: dict) -> bool:
    """
    Validation of the created data
    """
    schema_path: str = 'configs/canonical_schema.json'

    with open(schema_path) as f:
        data_schema = json.load(f)

    validator: Validator = Draft202012Validator(data_schema)
    errors: list[ValidationError] = list(validator.iter_errors(data))

    if len(errors) > 0:
        logger.error(f'Validation errors occurred ({len(errors)} total):')
        for error in errors:
            error_path = '.'.join(str(p) for p in error.absolute_path)
            logger.error(f'  Path: {error_path}')
            logger.error(f'  Message: {error.message}')
            logger.error(f'  Schema path: {list(error.absolute_schema_path)}')
            logger.error(f'  Failed value: {error.instance}')
            logger.error('---')
        return False

    return True
