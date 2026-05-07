import yaml
from pathlib import Path
from datetime import datetime
import logging

from src.data_processing.types import RetrieveTypesConfig
from src.retrieval import dgs_types_retriever
from src.utils.logging_utils import setup_logger


def main():

    # Load configuration from YAML file
    with open('configs/retrieving_types_config.yaml') as yaml_file:
        cfg_dict = yaml.safe_load(yaml_file)

    # Create config object from config dictionary
    cfg = RetrieveTypesConfig(
        url=cfg_dict['url'],
        data_dir=Path(cfg_dict['data_dir']),
        output_file=cfg_dict['output_file'],
        request_timeout=cfg_dict['request_timeout'],
        user_agent=cfg_dict['user_agent'],
    )

    now: str = datetime.now().strftime('%Y%m%d%H%M%S')
    logger_path: Path = cfg.data_dir / f'DGS_Korpus_types_retrieval_{now}.log'

    logger_path.parent.mkdir(parents=True, exist_ok=True)
    logger: logging.Logger = setup_logger(logger_path)

    logger.info('Retrieving DGS Korpus Types...')

    # Retrieve data
    result_stats: dict = dgs_types_retriever.retrieve(cfg)

    logger.info(f'Retrieving Types finished. Stats: {result_stats}')

if __name__ == '__main__':
    main()
