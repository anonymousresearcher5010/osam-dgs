"""Entry point for retrieving the public DGS Korpus Release 3 data."""
import yaml
from pathlib import Path
from datetime import datetime
import logging

from src.data_processing.types import RetrievalConfig
from src.retrieval import dgs_data_retriever
from src.utils.logging_utils import setup_logger


def main():

    # Load configuration from YAML file
    with open('configs/retrieving_data_config.yaml') as yaml_file:
        cfg_dict = yaml.safe_load(yaml_file)

    # Create config object from config dictionary
    cfg = RetrievalConfig(
        url=cfg_dict['url'],
        data_dir=Path(cfg_dict['data_dir']),
        SRT_required=cfg_dict['SRT_required'],
        max_conversations=cfg_dict['max_conversations'],
        retrieve_video_ab=cfg_dict['retrieve_video_ab'],
        retrieve_video_long_shot=cfg_dict['retrieve_video_long_shot'],
        retrieve_ilex=cfg_dict['retrieve_ilex'],
        base_url=cfg_dict['base_url'],
        request_timeout=cfg_dict['request_timeout'],
        delay_seconds=cfg_dict['delay_seconds'],
        user_agent=cfg_dict['user_agent'],
        overwrite_logs=cfg_dict['overwrite_logs']
    )

    now: str = datetime.now().strftime('%Y%m%d%H%M%S')
    logger_path: Path = cfg.data_dir / f'DGS_Korpus_data_retrieval_{now}.log'
    logger_path.parent.mkdir(parents=True, exist_ok=True)
    logger: logging.Logger = setup_logger(logger_path, overwrite_log_file=cfg.overwrite_logs)

    logger.info('Retrieving public DGS Korpus Data...')

    # Retrieve data
    result_stats: dict = dgs_data_retriever.retrieve(cfg)

    logger.info(f'Retrieval finished. Stats: {result_stats}')

if __name__ == '__main__':
    main()
