import sys
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(log_path: Path, overwrite_log_file: bool = True, debug: bool = False ) -> logging.Logger:
    """
    Set up logging to both console and file.

    param log_path:             Path to the log file.
    param overwrite_log_file:   If True, overwrite the log file if it already exists.

    return: The logger object.
    """
    logger: logging.Logger = logging.getLogger()

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)

    if overwrite_log_file and log_path.exists():
        log_path.unlink()

    file = RotatingFileHandler(
        log_path,
        maxBytes=5_000_000,  # create a new log file when reaching 5 MB per file
        backupCount=0,       # unlimited log files
    )
    file.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file)

    return logger
