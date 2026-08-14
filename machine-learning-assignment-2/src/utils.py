# utils.py
import logging

from src.config import LOG_DIR

def setup_logger(logger_name):
    LOG_DIR.mkdir(parents=True,exist_ok=True)
    logger = logging.getLogger(logger_name)

    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    stream_handler = logging.StreamHandler()
    file_path = LOG_DIR / "application.log"
    file_handler = logging.FileHandler(filename=file_path,encoding="utf-8")

    # date -time | level | logger_name | message
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    formatter = logging.Formatter(fmt=log_format)

    stream_handler.setFormatter(fmt=formatter)
    file_handler.setFormatter(fmt=formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger

"""
python -c ""
"from src.utils import setup_logger; "
"logger = setup_logger('test'); "
"logger.info('Logger test successful.')"
""
"""