import logging
from colorlog import ColoredFormatter
from pathlib import Path


def get_debug_logger(name_str):
    # Logging (for debug)
    db_logger = logging.getLogger(name_str)
    db_logger.setLevel(logging.DEBUG)
    db_logger.handlers = []  # No duplicated handlers  
    db_logger.propagate = False  # workaround for duplicated logs in ipython

    # stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)

    stream_formatter = ColoredFormatter(
        "%(log_color)s[%(asctime)s][%(process)05d] %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "white,bold",
            "INFOV": "cyan,bold",
            "WARNING": "yellow",
            "ERROR": "red,bold",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={},
        style="%",
    )
    stream_handler.setFormatter(stream_formatter)
    db_logger.addHandler(stream_handler)

    # file handler
    file_handler = logging.FileHandler(f"debug_warnings.log")
    file_handler.setLevel(logging.INFO)
    
    file_formatter = logging.Formatter(
        "[%(asctime)s][%(process)05d][%(name)s][%(levelname)s] %(message)s",
        datefmt=None,
        style="%",
    )
    file_handler.setFormatter(file_formatter)
    db_logger.addHandler(file_handler)

    return db_logger