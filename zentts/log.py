"""Package logger. Set LOG_LEVEL=DEBUG for verbose engine output."""

import logging
import os


def _create_logger() -> logging.Logger:
    logger = logging.getLogger("zentts")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s")
        )
        logger.addHandler(handler)
    level = os.getenv("LOG_LEVEL", "WARNING").upper()
    logger.setLevel(getattr(logging, level, logging.WARNING))
    return logger


log = _create_logger()
