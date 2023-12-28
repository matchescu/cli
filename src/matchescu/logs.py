import logging
import os
import sys
from typing import Optional


def _configure_logging():
    level = os.environ.get("LOG_LEVEL", logging.INFO)
    logging.basicConfig(stream=sys.stdout, level=level)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    _configure_logging()
    return logging.getLogger(name) if name else logging.root
