import logging
import logging.handlers
import sys

from rich.logging import RichHandler

from .constants import LOG_FILEPATH

RICH_FORMAT = "| %(filename)s:%(lineno)s | %(message)s"
FILE_HANDLER_FORMAT = "[%(asctime)s]\t%(levelname)s\t| %(filename)s:%(lineno)s | %(message)s"

def get_file_handler(log_path: str = LOG_FILEPATH, level: int = logging.INFO):
    fh = logging.handlers.RotatingFileHandler(
        os.path.join(log_path, "gen_rate_ported.log"),
        maxBytes=10_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(FILE_HANDLER_FORMAT))
    return fh

def get_rich_handler(level: int = logging.INFO):
    rh = RichHandler(rich_tracebacks=True, markup=False)
    rh.setLevel(level)
    rh.setFormatter(logging.Formatter(RICH_FORMAT))
    return rh

def set_logger(log_path: str = LOG_FILEPATH, level: int = logging.INFO):
    logger = logging.getLogger("rich")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(get_file_handler(log_path, level))
    logger.addHandler(get_rich_handler(level))
    return logger

def get_logger():
    if logging.root.manager.loggerDict:
        return logging.getLogger("rich")
    return set_logger()

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger = get_logger()
    logger.error("Unexpected exception", exc_info=(exc_type, exc_value, exc_traceback))
    logger.error("Unexpected exception caught!")
