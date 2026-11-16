"""
logger.py  [One logging setup]
==============================
Purpose   Configure logging ONCE: time, level, module name, to the console and (optionally) a log file
Called by main.py and api.py at start-up (only the entry points)
Used by   every module through: logger = logging.getLogger(__name__)
Settings  config.yaml (logging.level, paths.log_file)

Why it matters
- Every message has a time, a severity and a source, so a run can be audited afterwards
- Console lines show up in Docker and on the cloud host; the file keeps the trail of each local run

Tests     tests/test_logger.py
"""

import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(level: str, log_file: Path | None = None) -> None:
    """Send every module's log lines to the terminal and, if given, to one log file.

    Inputs:  level, e.g. "INFO" (from config.yaml); log_file, e.g. logs/pipeline.log, or None for console only.
    Outputs: None. The root logger gets one format and its handlers, replacing any earlier set-up.
    Why:     modules only call logging.getLogger(__name__); where lines go is decided once, here.
    Raises:  ValueError if the level is not a logging level name.
    """
    name = str(level).upper()
    if name not in logging.getLevelNamesMapping():
        raise ValueError(f"logging.level is {level!r}. Use DEBUG, INFO, WARNING, ERROR or CRITICAL.")
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=name, format=LOG_FORMAT, handlers=handlers, force=True)
