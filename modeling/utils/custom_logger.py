"""
Custom logger for the drone sound profile detection pipeline.
- main.py uses log_handler (writes to modeling_training.log in repo root).
- Other scripts use get_logger(name, log_filename) to write to logs/<log_filename>
  with the same format. Run scripts from repo root so "modeling" is importable.
"""

import logging
import sys
from pathlib import Path

# Repo root (modeling/utils/ -> modeling/ -> repo root)
_SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPT_DIR.parent.parent
LOGS_DIR = REPO_ROOT / "logs"


class _PipelineFormatter(logging.Formatter):
    """Shared formatter: timestamp with milliseconds | level | message."""
    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        t = "%04d-%02d-%02d %02d:%02d:%02d,%03d" % (
            ct.tm_year, ct.tm_mon, ct.tm_mday,
            ct.tm_hour, ct.tm_min, ct.tm_sec,
            int(record.created * 1000) % 1000,
        )
        return t


# One formatter for all loggers
_LOG_FORMAT = _PipelineFormatter("%(asctime)s | %(levelname)s | %(message)s")


def get_logger(logger_name: str, log_filename: str | None = None) -> logging.Logger:
    """
    Return a logger that writes to a file (and console) with the shared format.
    - logger_name: unique name for the logger (e.g. "tiny_cnn_logger").
    - log_filename: name of the log file under logs/ (e.g. "tiny_cnn_training.log").
      If None, writes to modeling_training.log in repo root (for main.py).
    Log files are created/overwritten in logs/ (or repo root when log_filename is None).
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.hasHandlers():
        return logger

    if log_filename is None:
        log_path = REPO_ROOT / "modeling_training.log"
        file_mode = "a"
    else:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOGS_DIR / log_filename
        file_mode = "w"

    file_handler = logging.FileHandler(log_path, mode=file_mode, encoding="utf-8")
    file_handler.setFormatter(_LOG_FORMAT)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(_LOG_FORMAT)
    logger.addHandler(console_handler)

    return logger


# Default logger for main.py: writes to modeling_training.log in repo root
log_handler = get_logger("modeling_training", log_filename=None)
