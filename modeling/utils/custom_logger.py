"""
Custom logger for the drone sound profile detection pipeline.
Writes to modeling_training.log in the repo root and to console.
No config file required.
"""

import logging
import os
import sys
from pathlib import Path

# Repo root (modeling/utils/ -> modeling/ -> repo root)
_SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPT_DIR.parent.parent
LOG_FILE_NAME = "modeling_training.log"
LOG_FILE_PATH = REPO_ROOT / LOG_FILE_NAME

# Logger name for this project
LOGGER_NAME = "modeling_training"
log_handler = logging.getLogger(LOGGER_NAME)
log_handler.setLevel(logging.INFO)

# Prevent propagation to root logger (avoid duplicate console lines)
log_handler.propagate = False

# Formatter: timestamp | level | message
log_format = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# File handler: append to modeling_training.log in repo root
file_handler = logging.FileHandler(LOG_FILE_PATH, mode="a", encoding="utf-8")
file_handler.setFormatter(log_format)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_format)

if not log_handler.hasHandlers():
    log_handler.addHandler(file_handler)
    log_handler.addHandler(console_handler)

# Example usage (from main.py or any script with repo root on path):
#
# from modeling.utils.custom_logger import log_handler
#
# log_handler.info("Pipeline starting")
# log_handler.warning("Something to watch")
# log_handler.error("Something failed")
