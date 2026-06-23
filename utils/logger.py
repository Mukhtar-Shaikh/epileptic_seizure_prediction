import logging
import os
from datetime import datetime
from utils.config import RESULTS_DIR

LOG_DIR = os.path.join(RESULTS_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

_log_filename = os.path.join(LOG_DIR, f"eeg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")


def get_logger(name: str = "EEG") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(_log_filename)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


logger = get_logger("EEG")
