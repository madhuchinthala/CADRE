"""
Structured Logger for CADRE
============================
Provides consistent, colored logging across all modules.
Logs to both console and file.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str = "cadre",
    log_dir: str = "outputs/logs",
    level: int = logging.INFO,
    log_to_file: bool = True,
) -> logging.Logger:
    """
    Set up a logger with console and file handlers.

    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        log_to_file: Whether to also log to a file

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # ── Console handler ──
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_fmt = logging.Formatter(
        "%(asctime)s │ %(name)-20s │ %(levelname)-8s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # ── File handler ──
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            log_path / f"cadre_{timestamp}.log",
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter(
            "%(asctime)s │ %(name)s │ %(levelname)s │ %(filename)s:%(lineno)d │ %(message)s",
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

    return logger
