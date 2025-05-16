"""
@file: logging_setup.py
Logging setup module for the QA system.

This module provides a function to configure logging for the application, including:
- Rotating file logging with line buffering
- Console logging at a configurable level
- Automatic log directory creation
- Ensures all log records are flushed to disk immediately

Usage:
    from qa_system.logging_setup import setup_logging
    setup_logging()

"""

import logging
import logging.handlers
import os
from pathlib import Path

class LineBufferedRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """
    RotatingFileHandler with line buffering enabled.
    Ensures each log record is flushed to disk immediately.
    """
    def _open(self):
        return open(self.baseFilename, self.mode, encoding=self.encoding, buffering=1)


def setup_logging(LOG_FILE: str = "logs/qa_system.log", LEVEL: str = None, config_path: str = None) -> None:
    """
    Set up logging configuration for the QA system.

    This configures both file and console logging:
    - File logs are written to LOG_FILE, rotated at 10MB, with 5 backups, at the configured LEVEL.
    - Console logs are written at the specified LEVEL (default: INFO).
    - Log directory is created if it does not exist.
    - All file log records are flushed immediately for reliability.

    Args:
        LOG_FILE: Path to the log file (default: 'logs/qa_system.log')
        LEVEL: Logging level for output (DEBUG, INFO, WARNING, ERROR, CRITICAL). If None, will read from config.yaml LOGGING.LEVEL.
        config_path: Optional path to config.yaml to use for loading the log level if LEVEL is None.
    """
    if LEVEL is None:
        try:
            from qa_system.config import get_config
            config = get_config(config_path)
            LEVEL = config.get_nested('LOGGING.LEVEL', 'INFO')
        except Exception:
            LEVEL = 'INFO'

    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Set numeric level from LEVEL argument (default INFO)
    numeric_level = getattr(logging, LEVEL.upper(), logging.INFO)

    # Set root logger to the configured level
    logging.root.setLevel(numeric_level)

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )

    # Set up file handler with rotation (at configured level, line-buffered)
    file_handler = LineBufferedRotatingFileHandler(
        LOG_FILE,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(numeric_level)

    class FlushOnWriteFilter(logging.Filter):
        """
        Logging filter that flushes the file handler after every log record.
        """
        def filter(self, record):
            file_handler.flush()
            return True
    file_handler.addFilter(FlushOnWriteFilter())

    # Set up console handler (user-specified level)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    logging.root.handlers = []

    # Add handlers to root logger
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)

    # Log setup completion at INFO level or lower
    if numeric_level <= logging.INFO:
        logging.getLogger(__name__).info(f"Logging setup complete (level {LEVEL})") 