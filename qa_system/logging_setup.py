"""Logging setup module for the QA system."""

import logging
import logging.handlers
import os
from pathlib import Path

class LineBufferedRotatingFileHandler(logging.handlers.RotatingFileHandler):
    def _open(self):
        return open(self.baseFilename, self.mode, encoding=self.encoding, buffering=1)

def setup_logging(LOG_FILE: str = "logs/qa_system.log", LEVEL: str = "INFO") -> None:
    """Set up logging configuration.
    
    Args:
        LOG_FILE: Path to log file
        LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) for console
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"Called setup_logging(LOG_FILE={LOG_FILE}, LEVEL={LEVEL})")
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Set numeric level from LEVEL argument (default INFO)
    numeric_level = getattr(logging, LEVEL.upper(), logging.INFO)
    
    # Always capture all messages in the logfile (DEBUG and above)
    logging.root.setLevel(logging.DEBUG)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Set up file handler with rotation (always DEBUG level, line-buffered)
    file_handler = LineBufferedRotatingFileHandler(
        LOG_FILE,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    # Add a filter to flush after every log record
    class FlushOnWriteFilter(logging.Filter):
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
    
    # Create logger for this module
    logger = logging.getLogger(__name__)
    if numeric_level <= logging.INFO:
        logger.info(f"Logging setup complete (level {LEVEL})") 