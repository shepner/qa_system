"""
Logging setup module for the QA System.

This module provides centralized logging configuration with support for
file and console output, log rotation, and debug level control.

The logging configuration can be controlled through the config system:
- LOGGING.LOG_FILE: Path to the log file
- LOGGING.LEVEL: Logging level (default: "INFO")
- LOGGING.DEBUG: Enable debug logging (default: False)
- LOGGING.MAX_BYTES: Maximum log file size before rotation (default: 10MB)
- LOGGING.BACKUP_COUNT: Number of backup files to keep (default: 5)
"""

import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
from typing import Optional

# ANSI color codes for console output
COLORS = {
    'DEBUG': '\033[0;36m',    # Cyan
    'INFO': '\033[0;32m',     # Green
    'WARNING': '\033[0;33m',  # Yellow
    'ERROR': '\033[0;31m',    # Red
    'CRITICAL': '\033[0;35m', # Magenta
    'RESET': '\033[0m',       # Reset
}

class ColorFormatter(logging.Formatter):
    """Custom formatter adding colors to levelname for console output."""
    
    def format(self, record):
        # Add color to levelname if color is available for this level
        if record.levelname in COLORS:
            record.levelname = f"{COLORS[record.levelname]}{record.levelname}{COLORS['RESET']}"
        return super().format(record)

def setup_logging(
    log_file: str,
    log_level: str = "INFO",
    enable_debug: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB default
    backup_count: int = 5
) -> None:
    """Set up logging configuration with both file and console output.
    
    This function configures the root logger with both file and console handlers,
    supporting log rotation and debug level control. The file handler uses
    RotatingFileHandler to manage log file sizes.
    
    Args:
        log_file: Path to the log file. Will create parent directories if needed.
        log_level: Logging level (default: "INFO"). Used if enable_debug is False.
        enable_debug: Whether to enable debug logging (default: False).
                     If True, overrides log_level to DEBUG.
        max_bytes: Maximum size of each log file in bytes (default: 10MB).
                  When exceeded, the file is rotated.
        backup_count: Number of backup files to keep (default: 5).
                     Older files are removed.
        
    Returns:
        None
        
    Example:
        >>> from embed_files.config import get_config
        >>> config = get_config()
        >>> setup_logging(
        ...     log_file=config.get_nested('LOGGING.LOG_FILE'),
        ...     log_level=config.get_nested('LOGGING.LEVEL', default="INFO"),
        ...     enable_debug=config.get_nested('LOGGING.DEBUG', default=False),
        ...     max_bytes=config.get_nested('LOGGING.MAX_BYTES', default=10*1024*1024),
        ...     backup_count=config.get_nested('LOGGING.BACKUP_COUNT', default=5)
        ... )
        2024-03-20 14:30:45 - embed_files.logging_setup - INFO - Logging initialized: level=20, file=./logs/app.log
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create handlers
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    console_handler = logging.StreamHandler()
    
    # Set different formats for file and console
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = ColorFormatter('%(levelname)s - %(message)s')
    
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers = []  # Remove any existing handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Set log level
    level = logging.DEBUG if enable_debug else getattr(logging, log_level.upper())
    root_logger.setLevel(level)
    
    # Log initial setup
    logging.info(f"Logging initialized: level={level}, file={log_file}")