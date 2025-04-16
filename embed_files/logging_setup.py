"""
Logging setup module for the QA System.

This module provides centralized logging configuration with support for
file and console output, log rotation, and debug level control.
"""

import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
from typing import Optional

def setup_logging(
    log_file: str,
    log_level: str = "INFO",
    enable_debug: bool = False,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5
) -> None:
    """Set up logging configuration with both file and console output.
    
    Args:
        log_file: Path to the log file
        log_level: Logging level (default: INFO)
        enable_debug: Whether to enable debug logging (default: False)
        max_bytes: Maximum size of each log file in bytes (default: 10MB)
        backup_count: Number of backup files to keep (default: 5)
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
    
    # Set format for both handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
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