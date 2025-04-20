"""
Logging setup module for the file removal system.

This module provides centralized logging configuration with support for
file and console output, log rotation, and debug level control.

The logging configuration is controlled through the config system:
- LOGGING.LOG_FILE: Path to the log file (defaults to 'logs/remove_files.log')
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

from remove_files.config import get_config, Config

def setup_logging(
    LOG_FILE: str = "logs/remove_files.log",
    LOG_LEVEL: str = "INFO",
    DEBUG: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """Set up logging configuration with both file and console output.
    
    This function configures the root logger with both file and console handlers,
    supporting log rotation and debug level control. The file handler uses
    RotatingFileHandler to manage log file sizes.
    
    Args:
        LOG_FILE: Path to the log file (default: 'logs/remove_files.log')
        LOG_LEVEL: Logging level (default: "INFO")
        DEBUG: Enable debug logging (default: False)
        max_bytes: Maximum log file size before rotation (default: 10MB)
        backup_count: Number of backup files to keep (default: 5)
        
    Returns:
        None
        
    Example:
        >>> setup_logging()  # Uses default configuration
        >>> 
        >>> # Or with custom settings:
        >>> setup_logging(
        ...     LOG_FILE="custom.log",
        ...     LOG_LEVEL="DEBUG",
        ...     DEBUG=True
        ... )
    """
    # Create logs directory if it doesn't exist
    log_path = Path(LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create handlers
    file_handler = RotatingFileHandler(
        LOG_FILE,
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
    level = logging.DEBUG if DEBUG else getattr(logging, LOG_LEVEL.upper())
    root_logger.setLevel(level)
    
    # Log initial setup
    logging.info(f"File removal system logging initialized: level={level}, file={LOG_FILE}")

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name.
    
    Args:
        name: Name of the logger, typically __name__ of the calling module
        
    Returns:
        logging.Logger: Configured logger instance
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing file: example.pdf")
    """
    return logging.getLogger(name) 