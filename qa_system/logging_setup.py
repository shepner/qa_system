"""Centralized logging configuration with support for file and console output.

This module provides logging configuration with the following features:
- File output with rotation
- Color-coded console output
- Debug level control
- Structured logging with extra context

Configuration options:
- LOGGING.LOG_FILE: Path to log file (default: logs/qa_system.log)
- LOGGING.LEVEL: Logging level (default: INFO)
- LOGGING.DEBUG: Enable debug logging (default: False)
- LOGGING.MAX_BYTES: Maximum log file size before rotation (default: 10MB)
- LOGGING.BACKUP_COUNT: Number of backup files to keep (default: 5)
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Optional

# ANSI color codes for different log levels
COLORS = {
    'DEBUG': '\033[36m',    # Cyan
    'INFO': '\033[32m',     # Green
    'WARNING': '\033[33m',  # Yellow
    'ERROR': '\033[31m',    # Red
    'CRITICAL': '\033[35m', # Magenta
    'RESET': '\033[0m'      # Reset
}

class ColorFormatter(logging.Formatter):
    """Custom formatter adding colors to log levels in console output."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors for console output.
        
        Args:
            record: Log record to format
            
        Returns:
            Formatted log message with color codes
        """
        # Add color to level name if color is available
        if record.levelname in COLORS:
            record.levelname = f"{COLORS[record.levelname]}{record.levelname}{COLORS['RESET']}"
        
        # Format timestamp for consistency
        record.asctime = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        
        # Format message with extra context if available
        msg = record.getMessage()
        if hasattr(record, 'extra'):
            extra_str = ' '.join(f"{k}={v}" for k, v in record.extra.items())
            msg = f"{msg}: {extra_str}"
        
        # Construct the final message
        return f"{record.asctime} - {record.name} - {record.levelname} - {msg}"

def setup_logging(
    log_file: str = "logs/qa_system.log",
    log_level: str = "INFO",
    enable_debug: bool = False
) -> None:
    """Set up logging configuration with file and console output.
    
    Args:
        log_file: Path to log file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_debug: Override to enable debug logging
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Set debug level if enabled
    if enable_debug:
        log_level = "DEBUG"
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    formatter = ColorFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create and configure file handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Create and configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Log initialization
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging initialized",
        extra={
            'component': 'logging',
            'log_file': log_file,
            'log_level': log_level
        }
    )