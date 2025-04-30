"""Logging setup module for qa_system.

This module provides centralized logging configuration with support for:
- File and console output
- Log rotation
- Debug level control
- Colored console output
"""

import os
import json
import time
import logging
import logging.handlers
import threading
from typing import Optional, Callable, TypeVar, Any, Dict
from pathlib import Path
from functools import wraps

# Type variable for generic function type
F = TypeVar('F', bound=Callable[..., Any])

# ANSI color codes for console output
COLORS = {
    'DEBUG': '\033[0;36m',    # Cyan
    'INFO': '\033[0;32m',     # Green
    'WARNING': '\033[0;33m',  # Yellow
    'ERROR': '\033[0;31m',    # Red
    'CRITICAL': '\033[0;35m', # Magenta
    'RESET': '\033[0m'        # Reset
}

class StructuredFormatter(logging.Formatter):
    """Base formatter that handles structured logging with extra context."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the specified record.
        
        Args:
            record: LogRecord instance to format
            
        Returns:
            Formatted string with optional context
        """
        # Thread-safe copy of the record
        record_copy = logging.makeLogRecord(record.__dict__)
        
        # Format the basic message first
        message = super().format(record_copy)
        
        # Extract extra context that's not part of the standard LogRecord
        # Create a thread-safe copy of the record attributes
        record_dict = dict(record_copy.__dict__)
        
        extra_context = {
            key: value for key, value in record_dict.items()
            if key not in logging.LogRecord.__dict__ and
               key not in ('args', 'exc_info', 'exc_text', 'stack_info', 'msg', 'message')
        }
        
        # Add context if present, with safe JSON serialization
        if extra_context:
            try:
                context_str = json.dumps(extra_context, default=str)
                message = f"{message} | Context: {context_str}"
            except Exception as e:
                # If JSON serialization fails, add context in a simple format
                context_str = ", ".join(f"{k}={v!r}" for k, v in extra_context.items())
                message = f"{message} | Context: {context_str}"
                
        return message

class ColoredFormatter(StructuredFormatter):
    """Formatter adding colors to levelname for console output."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the specified record with colored output.
        
        Args:
            record: LogRecord instance to format
            
        Returns:
            Formatted string with colored level name
        """
        # Create a copy of the record to avoid modifying the original
        record_copy = logging.makeLogRecord(record.__dict__)
        
        # Add color to levelname if it exists in our color mapping
        if record_copy.levelname in COLORS:
            record_copy.levelname = f"{COLORS[record_copy.levelname]}{record_copy.levelname}{COLORS['RESET']}"
        return super().format(record_copy)

def setup_logging(LOG_FILE: str = "logs/qa_system.log", LEVEL: str = "INFO", debug: bool = False) -> None:
    """Set up logging configuration with file and console handlers.
    
    Args:
        LOG_FILE: Path to the log file (default: "logs/qa_system.log")
        LEVEL: Logging level (default: "INFO")
        debug: Flag to enable debug logging (overrides LEVEL setting)
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Set the base logging level
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug else getattr(logging, LEVEL.upper()))
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    file_formatter = StructuredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = ColoredFormatter('%(levelname)s - %(message)s')
    
    try:
        # Set up file handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG if debug else getattr(logging, LEVEL.upper()))
        
        # Set up console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.DEBUG if debug else getattr(logging, LEVEL.upper()))
        
        # Add handlers to root logger
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Disable propagation for the qa_system logger to prevent duplicate logs
        qa_logger = logging.getLogger('qa_system')
        qa_logger.propagate = False
        
        # Log initial message
        logging.getLogger(__name__).info("Logging initialized")
        
    except Exception as e:
        # Ensure we have at least console logging in case of file handler setup failure
        root_logger.addHandler(console_handler)
        logging.getLogger(__name__).error(f"Failed to setup file logging: {str(e)}")

def enable_debug_logging() -> None:
    """Temporarily enable debug logging."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    for handler in root_logger.handlers:
        handler.setLevel(logging.DEBUG)
    logging.getLogger(__name__).debug("Debug logging enabled")

def reset_logging_level(level: str = "INFO") -> None:
    """Reset to configured logging level.
    
    Args:
        level: Logging level to reset to (default: "INFO")
    """
    try:
        log_level = getattr(logging, level.upper())
    except AttributeError:
        log_level = logging.INFO
        logging.getLogger(__name__).warning(f"Invalid log level '{level}', using INFO")
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    for handler in root_logger.handlers:
        handler.setLevel(log_level)
    logging.getLogger(__name__).info(f"Logging level reset to {level}")

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance with the specified name.
    
    Args:
        name: Logger name (default: None, returns root logger)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    if name and name.startswith('qa_system'):
        logger.propagate = False
    return logger

def log_performance(logger: Optional[logging.Logger] = None) -> Callable[[F], F]:
    """Decorator to log function execution time.
    
    Args:
        logger: Logger instance to use (default: None, uses root logger)
        
    Returns:
        Decorator function that logs performance metrics
        
    Example:
        @log_performance()
        def process_document(path: str):
            # Processing logic here
            pass
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get logger instance
            log = logger or logging.getLogger()
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                log.debug(
                    f"{func.__name__} completed in {duration:.2f} seconds",
                    extra={
                        'duration': duration,
                        'function': func.__name__,
                        'status': 'success'
                    }
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                log.error(
                    f"{func.__name__} failed after {duration:.2f} seconds",
                    exc_info=True,
                    extra={
                        'duration': duration,
                        'function': func.__name__,
                        'status': 'error',
                        'error_type': type(e).__name__
                    }
                )
                raise
        return wrapper
    return decorator 