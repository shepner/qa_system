"""
Logging setup module for the QA System.

This module provides centralized logging configuration based on the settings
defined in config.yaml. It sets up file handlers for general logging,
error logging, and access logging with thread safety and proper error handling.
"""

import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
from typing import Dict, Any, Optional
from embed_files.config import Configuration

def validate_logging_config(config: Dict[str, Any]) -> None:
    """Validate logging configuration."""
    required_fields = ['LEVEL', 'LOG_FILE']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required logging configuration field: {field}")

def create_log_directory(path: str) -> None:
    """
    Safely create log directory if it doesn't exist.
    
    Args:
        path: Path to the log file
        
    Raises:
        OSError: If directory creation fails
    """
    try:
        log_dir = Path(path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create log directory {log_dir}: {e}")

def create_handler(filename: str, max_bytes: int = 10485760, backup_count: int = 5) -> RotatingFileHandler:
    """Create a rotating file handler with the specified parameters.
    
    Args:
        filename: Path to the log file
        max_bytes: Maximum size of each log file in bytes (default: 10MB)
        backup_count: Number of backup files to keep (default: 5)
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    handler = RotatingFileHandler(
        filename,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    return handler

def setup_logging(config: Dict[str, Any]) -> None:
    """Set up logging configuration."""
    validate_logging_config(config)
    
    log_level = config['LEVEL']
    log_file = config['LOG_FILE']
    enable_debug = config.get('ENABLE_DEBUG', False)
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Set debug level if enabled
    if enable_debug:
        logging.getLogger().setLevel(logging.DEBUG)