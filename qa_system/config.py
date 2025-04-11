"""
Configuration management for the QA system
"""
import os
import re
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dotenv import load_dotenv
import yaml
import logging

logger = logging.getLogger(__name__)

def interpolate_env_vars(value: str) -> str:
    """Replace ${VAR} or $VAR in string with environment variable values."""
    if not isinstance(value, str):
        return value
        
    pattern = r'\${([^}]+)}|\$([a-zA-Z_][a-zA-Z0-9_]*)'
    
    def replace_var(match):
        var_name = match.group(1) or match.group(2)
        return os.getenv(var_name, '')
        
    return re.sub(pattern, replace_var, value)

def process_config_values(config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively process configuration values, interpolating environment variables."""
    processed_config = {}
    
    for key, value in config.items():
        if isinstance(value, dict):
            processed_config[key] = process_config_values(value)
        elif isinstance(value, list):
            processed_config[key] = [interpolate_env_vars(item) if isinstance(item, str) else item for item in value]
        else:
            processed_config[key] = interpolate_env_vars(value)
            
    return processed_config

def flatten_config(config: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
    """Flatten nested configuration into dot notation."""
    flattened = {}
    
    for key, value in config.items():
        new_key = f"{prefix}{key}" if prefix else key
        
        if isinstance(value, dict):
            flattened.update(flatten_config(value, f"{new_key}_"))
        else:
            flattened[new_key] = value
            
    return flattened

def load_config(config_path: Union[str, Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load configuration from environment variables and a config file or dictionary.
    
    Args:
        config_path: Path to config file (with optional @ prefix for absolute path) or a configuration dictionary
        
    Returns:
        Dict containing the merged configuration
    """
    # Initialize config
    config = None
    
    # Handle different input types
    if isinstance(config_path, dict):
        config = config_path
    else:
        # Default to config/config.yaml if no path provided
        yaml_path = Path('config/config.yaml') if config_path is None else (
            Path(config_path[1:]).resolve() if isinstance(config_path, str) and config_path.startswith('@')
            else Path(config_path)
        )
        
        try:
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            logging.error(f"Configuration file not found at: {yaml_path.absolute()}")
            raise
        except Exception as e:
            logging.error(f"Error loading configuration: {str(e)}")
            raise

    if not config:
        raise ValueError("Failed to load configuration")

    # Process configuration values
    config = process_config_values(config)
    
    # Validate required environment variables
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("OPENAI_API_KEY environment variable is required")
        
    return config 