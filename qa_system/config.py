"""
Configuration management for the QA system
"""
import os
import re
from typing import Dict, Any
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

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from environment variables and config file.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        Configuration dictionary
    """
    # Load environment variables
    load_dotenv()
    
    # Create data directory if it doesn't exist
    data_dir = Path("./data/vectordb")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    config = {}
    
    # Load YAML config if provided
    if config_path:
        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    # Process and interpolate environment variables
                    config = process_config_values(yaml_config)
        except Exception as e:
            logger.error(f"Failed to load YAML config from {config_path}: {str(e)}")
            raise
    
    # Flatten the configuration for easier access
    flat_config = flatten_config(config)
    
    # Add security variables from environment if not in config
    security_vars = {
        'SECURITY_API_KEY': os.getenv('API_KEY'),
        'SECURITY_GOOGLE_CLOUD_PROJECT': os.getenv('GOOGLE_CLOUD_PROJECT'),
        'SECURITY_GOOGLE_APPLICATION_CREDENTIALS': os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    }
    
    # Update flat_config with environment variables if they exist
    for key, value in security_vars.items():
        if value is not None:
            flat_config[key] = value
    
    # Override with QA_ prefixed environment variables
    for key in flat_config:
        env_value = os.getenv(f"QA_{key}")
        if env_value is not None:
            # Convert string to appropriate type
            current_value = flat_config[key]
            if isinstance(current_value, bool):
                flat_config[key] = env_value.lower() in ('true', '1', 'yes')
            elif isinstance(current_value, int):
                flat_config[key] = int(env_value)
            elif isinstance(current_value, float):
                flat_config[key] = float(env_value)
            elif isinstance(current_value, list):
                flat_config[key] = env_value.split(',')
            else:
                flat_config[key] = env_value
    
    # Validate required configuration
    required_vars = [
        'SECURITY_API_KEY',
        'SECURITY_GOOGLE_CLOUD_PROJECT',
        'SECURITY_GOOGLE_APPLICATION_CREDENTIALS'
    ]
    
    missing = [var for var in required_vars if not flat_config.get(var)]
    if missing:
        raise ValueError(f"Missing required configuration: {', '.join(missing)}")
    
    return flat_config 