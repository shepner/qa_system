"""
Configuration module for the QA system.

This module provides a Config class for loading and accessing configuration values
from a YAML file, with support for environment variable substitution in the SECURITY section.
"""

import os
from pathlib import Path
import yaml
import logging

class Config:
    """
    Provides access to configuration values loaded from a dictionary.

    Supports nested access using dot notation (e.g., 'LOGGING.LEVEL').
    """
    def __init__(self, config_data: dict):
        """
        Initialize the Config object.

        Args:
            config_data: Dictionary containing configuration data.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Called Config.__init__(config_data={config_data})")
        self._config = config_data
        
    def get_nested(self, path: str, default=None):
        """
        Retrieve a nested configuration value using dot notation.

        Args:
            path: Configuration path using dot notation (e.g., 'LOGGING.LEVEL').
            default: Value to return if the path does not exist.

        Returns:
            The configuration value at the specified path, or the default if not found.
        """
        current = self._config
        for key in path.split('.'):
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                self.logger.info(f"Config.get_nested({path}) not found, returning default={default!r}")
                return default
        self.logger.info(f"Config.get_nested({path}) -> {current!r}")
        return current

def get_config(config_path: str = "./config/config.yaml") -> Config:
    """
    Load configuration from a YAML file and return a Config object.

    Args:
        config_path: Path to the configuration YAML file.

    Returns:
        Config: An instance of the Config class with loaded configuration data.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the configuration file is invalid YAML.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Called get_config(config_path={config_path})")
    config_path = Path(config_path) if config_path else Path("./config/config.yaml")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path) as f:
        config_data = yaml.safe_load(f)
        
    # Substitute environment variables in the SECURITY section
    if 'SECURITY' in config_data:
        for key, value in config_data['SECURITY'].items():
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                env_var = value[2:-1]
                config_data['SECURITY'][key] = os.getenv(env_var)
                
    return Config(config_data) 