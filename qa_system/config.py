"""Configuration module for the QA system."""

import os
from pathlib import Path
import yaml
import logging

class Config:
    """Configuration class that provides access to configuration values."""
    
    def __init__(self, config_data: dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(f"Called Config.__init__(config_data={config_data})")
        self._config = config_data
        
    def get_nested(self, path: str, default=None):
        """Get a nested configuration value using dot notation.
        
        Args:
            path: Configuration path using dot notation (e.g., 'LOGGING.LEVEL')
            default: Default value if path doesn't exist
            
        Returns:
            Configuration value or default if not found
        """
        self.logger.debug(f"Called Config.get_nested(path={path}, default={default})")
        current = self._config
        for key in path.split('.'):
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

def get_config(config_path: str = "./config/config.yaml") -> Config:
    """Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config object
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        yaml.YAMLError: If configuration file is invalid
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"Called get_config(config_path={config_path})")
    config_path = Path(config_path) if config_path else Path("./config/config.yaml")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path) as f:
        config_data = yaml.safe_load(f)
        
    # Load environment variables
    if 'SECURITY' in config_data:
        for key, value in config_data['SECURITY'].items():
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                env_var = value[2:-1]
                config_data['SECURITY'][key] = os.getenv(env_var)
                
    return Config(config_data) 