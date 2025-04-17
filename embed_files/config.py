"""
Configuration loader for the QA system.
"""

import logging
import os
from pathlib import Path
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class Config:
    """Configuration class for the QA system."""
    SECURITY: Dict[str, Any] = field(default_factory=dict)
    VECTOR_STORE: Dict[str, Any] = field(default_factory=dict)
    DOCUMENT_PROCESSING: Dict[str, Any] = field(default_factory=dict)
    EMBEDDING_MODEL: Dict[str, Any] = field(default_factory=dict)
    LOGGING: Dict[str, Any] = field(default_factory=dict)
    FILE_SCANNER: Dict[str, Any] = field(default_factory=lambda: {
        'allowed_extensions': ['*'],  # Default to all files
        'exclude_patterns': ['.*', '__pycache__', '*.pyc'],  # Default exclude patterns
        'hash_algorithm': 'sha256'  # Default hash algorithm
    })

    @classmethod
    def get_instance(cls, config_path: Optional[str] = None) -> 'Config':
        """Get or create the singleton instance of Config."""
        global _config_instance
        if _config_instance is None:
            _config_instance = cls.load(config_path)
        elif config_path is not None:
            # Only reload if a specific path is provided
            _config_instance = cls.load(config_path)
        return _config_instance

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> 'Config':
        """Load configuration from file and environment variables."""
        logger = logging.getLogger(__name__)
        
        # Use default config path if none provided
        if config_path is None:
            config_path = os.getenv('CONFIG_PATH', './config/config.yaml')
        
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Config file {config_path} not found, using default values")
                return cls()

            with open(config_file, 'r') as f:
                yaml_config = yaml.safe_load(f) or {}
                
            # Create instance with YAML config
            instance = cls(**yaml_config)
            
            # Override with environment variables
            instance._load_env_vars()
            
            return instance
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise

    def _load_env_vars(self) -> None:
        """Load configuration overrides from environment variables."""
        env_overrides = self._get_env_overrides()
        
        # Update instance attributes
        for section in ['SECURITY', 'VECTOR_STORE', 'DOCUMENT_PROCESSING', 'EMBEDDING_MODEL', 'LOGGING', 'FILE_SCANNER']:
            if hasattr(self, section):
                section_overrides = {
                    k.split('_', 1)[1]: v 
                    for k, v in env_overrides.items() 
                    if k.startswith(f"{section}_")
                }
                if section_overrides:
                    current = getattr(self, section)
                    current.update(section_overrides)

    def _get_env_overrides(self) -> Dict[str, Any]:
        """Get configuration overrides from environment variables."""
        return {
            k.replace('QA_', ''): v
            for k, v in os.environ.items()
            if k.startswith('QA_')
        }

    def get_nested(self, section: str, default: Any = None) -> Optional[Dict[str, Any]]:
        """Get a nested configuration section by name.
        
        Args:
            section: The name of the configuration section to retrieve.
            default: The default value to return if the section doesn't exist.
            
        Returns:
            The configuration section if it exists, otherwise the default value.
        """
        try:
            # Split the section path by dots
            parts = section.split('.')
            value = self
            for part in parts:
                value = getattr(value, part, None) if hasattr(value, part) else value.get(part, None)
                if value is None:
                    return default
            return value
        except Exception:
            return default

# Global configuration instance
_config_instance: Optional[Config] = None

def get_config(config_path: Optional[str] = None) -> Config:
    """Get the global configuration instance."""
    return Config.get_instance(config_path)