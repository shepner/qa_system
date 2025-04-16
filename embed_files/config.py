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

    @classmethod
    def get_instance(cls, config_path: Optional[str] = None) -> 'Config':
        """Get or create the singleton instance of Config."""
        global _config_instance
        if _config_instance is None or config_path is not None:
            _config_instance = cls.load(config_path)
        return _config_instance

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> 'Config':
        """Load configuration from file and environment variables."""
        logger = logging.getLogger(__name__)
        
        # Use default config path if none provided
        if config_path is None:
            config_path = os.getenv('CONFIG_PATH', './config/config.yaml')
        
        logger.info(f"Loading configuration from {config_path}")
        
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
        flattened = self._flatten_config()
        flattened.update(env_overrides)
        
        # Update instance attributes
        for section in ['SECURITY', 'VECTOR_STORE', 'DOCUMENT_PROCESSING', 'EMBEDDING_MODEL', 'LOGGING']:
            if hasattr(self, section):
                section_overrides = {
                    k.split('_', 1)[1]: v 
                    for k, v in env_overrides.items() 
                    if k.startswith(f"{section}_")
                }
                getattr(self, section).update(section_overrides)

    def _flatten_config(self) -> Dict[str, Any]:
        """Flatten nested configuration into dot notation."""
        flattened = {}
        for section in ['SECURITY', 'VECTOR_STORE', 'DOCUMENT_PROCESSING', 'EMBEDDING_MODEL', 'LOGGING']:
            if hasattr(self, section):
                section_data = getattr(self, section)
                for key, value in section_data.items():
                    flattened[f"{section}_{key}"] = value
        return flattened

    def _get_env_overrides(self) -> Dict[str, Any]:
        """Get configuration overrides from environment variables."""
        return {
            k.replace('QA_', ''): v
            for k, v in os.environ.items()
            if k.startswith('QA_')
        }

    def get_nested(self, section: str) -> Optional[Dict[str, Any]]:
        """Get a nested configuration section by name."""
        if not hasattr(self, section):
            return None
        return getattr(self, section)

# Global configuration instance
_config_instance: Optional[Config] = None

def get_config(config_path: Optional[str] = None) -> Config:
    """Get the global configuration instance."""
    return Config.get_instance(config_path)