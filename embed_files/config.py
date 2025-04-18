"""
Configuration loader for the QA system.

This module handles loading and managing configuration settings from:
1. YAML configuration files
2. Environment variables (with QA_ prefix and direct Google Cloud vars)
3. Default values
"""

import os
from pathlib import Path
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, List

@dataclass
class Config:
    """Configuration class for the QA system.
    
    Attributes:
        SECURITY: Security-related settings including API credentials
        VECTOR_STORE: Vector database configuration
        DOCUMENT_PROCESSING: Document processing settings
        EMBEDDING_MODEL: Model configuration for embeddings
        LOGGING: Logging configuration
        FILE_SCANNER: File scanning and processing settings
    """
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

    def __post_init__(self):
        """Initialize default values after instance creation."""
        # Initialize SECURITY settings from environment variables
        self.SECURITY = {
            'GOOGLE_API_KEY': os.getenv('QA_SECURITY_GOOGLE_API_KEY') or os.getenv('GOOGLE_API_KEY', ''),
            'GOOGLE_CLOUD_PROJECT': os.getenv('QA_SECURITY_GOOGLE_CLOUD_PROJECT') or os.getenv('GOOGLE_CLOUD_PROJECT', ''),
            'GOOGLE_CLOUD_REGION': os.getenv('QA_SECURITY_GOOGLE_CLOUD_REGION') or os.getenv('GOOGLE_CLOUD_REGION', 'us-central1')
        }
        
        # Initialize EMBEDDING_MODEL settings
        self.EMBEDDING_MODEL = {
            'MODEL_NAME': os.getenv('QA_EMBEDDING_MODEL_NAME') or os.getenv('GOOGLE_EMBEDDING_MODEL', 'embedding-001'),
            'BATCH_SIZE': int(os.getenv('QA_EMBEDDING_MODEL_BATCH_SIZE', '15')),
            'MAX_LENGTH': int(os.getenv('QA_EMBEDDING_MODEL_MAX_LENGTH', '3072'))
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key."""
        # First try to get from class attributes
        if hasattr(self, key):
            return getattr(self, key)
            
        # Then try to get from nested dictionaries
        for section in [self.SECURITY, self.VECTOR_STORE, self.DOCUMENT_PROCESSING,
                       self.EMBEDDING_MODEL, self.LOGGING, self.FILE_SCANNER]:
            if key in section:
                return section[key]
                
        return default

    def get_nested(self, path: str, default: Any = None) -> Any:
        """Get a nested configuration value using dot notation."""
        parts = path.split('.')
        current = self
        
        try:
            for part in parts:
                if hasattr(current, part):
                    current = getattr(current, part)
                elif isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return default
            return current
        except (AttributeError, KeyError):
            return default

    @classmethod
    def get_instance(cls, config_path: Optional[str] = None) -> 'Config':
        """Get or create the singleton instance of Config."""
        global _config_instance
        
        if _config_instance is None:
            _config_instance = cls.load(config_path)
        elif config_path is not None:
            _config_instance = cls.load(config_path)
        return _config_instance

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> 'Config':
        """Load configuration from file and environment variables."""
        # Use default config path if none provided
        if config_path is None:
            config_path = os.getenv('CONFIG_PATH', './config/config.yaml')
        
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                return cls()  # Return default config if file doesn't exist

            # Read YAML configuration
            with open(config_file, 'r') as f:
                yaml_config = yaml.safe_load(f) or {}
                
            # Create instance with YAML config
            instance = cls(**yaml_config)
            
            # Override with environment variables
            instance._load_env_vars()
            
            return instance
            
        except Exception as e:
            raise RuntimeError(f"Error loading configuration: {str(e)}") from e

    def _load_env_vars(self) -> None:
        """Load environment variables and update configuration."""
        # Get all environment variables starting with QA_
        qa_env_vars = {k: v for k, v in os.environ.items() if k.startswith('QA_')}
        
        for env_var, value in qa_env_vars.items():
            # Remove QA_ prefix and split into parts
            parts = env_var[3:].split('_', 1)  # QA_SECTION_KEY -> [SECTION, KEY]
            
            if len(parts) < 2:
                continue  # Skip malformed variables silently
                
            section, key = parts
            
            # Check if this section exists in our config
            if not hasattr(self, section):
                continue  # Skip unknown sections silently
            
            # Initialize section dict if needed
            if getattr(self, section) is None:
                setattr(self, section, {})
            
            # Update the configuration
            section_dict = getattr(self, section)
            section_dict[key] = value

def get_config(config_path: Optional[str] = None) -> Config:
    """Get the global configuration instance."""
    return Config.get_instance(config_path)

# Global configuration instance
_config_instance: Optional[Config] = None
