"""
Configuration loader for the file removal system.

This module handles loading and managing configuration settings from:
1. YAML configuration files
2. Environment variables (with QA_ prefix)
3. Default values
"""

import os
from pathlib import Path
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union

# Debug configuration
debug_config = False

@dataclass
class Config:
    """Configuration class for the file removal system.
    
    Attributes:
        LOGGING: Logging configuration
        FILE_MATCHER: File matching settings
        REMOVAL_VALIDATION: Validation settings
        VECTOR_STORE: Vector database configuration
    """
    LOGGING: Dict[str, Any] = field(default_factory=lambda: {
        'LEVEL': 'INFO',
        'LOG_FILE': 'logs/remove_files.log'
    })
    FILE_MATCHER: Dict[str, Any] = field(default_factory=lambda: {
        'RECURSIVE': True,
        'CASE_SENSITIVE': False
    })
    REMOVAL_VALIDATION: Dict[str, Any] = field(default_factory=lambda: {
        'REQUIRE_CONFIRMATION': True
    })
    VECTOR_STORE: Dict[str, Any] = field(default_factory=lambda: {
        'TYPE': 'chroma',
        'PERSIST_DIRECTORY': 'data/vector_store',
        'COLLECTION_NAME': 'documents'
    })

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create a Config instance from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values
            
        Returns:
            Config instance with values from dictionary
        """
        instance = cls()
        
        # Update each section if it exists in the config_dict
        if 'LOGGING' in config_dict:
            instance.LOGGING.update(config_dict['LOGGING'])
        if 'FILE_MATCHER' in config_dict:
            instance.FILE_MATCHER.update(config_dict['FILE_MATCHER'])
        if 'REMOVAL_VALIDATION' in config_dict:
            instance.REMOVAL_VALIDATION.update(config_dict['REMOVAL_VALIDATION'])
        if 'VECTOR_STORE' in config_dict:
            instance.VECTOR_STORE.update(config_dict['VECTOR_STORE'])
            
        return instance

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key."""
        value = None
        
        # First try to get from class attributes
        if hasattr(self, key):
            value = getattr(self, key)
        else:
            # Then try to get from nested dictionaries
            for section in [self.LOGGING, self.FILE_MATCHER, 
                          self.REMOVAL_VALIDATION, self.VECTOR_STORE]:
                if key in section:
                    value = section[key]
                    break
        
        if debug_config:
            print(f"\n=== Config Get ===")
            print(f"Key: {key}")
            print(f"Value: {value if value is not None else default}")
            
        return value if value is not None else default

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
                    if debug_config:
                        print(f"\n=== Config Get Nested (Not Found) ===")
                        print(f"Path: {path}")
                        print(f"Default Value: {default}")
                    return default
                    
            if debug_config:
                print(f"\n=== Config Get Nested ===")
                print(f"Path: {path}")
                print(f"Value: {current}")
                
            return current
        except (AttributeError, KeyError):
            if debug_config:
                print(f"\n=== Config Get Nested (Error) ===")
                print(f"Path: {path}")
                print(f"Default Value: {default}")
            return default

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> 'Config':
        """Load configuration from file and environment variables."""
        if debug_config:
            print("\n=== Loading Configuration ===")
            print(f"Config Path: {config_path}")
        
        # Use default config path if none provided
        if config_path is None:
            config_path = os.getenv('CONFIG_PATH', './config/config.yaml')
            if debug_config:
                print(f"Using default config path: {config_path}")
        
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                if debug_config:
                    print(f"Config file not found, using default configuration")
                return cls()  # Return default config if file doesn't exist

            # Read YAML configuration
            with open(config_file, 'r') as f:
                yaml_config = yaml.safe_load(f) or {}
                if debug_config:
                    print("\nLoaded YAML configuration:")
                    print(yaml_config)
                
            # Create instance with YAML config using from_dict
            instance = cls.from_dict(yaml_config)
            
            # Override with environment variables
            instance._load_env_vars()
            
            return instance
            
        except Exception as e:
            if debug_config:
                print(f"\nError loading configuration: {str(e)}")
            raise RuntimeError(f"Error loading configuration: {str(e)}") from e

    def _load_env_vars(self) -> None:
        """Load environment variables and update configuration."""
        if debug_config:
            print("\n=== Loading Environment Variables ===")
        
        # Get all environment variables starting with QA_
        qa_env_vars = {k: v for k, v in os.environ.items() if k.startswith('QA_')}
        
        if debug_config:
            print("Found QA_ environment variables:")
            print(qa_env_vars)
        
        for env_var, value in qa_env_vars.items():
            # Remove QA_ prefix and split into parts
            parts = env_var[3:].split('_', 1)  # QA_SECTION_KEY -> [SECTION, KEY]
            
            if len(parts) < 2:
                if debug_config:
                    print(f"Skipping malformed variable: {env_var}")
                continue
                
            section, key = parts
            
            # Check if this section exists in our config
            if not hasattr(self, section):
                if debug_config:
                    print(f"Skipping unknown section: {section}")
                continue
            
            # Initialize section dict if needed
            if getattr(self, section) is None:
                setattr(self, section, {})
            
            # Update the configuration
            section_dict = getattr(self, section)
            section_dict[key] = value
            
            if debug_config:
                print(f"Updated {section}.{key} = {value}")

def get_config(config_path: Optional[str] = None) -> Config:
    """Get a new configuration instance."""
    return Config.load(config_path) 