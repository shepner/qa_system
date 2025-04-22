"""
Configuration loader for the QA system.

This module handles loading and managing configuration settings from:
1. YAML configuration files
2. Environment variables (with QA_ prefix and direct Google Cloud vars)
3. Default values

Configuration is validated against defined schemas and types.
Environment variables override file-based configuration.
Sensitive values should be provided via environment variables.
"""

import os
from pathlib import Path
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, List, Type
import logging
import json

class ConfigError(Exception):
    """Base exception for configuration-related errors."""
    pass

class ConfigValidationError(ConfigError):
    """Raised when configuration validation fails."""
    pass

class ConfigLoadError(ConfigError):
    """Raised when configuration loading fails."""
    pass

class ConfigSecurityError(ConfigError):
    """Raised when security-related configuration is invalid or missing."""
    pass

# Debug configuration
debug_config = False

# Configuration schemas defining required fields and types
CONFIG_SCHEMAS = {
    'LOGGING': {
        'required': ['LEVEL', 'LOG_FILE'],
        'optional': [],
        'types': {
            'LEVEL': str,
            'LOG_FILE': str
        },
        'defaults': {
            'LEVEL': 'INFO',
            'LOG_FILE': 'logs/qa_system.log'
        }
    },
    'SECURITY': {
        'required': ['GOOGLE_APPLICATION_CREDENTIALS', 'GOOGLE_CLOUD_PROJECT'],
        'optional': ['GOOGLE_VISION_API_KEY'],
        'types': {
            'GOOGLE_APPLICATION_CREDENTIALS': str,
            'GOOGLE_CLOUD_PROJECT': str,
            'GOOGLE_VISION_API_KEY': str
        }
    },
    'FILE_SCANNER': {
        'required': ['DOCUMENT_PATH', 'ALLOWED_EXTENSIONS', 'EXCLUDE_PATTERNS'],
        'optional': ['HASH_ALGORITHM', 'SKIP_EXISTING'],
        'types': {
            'DOCUMENT_PATH': str,
            'ALLOWED_EXTENSIONS': list,
            'EXCLUDE_PATTERNS': list,
            'HASH_ALGORITHM': str,
            'SKIP_EXISTING': bool
        },
        'defaults': {
            'DOCUMENT_PATH': './docs',
            'ALLOWED_EXTENSIONS': ['*'],
            'EXCLUDE_PATTERNS': ['.*', '__pycache__', '*.pyc'],
            'HASH_ALGORITHM': 'sha256',
            'SKIP_EXISTING': True
        }
    },
    'DATA_REMOVER': {
        'required': ['RECURSIVE', 'CASE_SENSITIVE', 'REQUIRE_CONFIRMATION'],
        'optional': [],
        'types': {
            'RECURSIVE': bool,
            'CASE_SENSITIVE': bool,
            'REQUIRE_CONFIRMATION': bool
        },
        'defaults': {
            'RECURSIVE': True,
            'CASE_SENSITIVE': False,
            'REQUIRE_CONFIRMATION': True
        }
    },
    'DOCUMENT_PROCESSING': {
        'required': ['MAX_CHUNK_SIZE', 'MIN_CHUNK_SIZE', 'CHUNK_OVERLAP'],
        'optional': ['CONCURRENT_TASKS', 'BATCH_SIZE', 'PRESERVE_SENTENCES',
                    'PDF_HEADER_RECOGNITION', 'VISION_API'],
        'types': {
            'MAX_CHUNK_SIZE': int,
            'MIN_CHUNK_SIZE': int,
            'CHUNK_OVERLAP': int,
            'CONCURRENT_TASKS': int,
            'BATCH_SIZE': int,
            'PRESERVE_SENTENCES': bool,
            'PDF_HEADER_RECOGNITION': dict,
            'VISION_API': dict
        },
        'defaults': {
            'MAX_CHUNK_SIZE': 3072,
            'MIN_CHUNK_SIZE': 1024,
            'CHUNK_OVERLAP': 768,
            'CONCURRENT_TASKS': 6,
            'BATCH_SIZE': 50,
            'PRESERVE_SENTENCES': True,
            'PDF_HEADER_RECOGNITION': {
                'ENABLED': True,
                'MIN_FONT_SIZE': 12,
                'PATTERNS': [],
                'MAX_HEADER_LENGTH': 100
            },
            'VISION_API': {
                'ENABLED': True,
                'FEATURES': [],
                'MAX_RESULTS': 50
            }
        }
    },
    'EMBEDDING_MODEL': {
        'required': ['MODEL_NAME'],
        'optional': ['BATCH_SIZE', 'MAX_LENGTH', 'DIMENSIONS'],
        'types': {
            'MODEL_NAME': str,
            'BATCH_SIZE': int,
            'MAX_LENGTH': int,
            'DIMENSIONS': int
        },
        'defaults': {
            'MODEL_NAME': 'models/embedding-001',
            'BATCH_SIZE': 15,
            'MAX_LENGTH': 3072,
            'DIMENSIONS': 768
        }
    },
    'VECTOR_STORE': {
        'required': ['TYPE', 'PERSIST_DIRECTORY', 'COLLECTION_NAME'],
        'optional': ['DISTANCE_METRIC', 'TOP_K', 'DIMENSIONS'],
        'types': {
            'TYPE': str,
            'PERSIST_DIRECTORY': str,
            'COLLECTION_NAME': str,
            'DISTANCE_METRIC': str,
            'TOP_K': int,
            'DIMENSIONS': int
        },
        'defaults': {
            'TYPE': 'chroma',
            'PERSIST_DIRECTORY': './data/vector_store',
            'COLLECTION_NAME': 'qa_documents',
            'DISTANCE_METRIC': 'cosine',
            'TOP_K': 40,
            'DIMENSIONS': 768
        }
    }
}

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
        DATA_REMOVER: File matcher configuration
        
    Raises:
        ValueError: If required configuration fields are missing or have invalid types
        RuntimeError: If configuration loading fails
    """
    SECURITY: Dict[str, Any] = field(default_factory=dict)
    VECTOR_STORE: Dict[str, Any] = field(default_factory=dict)
    DOCUMENT_PROCESSING: Dict[str, Any] = field(default_factory=dict)
    EMBEDDING_MODEL: Dict[str, Any] = field(default_factory=dict)
    LOGGING: Dict[str, Any] = field(default_factory=dict)
    FILE_SCANNER: Dict[str, Any] = field(default_factory=dict)
    DATA_REMOVER: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize and validate configuration after instance creation."""
        self._apply_secure_defaults()
        self._validate_config()
        self._load_env_vars()
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration against defined schemas.
        
        Raises:
            ValueError: If required fields are missing or have invalid types
        """
        logger = logging.getLogger(__name__)
        
        for section, schema in CONFIG_SCHEMAS.items():
            if not hasattr(self, section):
                raise ValueError(f"Missing required configuration section: {section}")
                
            section_config = getattr(self, section)
            if not isinstance(section_config, dict):
                raise ValueError(f"Configuration section {section} must be a dictionary")
                
            # Check required fields
            missing_fields = [
                field for field in schema['required']
                if field not in section_config
            ]
            if missing_fields:
                # Apply defaults if available
                for field in missing_fields:
                    if 'defaults' in schema and field in schema['defaults']:
                        section_config[field] = schema['defaults'][field]
                        logger.debug(f"Applied default value for {section}.{field}: {schema['defaults'][field]}")
                    else:
                        raise ValueError(f"Missing required field(s) in {section}: {missing_fields}")
            
            # Validate types
            for field, value in section_config.items():
                if field in schema['types']:
                    expected_type = schema['types'][field]
                    if not isinstance(value, expected_type):
                        try:
                            # Attempt type conversion
                            section_config[field] = expected_type(value)
                            logger.debug(f"Converted {section}.{field} to {expected_type}")
                        except (ValueError, TypeError):
                            raise ValueError(
                                f"Invalid type for {section}.{field}: "
                                f"expected {expected_type.__name__}, got {type(value).__name__}"
                            )

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key."""
        value = None
        
        # First try to get from class attributes
        if hasattr(self, key):
            value = getattr(self, key)
        else:
            # Then try to get from nested dictionaries
            for section in [self.SECURITY, self.VECTOR_STORE, self.DOCUMENT_PROCESSING,
                          self.EMBEDDING_MODEL, self.LOGGING, self.FILE_SCANNER,
                          self.DATA_REMOVER]:
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
    def get_instance(cls, config_path: Optional[str] = None) -> 'Config':
        """Get or create the singleton instance of Config."""
        global _config_instance
        
        if debug_config:
            print("\n=== Get Instance ===")
            print(f"Config Path: {config_path}")
            print(f"Existing Instance: {'Yes' if _config_instance else 'No'}")
        
        if _config_instance is None:
            _config_instance = cls.load(config_path)
        elif config_path is not None:
            _config_instance = cls.load(config_path)
        return _config_instance

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> 'Config':
        """Load configuration from file and environment variables.
        
        Args:
            config_path: Optional path to the configuration file.
            
        Returns:
            Config: A configured instance.
            
        Raises:
            ConfigLoadError: If there are issues loading the configuration file
            ConfigSecurityError: If security-related configuration is invalid
            ConfigValidationError: If configuration validation fails
        """
        logger = logging.getLogger(__name__)
        
        if debug_config:
            logger.debug("=== Loading Configuration ===")
            logger.debug(f"Config Path: {config_path}")
        
        # Use default config path if none provided
        if config_path is None:
            config_path = os.getenv('CONFIG_PATH', './config/config.yaml')
            if debug_config:
                logger.debug(f"Using default config path: {config_path}")
        
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Config file not found at {config_path}, using default configuration")
                return cls()  # Return default config if file doesn't exist

            # Read YAML configuration
            try:
                with open(config_file, 'r') as f:
                    yaml_config = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise ConfigLoadError(f"Invalid YAML in configuration file: {str(e)}")
            except PermissionError:
                raise ConfigLoadError(f"Permission denied reading configuration file: {config_path}")
            
            if debug_config:
                logger.debug("Loaded YAML configuration:")
                logger.debug(yaml_config)
            
            # Create instance with YAML config
            try:
                instance = cls(**yaml_config)
            except TypeError as e:
                raise ConfigValidationError(f"Invalid configuration structure: {str(e)}")
            
            # Override with environment variables
            instance._load_env_vars()
            
            # Validate security configuration
            instance._validate_security_config()
            
            return instance
            
        except (ConfigError, ValueError) as e:
            # Re-raise known configuration errors
            raise
        except Exception as e:
            if debug_config:
                logger.exception("Unexpected error loading configuration")
            raise ConfigLoadError(f"Error loading configuration: {str(e)}") from e

    def _validate_security_config(self) -> None:
        """Validate security-related configuration settings.
        
        Raises:
            ConfigSecurityError: If security validation fails
        """
        logger = logging.getLogger(__name__)
        
        # Check for required security credentials
        if not self.SECURITY.get('GOOGLE_APPLICATION_CREDENTIALS'):
            raise ConfigSecurityError(
                "Missing Google Cloud credentials. Set GOOGLE_APPLICATION_CREDENTIALS "
                "environment variable or configure in SECURITY section."
            )
        
        if not self.SECURITY.get('GOOGLE_CLOUD_PROJECT'):
            raise ConfigSecurityError(
                "Missing Google Cloud project. Set GOOGLE_CLOUD_PROJECT "
                "environment variable or configure in SECURITY section."
            )
        
        # Validate credentials file exists if specified
        creds_path = self.SECURITY['GOOGLE_APPLICATION_CREDENTIALS']
        if not os.path.isfile(creds_path):
            raise ConfigSecurityError(
                f"Google Cloud credentials file not found at: {creds_path}"
            )
        
        # Check file permissions on credentials
        try:
            creds_stat = os.stat(creds_path)
            if creds_stat.st_mode & 0o077:  # Check if group/others have any access
                logger.warning(
                    f"Insecure permissions on credentials file: {creds_path}. "
                    "Recommend changing to 600 (user read/write only)."
                )
        except OSError as e:
            raise ConfigSecurityError(
                f"Unable to check credentials file permissions: {str(e)}"
            )

    def _load_env_vars(self) -> None:
        """Load and validate environment variables.
        
        Environment variables with QA_ prefix override file-based configuration.
        Handles type conversion and validation for complex data types.
        
        Raises:
            ConfigValidationError: If environment variable type conversion fails
        """
        logger = logging.getLogger(__name__)
        
        # Define environment variable mappings and types
        ENV_MAPPINGS = {
            'SECURITY': {
                'QA_SECURITY_GOOGLE_APPLICATION_CREDENTIALS': ('GOOGLE_APPLICATION_CREDENTIALS', str),
                'QA_SECURITY_GOOGLE_CLOUD_PROJECT': ('GOOGLE_CLOUD_PROJECT', str),
                'QA_SECURITY_GOOGLE_CLOUD_REGION': ('GOOGLE_CLOUD_REGION', str),
                'QA_SECURITY_GOOGLE_VISION_API_KEY': ('GOOGLE_VISION_API_KEY', str)
            }
        }
        
        # Load environment variables with type conversion
        for section, mappings in ENV_MAPPINGS.items():
            section_config = getattr(self, section)
            
            for env_var, (config_key, expected_type) in mappings.items():
                value = os.getenv(env_var)
                
                # Also check for non-prefixed Google Cloud variables
                if section == 'SECURITY' and value is None:
                    non_prefixed = env_var.replace('QA_SECURITY_', '')
                    value = os.getenv(non_prefixed)
                
                if value is not None:
                    try:
                        if expected_type == bool:
                            value = value.lower() in ('true', '1', 'yes', 'on')
                        elif expected_type == list:
                            value = [item.strip() for item in value.split(',')]
                        elif expected_type == dict:
                            try:
                                value = json.loads(value)
                            except json.JSONDecodeError as e:
                                raise ConfigValidationError(
                                    f"Invalid JSON in environment variable {env_var}: {str(e)}"
                                )
                        else:
                            value = expected_type(value)
                            
                        section_config[config_key] = value
                        if debug_config:
                            logger.debug(f"Loaded environment variable {env_var}")
                    except (ValueError, TypeError) as e:
                        raise ConfigValidationError(
                            f"Failed to convert environment variable {env_var} "
                            f"to type {expected_type.__name__}: {str(e)}"
                        )

    def _apply_secure_defaults(self) -> None:
        """Apply secure default values for sensitive configuration settings."""
        # Apply defaults from CONFIG_SCHEMAS
        for section, schema in CONFIG_SCHEMAS.items():
            if 'defaults' in schema:
                section_config = getattr(self, section)
                for key, value in schema['defaults'].items():
                    if key not in section_config:
                        section_config[key] = value

def get_config(config_path: Optional[str] = None) -> Config:
    """Get the global configuration instance."""
    if debug_config:
        print("\n=== Getting Global Config ===")
        print(f"Requested Config Path: {config_path}")
    return Config.get_instance(config_path)

# Global configuration instance
_config_instance: Optional[Config] = None
