"""
Configuration loader for QA System.

This module handles loading and validating configuration from YAML files
and environment variables. It implements a singleton pattern to ensure
consistent configuration access across the application.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, TypeVar, Type
from dataclasses import dataclass, field
import yaml
from dotenv import load_dotenv
from functools import lru_cache
import logging
import re

T = TypeVar('T')

class ConfigurationError(Exception):
    """Raised when there is an error in configuration loading or validation."""
    pass

@dataclass
class SecurityConfig:
    """Security-related configuration."""
    google_cloud_project: str
    api_key: Optional[str] = None

@dataclass
class VectorStoreConfig:
    """Vector store configuration."""
    type: str
    persist_directory: str
    collection_name: str
    distance_metric: str = "cosine"
    n_jobs: int = -1

@dataclass
class DocumentProcessingConfig:
    """Document processing configuration."""
    batch_size: int
    supported_formats: List[str]
    excluded_patterns: List[str]
    chunk_size: int
    chunk_overlap: int

@dataclass
class EmbeddingModelConfig:
    """Embedding model configuration."""
    type: str
    model_name: str
    batch_size: int
    max_retries: int = 3

@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str
    file_path: str
    max_size: int
    backup_count: int

class Configuration:
    """Singleton configuration class."""
    _instance = None
    _initialized = False

    def __new__(cls) -> 'Configuration':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if Configuration._initialized:
            return
        Configuration._initialized = True
        self._config = None
        self.security: SecurityConfig = None
        self.vector_store: VectorStoreConfig = None
        self.document_processing: DocumentProcessingConfig = None
        self.embedding_model: EmbeddingModelConfig = None
        self.logging: LoggingConfig = None
        self.logger = logging.getLogger(__name__)

    def _validate_env_vars(self) -> None:
        """Validate required environment variables."""
        required_vars = ['GOOGLE_APPLICATION_CREDENTIALS']
        missing_vars = []

        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            raise ConfigurationError(f"Missing or empty required environment variables: {', '.join(missing_vars)}")

        # Verify the credentials file exists
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if not os.path.isfile(credentials_path):
            raise ConfigurationError(f"Google Cloud credentials file not found at: {credentials_path}")

    def _load_env_vars(self, env_path: Optional[str] = None) -> None:
        """Load environment variables from .env file."""
        if env_path and os.path.exists(env_path):
            load_dotenv(env_path)
        else:
            for env_file in ['.env', 'secrets/.env']:
                if os.path.exists(env_file):
                    load_dotenv(env_file)
                    break

        self._validate_env_vars()

    def _validate_config_schema(self, config: Dict[str, Any]) -> None:
        """Validate the configuration schema.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        required_sections = {
            'SECURITY': ['google_cloud_project'],
            'VECTOR_STORE': ['TYPE', 'PERSIST_DIRECTORY', 'COLLECTION_NAME'],
            'DOCUMENT_PROCESSING': ['CHUNK_SIZE', 'CHUNK_OVERLAP'],
            'EMBEDDING_MODEL': ['TYPE', 'MODEL_NAME', 'BATCH_SIZE'],
            'LOGGING': ['LEVEL', 'FORMAT']
        }

        # Check for missing sections
        missing_sections = [section for section in required_sections if section not in config]
        if missing_sections:
            raise ConfigurationError(f"Missing required configuration sections: {', '.join(missing_sections)}")

        # Check for missing fields in each section
        missing_fields = []
        for section, fields in required_sections.items():
            for field in fields:
                if field not in config[section]:
                    missing_fields.append(f"{section}.{field}")

        if missing_fields:
            raise ConfigurationError(f"Missing required configuration fields: {', '.join(missing_fields)}")

        # Validate Google Cloud project ID format
        project_id = config['SECURITY']['google_cloud_project']
        if not re.match(r'^[a-z][-a-z0-9]{4,28}[a-z0-9]$', project_id):
            raise ConfigurationError(
                "Invalid Google Cloud project ID format. Must start with a letter, "
                "contain only lowercase letters, numbers, or hyphens, "
                "be between 6-30 characters, and end with a letter or number."
            )

    @staticmethod
    def _substitute_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively substitute environment variables in configuration values."""
        if isinstance(config, dict):
            return {k: Configuration._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [Configuration._substitute_env_vars(v) for v in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            env_var = config[2:-1]
            value = os.getenv(env_var)
            if not value or value.strip() == '':
                raise ConfigurationError(f"Environment variable not found or empty: {env_var}")
            return value
        return config

    def load_config(self, config_path: str = "./config/config.yaml"):
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            raise ConfigurationError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)

        self._validate_config_schema(self._config)
        self._load_env_vars()
        self._initialize_components()

    def _initialize_components(self):
        """Initialize configuration components."""
        self.security = SecurityConfig(**self._config['SECURITY'])
        self.vector_store = VectorStoreConfig(**self._config['VECTOR_STORE'])
        self.document_processing = DocumentProcessingConfig(**self._config['DOCUMENT_PROCESSING'])
        self.embedding_model = EmbeddingModelConfig(**self._config['EMBEDDING_MODEL'])
        self.logging = LoggingConfig(**self._config['LOGGING'])

    def get_nested(self, *keys: str, default: Any = None) -> Any:
        """Safely get nested configuration values."""
        value = self._config
        for key in keys:
            if not isinstance(value, dict):
                return default
            value = value.get(key, default)
            if value is None:
                return default
        return value

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Get configuration value by key."""
        try:
            value = self._config.get(key, default)
            self.logger.debug("Retrieved configuration value for key '%s'", key)
            return value
        except Exception as e:
            self.logger.error("Error retrieving configuration value for key '%s': %s", key, str(e))
            raise

    def __getitem__(self, key: str) -> Any:
        """Get configuration value by key using dictionary syntax."""
        try:
            return self._config[key]
        except KeyError as e:
            self.logger.error("Configuration key not found: %s", key)
            raise
        except Exception as e:
            self.logger.error("Error accessing configuration key '%s': %s", key, str(e))
            raise

@lru_cache()
def get_config() -> Configuration:
    """Get the singleton configuration instance."""
    return Configuration()

# Example usage:
# config = get_config()
# config.load_config('config/config.yaml')
# vector_store_config = config.vector_store
# security_config = config.security
# nested_value = config.get_nested('VECTOR_STORE', 'EMBEDDINGS', 'DIMENSION', default=768) 