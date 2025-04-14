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

T = TypeVar('T')

class ConfigurationError(Exception):
    """Raised when there is an error in configuration loading or validation."""
    pass

@dataclass(frozen=True)
class SecurityConfig:
    """Security-related configuration."""
    google_vision_api_key: str
    google_cloud_project: str
    google_application_credentials: str
    auth_required: bool = True

@dataclass(frozen=True)
class VectorStoreConfig:
    """Vector store configuration."""
    type: str
    persist_directory: str
    collection_name: str
    distance_metric: str = "cosine"
    top_k: int = 40
    dimensions: int = 768
    dedup_threshold: float = 0.95
    normalize_vectors: bool = True
    use_mmap: bool = True
    max_cache_size: int = 1024
    max_segments: int = 5
    search_threads: int = 0
    batch_size: int = 100
    enable_metrics: bool = True
    stats_interval: int = 300

@dataclass(frozen=True)
class EmbeddingModelConfig:
    """Embedding model configuration."""
    type: str
    model_name: str
    batch_size: int
    max_length: int
    dimensions: int
    strip_html: bool = True
    normalize_whitespace: bool = True
    max_retries: int = 3
    retry_base_delay: int = 2
    enable_metrics: bool = True
    enable_logging: bool = True
    log_level: str = "INFO"

@dataclass(frozen=True)
class QueryEngineConfig:
    """Query engine configuration."""
    system_prompt: str
    model: str
    top_k_results: int
    max_query_length: int
    max_output_tokens: int
    temperature: float
    top_p: float
    response_mode: str
    min_relevance_score: float
    max_context_docs: int
    enable_query_expansion: bool = True

@dataclass(frozen=True)
class AppConfig:
    """Application configuration."""
    base_dir: str
    timeout: int
    max_retries: int
    api_port: int

class Configuration:
    """Singleton configuration class."""
    _instance = None
    _initialized = False

    def __new__(cls) -> 'Configuration':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if Configuration._initialized:
            return
        Configuration._initialized = True
        self._config = {}
        self._loaded = False

    @staticmethod
    def _validate_env_vars() -> List[str]:
        """Validate required environment variables."""
        required_env_vars = [
            'GOOGLE_CLOUD_PROJECT',
            'GOOGLE_APPLICATION_CREDENTIALS',
            'GOOGLE_VISION_API_KEY',
            'PROMPTS'  # For system prompts
        ]
        return [var for var in required_env_vars if not os.getenv(var)]

    def _load_env_vars(self, env_path: Optional[str] = None) -> None:
        """Load environment variables from .env file."""
        if env_path and os.path.exists(env_path):
            load_dotenv(env_path)
        else:
            for env_file in ['.env', 'secrets/.env']:
                if os.path.exists(env_file):
                    load_dotenv(env_file)
                    break

        missing_vars = self._validate_env_vars()
        if missing_vars:
            raise ConfigurationError(f"Missing required environment variables: {', '.join(missing_vars)}")

    def _validate_config_schema(self, config: Dict[str, Any]) -> None:
        """Validate configuration schema."""
        required_sections = [
            'SECURITY',
            'VECTOR_STORE',
            'DOCUMENT_PROCESSING',
            'EMBEDDING_MODEL',
            'QUERY_ENGINE',
            'LOGGING',
            'APP'
        ]
        
        missing_sections = [section for section in required_sections if section not in config]
        if missing_sections:
            raise ConfigurationError(f"Missing required configuration sections: {', '.join(missing_sections)}")

    @staticmethod
    def _substitute_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively substitute environment variables in configuration values."""
        if isinstance(config, dict):
            return {k: Configuration._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [Configuration._substitute_env_vars(v) for v in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            env_var = config[2:-1]
            if env_var not in os.environ:
                raise ConfigurationError(f"Environment variable not found: {env_var}")
            return os.environ[env_var]
        return config

    def load(self, config_path: str, env_path: Optional[str] = None) -> None:
        """Load configuration from files."""
        if self._loaded:
            return

        self._load_env_vars(env_path)

        try:
            config_path = Path(config_path)
            if not config_path.exists():
                raise ConfigurationError(f"Configuration file not found: {config_path}")
                
            with open(config_path) as f:
                config = yaml.safe_load(f)

            self._validate_config_schema(config)
            self._config = self._substitute_env_vars(config)
            self._loaded = True

        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing YAML configuration: {e}")

    @property
    def security(self) -> SecurityConfig:
        """Get security configuration."""
        return SecurityConfig(**self._config['SECURITY'])

    @property
    def vector_store(self) -> VectorStoreConfig:
        """Get vector store configuration."""
        return VectorStoreConfig(**self._config['VECTOR_STORE'])

    @property
    def embedding_model(self) -> EmbeddingModelConfig:
        """Get embedding model configuration."""
        return EmbeddingModelConfig(**self._config['EMBEDDING_MODEL'])

    @property
    def query_engine(self) -> QueryEngineConfig:
        """Get query engine configuration."""
        return QueryEngineConfig(**self._config['QUERY_ENGINE'])

    @property
    def app(self) -> AppConfig:
        """Get application configuration."""
        return AppConfig(**self._config['APP'])

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

@lru_cache()
def get_config() -> Configuration:
    """Get the singleton configuration instance."""
    return Configuration()

# Example usage:
# config = get_config()
# config.load('config/config.yaml')
# vector_store_config = config.vector_store
# security_config = config.security
# nested_value = config.get_nested('VECTOR_STORE', 'EMBEDDINGS', 'DIMENSION', default=768) 