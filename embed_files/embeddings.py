"""
Embedding model implementation for generating text embeddings using Google's Vertex AI API.
"""

from typing import List, Dict, Any
import os
import re
import time
from datetime import datetime
import logging
import httpx
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
from .config import Configuration, ConfigurationError

class EmbeddingModel:
    def __init__(self, config: Configuration):
        """Initialize embedding model with configuration."""
        self.config = config.get_nested("EMBEDDING_MODEL")
        if self.config is None:
            raise ConfigurationError("Missing EMBEDDING_MODEL configuration section")
        
        # Store security config separately
        self.security_config = config.get_nested("SECURITY")
        if self.security_config is None:
            raise ConfigurationError("Missing SECURITY configuration section")
        
        # Validate required configuration parameters
        required_params = [
            "MODEL_NAME",
            "PREPROCESSING",
            "RATE_LIMITS",
            "ERROR_HANDLING",
            "MAX_LENGTH",
            "BATCH_SIZE",
            "MONITORING"
        ]
        missing_params = [param for param in required_params if param not in self.config]
        if missing_params:
            raise ConfigurationError(f"Missing required configuration parameters: {', '.join(missing_params)}")
            
        self.logger = logging.getLogger(__name__)
        self._setup_client()
        self.logger.info(f"Embedding model initialized with model: {self.config['MODEL_NAME']}")
        self._request_count = 0
        self._last_reset = datetime.now()
        self._window_requests = 0
        self._window_start = datetime.now()

    def _setup_client(self) -> None:
        """Set up Google Cloud client."""
        try:
            # Validate security configuration
            self.logger.debug(f"Security config type: {type(self.security_config)}")
            self.logger.debug(f"Security config contents: {self.security_config}")
            
            if not isinstance(self.security_config, dict):
                raise ConfigurationError("SECURITY configuration must be a dictionary")
                
            project_id = self.security_config.get("GOOGLE_CLOUD_PROJECT")
            self.logger.debug(f"Retrieved project_id: {project_id}")
            
            if not project_id:
                raise ConfigurationError("GOOGLE_CLOUD_PROJECT must be set in security configuration")
                
            location = self.config.get("LOCATION", "us-central1")
            self.logger.debug(f"Using location: {location}")
            
            self.logger.info(f"Initializing Vertex AI client with project_id={project_id} and location={location}")
            
            try:
                # Initialize Vertex AI with project and location
                self.logger.debug("Attempting to initialize Vertex AI...")
                aiplatform.init(
                    project=project_id,
                    location=location
                )
                self.logger.debug("Successfully initialized Vertex AI")
            except ValueError as e:
                self.logger.error(f"ValueError during Vertex AI initialization: {str(e)}")
                raise ConfigurationError(f"Invalid project ID or location: {str(e)}")
            except Exception as e:
                self.logger.error(f"Unexpected error during Vertex AI initialization: {str(e)}")
                raise ConfigurationError(f"Failed to initialize Vertex AI client: {str(e)}")
            
            model_name = self.config.get("MODEL_NAME")
            self.logger.debug(f"Retrieved model_name: {model_name}")
            
            if not model_name:
                raise ConfigurationError("MODEL_NAME must be set in EMBEDDING_MODEL configuration")
                
            self.logger.info(f"Initializing text embedding model: {model_name}")
            
            try:
                # Initialize the text embedding model
                self.logger.debug(f"Attempting to initialize text embedding model with name: {model_name}")
                self.client = TextEmbeddingModel.from_pretrained(model_name)
                self.logger.debug(f"Successfully initialized embedding model client in project {project_id}")
            except ValueError as e:
                self.logger.error(f"ValueError during model initialization: {str(e)}")
                raise ConfigurationError(f"Invalid model name {model_name}: {str(e)}")
            except Exception as e:
                self.logger.error(f"Unexpected error during model initialization: {str(e)}")
                raise ConfigurationError(f"Failed to initialize text embedding model {model_name}: {str(e)}")
            
        except ConfigurationError:
            self.logger.error("Configuration error occurred during setup", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during client setup: {str(e)}", exc_info=True)
            raise ConfigurationError(f"Failed to initialize embedding model: {str(e)}")

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text according to configuration."""
        if self.config["PREPROCESSING"]["STRIP_HTML"]:
            # Simple HTML tag removal - can be enhanced with proper HTML parsing if needed
            text = re.sub(r'<[^>]+>', '', text)
        
        if self.config["PREPROCESSING"]["NORMALIZE_WHITESPACE"]:
            # Normalize whitespace and newlines
            text = ' '.join(text.split())
        
        if self.config["PREPROCESSING"]["LOWERCASE"]:
            text = text.lower()
        
        return text

    def _check_rate_limits(self) -> None:
        """Check and enforce rate limits."""
        rate_limits = self.config["RATE_LIMITS"]
        max_requests = rate_limits.get("MAX_REQUESTS_PER_MINUTE", 100)
        window_size = rate_limits.get("WINDOW_SIZE_SECONDS", 60)
        
        current_time = datetime.now()
        window_elapsed = (current_time - self._window_start).total_seconds()
        
        if window_elapsed >= window_size:
            self._window_requests = 0
            self._window_start = current_time
        
        if self._window_requests >= max_requests:
            wait_time = window_size - window_elapsed
            if wait_time > 0:
                self.logger.warning(f"Rate limit reached. Waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
                self._window_requests = 0
                self._window_start = datetime.now()
        
        self._window_requests += 1

    def _retry_with_backoff(self, func) -> Any:
        """Execute function with exponential backoff retry."""
        max_retries = self.config["ERROR_HANDLING"].get("MAX_RETRIES", 3)
        base_delay = self.config["ERROR_HANDLING"].get("BASE_DELAY_SECONDS", 1)
        max_delay = self.config["ERROR_HANDLING"].get("MAX_DELAY_SECONDS", 32)
        
        retries = 0
        while True:
            try:
                return func()
            except httpx.HTTPError as e:
                if retries >= max_retries:
                    self.logger.error(f"Max retries ({max_retries}) exceeded: {str(e)}")
                    raise
                    
                delay = min(base_delay * (2 ** retries), max_delay)
                self.logger.warning(f"Request failed, retrying in {delay} seconds: {str(e)}")
                time.sleep(delay)
                retries += 1
            except Exception as e:
                self.logger.error(f"Unexpected error during embedding generation: {str(e)}")
                raise
                
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []

        embeddings = []
        batch_size = self.config["BATCH_SIZE"]
        
        # Preprocess all texts
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # Process in batches
        for i in range(0, len(processed_texts), batch_size):
            batch = processed_texts[i:i + batch_size]
            
            # Check rate limits before each batch
            self._check_rate_limits()
            
            try:
                # Use retry mechanism for the embedding generation
                batch_embeddings = self._retry_with_backoff(
                    lambda: self.client.get_embeddings(batch)
                )
                
                if self.config["MONITORING"].get("ENABLE_METRICS", False):
                    self.logger.info(
                        f"Generated embeddings for batch of {len(batch)} texts. "
                        f"Dimensions: {len(batch_embeddings[0])}"
                    )
                
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                self.logger.error(f"Failed to generate embeddings for batch: {str(e)}")
                raise
        
        return embeddings

class RateLimitError(Exception):
    """Exception raised when rate limits are exceeded."""
    pass 