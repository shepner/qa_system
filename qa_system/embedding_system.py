"""
Embedding system for generating text embeddings using Google's Gemini model.

This module handles:
- Text embedding generation
- Batch processing
- Error handling and retries
- Token counting and validation

Authentication Setup:
    This module requires proper authentication setup using Google Cloud credentials:
    1. Enable the Google Generative Language API in your Google Cloud project
    2. Set up application default credentials (ADC):
       - Create OAuth client ID credentials in Google Cloud Console
       - Download the client_secret.json file
       - Run: gcloud auth application-default login --client-id-file=client_secret.json \
         --scopes='https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/generative-language.retriever'
    3. Set GOOGLE_APPLICATION_CREDENTIALS environment variable to point to your credentials file
    
    For detailed OAuth setup instructions, see:
    https://ai.google.dev/gemini-api/docs/oauth#set-application-default

API Reference:
    https://googleapis.github.io/python-genai/
"""

import logging
from typing import List, Dict, Any, Optional
import time
from datetime import datetime
import os
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

from qa_system.config import get_config, ConfigLoadError
from qa_system.exceptions import EmbeddingError

# Configure module-level logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Let the application configure logging

class EmbeddingGenerator:
    """
    A class for generating text embeddings using Google's Generative AI model.
    
    This class handles:
    - Initialization of the embedding model
    - Generation of embeddings for single texts
    - Batch processing of multiple texts
    - Error handling and retries
    - Token validation and logging
    
    API Reference:
        https://googleapis.github.io/python-genai/
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the embedding system.
        
        Args:
            logger (Optional[logging.Logger]): Logger instance to use. If None, creates a new logger.
        """
        # Setup logging first
        self.logger = logger or logging.getLogger(__name__)
        
        try:
            # Load configuration
            self.config = get_config()
            self.logger.debug(
                "Loaded base configuration",
                extra={
                    'component': 'embedding_system',
                    'operation': 'init',
                    'config_keys': list(self.config.keys()) if hasattr(self.config, 'keys') else None,
                    'config_type': type(self.config).__name__
                }
            )
            
            # Get model configuration
            model_config = self.config.get_nested('EMBEDDING_MODEL', {})
            self.logger.debug(
                "Loaded model configuration",
                extra={
                    'component': 'embedding_system',
                    'operation': 'init',
                    'model_config': model_config,
                    'model_config_type': type(model_config).__name__
                }
            )
            
            self.model_name = model_config.get('MODEL_NAME', 'models/embedding-001')
            self.batch_size = model_config.get('BATCH_SIZE', 15)
            self.max_length = model_config.get('MAX_LENGTH', 3072)
            self.dimensions = model_config.get('DIMENSIONS', 768)
            self.task_type = model_config.get('TASK_TYPE', 'RETRIEVAL_DOCUMENT')
            self.auto_truncate = model_config.get('AUTO_TRUNCATE', True)
            
            self.logger.debug(
                "Model parameters configured",
                extra={
                    'component': 'embedding_system',
                    'operation': 'init',
                    'model_name': self.model_name,
                    'batch_size': self.batch_size,
                    'max_length': self.max_length,
                    'dimensions': self.dimensions,
                    'task_type': self.task_type,
                    'auto_truncate': self.auto_truncate
                }
            )
            
            # Get security configuration
            security_config = self.config.get_nested('SECURITY', {})
            self.logger.debug(
                "Loaded security configuration",
                extra={
                    'component': 'embedding_system',
                    'operation': 'init',
                    'security_config_keys': list(security_config.keys()),
                    'has_credentials_config': 'GOOGLE_APPLICATION_CREDENTIALS' in security_config,
                    'has_project_id': 'GOOGLE_CLOUD_PROJECT' in security_config,
                    'security_config_type': type(security_config).__name__,
                    'security_config': security_config  # Log full config for debugging
                }
            )
            
            # Get API key from environment or config
            api_key = os.getenv('GOOGLE_API_KEY')
            self.logger.debug(
                "Checking environment for API key",
                extra={
                    'component': 'embedding_system',
                    'operation': 'init',
                    'api_key_in_env': api_key is not None,
                    'api_key_length': len(api_key) if api_key else 0,
                    'env_vars': {k: v for k, v in os.environ.items() if 'GOOGLE' in k},  # Log all Google-related env vars
                    'env_vars_type': {k: type(v).__name__ for k, v in os.environ.items() if 'GOOGLE' in k}
                }
            )
            
            if not api_key:
                api_key = security_config.get('GOOGLE_APPLICATION_CREDENTIALS')
                self.logger.debug(
                    "Falling back to config for API key",
                    extra={
                        'component': 'embedding_system',
                        'operation': 'init',
                        'api_key_in_config': api_key is not None,
                        'api_key_length': len(api_key) if api_key else 0,
                        'api_key_type': type(api_key).__name__ if api_key else None,
                        'api_key_value': api_key  # Log actual key for debugging
                    }
                )
            
            if not api_key:
                error_msg = "GOOGLE_API_KEY not configured in environment or security config"
                self.logger.error(
                    error_msg,
                    extra={
                        'component': 'embedding_system',
                        'operation': 'init',
                        'error_type': 'ConfigurationError',
                        'security_config': security_config,
                        'env_vars': dict(os.environ),
                        'config_type': type(self.config).__name__,
                        'security_config_type': type(security_config).__name__
                    }
                )
                raise EmbeddingError(error_msg)
            
            # Initialize the Google Generative AI client
            self.logger.debug(
                "Configuring Google Generative AI client",
                extra={
                    'component': 'embedding_system',
                    'operation': 'init',
                    'api_key_length': len(api_key),
                    'api_key_prefix': api_key[:4] if api_key else None,
                    'api_key_type': type(api_key).__name__,
                    'api_key': api_key  # Log actual key for debugging
                }
            )
            
            try:
                genai.configure(api_key=api_key)
                self.logger.debug(
                    "Google Generative AI client configured successfully",
                    extra={
                        'component': 'embedding_system',
                        'operation': 'init',
                        'genai_version': genai.__version__,
                        'genai_config': {
                            'api_key_set': bool(genai.get_api_key()),
                            'default_model': genai.get_default_model()
                        }
                    }
                )
                
                self.logger.debug(
                    "Attempting to get model",
                    extra={
                        'component': 'embedding_system',
                        'operation': 'init',
                        'model_name': self.model_name,
                        'available_models': [str(m) for m in genai.list_models()]
                    }
                )
                self.model = genai.get_model(self.model_name)
                
                self.logger.debug(
                    "Model retrieved successfully",
                    extra={
                        'component': 'embedding_system',
                        'operation': 'init',
                        'model': str(self.model),
                        'model_type': type(self.model).__name__
                    }
                )
                
            except Exception as e:
                self.logger.error(
                    f"Failed to initialize Google Generative AI client: {str(e)}",
                    extra={
                        'component': 'embedding_system',
                        'operation': 'init',
                        'error_type': type(e).__name__,
                        'error_message': str(e),
                        'api_key_length': len(api_key) if api_key else 0,
                        'api_key_type': type(api_key).__name__,
                        'api_key': api_key,  # Log actual key for debugging
                        'traceback': str(e.__traceback__)
                    },
                    exc_info=True
                )
                raise
            
            # Log debug information about the model
            self.logger.debug(
                "Initialized embedding model",
                extra={
                    'component': 'embedding_system',
                    'operation': 'init',
                    'model_name': self.model_name,
                    'genai_version': genai.__version__,
                    'dimensions': self.dimensions,
                    'task_type': self.task_type
                }
            )
            
            self.logger.info(
                "Embedding system initialized",
                extra={
                    'component': 'embedding_system',
                    'model': self.model_name,
                    'dimensions': self.dimensions
                }
            )
            
        except ConfigLoadError as e:
            self.logger.error(
                f"Configuration loading failed: {str(e)}",
                extra={
                    'component': 'embedding_system',
                    'error_type': 'ConfigLoadError'
                },
                exc_info=True
            )
            raise EmbeddingError(f"Failed to load configuration: {str(e)}")
            
        except Exception as e:
            self.logger.error(
                f"Failed to initialize embedding system: {str(e)}",
                extra={
                    'component': 'embedding_system',
                    'operation': 'init',
                    'error_type': type(e).__name__
                },
                exc_info=True
            )
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(EmbeddingError),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING)
    )
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text using Google's Generative AI model.
        
        Args:
            text (str): The text to generate an embedding for.
            
        Returns:
            List[float]: The generated embedding vector.
            
        Raises:
            EmbeddingError: If embedding generation fails.
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")
            
        start_time = time.time()
        
        try:
            # Log debug information before generating embedding
            self.logger.debug(
                "Generating embedding",
                extra={
                    'component': 'embedding_system',
                    'operation': 'generate_embedding',
                    'text_length': len(text),
                    'text_preview': text[:100],
                    'model_name': self.model_name
                }
            )
            
            # Generate embedding using the model
            result = self.model.embed_content(text)
            
            if not result or not result.embedding:
                raise EmbeddingError("No embedding in response")
                
            # Extract embedding from response
            embedding = result.embedding
            
            # Validate dimensions
            if len(embedding) != self.dimensions:
                raise EmbeddingError(
                    f"Unexpected embedding dimensions: {len(embedding)} != {self.dimensions}"
                )
            
            # Log success with timing
            duration = time.time() - start_time
            self.logger.debug(
                "Generated embedding successfully",
                extra={
                    'component': 'embedding_system',
                    'operation': 'generate_embedding',
                    'duration': duration,
                    'text_length': len(text),
                    'embedding_dimensions': len(embedding)
                }
            )
            
            return embedding
            
        except Exception as e:
            self.logger.error(
                f"Failed to generate embedding: {str(e)}",
                extra={
                    'component': 'embedding_system',
                    'operation': 'generate_embedding',
                    'error_type': type(e).__name__,
                    'text_length': len(text) if text else 0
                },
                exc_info=True
            )
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}")
    
    def generate_embeddings_batch(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts (List[str]): List of texts to generate embeddings for.
            metadata (Optional[List[Dict[str, Any]]]): Optional metadata for each text.
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - embeddings: List of generated embeddings
                - processed_metadata: List of metadata with embedding info
                - stats: Statistics about the batch processing
        """
        if not texts:
            raise ValueError("Input texts list cannot be empty")
            
        # Initialize results
        embeddings = []
        processed_metadata = []
        
        # Initialize statistics
        stats = {
            'total_texts': len(texts),
            'successful_texts': 0,
            'failed_texts': 0,
            'total_tokens': 0,
            'batches': 0,
            'start_time': datetime.now().isoformat(),
            'errors': []
        }
        
        # Process texts in batches
        for i in range(0, len(texts), self.batch_size):
            batch_start_time = time.time()
            batch = texts[i:i + self.batch_size]
            batch_metadata = metadata[i:i + self.batch_size] if metadata else None
            
            try:
                # Log batch processing start
                self.logger.debug(
                    f"Processing batch {stats['batches'] + 1}",
                    extra={
                        'component': 'embedding_system',
                        'operation': 'batch_process',
                        'batch_size': len(batch),
                        'batch_number': stats['batches'] + 1
                    }
                )
                
                # Generate embeddings for the batch
                batch_results = [
                    self.model.embed_content(text)
                    for text in batch
                ]
                
                # Process batch results
                for j, result in enumerate(batch_results):
                    if not result or not result.embedding:
                        raise EmbeddingError(f"No embedding in response for text {i + j}")
                        
                    # Validate dimensions
                    if len(result.embedding) != self.dimensions:
                        raise EmbeddingError(
                            f"Unexpected embedding dimensions for text {i + j}: "
                            f"{len(result.embedding)} != {self.dimensions}"
                        )
                    
                    # Add embedding to results
                    embeddings.append(result.embedding)
                    
                    # Update metadata
                    if batch_metadata:
                        text_metadata = batch_metadata[j].copy()
                        text_metadata.update({
                            'embedding_generated': datetime.now().isoformat(),
                            'embedding_dimensions': len(result.embedding),
                            'text_length': len(batch[j])
                        })
                        processed_metadata.append(text_metadata)
                    
                    stats['successful_texts'] += 1
                    stats['total_tokens'] += len(batch[j].split())
                
                # Update batch statistics
                batch_duration = time.time() - batch_start_time
                stats['batches'] += 1
                
                # Log batch completion
                self.logger.debug(
                    f"Completed batch {stats['batches']}",
                    extra={
                        'component': 'embedding_system',
                        'operation': 'batch_process',
                        'batch_duration': batch_duration,
                        'successful_texts': len(batch),
                        'batch_number': stats['batches']
                    }
                )
                
            except Exception as e:
                # Log batch error
                error_msg = f"Batch {stats['batches'] + 1} failed: {str(e)}"
                self.logger.error(
                    error_msg,
                    extra={
                        'component': 'embedding_system',
                        'operation': 'batch_process',
                        'batch_number': stats['batches'] + 1,
                        'error_type': type(e).__name__
                    },
                    exc_info=True
                )
                
                # Update error statistics
                stats['failed_texts'] += len(batch)
                stats['errors'].append({
                    'batch': stats['batches'] + 1,
                    'error': str(e),
                    'error_type': type(e).__name__
                })
                
                # Add None for failed embeddings
                embeddings.extend([None] * len(batch))
                if batch_metadata:
                    for meta in batch_metadata:
                        meta_copy = meta.copy()
                        meta_copy['embedding_error'] = str(e)
                        processed_metadata.append(meta_copy)
        
        # Add completion time to stats
        stats['end_time'] = datetime.now().isoformat()
        stats['duration'] = (
            datetime.fromisoformat(stats['end_time']) -
            datetime.fromisoformat(stats['start_time'])
        ).total_seconds()
        
        # Log completion
        self.logger.info(
            "Batch processing completed",
            extra={
                'component': 'embedding_system',
                'operation': 'batch_process',
                'total_texts': stats['total_texts'],
                'successful_texts': stats['successful_texts'],
                'failed_texts': stats['failed_texts'],
                'duration': stats['duration']
            }
        )
        
        return {
            'embeddings': embeddings,
            'processed_metadata': processed_metadata if metadata else None,
            'stats': stats
        }
    
    def generate_embeddings(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate embeddings for a single text.
        
        Args:
            text (str): Text to generate embeddings for.
            metadata (Optional[Dict[str, Any]]): Optional metadata for the text.
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - embedding: Generated embedding vector
                - metadata: Enhanced metadata with embedding info
                - model_info: Information about the model used
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")
            
        try:
            # Log embedding generation start
            self.logger.debug(
                "Generating embedding for text",
                extra={
                    'component': 'embedding_system',
                    'operation': 'generate_embedding',
                    'text_length': len(text)
                }
            )
            
            # Generate embedding
            result = self.model.embed_content(text)
            
            if not result or not result.embedding:
                raise EmbeddingError("No embedding in response")
                
            # Validate dimensions
            if len(result.embedding) != self.dimensions:
                raise EmbeddingError(
                    f"Unexpected embedding dimensions: {len(result.embedding)} != {self.dimensions}"
                )
            
            # Process metadata
            if metadata:
                metadata = metadata.copy()
                metadata.update({
                    'embedding_generated': datetime.now().isoformat(),
                    'embedding_dimensions': len(result.embedding),
                    'text_length': len(text)
                })
            
            # Log successful generation
            self.logger.debug(
                "Successfully generated embedding",
                extra={
                    'component': 'embedding_system',
                    'operation': 'generate_embedding',
                    'embedding_dimensions': len(result.embedding)
                }
            )
            
            return {
                'embedding': result.embedding,
                'metadata': metadata,
                'model_info': {
                    'name': self.model_name,
                    'task_type': self.task_type,
                    'dimensions': self.dimensions
                }
            }
            
        except Exception as e:
            # Log error
            self.logger.error(
                f"Embedding generation failed: {str(e)}",
                extra={
                    'component': 'embedding_system',
                    'operation': 'generate_embedding',
                    'error_type': type(e).__name__,
                    'text_length': len(text)
                },
                exc_info=True
            )
            
            # Update metadata with error
            if metadata:
                metadata = metadata.copy()
                metadata.update({
                    'embedding_error': str(e),
                    'error_type': type(e).__name__
                })
            
            raise EmbeddingError(f"Embedding generation failed: {str(e)}") 