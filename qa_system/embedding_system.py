"""Embedding generation system using Google's Gemini models.

This module handles:
- Text embedding generation
- Image embedding generation
- Batch processing
- Error handling and retries
"""

import logging
from typing import Dict, Any, List, Optional, Literal, Union
import time
from datetime import datetime
import math
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from qa_system.config import get_config
from .exceptions import (
    EmbeddingError,
    ValidationError,
    handle_exception
)

# Define valid task types as per Gemini API
TaskType = Literal[
    'SEMANTIC_SIMILARITY',
    'CLASSIFICATION',
    'CLUSTERING',
    'RETRIEVAL_DOCUMENT',
    'RETRIEVAL_QUERY',
    'QUESTION_ANSWERING',
    'FACT_VERIFICATION',
    'CODE_RETRIEVAL_QUERY'
]

class EmbeddingGenerator:
    """Handles generation of embeddings using Google's Gemini models."""
    
    DEFAULT_TASK_TYPE: TaskType = 'RETRIEVAL_DOCUMENT'
    
    # Required metadata fields from Document Processors
    REQUIRED_DOC_METADATA = {
        'path',
        'file_type',
        'filename',
        'checksum'
    }
    
    # Required fields for image data from Vision Processor
    REQUIRED_IMAGE_FIELDS = {
        'text_content',
        'labels',
        'vision_analysis'
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the embedding generator.
        
        Args:
            config_path: Optional path to configuration file
            
        Raises:
            RuntimeError: If initialization fails
        """
        try:
            # Load configuration
            self.config = get_config(config_path)
            
            # Setup logging
            self.logger = logging.getLogger(__name__)
            
            # Get model configuration
            model_config = self.config.get_nested('EMBEDDING_MODEL', {})
            self.model_name = model_config.get('MODEL_NAME', 'gemini-embedding-exp-03-07')
            self.batch_size = model_config.get('BATCH_SIZE', 15)
            self.max_length = model_config.get('MAX_LENGTH', 3072)
            self.dimensions = model_config.get('DIMENSIONS', 768)
            
            # Initialize Gemini
            credentials = self.config.get_nested('SECURITY.GOOGLE_APPLICATION_CREDENTIALS')
            project_id = self.config.get_nested('SECURITY.GOOGLE_CLOUD_PROJECT')
            
            if not credentials or not project_id:
                raise RuntimeError("Missing required Google Cloud credentials")
                
            genai.configure(
                project_id=project_id,
                credentials=credentials
            )
            
            # Initialize model
            self.model = genai.GenerativeModel(self.model_name)
            
            self.logger.info(
                "Embedding generator initialized",
                extra={
                    'component': 'embedding_generator',
                    'model': self.model_name,
                    'dimensions': self.dimensions
                }
            )
            
        except Exception as e:
            error_details = handle_exception(
                e,
                "Failed to initialize embedding generator",
                reraise=False
            )
            raise RuntimeError(
                f"Embedding generator initialization failed: {error_details['message']}"
            ) from e
    
    def validate_metadata(self, metadata: Dict[str, Any], is_document: bool = True) -> None:
        """Validate metadata contains required fields.
        
        Args:
            metadata: Metadata dictionary to validate
            is_document: Whether this is document metadata (vs image)
            
        Raises:
            ValidationError: If metadata is invalid
        """
        if not isinstance(metadata, dict):
            raise ValidationError("Metadata must be a dictionary")
        
        required_fields = self.REQUIRED_DOC_METADATA if is_document else self.REQUIRED_IMAGE_FIELDS
        missing_fields = required_fields - set(metadata.keys())
        
        if missing_fields:
            raise ValidationError(
                f"Missing required metadata fields: {', '.join(missing_fields)}"
            )
    
    def normalize_text(self, text: Union[str, bytes, List[str]]) -> str:
        """Normalize text input to string format.
        
        Args:
            text: Input text in various formats
            
        Returns:
            Normalized string
            
        Raises:
            ValidationError: If text cannot be normalized
        """
        try:
            if isinstance(text, bytes):
                return text.decode('utf-8')
            elif isinstance(text, list):
                return ' '.join(str(item) for item in text)
            elif isinstance(text, str):
                return text
            else:
                raise ValidationError(
                    f"Unsupported text type: {type(text)}"
                )
        except Exception as e:
            raise ValidationError(
                f"Failed to normalize text: {str(e)}"
            )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def generate_embedding(
        self,
        text: Union[str, bytes, List[str]],
        task_type: Optional[TaskType] = None
    ) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to generate embedding for (string, bytes, or list of strings)
            task_type: Optional task type for optimizing embeddings
                      Defaults to RETRIEVAL_DOCUMENT
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            ValidationError: If text is empty or too long
            EmbeddingError: If embedding generation fails
        """
        try:
            # Normalize text input
            normalized_text = self.normalize_text(text)
            
            if not normalized_text:
                raise ValidationError("Text cannot be empty")
                
            # Truncate if needed
            if len(normalized_text) > self.max_length:
                self.logger.warning(
                    "Text exceeds maximum length, truncating",
                    extra={
                        'component': 'embedding_generator',
                        'text_length': len(normalized_text),
                        'max_length': self.max_length
                    }
                )
                normalized_text = normalized_text[:self.max_length]
            
            start_time = time.time()
            
            # Generate embedding with task type
            embedding = self.model.embed_content(
                normalized_text,
                task_type=task_type or self.DEFAULT_TASK_TYPE,
                dimensions=self.dimensions
            )
            
            duration = time.time() - start_time
            
            self.logger.debug(
                "Generated embedding",
                extra={
                    'component': 'embedding_generator',
                    'text_length': len(normalized_text),
                    'task_type': task_type or self.DEFAULT_TASK_TYPE,
                    'duration_seconds': duration
                }
            )
            
            return embedding.values
            
        except Exception as e:
            error_details = handle_exception(
                e,
                "Failed to generate embedding",
                reraise=False
            )
            raise EmbeddingError(
                f"Embedding generation failed: {error_details['message']}"
            ) from e
    
    def generate_batch_embeddings(
        self,
        texts: List[Union[str, bytes, List[str]]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        task_type: Optional[TaskType] = None
    ) -> Dict[str, Any]:
        """Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            metadata: Optional list of metadata dictionaries
            task_type: Optional task type for optimizing embeddings
                      Defaults to RETRIEVAL_DOCUMENT
            
        Returns:
            Dictionary containing:
            - embeddings: List of embedding vectors
            - metadata: Enhanced metadata list
            - model_info: Model information
            
        Raises:
            ValidationError: If input validation fails
            EmbeddingError: If batch processing fails
        """
        try:
            if not texts:
                raise ValidationError("Texts list cannot be empty")
                
            if metadata and len(texts) != len(metadata):
                raise ValidationError(
                    f"Number of texts ({len(texts)}) must match "
                    f"number of metadata entries ({len(metadata)})"
                )
            
            # Validate metadata if provided
            if metadata:
                for meta in metadata:
                    self.validate_metadata(meta)
            
            start_time = time.time()
            embeddings = []
            processed_metadata = []
            
            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                batch_embeddings = []
                
                for text in batch_texts:
                    embedding = self.generate_embedding(text, task_type)
                    batch_embeddings.append(embedding)
                
                # Validate batch embeddings
                self.validate_embeddings(batch_embeddings)
                embeddings.extend(batch_embeddings)
                
                # Update metadata
                if metadata:
                    batch_metadata = metadata[i:i + self.batch_size]
                    for j, meta in enumerate(batch_metadata):
                        meta.update({
                            'embedding_timestamp': datetime.now().isoformat(),
                            'embedding_model': self.model_name,
                            'embedding_dimensions': self.dimensions,
                            'task_type': task_type or self.DEFAULT_TASK_TYPE,
                            'text_length': len(self.normalize_text(batch_texts[j])),
                            'batch_index': i + j
                        })
                        processed_metadata.append(meta)
            
            duration = time.time() - start_time
            
            self.logger.info(
                "Batch embedding generation complete",
                extra={
                    'component': 'embedding_generator',
                    'text_count': len(texts),
                    'task_type': task_type or self.DEFAULT_TASK_TYPE,
                    'duration_seconds': duration
                }
            )
            
            # Prepare for vector store
            return self.prepare_for_vector_store(embeddings, metadata)
            
        except Exception as e:
            error_details = handle_exception(
                e,
                "Failed to generate batch embeddings",
                reraise=False
            )
            raise EmbeddingError(
                f"Batch embedding generation failed: {error_details['message']}"
            ) from e
    
    def generate_image_embeddings(
        self,
        image_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        task_type: Optional[TaskType] = None
    ) -> Dict[str, Any]:
        """Generate embeddings for image content.
        
        Args:
            image_data: Dictionary containing:
                - text_content: Text extracted from image
                - labels: Detected object labels
                - vision_analysis: Vision API analysis results
            metadata: Optional metadata dictionary
            task_type: Optional task type for optimizing embeddings
                      Defaults to RETRIEVAL_DOCUMENT
            
        Returns:
            Dictionary containing:
            - embeddings: List of embedding vectors
            - metadata: Enhanced metadata
            - model_info: Model information
            
        Raises:
            ValidationError: If input validation fails
            EmbeddingError: If embedding generation fails
        """
        try:
            if not image_data:
                raise ValidationError("Image data cannot be empty")
            
            # Validate image data fields
            self.validate_metadata(image_data, is_document=False)
            
            # Validate metadata if provided
            if metadata:
                self.validate_metadata(metadata)
            
            start_time = time.time()
            
            # Combine text content for embedding
            text_parts = []
            
            # Add OCR text if available
            if 'text_content' in image_data:
                text_parts.append(str(image_data['text_content']))
            
            # Add detected labels
            if 'labels' in image_data:
                text_parts.append(
                    "Image contains: " + ", ".join(str(label) for label in image_data['labels'])
                )
            
            # Add vision analysis summary if available
            if 'vision_analysis' in image_data:
                analysis = image_data['vision_analysis']
                if isinstance(analysis, dict):
                    for key, value in analysis.items():
                        if isinstance(value, (str, list)):
                            text_parts.append(f"{key}: {str(value)}")
            
            # Generate embedding for combined text
            combined_text = " ".join(text_parts)
            embedding = self.generate_embedding(combined_text, task_type)
            
            # Validate the embedding
            embeddings = [embedding]
            self.validate_embeddings(embeddings)
            
            # Update metadata
            if metadata:
                metadata.update({
                    'embedding_timestamp': datetime.now().isoformat(),
                    'embedding_model': self.model_name,
                    'embedding_dimensions': self.dimensions,
                    'task_type': task_type or self.DEFAULT_TASK_TYPE,
                    'content_type': 'image',
                    'text_length': len(combined_text)
                })
            
            duration = time.time() - start_time
            
            self.logger.info(
                "Image embedding generation complete",
                extra={
                    'component': 'embedding_generator',
                    'text_length': len(combined_text),
                    'task_type': task_type or self.DEFAULT_TASK_TYPE,
                    'duration_seconds': duration
                }
            )
            
            # Prepare for vector store
            return self.prepare_for_vector_store(embeddings, metadata)
            
        except Exception as e:
            error_details = handle_exception(
                e,
                "Failed to generate image embedding",
                reraise=False
            )
            raise EmbeddingError(
                f"Image embedding generation failed: {error_details['message']}"
            ) from e
    
    def validate_embeddings(
        self,
        embeddings: List[List[float]],
        expected_dimensions: Optional[int] = None
    ) -> bool:
        """Validates generated embeddings meet requirements.
        
        Args:
            embeddings: List of embedding vectors to validate
            expected_dimensions: Expected dimensionality of vectors (defaults to self.dimensions)
            
        Returns:
            bool: True if embeddings are valid
            
        Raises:
            ValidationError: If embeddings fail validation
        """
        try:
            if not embeddings:
                raise ValidationError("Embeddings list cannot be empty")
            
            expected_dimensions = expected_dimensions or self.dimensions
            
            for i, embedding in enumerate(embeddings):
                # Check if embedding is a list of floats
                if not isinstance(embedding, list) or not all(isinstance(x, float) for x in embedding):
                    raise ValidationError(
                        f"Embedding {i} is not a list of floats"
                    )
                
                # Validate dimensions
                if len(embedding) != expected_dimensions:
                    raise ValidationError(
                        f"Embedding {i} has incorrect dimensions: "
                        f"{len(embedding)} != {expected_dimensions}"
                    )
                
                # Check for NaN or Inf values
                if any(not isinstance(x, float) or math.isnan(x) or math.isinf(x) for x in embedding):
                    raise ValidationError(
                        f"Embedding {i} contains invalid values (NaN or Inf)"
                    )
            
            return True
            
        except Exception as e:
            error_details = handle_exception(
                e,
                "Failed to validate embeddings",
                reraise=False
            )
            raise ValidationError(
                f"Embedding validation failed: {error_details['message']}"
            ) from e
    
    def prepare_for_vector_store(
        self,
        embeddings: List[List[float]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Formats embeddings and metadata for vector store storage.
        
        Args:
            embeddings: List of embedding vectors
            metadata: Associated metadata
            
        Returns:
            Dictionary formatted for vector store insertion
            
        Raises:
            ValidationError: If preparation fails
        """
        try:
            # Validate embeddings first
            self.validate_embeddings(embeddings)
            
            # Generate unique IDs if needed
            embedding_ids = [
                f"emb_{i}_{int(time.time())}"
                for i in range(len(embeddings))
            ]
            
            # Prepare metadata
            processed_metadata = []
            for i in range(len(embeddings)):
                entry_metadata = {
                    'embedding_id': embedding_ids[i],
                    'embedding_index': i,
                    'embedding_timestamp': datetime.now().isoformat(),
                    'embedding_model': self.model_name,
                    'embedding_dimensions': self.dimensions
                }
                
                # Add provided metadata if available
                if metadata:
                    entry_metadata.update(metadata)
                
                processed_metadata.append(entry_metadata)
            
            return {
                'ids': embedding_ids,
                'embeddings': embeddings,
                'metadata': processed_metadata,
                'model_info': {
                    'name': self.model_name,
                    'dimensions': self.dimensions,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            error_details = handle_exception(
                e,
                "Failed to prepare embeddings for vector store",
                reraise=False
            )
            raise ValidationError(
                f"Vector store preparation failed: {error_details['message']}"
            ) from e 