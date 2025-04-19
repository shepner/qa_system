"""
Embedding generation system for the QA system.

This module handles the generation of embeddings for both text and image content
using Google's models. It supports batch processing, concurrent operations,
and provides comprehensive validation and error handling.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import asyncio
from pathlib import Path
import numpy as np
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
import os
from .config import get_config, Config
import tenacity
from .vector_system import VectorStore  # Add VectorStore import

# Initialize logger with module name for better traceability
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generates vector embeddings for document chunks and processed image data using Google's models."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the embedding generator with configuration settings.
        
        Args:
            config_path: Optional path to config file. If not provided, uses default config path.
        """
        self.logger = logger
        
        # Get configuration from config.py
        self.config = get_config(config_path)
        
        # Initialize VectorStore
        self.vector_store = VectorStore(config_path)
        self.logger.info("Initialized VectorStore for embedding storage")
        
        # Get embedding model configuration
        embedding_config = self.config.get_nested('EMBEDDING_MODEL', {})
        self.model_name = embedding_config.get('MODEL_NAME', 'embedding-001')
        self.batch_size = embedding_config.get('BATCH_SIZE', 15)
        self.max_length = embedding_config.get('MAX_LENGTH', 3072)
        self.dimensions = embedding_config.get('DIMENSIONS', 768)

        # Get security configuration
        security_config = self.config.get_nested('SECURITY', {})
        api_key = security_config.get('GOOGLE_API_KEY')

        # Configure Google AI client
        try:
            genai.configure(
                api_key=api_key,
                transport="rest"
            )
            self.logger.info(f"Initialized EmbeddingGenerator with model {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to configure Google AI client: {str(e)}", exc_info=True)
            raise RuntimeError(f"Google AI client configuration failed: {str(e)}")

        # Initialize the embedding model
        try:
            self.model = genai.GenerativeModel(self.model_name)
            self.logger.debug("Successfully initialized embedding model")
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {str(e)}", exc_info=True)
            raise RuntimeError(f"Model initialization failed: {str(e)}")
        
        # Configure retry settings
        self.max_retries = embedding_config.get('MAX_RETRIES', 5)
        self.min_wait = embedding_config.get('MIN_WAIT', 4)
        self.max_wait = embedding_config.get('MAX_WAIT', 60)
        
    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=60),
        stop=tenacity.stop_after_attempt(5),
        retry=tenacity.retry_if_exception_type((
            ValueError,
            ConnectionError,
            TimeoutError,
            Exception
        )),
        before=tenacity.before_log(logging.getLogger(__name__), logging.DEBUG),
        after=tenacity.after_log(logging.getLogger(__name__), logging.DEBUG),
        reraise=True
    )
    def _generate_embedding_batch(
        self,
        chunks: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Generate embeddings for a batch of text chunks.
        
        Args:
            chunks: List of text chunks to generate embeddings for
            metadata: Optional metadata to include with each embedding
            
        Returns:
            List of dictionaries containing embeddings and metadata
            
        Raises:
            ValueError: If chunks are invalid
            RuntimeError: If embedding generation fails
        """
        try:
            self.logger.debug(f"Processing batch of {len(chunks)} chunks")
            start_time = datetime.utcnow()

            # Extract text content from chunks if they're dictionaries
            text_chunks = []
            for chunk in chunks:
                if isinstance(chunk, dict):
                    text_chunks.append(chunk.get('content', ''))
                else:
                    text_chunks.append(chunk)

            # Sanitize text chunks
            sanitized_chunks = []
            for chunk in text_chunks:
                # Remove or replace problematic characters
                sanitized = (
                    chunk.replace('\x00', '')  # Remove null bytes
                    .replace('®', '(R)')       # Replace registered trademark
                    .replace('™', '(TM)')      # Replace trademark
                    .replace('©', '(C)')       # Replace copyright
                    .encode('ascii', 'ignore').decode('ascii')  # Remove non-ASCII chars
                )
                # Normalize whitespace
                sanitized = ' '.join(sanitized.split())
                sanitized_chunks.append(sanitized)

            # Generate embeddings for the batch using the latest Gemini API
            embeddings = []
            for chunk in sanitized_chunks:
                if not chunk.strip():
                    self.logger.warning("Empty chunk detected after sanitization, skipping")
                    continue

                try:
                    # Use the latest embedding API with validation
                    response = genai.embed_content(
                        model=self.model_name,
                        content=chunk,
                        task_type="retrieval_document",  # Specify task type for optimal embeddings
                        title="document_chunk"  # Optional title for better context
                    )
                    
                    # Validate response format
                    if not response:
                        raise ValueError("Empty response from embedding API")
                    
                    # Handle both dictionary and object response formats
                    if isinstance(response, dict):
                        embedding = response.get('embedding')
                    else:
                        embedding = response.embedding if hasattr(response, 'embedding') else None
                        
                    if embedding is None:
                        self.logger.error(f"Unexpected response format: {response}")
                        raise ValueError("Response missing embedding data")
                    
                    # Validate embedding format and dimensions
                    if not isinstance(embedding, (list, np.ndarray)):
                        raise ValueError(f"Invalid embedding type: {type(embedding)}")
                        
                    if len(embedding) != self.dimensions:
                        raise ValueError(f"Invalid embedding dimensions: got {len(embedding)}, expected {self.dimensions}")
                    
                    embeddings.append(embedding)
                    
                except Exception as e:
                    self.logger.error(f"Error generating embedding for chunk: {str(e)}")
                    self.logger.debug(f"Problematic chunk (first 100 chars): {chunk[:100]}...")
                    raise RuntimeError(f"Embedding generation failed for chunk: {str(e)}")
            
            # Validate and process results
            results = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Create result dictionary with comprehensive metadata
                result = {
                    'chunk_number': i if not isinstance(chunk, dict) else chunk.get('chunk_number', i),
                    'chunk_text': chunk if not isinstance(chunk, dict) else chunk.get('content', ''),
                    'embedding': embedding,
                    'embedding_model': self.model_name,
                    'embedding_timestamp': datetime.utcnow().isoformat(),
                    'embedding_dimensions': len(embedding),
                    'processing_info': {
                        'sanitized': True,
                        'original_length': len(chunk if isinstance(chunk, str) else chunk.get('content', '')),
                        'sanitized_length': len(sanitized_chunks[i]),
                        'processing_time': (datetime.utcnow() - start_time).total_seconds()
                    }
                }
                
                # Add metadata if provided
                if metadata:
                    result['metadata'] = metadata.copy()
                    
                # Add any additional chunk metadata if chunk is a dictionary
                if isinstance(chunk, dict):
                    chunk_metadata = chunk.copy()
                    chunk_metadata.pop('content', None)  # Remove content as it's already in chunk_text
                    if 'metadata' in result:
                        result['metadata'].update(chunk_metadata)
                    else:
                        result['metadata'] = chunk_metadata
                    
                results.append(result)

            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            self.logger.info(f"Successfully generated embeddings for {len(chunks)} chunks in {processing_time:.2f} seconds")
            self.logger.debug(f"Batch processing metrics - Average time per chunk: {processing_time/len(chunks):.3f}s")
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings for batch: {str(e)}", exc_info=True)
            raise RuntimeError(f"Embedding generation failed: {str(e)}")
            
    async def generate_embeddings(
        self,
        chunks: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Generate embeddings for a list of text chunks.
        
        Args:
            chunks: List of text chunks to generate embeddings for
            metadata: Optional metadata to include with each embedding
            
        Returns:
            List of dictionaries containing:
            - chunk_number: Sequential number of the chunk
            - chunk_text: Original text content
            - embedding: Generated vector embedding
            - embedding_model: Name of the model used
            - embedding_timestamp: When the embedding was generated
            - embedding_dimensions: Size of the embedding vector
            - metadata: Optional metadata dictionary
            
        Raises:
            ValueError: If chunks are invalid
            RuntimeError: If embedding generation fails
        """
        if not chunks:
            self.logger.warning("No chunks provided for embedding generation")
            return []
            
        # Process chunks in batches
        results = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            self.logger.debug(f"Processing batch {i//self.batch_size + 1} of {(len(chunks)-1)//self.batch_size + 1}")
            
            # Since _generate_embedding_batch is no longer async, we need to run it in a thread pool
            batch_results = await asyncio.get_event_loop().run_in_executor(
                None, self._generate_embedding_batch, batch, metadata
            )
            results.extend(batch_results)
            
        return results
        
    async def generate_image_embeddings(
        self,
        processed_image_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate embeddings for processed image data.
        
        Args:
            processed_image_data: Dictionary containing processed image data and analysis
            metadata: Optional metadata to include with the embedding
            
        Returns:
            Dictionary containing:
            - embedding: Generated vector embedding
            - embedding_model: Name of the model used
            - embedding_timestamp: When the embedding was generated
            - embedding_dimensions: Size of the embedding vector
            - metadata: Optional metadata dictionary
            
        Raises:
            ValueError: If image data is invalid
            RuntimeError: If embedding generation fails
        """
        if not processed_image_data:
            raise ValueError("No image data provided")
            
        try:
            # Extract text content from image analysis
            text_content = []
            
            # Add OCR text if available
            if 'text_annotations' in processed_image_data:
                text_content.append(processed_image_data['text_annotations'])
                
            # Add detected labels with confidence scores
            if 'labels' in processed_image_data:
                text_content.extend(
                    f"{label['description']} ({label.get('confidence', 0):.2f})"
                    for label in processed_image_data['labels']
                )
                
            # Add detected objects with locations
            if 'objects' in processed_image_data:
                text_content.extend(
                    f"{obj['name']} at {obj.get('location', 'unknown position')}"
                    for obj in processed_image_data['objects']
                )
                
            # Combine all text content with structured formatting
            combined_text = (
                "Image Analysis Results:\n"
                f"OCR Text: {text_content[0] if text_content else 'None'}\n"
                f"Labels: {', '.join(text_content[1:])}"
            )
            
            # Generate embedding for the combined text using task-specific parameters
            response = genai.embed_content(
                model=self.model_name,
                content=combined_text,
                task_type="retrieval_document",
                title="image_analysis",  # Specify content type
            )
            
            if not response or not hasattr(response, 'embedding'):
                raise RuntimeError("Failed to generate embedding for image")
                
            # Create result with comprehensive metadata
            result = {
                'embedding': response.embedding,
                'embedding_model': self.model_name,
                'embedding_timestamp': datetime.utcnow().isoformat(),
                'embedding_dimensions': len(response.embedding),
                'content_type': 'image_analysis',
                'analysis_summary': {
                    'has_text': bool(processed_image_data.get('text_annotations')),
                    'label_count': len(processed_image_data.get('labels', [])),
                    'object_count': len(processed_image_data.get('objects', [])),
                }
            }
            
            # Add provided metadata
            if metadata:
                result['metadata'] = metadata
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating image embedding: {str(e)}")
            raise RuntimeError(f"Image embedding generation failed: {str(e)}")
            
    def prepare_for_vector_store(
        self,
        embeddings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Format embeddings and metadata for storage in the vector database.
        
        Args:
            embeddings: List of embedding results from generate_embeddings
            
        Returns:
            Dictionary containing:
            - embeddings: List of vector embeddings
            - metadata: List of corresponding metadata
            - ids: List of unique identifiers
            
        Raises:
            ValueError: If embeddings are invalid
        """
        if not embeddings:
            return {'embeddings': [], 'metadata': [], 'ids': []}
            
        try:
            # Separate embeddings and metadata
            vectors = []
            metadata_list = []
            ids = []
            
            for i, result in enumerate(embeddings):
                # Extract and validate the embedding vector
                if 'embedding' not in result:
                    raise ValueError(f"Missing embedding in result {i}")
                    
                embedding = result['embedding']
                if not isinstance(embedding, (list, np.ndarray)):
                    raise ValueError(f"Invalid embedding type in result {i}")
                    
                if len(embedding) != self.dimensions:
                    raise ValueError(f"Invalid embedding dimensions in result {i}: got {len(embedding)}, expected {self.dimensions}")
                    
                vectors.append(embedding)
                
                # Prepare comprehensive metadata
                metadata_entry = {
                    'chunk_number': result.get('chunk_number', i),
                    'chunk_text': result.get('chunk_text', ''),
                    'embedding_model': result.get('embedding_model', self.model_name),
                    'embedding_timestamp': result.get('embedding_timestamp', datetime.utcnow().isoformat()),
                    'embedding_dimensions': len(embedding),
                    'content_type': result.get('content_type', 'text'),  # Distinguish between text and image content
                    'processing_info': {
                        'batch_processed': True,
                        'processing_timestamp': datetime.utcnow().isoformat(),
                        'model_version': self.model_name.split('/')[-1],
                    }
                }
                
                # Add analysis summary for image content
                if result.get('content_type') == 'image_analysis':
                    metadata_entry['analysis_summary'] = result.get('analysis_summary', {})
                
                # Add any additional metadata
                if 'metadata' in result:
                    metadata_entry.update(result['metadata'])
                    
                metadata_list.append(metadata_entry)
                
                # Generate unique ID with timestamp for better tracking
                timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                unique_id = f"{metadata_entry.get('path', '')}_{metadata_entry['chunk_number']}_{timestamp}"
                ids.append(unique_id)
                
            self.logger.info(f"Prepared {len(vectors)} embeddings for vector store storage")
            self.logger.debug(f"Vector dimensions: {self.dimensions}, Metadata fields: {list(metadata_list[0].keys() if metadata_list else [])}")
            
            # Store embeddings in vector store
            self.vector_store.add_embeddings(
                embeddings=vectors,
                metadata=metadata_list,
                ids=ids
            )
            self.logger.info("Successfully stored embeddings in vector store")
                
            return {
                'embeddings': vectors,
                'metadata': metadata_list,
                'ids': ids
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing embeddings for storage: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to prepare embeddings for storage: {str(e)}") 