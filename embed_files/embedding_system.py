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

# Initialize logger with module name for better traceability
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generates vector embeddings for document chunks and processed image data using Google's models."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the embedding generator with configuration settings.
        
        Args:
            config: Configuration dictionary containing model settings and security credentials
        """
        self.config = config
        self.logger = logger

        # DEBUG: Log the entire config dictionary and its type
        self.logger.info("Received configuration:")
        self.logger.info(f"Config type: {type(config)}")
        self.logger.info(f"Config keys: {list(config.keys())}")
        self.logger.info(f"Full config: {config}")

        # Get embedding model configuration
        embedding_config = config.get('EMBEDDING_MODEL', {})
        self.logger.info(f"Embedding config: {embedding_config}")  # DEBUG
        self.model_name = embedding_config.get('MODEL_NAME', 'embedding-001')
        self.batch_size = embedding_config.get('BATCH_SIZE', 15)
        self.max_length = embedding_config.get('MAX_LENGTH', 3072)
        self.dimensions = embedding_config.get('DIMENSIONS', 768)

        # Extract security configuration
        security_config = config.get('SECURITY', {})
        self.logger.info(f"Security config: {security_config}")  # DEBUG
        
        # Try to get credentials directly from config if not in security section
        api_key = security_config.get('GOOGLE_API_KEY') or config.get('GOOGLE_API_KEY')
        project_id = security_config.get('GOOGLE_CLOUD_PROJECT') or config.get('GOOGLE_CLOUD_PROJECT')
        region = security_config.get('GOOGLE_CLOUD_REGION', 'us-central1')

        # DEBUG: Log credential information (without exposing actual API key)
        self.logger.info(f"API Key present: {bool(api_key)}")
        self.logger.info(f"Project ID: {project_id}")
        self.logger.info(f"Region: {region}")
        self.logger.info(f"Direct environment check - API Key present: {bool(os.getenv('GOOGLE_API_KEY'))}, Project ID present: {bool(os.getenv('GOOGLE_CLOUD_PROJECT'))}")  # DEBUG

        if not api_key or not project_id:
            self.logger.error("Missing required Google Cloud credentials")
            raise ValueError("Missing required Google Cloud credentials. Please set GOOGLE_API_KEY and GOOGLE_CLOUD_PROJECT.")

        # Configure Google AI client
        try:
            genai.configure(
                api_key=api_key,
                transport="rest",
                project_id=project_id,
                location=region
            )
            self.logger.info(f"Initialized EmbeddingGenerator with model {self.model_name} in project {project_id} (region: {region})")
        except Exception as e:
            self.logger.error(f"Failed to configure Google AI client: {str(e)}", exc_info=True)
            raise RuntimeError(f"Google AI client configuration failed: {str(e)}")

        # Initialize the model
        self.model = genai.GenerativeModel(self.model_name)
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _generate_embedding_batch(
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

            # Generate embeddings for the batch using the correct API method
            embeddings = await genai.embed_content(
                model=self.model_name,
                content=text_chunks,
                task_type="retrieval_document"
            )
            
            # Validate and process results
            results = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                if not isinstance(embedding, (list, np.ndarray)) or len(embedding) != self.dimensions:
                    self.logger.error(f"Invalid embedding generated for chunk {i}: wrong format or dimensions")
                    raise ValueError(f"Invalid embedding generated for chunk {i}")
                    
                # Create result dictionary
                result = {
                    'chunk_number': i if not isinstance(chunk, dict) else chunk.get('chunk_number', i),
                    'chunk_text': chunk if not isinstance(chunk, dict) else chunk.get('content', ''),
                    'embedding': embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                    'embedding_model': self.model_name,
                    'embedding_timestamp': datetime.utcnow().isoformat(),
                    'embedding_dimensions': self.dimensions
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
            
            batch_results = await self._generate_embedding_batch(batch, metadata)
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
                
            # Add detected labels
            if 'labels' in processed_image_data:
                text_content.extend(label['description'] for label in processed_image_data['labels'])
                
            # Add detected objects
            if 'objects' in processed_image_data:
                text_content.extend(obj['name'] for obj in processed_image_data['objects'])
                
            # Combine all text content
            combined_text = ' '.join(text_content)
            
            # Generate embedding for the combined text
            embedding_results = await self.generate_embeddings([combined_text], metadata)
            
            if not embedding_results:
                raise RuntimeError("Failed to generate embedding for image")
                
            return embedding_results[0]
            
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
                # Extract the embedding vector
                if 'embedding' not in result:
                    raise ValueError(f"Missing embedding in result {i}")
                vectors.append(result['embedding'])
                
                # Prepare metadata
                metadata_entry = {
                    'chunk_number': result.get('chunk_number', i),
                    'chunk_text': result.get('chunk_text', ''),
                    'embedding_model': result.get('embedding_model', self.model_name),
                    'embedding_timestamp': result.get('embedding_timestamp', datetime.utcnow().isoformat()),
                    'embedding_dimensions': result.get('embedding_dimensions', self.dimensions)
                }
                
                # Add any additional metadata
                if 'metadata' in result:
                    metadata_entry.update(result['metadata'])
                    
                metadata_list.append(metadata_entry)
                
                # Generate unique ID
                unique_id = f"{metadata_entry.get('path', '')}_{metadata_entry['chunk_number']}"
                ids.append(unique_id)
                
            return {
                'embeddings': vectors,
                'metadata': metadata_list,
                'ids': ids
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing embeddings for storage: {str(e)}")
            raise ValueError(f"Failed to prepare embeddings for storage: {str(e)}") 