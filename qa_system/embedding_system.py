"""
Embedding system for generating text embeddings using Google's Gemini model.

This module handles:
- Text embedding generation
- Batch processing
- Error handling and retries
- Token counting and validation
"""

import logging
from typing import List, Dict, Any, Optional
import time
from datetime import datetime
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from qa_system.config import get_config

class EmbeddingSystem:
    """Handles text embedding generation using Google's Gemini model."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the embedding system.
        
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
            self.model_name = model_config.get('MODEL_NAME', 'models/embedding-001')
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
                "Embedding system initialized",
                extra={
                    'component': 'embedding_system',
                    'model': self.model_name,
                    'dimensions': self.dimensions
                }
            )
            
        except Exception as e:
            self.logger.error(
                f"Embedding system initialization failed: {str(e)}",
                extra={
                    'component': 'embedding_system',
                    'error_type': type(e).__name__
                },
                exc_info=True
            )
            raise RuntimeError(f"Embedding system initialization failed: {str(e)}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of embedding values
            
        Raises:
            ValueError: If text is invalid
            RuntimeError: If embedding generation fails
        """
        try:
            if not text or not text.strip():
                raise ValueError("Empty text provided for embedding")
                
            start_time = time.time()
            
            # Generate embedding
            embedding = self.model.embed_content(
                text,
                task_type="retrieval_document",
                title="Document chunk"
            )
            
            duration = time.time() - start_time
            self.logger.debug(
                "Generated embedding",
                extra={
                    'component': 'embedding_system',
                    'operation': 'generate_embedding',
                    'text_length': len(text),
                    'duration_seconds': duration
                }
            )
            
            return embedding.values
            
        except Exception as e:
            self.logger.error(
                f"Error generating embedding: {str(e)}",
                extra={
                    'component': 'embedding_system',
                    'operation': 'generate_embedding',
                    'error_type': type(e).__name__,
                    'text_length': len(text)
                },
                exc_info=True
            )
            raise
    
    def generate_embeddings_batch(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to generate embeddings for
            metadata: Optional list of metadata dictionaries
            
        Returns:
            Dictionary containing:
            - embeddings: List of embedding vectors
            - metadata: Enhanced metadata with embedding info
            - stats: Processing statistics
            
        Raises:
            ValueError: If input validation fails
            RuntimeError: If batch processing fails
        """
        try:
            if not texts:
                raise ValueError("Empty text list provided")
                
            if metadata and len(metadata) != len(texts):
                raise ValueError("Metadata length must match texts length")
                
            start_time = time.time()
            embeddings = []
            enhanced_metadata = []
            stats = {
                'total_texts': len(texts),
                'successful': 0,
                'failed': 0,
                'total_tokens': 0,
                'start_time': datetime.utcnow().isoformat(),
                'batch_size': self.batch_size
            }
            
            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                batch_metadata = metadata[i:i + self.batch_size] if metadata else None
                
                for j, text in enumerate(batch_texts):
                    try:
                        # Generate embedding
                        embedding = self.generate_embedding(text)
                        embeddings.append(embedding)
                        
                        # Enhance metadata
                        meta = batch_metadata[j] if batch_metadata else {}
                        meta.update({
                            'embedding_generated_at': datetime.utcnow().isoformat(),
                            'text_length': len(text),
                            'embedding_model': self.model_name,
                            'embedding_dimensions': self.dimensions
                        })
                        enhanced_metadata.append(meta)
                        
                        stats['successful'] += 1
                        
                    except Exception as e:
                        stats['failed'] += 1
                        self.logger.error(
                            f"Error processing text in batch: {str(e)}",
                            extra={
                                'component': 'embedding_system',
                                'operation': 'generate_embeddings_batch',
                                'batch_index': i,
                                'text_index': j,
                                'error_type': type(e).__name__
                            }
                        )
                        # Add failed item metadata
                        if batch_metadata:
                            meta = batch_metadata[j]
                            meta['error'] = str(e)
                            enhanced_metadata.append(meta)
                        
            # Update final statistics
            stats.update({
                'end_time': datetime.utcnow().isoformat(),
                'duration_seconds': time.time() - start_time,
                'average_time_per_text': (time.time() - start_time) / len(texts)
            })
            
            self.logger.info(
                "Batch embedding generation complete",
                extra={
                    'component': 'embedding_system',
                    'operation': 'generate_embeddings_batch',
                    'stats': stats
                }
            )
            
            return {
                'embeddings': embeddings,
                'metadata': enhanced_metadata,
                'stats': stats
            }
            
        except Exception as e:
            self.logger.error(
                f"Batch embedding generation failed: {str(e)}",
                extra={
                    'component': 'embedding_system',
                    'operation': 'generate_embeddings_batch',
                    'error_type': type(e).__name__
                },
                exc_info=True
            )
            raise 