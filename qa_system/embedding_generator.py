"""
Embedding generation using Google's Gemini API
"""
import logging
import os
from typing import List, Dict, Any, Optional
import numpy as np
import google.generativeai as genai
from vertexai import generative_models
import vertexai

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generates embeddings for text using Google's Gemini API."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the embedding generator.
        
        Args:
            config: Configuration dictionary containing embedding model settings
        """
        self.config = config
        
        # Get embedding model configuration
        embedding_config = config.get("EMBEDDING_MODEL", {})
        if embedding_config.get("TYPE") != "gemini":
            raise ValueError("Embedding model type must be 'gemini'")
            
        self.model_name = embedding_config.get("MODEL_NAME", "models/gemini-embedding-exp-03-07")
        self.max_length = embedding_config.get("MAX_LENGTH", 8192)
        self.dimensions = embedding_config.get("DIMENSIONS", 768)
        self.batch_size = embedding_config.get("BATCH_SIZE", 10)
        
        # Initialize Vertex AI with project and location
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = "us-central1"  # default location
        vertexai.init(project=project_id, location=location)
        
        # Initialize Gemini
        genai.configure(transport="rest")
        logger.info(f"Initialized embedding generator with model {self.model_name}")
        
    async def generate_embeddings(
        self, 
        texts: List[str],
        task_type: str = "retrieval_document"
    ) -> List[np.ndarray]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            task_type: Type of embedding to generate ("retrieval_document" or "retrieval_query")
            
        Returns:
            List of embeddings as numpy arrays
        """
        try:
            embeddings = []
            
            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                
                # Truncate texts if needed
                processed_batch = []
                for text in batch:
                    if len(text) > self.max_length:
                        logger.warning(
                            f"Text length {len(text)} exceeds max_length {self.max_length}, truncating..."
                        )
                        processed_batch.append(text[:self.max_length])
                    else:
                        processed_batch.append(text)
                
                # Generate embeddings for batch
                batch_embeddings = []
                for text in processed_batch:
                    # Get embedding using the embedding model
                    result = genai.embed_content(
                        model=self.model_name,
                        content=text,
                        task_type="retrieval_document"
                    )
                    
                    # Extract embedding values from the response
                    if isinstance(result, dict):
                        embedding_values = result.get('embedding', [])
                    else:
                        embedding_values = result
                    
                    # Convert to numpy array
                    embedding_array = np.array(embedding_values, dtype=np.float32)
                    
                    # Verify embedding dimensions
                    if embedding_array.shape[0] != self.dimensions:
                        raise ValueError(
                            f"Embedding dimension mismatch. Expected {self.dimensions}, "
                            f"got {embedding_array.shape[0]}"
                        )
                    
                    batch_embeddings.append(embedding_array)
                
                embeddings.extend(batch_embeddings)
                
                logger.debug(f"Generated embeddings for batch of {len(batch)} texts")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise
            
    async def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a query string.
        
        Args:
            query: Query text to generate embedding for
            
        Returns:
            Query embedding as numpy array
        """
        try:
            embeddings = await self.generate_embeddings(
                texts=[query],
                task_type="retrieval_query"
            )
            return embeddings[0]
            
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {str(e)}")
            raise
            
    async def generate_document_embeddings(
        self,
        chunks: List[Any]
    ) -> List[Dict[str, Any]]:
        """Generate embeddings for document chunks.
        
        Args:
            chunks: List of document chunks (either strings or dicts with 'text' field)
            
        Returns:
            List of chunks with embeddings added
        """
        try:
            # Extract texts, handling both string and dict chunks
            texts = []
            for chunk in chunks:
                if isinstance(chunk, str):
                    texts.append(chunk)
                elif isinstance(chunk, dict) and "text" in chunk:
                    texts.append(chunk["text"])
                else:
                    raise ValueError(f"Invalid chunk format: {type(chunk)}. Expected string or dict with 'text' key.")
            
            # Generate embeddings
            embeddings = await self.generate_embeddings(
                texts=texts,
                task_type="retrieval_document"
            )
            
            # Add embeddings to chunks
            result = []
            for chunk, embedding in zip(chunks, embeddings):
                if isinstance(chunk, str):
                    # Create new chunk dict for string chunks
                    result.append({
                        "text": chunk,
                        "embedding": embedding
                    })
                else:
                    # Update existing chunk dict
                    chunk["embedding"] = embedding
                    result.append(chunk)
                
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate document embeddings: {str(e)}")
            raise 