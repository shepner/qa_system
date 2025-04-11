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
import asyncio

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
        self.batch_size = embedding_config.get("BATCH_SIZE", 3)  # Further reduced batch size
        
        # Get rate limit settings with much more conservative defaults
        rate_limits = embedding_config.get("RATE_LIMITS", {})
        self.max_requests_per_minute = rate_limits.get("MAX_REQUESTS_PER_MINUTE", 10)  # Reduced from 20
        self.max_concurrent = rate_limits.get("MAX_CONCURRENT_REQUESTS", 2)  # Reduced from 3
        self.backoff_factor = rate_limits.get("BACKOFF_FACTOR", 4)  # Increased from 2
        self.max_retries = rate_limits.get("MAX_RETRIES", 5)
        self.initial_delay = rate_limits.get("INITIAL_RETRY_DELAY", 5)  # Increased from 2
        self.max_delay = rate_limits.get("MAX_RETRY_DELAY", 300)  # Increased from 120
        
        # Calculate minimum delay between requests - add 20% buffer
        self.min_delay = (60.0 / self.max_requests_per_minute) * 1.2
        
        # Add request tracking
        self._last_request_time = 0
        self._request_times = []
        self._request_lock = asyncio.Lock()
        
        # Initialize Vertex AI with project and location
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = "us-central1"  # default location
        vertexai.init(project=project_id, location=location)
        
        # Initialize Gemini
        genai.configure(transport="rest")
        logger.info(f"Initialized embedding generator with model {self.model_name}")
        
    async def _wait_for_rate_limit(self):
        """Ensure we don't exceed rate limits by tracking request times."""
        async with self._request_lock:
            current_time = asyncio.get_event_loop().time()
            
            # Remove old request times
            cutoff = current_time - 60
            self._request_times = [t for t in self._request_times if t > cutoff]
            
            # If we've hit the limit, wait until we can make another request
            if len(self._request_times) >= self.max_requests_per_minute:
                wait_time = self._request_times[0] - cutoff + self.min_delay
                if wait_time > 0:
                    logger.info(f"Rate limit reached, waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                    
            # Add the new request time
            self._request_times.append(current_time)
            
            # Ensure minimum delay between requests
            if self._last_request_time:
                elapsed = current_time - self._last_request_time
                if elapsed < self.min_delay:
                    await asyncio.sleep(self.min_delay - elapsed)
            
            self._last_request_time = current_time

    async def process_text(self, text: str, task_type: str) -> np.ndarray:
        """Process a single text with rate limiting and retries."""
        async with self._request_lock:
            for retry in range(self.max_retries):
                try:
                    # Wait for rate limit before making request
                    await self._wait_for_rate_limit()
                    
                    # Truncate text if needed
                    if len(text) > self.max_length:
                        logger.warning(
                            f"Text length {len(text)} exceeds max_length {self.max_length}, truncating..."
                        )
                        text = text[:self.max_length]
                    
                    # Get embedding using the embedding model
                    result = genai.embed_content(
                        model=self.model_name,
                        content=text,
                        task_type=task_type
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
                    
                    return embedding_array
                    
                except Exception as e:
                    if retry < self.max_retries - 1:
                        delay = min(self.initial_delay * (self.backoff_factor ** retry), self.max_delay)
                        logger.warning(f"Retry {retry + 1}/{self.max_retries} after error: {str(e)}. Waiting {delay}s...")
                        await asyncio.sleep(delay)
                    else:
                        raise

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
            
            # Process in batches with rate limiting
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            async def process_batch(batch: List[str]) -> List[np.ndarray]:
                async with semaphore:
                    batch_embeddings = []
                    for text in batch:
                        embedding = await self.process_text(text, task_type)
                        batch_embeddings.append(embedding)
                    return batch_embeddings
            
            # Process texts in smaller batches
            tasks = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                tasks.append(process_batch(batch))
                
                # Add extra delay between batches
                if i + self.batch_size < len(texts):
                    await asyncio.sleep(self.min_delay * 3)  # Triple delay between batches
            
            # Wait for all embeddings with progress logging
            total_texts = len(texts)
            completed = 0
            for batch_embeddings in await asyncio.gather(*tasks):
                embeddings.extend(batch_embeddings)
                completed += len(batch_embeddings)
                if completed % 10 == 0 or completed == total_texts:
                    logger.info(f"Generated embeddings for {completed}/{total_texts} texts")
            
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