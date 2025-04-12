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
import time

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Handles generation of embeddings using Google's Generative AI API."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the embedding generator with configuration."""
        self.config = config
        self._setup_logging()
        self._initialize_rate_limiter()
        
        # Track API usage statistics
        self.request_count = 0
        self.retry_count = 0
        self.total_tokens = 0
        self.start_time = None
        
        # Get embedding model configuration
        embedding_config = config.get("EMBEDDING_MODEL", {})
        if embedding_config.get("TYPE") != "gemini":
            raise ValueError("Embedding model type must be 'gemini'")
            
        self.model_name = embedding_config.get("MODEL_NAME", "models/gemini-embedding-exp-03-07")
        self.max_length = embedding_config.get("MAX_LENGTH", 8192)
        self.dimensions = embedding_config.get("DIMENSIONS", 768)
        self.batch_size = embedding_config.get("BATCH_SIZE", 3)  # Further reduced batch size
        
        # Initialize Vertex AI with project and location
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = "us-central1"  # default location
        vertexai.init(project=project_id, location=location)
        
        # Initialize Gemini
        genai.configure(transport="rest")
        logger.info(f"Initialized embedding generator with model {self.model_name}")
        
    def _setup_logging(self):
        """Configure detailed logging for API operations."""
        logger.info("Initializing EmbeddingGenerator with configuration")
        
    def _initialize_rate_limiter(self):
        """Initialize rate limiting parameters."""
        rate_limits = self.config.get("RATE_LIMITS", {})
        self.max_rpm = rate_limits.get("MAX_REQUESTS_PER_MINUTE", 60)  # Free tier: 60 RPM
        self.max_rpd = rate_limits.get("MAX_REQUESTS_PER_DAY", 1000)   # Free tier: 1000 RPD
        self.request_timestamps = []
        self.minute_window = []  # Track requests in current minute
        self.day_window = []     # Track requests in current day
        self.last_request_time = 0  # Track time of last request
        logger.info(f"Rate limits configured - RPM: {self.max_rpm}, RPD: {self.max_rpd}")
        
    async def _wait_for_rate_limit(self):
        """Wait if necessary to comply with rate limits.
        
        Implements intelligent rate limiting with rolling windows for both minute and day limits.
        """
        current_time = time.time()
        
        # Clean up expired timestamps
        minute_ago = current_time - 60
        day_ago = current_time - 86400
        
        self.minute_window = [t for t in self.minute_window if t > minute_ago]
        self.day_window = [t for t in self.day_window if t > day_ago]
        
        # Calculate current usage
        rpm_usage = len(self.minute_window)
        rpd_usage = len(self.day_window)
        
        # Log current usage every 5 requests
        if rpm_usage % 5 == 0:
            logger.info(f"Current usage - RPM: {rpm_usage}/{self.max_rpm}, RPD: {rpd_usage}/{self.max_rpd}")
        
        # Calculate required delays
        rpm_delay = 0
        rpd_delay = 0
        
        # RPM limit check (start throttling at 80%)
        if rpm_usage >= int(0.8 * self.max_rpm):
            required_gap = 60.0 / self.max_rpm  # Minimum time between requests
            if self.minute_window:
                time_since_last = current_time - self.minute_window[-1]
                rpm_delay = max(0, required_gap - time_since_last)
        
        # RPD limit check (aggressive throttling at 95%)
        if rpd_usage >= int(0.95 * self.max_rpd):
            seconds_left_in_day = 86400 - (current_time - day_ago)
            requests_left = self.max_rpd - rpd_usage
            if requests_left > 0:
                rpd_delay = seconds_left_in_day / requests_left
            else:
                logger.warning("Daily quota nearly exhausted, implementing aggressive throttling")
                rpd_delay = 300  # 5 minute delay when near quota
        
        # Use the longer of the two delays
        delay = max(rpm_delay, rpd_delay)
        
        if delay > 0:
            logger.info(f"Rate limit delay: {delay:.2f}s (RPM delay: {rpm_delay:.2f}s, RPD delay: {rpd_delay:.2f}s)")
            await asyncio.sleep(delay)
        
        # Record the request
        current_time = time.time()
        self.minute_window.append(current_time)
        self.day_window.append(current_time)
        self.last_request_time = current_time
        
    async def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        if not self.start_time:
            self.start_time = time.time()
            
        # Track request
        self.request_count += 1
        request_start = time.time()
        
        try:
            # Wait for rate limit if needed
            await self._wait_for_rate_limit()
            
            # Generate embedding
            embedding = await self._generate_embedding_with_retries(text)
            
            # Log success metrics
            request_time = time.time() - request_start
            total_time = time.time() - self.start_time
            avg_rate = self.request_count / total_time if total_time > 0 else 0
            
            logger.info(
                f"Embedding generated successfully "
                f"[Request: {self.request_count}, "
                f"Time: {request_time:.2f}s, "
                f"Avg Rate: {avg_rate:.1f} req/s]"
            )
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise
            
    async def _generate_embedding_with_retries(self, text: str) -> np.ndarray:
        """Generate embedding with retry logic."""
        max_retries = self.config.get("MAX_RETRIES", 3)
        base_delay = self.config.get("INITIAL_RETRY_DELAY", 1.0)
        max_delay = self.config.get("MAX_RETRY_DELAY", 60.0)
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    logger.warning(f"Retry attempt {attempt}/{max_retries} after {delay:.1f}s delay")
                    await asyncio.sleep(delay)
                    self.retry_count += 1
                
                return await self._make_embedding_request(text)
                
            except Exception as e:
                if attempt == max_retries:
                    logger.error(
                        f"Failed after {max_retries} retries. "
                        f"Total retries: {self.retry_count}"
                    )
                    raise
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                
    async def _make_embedding_request(self, text: str) -> np.ndarray:
        """Make the actual API request for embedding generation."""
        # Add token counting
        token_count = len(text.split())  # Simple approximation
        self.total_tokens += token_count
        
        logger.debug(
            f"Making embedding request "
            f"[Tokens: {token_count}, "
            f"Total Tokens: {self.total_tokens}]"
        )
        
        # Log text length and truncation if needed
        text_length = len(text)
        if text_length > self.max_length:
            logger.info(
                f"Text length {text_length} exceeds max_length {self.max_length}, truncating..."
            )
            text = text[:self.max_length]
        
        logger.debug(f"Generating embedding for text of length {len(text)}")
        
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
        
        logger.debug(f"Successfully generated embedding with dimension {self.dimensions}")
        return embedding_array

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
            total_texts = len(texts)
            logger.info(f"Starting embedding generation for {total_texts} texts")
            logger.info(f"Using batch size of {self.batch_size}, max {self.max_rpm} concurrent requests")
            
            # Process in batches with rate limiting
            semaphore = asyncio.Semaphore(self.max_rpm)
            
            async def process_batch(batch: List[str], batch_num: int) -> List[np.ndarray]:
                async with semaphore:
                    batch_embeddings = []
                    batch_size = len(batch)
                    logger.info(f"Processing batch {batch_num}, size {batch_size}")
                    
                    for i, text in enumerate(batch, 1):
                        start_time = asyncio.get_event_loop().time()
                        embedding = await self.generate_embedding(text)
                        end_time = asyncio.get_event_loop().time()
                        
                        batch_embeddings.append(embedding)
                        logger.info(
                            f"Batch {batch_num}: {i}/{batch_size} complete "
                            f"(took {end_time - start_time:.2f}s)"
                        )
                    
                    return batch_embeddings
            
            # Process texts in smaller batches
            tasks = []
            for batch_num, i in enumerate(range(0, len(texts), self.batch_size), 1):
                batch = texts[i:i + self.batch_size]
                tasks.append(process_batch(batch, batch_num))
                
                # Add extra delay between batches
                if i + self.batch_size < len(texts):
                    delay = 1  # Single delay between batches
                    logger.info(f"Waiting {delay:.2f}s before starting next batch...")
                    await asyncio.sleep(delay)
            
            # Wait for all embeddings with progress logging
            total_batches = len(tasks)
            completed = 0
            start_time = asyncio.get_event_loop().time()
            
            for batch_embeddings in await asyncio.gather(*tasks):
                embeddings.extend(batch_embeddings)
                completed += len(batch_embeddings)
                elapsed = asyncio.get_event_loop().time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                
                logger.info(
                    f"Progress: {completed}/{total_texts} texts embedded "
                    f"({(completed/total_texts)*100:.1f}%) "
                    f"[{elapsed:.1f}s elapsed, {rate:.1f} texts/s]"
                )
            
            total_time = asyncio.get_event_loop().time() - start_time
            logger.info(
                f"Embedding generation complete: {total_texts} texts in {total_time:.1f}s "
                f"({total_texts/total_time:.1f} texts/s average)"
            )
            
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