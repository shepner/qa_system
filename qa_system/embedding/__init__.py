"""
@file: embedding/__init__.py
Embedding generation utilities for the QA System.

This module provides a thread-safe, process-wide rate limiter and an EmbeddingGenerator class
for generating vector embeddings for document chunks using the Gemini API. It is designed to be
configurable, robust, and suitable for high-throughput embedding generation with proper rate limiting.

Classes:
    _RateLimiter: Thread-safe token bucket rate limiter for API call limiting.
    EmbeddingGenerator: Generates vector embeddings for text using Gemini API.

Exceptions:
    EmbeddingError: Raised for embedding generation failures.
    RateLimitError: Raised for rate limit errors.

Dependencies:
    - google-generativeai (google-genai)
    - Configuration object with get_nested method
    - GEMINI_API_KEY environment variable

Example usage:
    config = ...  # Your configuration object
    generator = EmbeddingGenerator(config)
    result = generator.generate_embeddings(["text1", "text2"], metadata={...})
"""

import logging
from typing import List, Dict, Any
import os
import threading
import time
from qa_system.exceptions import EmbeddingError, RateLimitError
from PIL import Image

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None  # Will error at runtime if used

class _RateLimiter:
    """
    Thread-safe token bucket rate limiter for process-wide API call limiting.
    Allows up to max_calls per period_seconds.
    
    Args:
        max_calls (int): Maximum number of calls allowed per period.
        period_seconds (float): Time window for rate limiting in seconds.
    """
    def __init__(self, max_calls: int, period_seconds: float):
        self.max_calls = max_calls
        self.period = period_seconds
        self._lock = threading.Lock()
        self._tokens = max_calls
        self._last_refill = time.monotonic()
        self._window_start = time.monotonic()
        self._request_count = 0
        self._total_requests = 0
        self._period_start = time.monotonic()

    def acquire(self):
        """
        Acquire a token for making an API call. Blocks if rate limit is exceeded.
        """
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                rate_per_sec = self.max_calls / self.period
                tokens_to_add = elapsed * rate_per_sec
                if tokens_to_add >= 1:
                    tokens_to_add_int = int(tokens_to_add)
                    self._tokens = min(self.max_calls, self._tokens + tokens_to_add_int)
                    self._last_refill += tokens_to_add_int / rate_per_sec
                if self._tokens > 0:
                    self._tokens -= 1
                    # --- Logging requests per minute ---
                    window_now = time.monotonic()
                    if window_now - self._window_start >= 60.0:
                        logging.info(f"[RateLimiter] Requests in last minute: {self._request_count}")
                        self._window_start = window_now
                        self._request_count = 0
                    self._request_count += 1
                    self._total_requests += 1
                    # Reset period if needed
                    if now - self._period_start >= self.period:
                        self._period_start = now
                        self._total_requests = 1  # This request
                    logging.debug(f"[RateLimiter] Request count this minute: {self._request_count}")
                    return
                # Not enough tokens, must wait
                sleep_time = max(0.01, self.period / self.max_calls)
            time.sleep(sleep_time)

    def get_period_request_count(self):
        """
        Return the number of requests submitted in the current rate limiter period.
        Returns:
            tuple: (request_count, period_seconds, period_start_time)
        """
        with self._lock:
            return self._total_requests, self.period, self._period_start

class EmbeddingGenerator:
    """
    EmbeddingGenerator generates vector embeddings for document chunks using the Gemini API.
    Integrates with configuration for model settings, batching, and rate limiting.

    Args:
        config: Configuration object (must support get_nested)

    Raises:
        RuntimeError: If GEMINI_API_KEY is not set in the environment.
        ImportError: If google-genai is not installed.
    """
    def __init__(self, config):
        """
        Initialize the EmbeddingGenerator with configuration.

        Args:
            config: Configuration object (must support get_nested)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Called EmbeddingGenerator.__init__(config={config})")
        self.config = config
        self.model_name = self.config.get_nested('EMBEDDING_MODEL.MODEL_NAME', 'embedding-001')
        self.batch_size = self.config.get_nested('EMBEDDING_MODEL.BATCH_SIZE', 32)
        self.max_length = self.config.get_nested('EMBEDDING_MODEL.MAX_LENGTH', 3072)
        self.dimensions = self.config.get_nested('EMBEDDING_MODEL.DIMENSIONS', 768)
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY not set in environment.")
        if genai is None:
            raise ImportError("google-genai is not installed. Please install the google-generativeai package.")
        self.client = genai.Client(api_key=self.gemini_api_key)

        # Rate limiter settings from config
        rate_limiter_cfg = self.config.get_nested('EMBEDDING_MODEL.EMBEDDING_RATE_LIMITER', {})
        max_calls = rate_limiter_cfg.get('MAX_CALLS', 500)
        period_seconds = rate_limiter_cfg.get('PERIOD_SECONDS', 60.0)
        self._embedding_rate_limiter = _RateLimiter(max_calls=max_calls, period_seconds=period_seconds)

    def report_rate_limiter_usage(self):
        """
        Log the number of requests submitted in the current rate limiter period.
        """
        count, period, period_start = self._embedding_rate_limiter.get_period_request_count()
        now = time.monotonic()
        elapsed = now - period_start
        self.logger.info(f"[EmbeddingGenerator] Requests in current period: {count} / {self._embedding_rate_limiter.max_calls} (period: {period:.1f}s, elapsed: {elapsed:.1f}s)")

    def generate_embeddings(self, texts: List[str], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate embeddings for a list of text chunks using Gemini API.

        Args:
            texts (List[str]): List of text chunks to embed.
            metadata (Dict[str, Any]): Metadata dictionary for the document.

        Returns:
            dict: Dictionary with keys: 'vectors', 'texts', 'metadata'.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        self.logger.debug(f"Called EmbeddingGenerator.generate_embeddings(texts=<len {len(texts)}>, metadata={metadata})")
        if not texts:
            self.logger.warning("No texts provided for embedding generation.")
            return {'vectors': [], 'texts': [], 'metadata': []}
        vectors = []
        batch_size = self.batch_size or 32
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                while True:
                    try:
                        # --- Rate limiting before each API call ---
                        start_wait = time.monotonic()
                        self._embedding_rate_limiter.acquire()
                        waited = time.monotonic() - start_wait
                        if waited > 0.01:
                            self.logger.info(f"Rate limiter: waited {waited:.3f}s before Gemini API call to stay under limit.")
                        # Gemini API expects a list of strings
                        result = self.client.models.embed_content(
                            model=self.model_name,
                            contents=batch,
                            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                        )
                        # result.embeddings is a list of Embedding objects, each with a .values attribute
                        # If only one input, result.embeddings may be a single Embedding object
                        if hasattr(result, 'embeddings'):
                            embeddings = result.embeddings
                            # If only one embedding, wrap in list
                            if not isinstance(embeddings, list):
                                embeddings = [embeddings]
                            for emb in embeddings:
                                # emb.values is the vector
                                vectors.append(list(emb.values))
                        else:
                            raise EmbeddingError("No embeddings returned from Gemini API.")
                        break  # Success, exit retry loop
                    except Exception as e:
                        # Check for 429/resource exhausted
                        msg = str(e)
                        if (hasattr(e, 'args') and e.args and 'RESOURCE_EXHAUSTED' in str(e.args[0])) or 'RESOURCE_EXHAUSTED' in msg or '429' in msg:
                            self.logger.warning("Rate limit hit (429/RESOURCE_EXHAUSTED). Sleeping for 60 seconds before retrying batch.")
                            time.sleep(60)
                            continue  # Retry the same batch
                        else:
                            raise  # Not a rate limit error, re-raise
                # After each batch, log rate limiter usage
                self.report_rate_limiter_usage()
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            raise EmbeddingError(f"Failed to generate embeddings: {e}")
        return {
            'vectors': vectors,
            'texts': texts,
            'metadata': [metadata] * len(texts)
        }

    def generate_image_embeddings(self, image_path: str, metadata: dict) -> dict:
        """
        Generate an embedding for an image using Gemini Pro Vision.

        Args:
            image_path (str): Path to the image file.
            metadata (dict): Metadata dictionary for the image.

        Returns:
            dict: Dictionary with keys: 'vectors', 'texts', 'metadata'.
        """
        self.logger.debug(f"Called EmbeddingGenerator.generate_image_embeddings(image_path={image_path}, metadata={metadata})")
        try:
            # Load image as PIL Image
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                model = genai.GenerativeModel('gemini-pro-vision')
                # Use a prompt to get a description (can be improved for your use case)
                response = model.generate_content([img, "Describe this image for retrieval/semantic search."])
                # Try to extract a vector if available (future-proofing)
                vector = getattr(response, 'embedding', None)
                if vector is not None:
                    # If Gemini returns a vector directly, use it
                    return {
                        'vectors': [vector],
                        'texts': ["[IMAGE EMBEDDING]"],
                        'metadata': [metadata]
                    }
                # Otherwise, fall back to using the generated text description
                text = getattr(response, 'text', None) or str(response)
                text_embedding = self.generate_embeddings([text], metadata)
                return {
                    'vectors': text_embedding['vectors'],
                    'texts': [text],
                    'metadata': [metadata]
                }
        except Exception as e:
            self.logger.error(f"Failed to generate image embedding: {e}")
            raise EmbeddingError(f"Failed to generate image embedding: {e}")
